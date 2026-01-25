
import random
import numpy as np
import torch
import os
import pkbar
import time
from defs import get_device
import math


from client.dataset_factories import datasets
from client.client import Client

from utils.gns_utils import GNSEstimator, compute_exact_gns

from compressors.countsketch import CountSketchReceiver, CountSketchSender

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def get_suffix(args):
    shortcut = {'sequential': 'seq', 'label_per_client': 'lpc', 'iid': 'iid'}
    return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        args['dataset'], 
        args['net'], 
        args['compression_scheme'], 
        args['rounds'], 
        shortcut[args['data_per_client']], 
        args['clients'], 
        args['batch_size'], 
        args['lr'], 
        args['lr_type'], 
        args['steps'], 
        str(args['nbits']).replace('.', '')
    )

if __name__ == '__main__':

    ############################
    #######   Arguments   ######
    ############################ 

    seed = 123                                  # random seed for reproducibility
    gpu = 0                                     # GPU ID to use (0 for first GPU, -1 for CPU)
    num_rounds = 6000                           # number of communication rounds
    num_clients = 8                           # number of clients
    test_every = 50                            # test every X rounds
    lr = 0.03                                   # learning rate for the model
    lr_type = 'step_decay'                           # learning rate type ['const', 'step_decay', 'exp_decay']
    client_train_steps = 1                      # local training steps per client
    client_batch_size = 32                     # Batch size of a client (for both train and test)
    net = 'ComEffFlPaperCnnModel'                             # CNN model to use
    dataset = 'MNIST'                         # dataset to use
    error_feedback = False                      # -- to be implemented --
    nbits = 1.0                                 # Number of bits per coordinate for compression scheme
    compression_scheme = 'none'    # compression/decompression scheme ['none', 'vector_topk', 'chunk_topk_recompress', 'chunk_topk_single', 
                                                #                                   'csh', 'cshtopk_actual', 'cshtopk_estimate']
    sketch_col = 180000                         # number of columns for the sketch matrix
    sketch_row = 1                              # number of rows for the sketch matrix
    k = 25000                                   # top-k k value for any compression scheme  
    data_per_client = 'iid'              # data distribution scheme ['sequential', 'label_per_client', 'iid']
    folder = 'ringallreduce'                    # folder to save the results


    
    args_for_suffix = {
        'rounds': num_rounds,
        'dataset': dataset,
        'net': net,
        'compression_scheme': compression_scheme,
        'data_per_client': data_per_client,
        'clients': num_clients,
        'batch_size': client_batch_size,
        'lr': lr,
        'lr_type': lr_type,
        'steps': client_train_steps,
        'nbits': nbits
    }



    print("Running ring allreduce simulation with {} over {} clients with {} dataset ({} policy) and {} CNN model with learning rate of {} over {} rounds".format(compression_scheme, num_clients, dataset, data_per_client, net, lr, num_rounds))

    ##################################################
    ##################################################

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    ##################################################
    ##################################################

    device, device_ids = get_device(gpu)

    global_sketch_compressor = CountSketchSender(device=device)
    global_sketch_decompressor = CountSketchReceiver(device=device)

    best_acc = 0 # best test accuracy
    start_round = 0 

    ##################################################
    ################ Preparing Data ##################
    ##################################################

    print('==> Preparing data, this might take a while...')

    dataloader = datasets(dataset, num_clients, params={'data_per_client': data_per_client})

    ##################################################
    ################ Initialise Clients ##############
    ##################################################

    print('==> Initializing ...')

    clients = {}


    for client_id in range(num_clients):
        clients[client_id] = Client(client_id, model=net, lr=lr, lr_type=lr_type, dataloader=dataloader, device=device, device_ids=device_ids, 
                                    train_batch_size=client_batch_size, test_batch_size=client_batch_size,
                                    compression_scheme=compression_scheme, nbits=nbits,
                                    error_feedback=error_feedback, seed=seed, num_clients=num_clients,
                                    testset=dataloader.get_test_data(), k_value = k, sketch_col=sketch_col, sketch_row=sketch_row)
    
    # NEW: Synchronize initial weights
    print("==> Synchronizing client weights to ensure identical start...")
    w_0 = clients[0].net.state_dict()
    for client_id in range(1, num_clients):
        clients[client_id].net.load_state_dict(w_0)

    if compression_scheme == 'none':
        b = 32
    else:
        b = nbits
    
    bytes_per_client_per_round = (clients[0].pytorch_total_params * b)/8

    print("*** Number of params: {}".format(clients[0].pytorch_total_params))
    print ("*** Data Transfer Info: sending {} in each round per client, and in total of {} till the end of the training "
           "aggregated for all clients.".format(convert_size(bytes_per_client_per_round),
                                                convert_size(bytes_per_client_per_round*num_clients*num_rounds)))
    
    results = {
        'rounds': [],
        'time': [],
        'trainACCs': [],
        'testACCs': [],
        'cumulative_bandwidth': [],
        'cumulative_latency': [],
        'latency': [],
        'scatter_total_sent': [],
        'gather_total_sent': [],
        'G2_estimate': [],
        'S_estimate': [],
        'GNS_estimate': [],
        'L_Norm': [],
        'G_Norm': [],
        'Norms_ratio': []
    }

    if not os.path.isdir('results'):
        os.mkdir('results')

    if not os.path.isdir('results/{}'.format(folder)):
        os.mkdir('results/{}'.format(folder))

    ##################################################
    ##################################################
    # Initial initialisation of global gradients

    if compression_scheme in ['csh', 'cshtopk_actual', 'cshtopk_estimate']:
        zeros_tensor = torch.zeros(clients[0].pytorch_total_params, device=device)
        data = {}
        data['vec'] = zeros_tensor

        data['r'] = sketch_row
        data['c'] = sketch_col
        data['d'] = int(zeros_tensor.numel())

        global_sketch =  global_sketch_compressor.compress(data=data)
        global_gradients = list(torch.chunk(global_sketch['vec'], num_clients))
    else:
        zeros_tensor = torch.zeros(int((clients[0].pytorch_total_params)), device=device)
        global_gradients = list(torch.chunk(zeros_tensor, num_clients))

    ##################################################
    ##################################################
    
    clients_per_round = range(num_clients)
        
    total_time = 0
    model_test_accuracy = 0

    kbar = pkbar.Kbar(target=start_round + num_rounds, width=50, always_stateful=True)


    train_loss = []
    train_correct = []
    train_total = []
    train_acc_per_client = []
    client_train_total_per_client = [[]] * num_clients
    client_train_correct_per_client = [[]] * num_clients

    total_data_sent_scatter = []
    total_data_sent_gather = []
    max_data_sent = []

    cumulative_bandwidth = 0
    cumulative_latency = 0

    # Initialize GNS Estimator
    gns_est = GNSEstimator(ema_decay=0.999)
    # --------------------------


    for round in range(start_round + 1, start_round + 1 + num_rounds):
        scatter_data_sent = 0
        scatter_sending_rounds = 0
        
        max_data = 0
        
        gather_data_sent = 0
        gather_sending_rounds = 0

        start_time = time.time()

        updated_param = torch.cat(global_gradients) if isinstance(global_gradients, list) else global_gradients

        # Update global gradients with the updated parameters
        if compression_scheme in ['csh', 'cshtopk_actual', 'cshtopk_estimate']:
            zeros_tensor = torch.zeros(clients[0].pytorch_total_params, device=device)
            data = {}
            data['vec'] = zeros_tensor

            data['r'] = sketch_row
            data['c'] = sketch_col
            data['d'] = int(zeros_tensor.numel())

            global_sketch =  global_sketch_compressor.compress(data=data)
            global_gradients = list(torch.chunk(global_sketch['vec'], num_clients))
        else:
            zeros_tensor = torch.zeros(clients[0].pytorch_total_params, device=device)
            global_gradients = list(torch.chunk(zeros_tensor, num_clients))

        state_0 = clients[0].net.state_dict()
        for client_id in clients_per_round:
            if client_id != 0:
                clients[client_id].net.load_state_dict(state_0)

        # Update client networks and perform a training step
        for client_id in clients_per_round:
            clients[client_id].update_net(updated_param)
            clients[client_id].train(round, client_train_steps)


        # Reduce-Scatter (+ AllGather)
        for scatter_round in range(num_clients):
            for chunk in global_gradients:
                data = np.count_nonzero(chunk.cpu())
                scatter_data_sent += data
                max_data = max(max_data, data)

            total_data_sent_scatter.append(scatter_data_sent)
            max_data_sent.append(max_data)
            scatter_sending_rounds += 1

            for client_id in clients_per_round:
                chunk_index = (client_id - scatter_round) % num_clients
                gradient_chunk = clients[client_id].get_gradient_chunk(chunk_index)
                global_gradients[chunk_index] += gradient_chunk
        
        # AllGather Data Sent
        for chunk in global_gradients:
            data = np.count_nonzero(chunk.cpu()) * (num_clients-1)
            gather_data_sent += data
            max_data = max(max_data, data)
        
        


        ########################################################
        ########################################################

        # Variable to hold the true global norm for GNS
        true_global_sq_norm = 0.0

        # Special handling for the 'cshtopk_actual' compression scheme

        if compression_scheme == 'cshtopk_actual':
            global_gradients = torch.cat(global_gradients)

            # Capture Norm of the Global Sketch
            true_global_sq_norm = (torch.linalg.vector_norm(global_gradients) ** 2).item()

            global_sketch['vec'] = global_gradients
            decompressed_sketch = global_sketch_decompressor.decompress(global_sketch)

            _, top_k_indices = torch.topk(decompressed_sketch.abs(), k=k)
            top_k_gradients = torch.zeros(clients[0].pytorch_total_params, device=device)
            
            for client_id in clients:
                top_k_gradients[top_k_indices] += \
                    clients[client_id].get_gradients_from_indices(top_k_indices)
            
            gather_data_sent += k * (num_clients-1)

            
            global_gradients = list(torch.chunk(top_k_gradients,num_clients))

            for chunk in global_gradients:
                data = np.count_nonzero(chunk.cpu())
                gather_data_sent += data
                max_data = max(max_data, data)

        # Special handling for the 'cshtopk_estimate' compression scheme

        if compression_scheme == 'cshtopk_estimate':
            global_gradients = torch.cat(global_gradients)

            # Capture Norm of the Global Sketch
            true_global_sq_norm = (torch.linalg.vector_norm(global_gradients) ** 2).item()

            global_sketch['vec'] = global_gradients
            decompressed_sketch = global_sketch_decompressor.decompress(global_sketch)
            
            _, top_k_indices = torch.topk(decompressed_sketch.abs(), k=k)
            top_k_values, _ = torch.topk(decompressed_sketch, k=k)

            top_k_gradients = torch.zeros(clients[0].pytorch_total_params, device=device)

            top_k_gradients[top_k_indices] = top_k_values

            gather_data_sent += np.count_nonzero(top_k_gradients.cpu()) * (num_clients-1)
            
            global_gradients = top_k_gradients
        
        ########################################################
        ########################################################

        
        total_data_sent_gather.append(gather_data_sent)
        max_data_sent.append(max_data)
        gather_sending_rounds += num_clients-1

        #### Gradient Noise Scale (B_simple) Estimator ####

        local_sq_norms = []
        
        for c_id in clients_per_round:
            raw_grad = clients[c_id].get_gradient()

            # CASE 1: Count Sketch (Dictionary)
            if isinstance(raw_grad, dict):
                # Use the sketch vector. 
                # For cshtopk, this matches 'true_global_sq_norm' (both are sketches).
                g_vec = raw_grad['vec']
                local_sq_norms.append(torch.linalg.vector_norm(g_vec) ** 2)

            # CASE 2: Chunk TopK (Dense -> needs Compression)
            elif 'chunk_topk' in compression_scheme:
                # Local is Dense, Global is Sparse. Must compress Local to match.
                # We perform a "simulation" of the compression on the dense grad 
                # to get the "Local Sparse Norm".
                dense_vec = raw_grad
                
                # Determine k (assuming args_for_suffix logic or global k)
                # k was defined in main scope
                
                # Fast local TopK norm:
                # We don't need indices, just the values of the top k elements.
                # Note: chunk_topk distributes k across chunks, but sum of topk(chunks) 
                # is roughly topk(full). For simplicity/speed, we topk the full vector.
                top_k_values, _ = torch.topk(dense_vec.abs(), k=k)
                local_sq_norms.append(torch.norm(top_k_values)**2)

            # CASE 3: Standard / Vector TopK
            else:
                # vector_topk returns sparse, none returns dense.
                # Both match their global counterparts naturally.
                local_sq_norms.append(torch.linalg.vector_norm(raw_grad) ** 2)
        
        avg_small_sq_norm = torch.stack(local_sq_norms).mean().item()


        # Compute Effective Large-Batch Gradient Norm
        # Check for cshtopk modes
        if true_global_sq_norm > 0.0:
            # true_global_sq_norm is ||Sum(g_i)||^2.
            # We need ||Mean(g_i)||^2 = ||Sum(g_i) / N||^2 = ||Sum||^2 / N^2.
            big_sq_norm = true_global_sq_norm / (num_clients ** 2)
        else:
            # For other schemes, global_gradients is correct (Dense or Sparse)
            if isinstance(global_gradients, list):
                global_grad_vec = torch.cat(global_gradients)
            else:
                global_grad_vec = global_gradients

            # Average over clients to get the Mean Gradient
            global_grad_mean = global_grad_vec / num_clients
            big_sq_norm = (torch.linalg.vector_norm(global_grad_mean) ** 2).item()


        # Update Estimator
        gns_est.update(
            avg_small_sq_norm, 
            big_sq_norm, 
            B_small=client_batch_size, 
            B_big=client_batch_size * num_clients
        )
        
        smoothed_GNS, running_G2, running_S = gns_est.get_stats()

        results['GNS_estimate'].append(smoothed_GNS)
        results['G2_estimate'].append(running_G2)
        results['S_estimate'].append(running_S)
        results['L_Norm'].append(avg_small_sq_norm)
        results['G_Norm'].append(big_sq_norm)
        results['Norms_ratio'].append(avg_small_sq_norm / (big_sq_norm + 1e-12))
        
        # ================================================

        for client_id in clients_per_round:
            client_train_loss, client_train_correct, client_train_total = clients[client_id].get_train_stats()
            train_loss.append(client_train_loss)
            train_correct.append(client_train_correct)
            train_total.append(client_train_total)

        end_time = time.time()

        curr_train_loss = sum(train_loss[-100:]) / (len(clients_per_round) * 100)
        train_acc = 100. * sum(train_correct[-100:]) / sum(train_total[-100:])

        bandwidth_scatter = sum(total_data_sent_scatter[-scatter_sending_rounds:]) / scatter_sending_rounds
        bandwidth_gather = sum(total_data_sent_scatter[-gather_sending_rounds:]) / gather_sending_rounds
        latency = sum(max_data_sent[-(scatter_sending_rounds+gather_sending_rounds):]) / (scatter_sending_rounds+gather_sending_rounds)

        total_time += (end_time - start_time)
        curr_net_params = clients[0].__get_params__()

        if len(results['time']) == 0:
            cumulative_bandwidth = bandwidth_scatter + bandwidth_gather
            cumulative_latency = latency
        else:
            cumulative_bandwidth += bandwidth_scatter + bandwidth_gather
            cumulative_latency += latency

        if round % test_every == 0:
            model_test_accuracy, model_test_correct, model_test_total = clients[0].test_model()
            results['rounds'].append(round)
            results['trainACCs'].append(train_acc)
            results['testACCs'].append(model_test_accuracy)
            results['time'].append(total_time)
            results['cumulative_bandwidth'].append(cumulative_bandwidth)
            results['cumulative_latency'].append(cumulative_latency)
            results['scatter_total_sent'] = total_data_sent_scatter
            results['gather_total_sent'] = total_data_sent_gather

            # --- DEBUG: Check for Divergence ---
            # 1. Check Learnable Weights (Should be 0.0)
            w0 = list(clients[0].net.parameters())
            w1 = list(clients[1].net.parameters())
            weight_diff = sum((p0 - p1).norm().item() for p0, p1 in zip(w0, w1))
    
            # 2. Check Batch Norm Stats (Likely > 0.0)
            s0 = clients[0].net.state_dict()
            s1 = clients[1].net.state_dict()
            bn_diff = 0.0
            for key in s0:
                # BN stats are usually named 'running_mean' or 'running_var'
                if "running" in key: 
                    bn_diff += (s0[key].float() - s1[key].float()).norm().item()
            
            print(f"[Round {round}] Weight Diff: {weight_diff:.6f} | BN Stats Diff: {bn_diff:.6f}")


        kbar.update(round, values=[
            ("train accuracy", train_acc),
            ("test accuracy", model_test_accuracy),
            ("Bandwidth", bandwidth_scatter),
            ("Latency", latency),
            ("GNS_estimate", smoothed_GNS),
            ("G_estimate", running_G2),
            ("S_estimate", running_S),
            ("Norms_ratio", avg_small_sq_norm / (big_sq_norm + 1e-12))
        ])

    

    data = {
        'results': results,
    }

    suffix = get_suffix(args_for_suffix)

    try :
        torch.save(data, './results/' + folder + '/' + compression_scheme + '/' + 'results_' + suffix + '.pt')
    except:    
        torch.save(data, './results/' + folder + '/' + 'results_' + suffix + '.pt')
    
    # Save directory for GNS analysis
    save_dir = './results/' + folder + '/' + compression_scheme
    if not os.path.exists(save_dir):
        # Fallback if specific compression folder wasn't made
        if os.path.isdir('./results/' + folder):
            save_dir = './results/' + folder
        else:
            save_dir = './results' # Last resort
                

    





