import torch.utils
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as torch_optim
import numpy as np
import math
from itertools import cycle

# Local imports
from . import model_factories
from compressors.sparsification import VectorTopK, ChunkTopK
from compressors.countsketch import CountSketchSender, CountSketchReceiver

###############################################################
#                        Client Class                         #
###############################################################

class Client:
    def __init__(self, client_id, model, lr, lr_type, optim, dataloader, device, device_ids,
                 train_batch_size, test_batch_size, compression_scheme, nbits,
                 error_feedback, seed, num_clients, testset, k_value, sketch_col, sketch_row):
        
        self.client_id = client_id
        self.device_ids = device_ids
        self.seed = seed
        self.num_clients = num_clients
        self.compression_scheme = compression_scheme
        self.nbits = nbits
        self.error_feedback = error_feedback
        self.sketch_col = sketch_col
        self.sketch_row = sketch_row

        self.k_value = k_value

        # Initialize gradient containers
        self.curr_gradient_chunks = []
        self.curr_gradient = []
        self.sketched_gradient = {}
        self.unsketched_gradient = []

        # Fetch client-specific training data
        self.train_data = dataloader.get_client_train_data(client_id)

        # Build model and set model name
        self.net = getattr(model_factories, model)(dataloader.num_classes)
        self.model_name = model

        # Create testloader only for client 0
        if client_id == 0:
            self.testloader = torch.utils.data.DataLoader(
                testset, batch_size=test_batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

        ############################
        #         Cuda Setup       #
        ############################

        self.net = self.net.to(device)
        if 'cuda' in device:
            # Enable DataParallel if more than one GPU is provided
            if len(device_ids) > 1:
                self.net = torch.nn.DataParallel(self.net, device_ids=device_ids)
            self.device = next(self.net.parameters()).device
            cudnn.benchmark = True
        else:
            self.device = device

        print('==> [Client {}] Building {} model with {} instances and {} classes on {}...'
              .format(client_id, model, len(self.train_data), dataloader.num_classes, self.device))

        ##################################################
        #     Trainable Parameters & Initializations     #
        ##################################################

        pytorch_total_params = sum(p.numel() for p in self.net.parameters())
        self.pytorch_total_params = pytorch_total_params
        # print(f"Gradient Length (Total Params): {self.pytorch_total_params}")
        self.initial_params = torch.zeros(pytorch_total_params, device=self.device)

        # Loss criterion and learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.lr_type = lr_type
        self.optim = optim

        # Create local dataloader for training
        self.trainloader = torch.utils.data.DataLoader(
            self.train_data, batch_size=train_batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=False
        )

        ############################
        #    Compression Setup     #
        ############################

        if self.compression_scheme == 'none':
            pass
        elif self.compression_scheme == 'vector_topk':
            self.compressor = VectorTopK(self.device)
        elif self.compression_scheme == 'chunk_topk_recompress':
            self.compressor = ChunkTopK(self.device)
        elif self.compression_scheme == 'chunk_topk_single':
            self.compressor = ChunkTopK(self.device)
        elif self.compression_scheme == 'chunk_topk_recompress':
            self.compressor = ChunkTopK(self.device)
        elif self.compression_scheme == 'csh':
            self.compressor = CountSketchSender(self.device)
            self.decompressor = CountSketchReceiver(self.device)
        elif self.compression_scheme == 'cshtopk_actual':
            self.compressor = CountSketchSender(self.device)
            self.decompressor = CountSketchReceiver(self.device)
        elif self.compression_scheme == 'cshtopk_estimate':
            self.compressor = CountSketchSender(self.device)
            self.decompressor = CountSketchReceiver(self.device)
        else:
            raise("Compression scheme not defined")

        # Optimiser setup (default is SGD, momentum is preferred for ResNet9 + CIFAR10 setup)
        # Local updates allow the momentum to accelerate convergence significantly.
        if self.optim == "sgd":
            self.optimiser = torch_optim.SGD(self.net.parameters(), lr=self.lr)
            self.use_local_updates = True
        elif self.optim == "momentum":
            self.optimiser = torch_optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
            self.use_local_updates = False
        elif self.optim == "adam":
            self.optimiser = torch_optim.Adam(self.net.parameters(), lr=self.lr)
            self.use_local_updates = False
        else:
            raise ValueError(f"Optimiser mode '{self.optim}' not recognised. Use 'sgd' or 'momentum'.")

        # Error feedback initialization
        if self.error_feedback:
            self.error = torch.zeros(pytorch_total_params)

    ###############################################################
    #                       Update & Train                        #
    ###############################################################
    
    def update_net(self, acc_grad):

        # acc_grad /= self.num_clients # OLD WAY: this causes in-place averaging issues
        acc_grad = acc_grad / self.num_clients # Average the gradients
            

        # if acc_grad is smaller in size then it was sketched.
        if acc_grad.numel() < self.pytorch_total_params:
            self.temp_sketch = self.compress(torch.zeros(self.pytorch_total_params, device=self.device))
            self.temp_sketch['vec'] = acc_grad
            acc_grad = self.decompressor.decompress(self.temp_sketch)



        if self.compression_scheme in ['cshtopk_estimate', 'cshtopk_actual']:
            self.initial_params = acc_grad
            start = 0
            for param in self.net.parameters():
                numel = param.numel()
                acc_grad_param = acc_grad[start:start + numel].view_as(param)
                
                # Get nonzero indices
                nonzero_indices = torch.nonzero(acc_grad_param, as_tuple=True)
                
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                # Update only nonzero indices
                param.grad[nonzero_indices] = acc_grad_param[nonzero_indices]
                
                start += numel
        else:
            self.initial_params = acc_grad
            start = 0
            for param in self.net.parameters():
                numel = param.numel()
                param.grad = acc_grad[start:start + numel].view_as(param)
                start += numel

    def train(self, round, steps):
        
        # --- LEGACY: Time-based Step Decay ---
        def step_decay(initial_lrate, round):
            drop = 0.8
            rounds_drop = 1000
            lrate = initial_lrate * math.pow(drop, math.floor((1 + round) / rounds_drop))
            return lrate

        def exp_decay(initial_lrate, round):
            k = 0.1
            rounds_drop = 100
            lrate = initial_lrate * math.exp(-k * (round//rounds_drop))
            return lrate

        def get_loop(steps):
            # if steps==0: train over the entire train data
            if steps:
                return steps
            else:
                return len(self.trainloader)


        # set lr
        if self.lr_type == 'const':
            lr = self.lr
        elif self.lr_type == 'step_decay':
            lr = step_decay(self.lr, round)
        elif self.lr_type == 'exp_decay':
            lr = exp_decay(self.lr, round)
        elif self.lr_type == 'acc_decay':
            # NEW: Server controls self.lr based on accuracy milestones.
            # We just use whatever value is currently set in self.lr
            lr = self.lr
        else:
            raise Exception('Undefined learning rate type. Received {}'.format(self.lr_type))

        for param_group in self.optimiser.param_groups:
            param_group['lr'] = lr

        # # Old code - refactored so that vanilla SGD is no longer hard-coded       
        # self.optimiser = optim.SGD(self.net.parameters(), lr=lr)
        # # self.optimiser = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.net.train()

        # step towards new gradients then empty it out to compute new ones
        self.optimiser.step()
        self.optimiser.zero_grad()

        # Reset training statistics
        self.train_loss = 0
        self.train_correct = 0
        self.train_total = 0

        # local SGD
        ds_iterator = iter(self.trainloader)
        for step in range(get_loop(steps)):
            try:
                client_inputs, client_targets = next(ds_iterator)
            except StopIteration:
                ds_iterator = iter(self.trainloader)
                client_inputs, client_targets = next(ds_iterator)
            client_inputs = client_inputs.to(self.device)
            client_targets = client_targets.to(self.device)

            client_outputs = self.net(client_inputs)
            client_loss = self.criterion(client_outputs, client_targets)
            client_loss.backward()

            if self.use_local_updates:
                self.optimiser.step()

            self.train_loss += client_loss.item()
            _, predicted = client_outputs.max(1)
            self.train_total += client_targets.size(0)
            self.train_correct += predicted.eq(client_targets).sum().item()
        
        self.train_loss /= get_loop(steps)

        self.prepare_gradients()

    def prepare_gradients(self):
        # Prepare gradients for sending based on the compression scheme
        if self.compression_scheme in ['csh', 'cshtopk_actual', 'cshtopk_estimate']:
            self.sketched_gradient = self.get_gradient()
            self.curr_gradient = self.sketched_gradient['vec']
        else:
            self.curr_gradient = self.get_gradient()
        
        if self.compression_scheme == 'chunk_topk_single':
            compressed = self.compressor.compress_chunked_vector(list(torch.chunk(self.curr_gradient, self.num_clients)), self.k_value)
            self.curr_gradient_chunks = compressed
        else:
            self.curr_gradient_chunks = list(torch.chunk(self.curr_gradient, self.num_clients))

    ###############################################################
    #                   Gradient Handling                       #
    ###############################################################

    def get_gradient_chunk(self, pos):
        if self.compression_scheme == 'chunk_topk_recompress':
            return self.compressor.compress_chunk(self.curr_gradient_chunks[pos], self.k_value)
        else:
            return self.curr_gradient_chunks[pos]
    
    def get_gradients_from_indices(self, indices):
        return self.unsketched_gradient[indices]
    
    def get_gradient(self):
        # Build the gradient vector from all parameters
        clients_grad_vec = []
        for param in self.net.parameters():
            x = param.grad.view(-1)
            clients_grad_vec.append(x)
        clients_grad_vec = torch.cat(clients_grad_vec)
        self.unsketched_gradient = clients_grad_vec
        return self.compress(clients_grad_vec)

    def compress(self, vec):
        # Prepare data for compression
        data = {
            'seed': hash(str(self.client_id)) % 2**16 + np.random.randint(2**16),
            'rotation_seed': self.seed,
            'vec': vec
        }
        nbits = self.nbits

        if self.compression_scheme == 'none':
            return vec
        elif self.compression_scheme == 'vector_topk':
            return self.compressor.compress(data, self.k_value)
        elif self.compression_scheme == 'chunk_topk_recompress':
            return vec
        elif self.compression_scheme == 'chunk_topk_single':
            return vec
        elif self.compression_scheme in ['csh', 'cshtopk_actual', 'cshtopk_estimate']:
            data["d"] = int(vec.numel())
            data["c"] = self.sketch_col 
            data["r"] = self.sketch_row
        
        return self.compressor.compress(data)

    ###############################################################
    #                   Testing & Parameter Access              #
    ###############################################################

    def get_train_stats(self):
        return self.train_loss, self.train_correct, self.train_total

    def test_model(self):
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.net(inputs)
                if self.model_name == 'LogisticRegression':
                    predicted = torch.where(outputs > 0.5, 1, 0)
                else:
                    _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        return 100. * test_correct / test_total, test_correct, test_total

    def __get_params__(self):
        # Return list of parameter data
        return [param.data for param in self.net.parameters()]

    def __get_flat_params__(self):
        # Return a flattened tensor of all parameters
        flat_params = [param.data.view(-1).clone() for param in self.net.parameters()]
        return torch.cat(flat_params)

    def __get_layer_info__(self):
        # Provide information about each layer
        layer_info = {'layer_dims': [], 'layer_sizes': []}
        for param in self.net.parameters():
            layer_info['layer_dims'].append(param.size())
            layer_info['layer_sizes'].append(param.numel())
        return layer_info

    def decompress_sketch(self, vec):
        self.sketched_gradient['vec'] = vec
        return self.decompressor.decompress(self.sketched_gradient)
