import torch

def compute_B_simple_estimator(local_grad, global_grad, B_small, B_big):
    """
    Implements the two-batch method for computing an unbiased estimator for the simplified Gradient Noise Scale (GNS)
    as described in Appendix A1 of An Empirical Model of Large-Batch Training by McCandlish et al.
    
    local_grad:    gradient from one client      (B_small = B)
    global_grad:   aggregated gradient (mean across all clients) (B_big = N*B)
    b_small:       local batch size
    b_big:         effective global batch size
    """

    N_small = torch.linalg.vector_norm(local_grad)**2
    N_big   = torch.linalg.vector_norm(global_grad)**2

    G2 = (B_big * N_big - B_small * N_small) / (B_big - B_small)
    S  = (N_small - N_big) / (1/B_small - 1/B_big)

    B_simple = S / G2

    return (B_simple.item(), G2.item(), S.item())


def compute_exact_gns(client, dataset, device, limit=None):
    """
    Calculates the EXACT Simple Noise Scale (B_simple) by iterating 
    the full dataset with batch_size=1.
    
    Args:
        client: A client instance (to use its model/weights)
        dataset: The full training dataset (e.g. dataloader.trainset)
        device: torch device
        limit: (Optional) If not None, calculate over a random subset of size 'limit' 
               to save time (e.g. 5000 is still 2500x better than the 2-batch trick).
    """
    # 1. Setup
    model = client.net
    model.eval() # Use eval to fix dropout masks if you want "deterministic" landscape properties
                 # OR use .train() if you want to measure the noise *including* dropout noise.
                 # McCandlish uses 0.4 Dropout for MNIST, so .train() is technically 
                 # what the optimizer faces.
    model.train() 
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create a loader for individual samples
    if limit:
        # Optional: Random subset for speed
        indices = torch.randperm(len(dataset))[:limit]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    else:
        # Full Exact Calculation
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"   [Exact GNS] Computing over {len(loader)} samples...")

    # 2. Accumulators
    sum_grad_vec = None
    sum_sq_norm = 0.0
    num_samples = 0

    # 3. Iterate
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()

        # Flatten and collect gradient for this sample
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads: continue
        
        flat_grad = torch.cat(grads)
        
        # Accumulate E[g] and E[|g|^2]
        if sum_grad_vec is None:
            sum_grad_vec = torch.zeros_like(flat_grad)
        
        sum_grad_vec += flat_grad
        sum_sq_norm += (torch.norm(flat_grad)**2).item()
        num_samples += 1

    # 4. Calculate Statistics
    # Mean Gradient Vector: G
    G_vec = sum_grad_vec / num_samples
    
    # Squared Norm of Mean Gradient: |G|^2
    G_sq_norm = (torch.norm(G_vec)**2).item()
    
    # Mean of Squared Norms: E[|g|^2]
    E_g_sq = sum_sq_norm / num_samples
    
    # Trace of Covariance: tr(Sigma) = E[|g|^2] - |E[g]|^2
    trace_sigma = E_g_sq - G_sq_norm
    
    # Simple Noise Scale: B = tr(Sigma) / |G|^2
    if G_sq_norm < 1e-10:
        return 0.0 # Avoid div by zero at convergence
        
    B_simple = trace_sigma / G_sq_norm
    
    return B_simple