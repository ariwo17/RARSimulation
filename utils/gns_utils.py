import torch

class GNSEstimator:
    def __init__(self, ema_decay=0.99):
        self.ema_decay = ema_decay
        self.running_S = 0.0
        self.running_G2 = 0.0
        self.step = 0

    def update(self, avg_small_sq_norm, big_sq_norm, B_small, B_big):
        """
        Updates the estimator with the current step's energy values.
        Implements the two-batch method for computing an unbiased estimator for the simplified Gradient Noise Scale (GNS) 
        as described in Appendix A1 of An Empirical Model of Large-Batch Training by McCandlish et al.
        
        avg_small_sq_norm: Average of squared norms of individual worker gradients (Variance Reduced)
        big_sq_norm: Squared norm of the global aggregated gradient
        """
        val_small = avg_small_sq_norm
        val_big = big_sq_norm

        # McCandlish Appendix A.1
        term_S = (val_small - val_big) / (1.0 / B_small - 1.0 / B_big)
        term_G2 = (B_big * val_big - B_small * val_small) / (B_big - B_small)

        if self.step == 0:
            self.running_S = term_S
            self.running_G2 = term_G2
        else:
            self.running_S = self.ema_decay * self.running_S + (1 - self.ema_decay) * term_S
            self.running_G2 = self.ema_decay * self.running_G2 + (1 - self.ema_decay) * term_G2
        
        self.step += 1

    def get_stats(self):
        if abs(self.running_G2) < 1e-12:
            return 0.0, self.running_G2, self.running_S
        
        gns = self.running_S / self.running_G2
        return gns, self.running_G2, self.running_S

def compute_exact_gns(client, dataset, device, limit=None):
    model = client.net
    model.train() 
    criterion = torch.nn.CrossEntropyLoss()
    
    if limit:
        indices = torch.randperm(len(dataset))[:limit]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"   [Exact GNS] Computing over {len(loader)} samples...")

    sum_grad_vec = None
    sum_sq_norm = 0.0
    num_samples = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()

        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads: continue
        
        flat_grad = torch.cat(grads)
        
        if sum_grad_vec is None:
            sum_grad_vec = torch.zeros_like(flat_grad)
        
        sum_grad_vec += flat_grad
        sum_sq_norm += (torch.norm(flat_grad)**2).item()
        num_samples += 1

    G_vec = sum_grad_vec / num_samples
    G_sq_norm = (torch.norm(G_vec)**2).item()
    E_g_sq = sum_sq_norm / num_samples
    trace_sigma = E_g_sq - G_sq_norm
    
    if G_sq_norm < 1e-12:
        return 0.0
        
    return trace_sigma / G_sq_norm