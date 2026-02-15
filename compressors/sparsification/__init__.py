import torch

# for testing
import numpy as np

class RandomK():
    def __init__(self, device='cpu'):
        self.device = device

    def compress(self, vec, k=100):
        d = vec.numel()
        # Ensure k does not exceed vector size
        k = min(k, d)
        
        # Generate random indices without replacement
        indices = torch.randperm(d, device=self.device)[:k]
        
        # Create sparse vector
        sparse_vec = torch.zeros_like(vec)
        
        # Unbiased scaling: (d/k) * value
        # This ensures E[compressed] = original
        sparse_vec[indices] = vec[indices] * (d / k)
        
        return sparse_vec

class VectorTopK():
    def __init__(self, device='cpu'):
        self.device = device

    def compress(self, data, k=100):
        vec = data['vec']

        sparse_vec = torch.zeros_like(vec)
        top_k_values, top_k_indices = torch.topk(vec.abs(), k)
        sparse_vec[top_k_indices] = vec[top_k_indices]
        return sparse_vec

class ChunkTopK():
    def __init__(self, device='cpu'):
        self.device = device
    
    def compress_chunk(self, chunk, k=100):
        sparse_chunk = torch.zeros_like(chunk)
        top_k_values, top_k_indices = torch.topk(chunk.abs(), k)
        sparse_chunk[top_k_indices] = chunk[top_k_indices]
        return sparse_chunk
    
    def compress_chunked_vector(self, vec, k=100):
        for i in range(len(vec)):
            chunk = vec[i]
            sparse_chunk = torch.zeros_like(chunk)
            top_k_values, top_k_indices = torch.topk(chunk.abs(), k)
            sparse_chunk[top_k_indices] = chunk[top_k_indices]
            vec[i] = sparse_chunk

        return vec