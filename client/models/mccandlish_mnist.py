import torch
import torch.nn as nn
import torch.nn.functional as F

class McCandlishMNIST(nn.Module):
    """
    Exact architecture from McCandlish et al. Appendix A.4.1 for MNIST.
    - Conv1: 32 filters, 5x5
    - Conv2: 64 filters, 5x5
    - Hidden: 1024 units
    - Dropout: 0.4
    """
    def __init__(self):
        super(McCandlishMNIST, self).__init__()
        # Input: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) 
        # 28-5+1 = 24 -> Pool(2) -> 12
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 12-5+1 = 8 -> Pool(2) -> 4
        
        # Flattened size: 64 * 4 * 4 = 1024
        self.fc1 = nn.Linear(1600, 1024) 
        self.fc2 = nn.Linear(1024, 10)
        
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out