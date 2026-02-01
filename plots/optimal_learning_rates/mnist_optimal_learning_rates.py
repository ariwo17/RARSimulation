import matplotlib.pyplot as plt
import numpy as np
import os

# Data
batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
optimal_lrs = [0.01, 0.02, 0.02, 0.04, 0.04, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.1, 0.1, 0.1]

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, optimal_lrs, marker='o', linestyle='-', linewidth=2, markersize=8, color='purple', label='Optimal LR')

# Log-Log Scale
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Effective Batch Size (Log Scale)', fontsize=12)
plt.ylabel('Optimal Learning Rate (Log Scale)', fontsize=12)
plt.title('Optimal Learning Rate vs. Batch Size for MNIST (ComEffFlPaperCnnModel)', fontsize=14)
plt.grid(True, which="both", ls="-", alpha=0.4)

# Custom y-ticks
plt.yticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.2], 
           ['0.01', '0.02', '0.03', '0.04', '0.05', '0.075', '0.1', '0.2'])

plt.legend()
plt.tight_layout()

# Define output directory and file
output_dir = 'plots/optimal_learning_rates/'
output_file = os.path.join(output_dir, 'mnist_learning_rates.png')

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save
plt.savefig(output_file)
print(f"Plot saved to {output_file}")