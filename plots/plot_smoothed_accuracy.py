import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metric(file_path, metric, alpha=0.1):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    results = data.get('results', {})
    if not results:
        print("Error: 'results' key not found.")
        return

    steps = results.get('rounds', [])
    if metric not in results:
        print(f"Error: Metric '{metric}' not found.")
        return
    
    y_values = results[metric]
    
    if len(steps) == 0 or len(y_values) == 0:
        print(f"Error: No data found.")
        return

    min_len = min(len(steps), len(y_values))
    steps = steps[:min_len]
    y_values = y_values[:min_len]

    s_raw = pd.Series(y_values)
    s_ewma = s_raw.ewm(alpha=alpha).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(steps, y_values, color='lightgray', linestyle='-', alpha=0.5, label='Raw')
    plt.plot(steps, s_ewma, color='blue', linewidth=2, label=f'Smoothed (alpha={alpha})')

    plt.xlabel('Optimization Steps (Rounds)')
    plt.ylabel(metric)
    plt.title(f'{metric} over Training\nFile: {os.path.basename(file_path)}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save logic
    save_path = file_path.replace('.pt', '.png')
    plt.savefig(save_path)
    plt.close() # Close memory to avoid warning
    
    print(f"Saved plot to: {save_path}")

if __name__ == '__main__':
    # Hardcoded Configuration
    # file_path = "results/ringallreduce/results_MNIST_ComEffFlPaperCnnModel_none_400_iid_4_32_0.005_step_decay_sgd_1_10.pt"
    # file_path = "results/ringallreduce/results_MNIST_ComEffFlPaperCnnModel_none_400_iid_4_32_0.005_const_sgd_1_10.pt"
    # file_path = "results/ringallreduce/debug/results_CIFAR10_ResNet9_none_14000_iid_8_16_0.1_acc_decay_momentum_1_10.pt"
    # file_path = "results/ringallreduce/debug/results_CIFAR10_ResNet9_none_6000_iid_8_1024_0.2_acc_decay_momentum_1_10.pt"
    # file_path = "results/ringallreduce/debug/results_CIFAR10_ResNet9_none_8000_iid_8_128_0.2_acc_decay_momentum_1_10.pt"
    # file_path = "results/ringallreduce/grid_search/results_CIFAR10_ResNet9_none_14000_iid_8_4_0.02_acc_decay_momentum_1_10.pt"
    file_path = "results/ringallreduce/grid_search/results_CIFAR10_ResNet9_none_8000_iid_8_1024_0.15_acc_decay_momentum_1_10.pt"
    metric = "testACCs" 
    alpha = 0.3

    plot_metric(file_path, metric, alpha)