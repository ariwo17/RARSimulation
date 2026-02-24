import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

# --- CONFIGURATION ---
METRIC_NAME = "trainACCs" 
SMOOTHING_ALPHA = 0.3
DATASET = "CIFAR10"
DATASET_PREFIX = DATASET.lower()
OUTPUT_PREFIX = f"{DATASET_PREFIX}_efficiency_csh_comparison"

# Format: (FILE_PATH, LEGEND_LABEL)
FILES_TO_COMPARE = [
    (
        "results/ringallreduce/grid_search/cifar10const/results_CIFAR10_ResNet9_none_14000_iid_8_64_0.075_const_momentum_1_10.pt", 
        "Baseline SGD (No compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c2451621.pt", 
        "Count Sketch (50$\%$ compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c1225810.pt", 
        "Count Sketch (25$\%$ compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c490324.pt", 
        "Count Sketch (10$\%$ compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c245162.pt", 
        "Count Sketch (5$\%$ compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c49032.pt", 
        "Count Sketch (1$\%$ compression)"
    ),
    (
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.075_const_momentum_1_10_r1_c9806.pt", 
        "Count Sketch (0.2$\%$ compression)"
    ),
    # (
    #     "results/ringallreduce/grid_search/gns0.999/results_MNIST_ComEffFlPaperCnnModel_none_4000_iid_8_16_0.05_const_sgd_1_10.pt", 
    #     "Baseline SGD (No compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c440000.pt", 
    #     "Count Sketch (50$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c220000.pt", 
    #     "Count Sketch (25$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c88000.pt", 
    #     "Count Sketch (10$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c44000.pt", 
    #     "Count Sketch (5$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c8800.pt", 
    #     "Count Sketch (1$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sketched_gns/results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c2000.pt", 
    #     "Count Sketch (0.2$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k438469.pt", 
    #     "Random-K (50$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k219234.pt", 
    #     "Random-K (25$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k87693.pt", 
    #     "Random-K (10$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k43846.pt", 
    #     "Random-K (5$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k26308.pt", 
    #     "Random-K (3$\%$ compression)"
    # ),
    #     (
    #     "results/ringallreduce/sparsified_gns/results_MNIST_ComEffFlPaperCnnModel_randomk_4000_iid_8_16_0.05_const_sgd_1_10_k21923.pt", 
    #     "Random-K (2.5$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_CIFAR10_ResNet9_randomk_14000_iid_8_64_0.075_const_momentum_1_10_k2451621.pt", 
    #     "Random-K (50$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_CIFAR10_ResNet9_randomk_14000_iid_8_64_0.075_const_momentum_1_10_k1225810.pt", 
    #     "Random-K (25$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_CIFAR10_ResNet9_randomk_14000_iid_8_64_0.075_const_momentum_1_10_k490324.pt", 
    #     "Random-K (10$\%$ compression)"
    # ),
    # (
    #     "results/ringallreduce/sparsified_gns/results_CIFAR10_ResNet9_randomk_14000_iid_8_64_0.075_const_momentum_1_10_k245162.pt", 
    #     "Random-K (5$\%$ compression)"
    # )
]

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping: File not found {file_path}")
        return None
    try:
        data = torch.load(file_path, map_location=torch.device('cpu'))
        return data.get('results', {})
    except Exception as e:
        print(f"Skipping: Failed to load {os.path.basename(file_path)} ({e})")
        return None

def get_readable_metric_name(metric):
    if metric == 'testACCs': return 'Test Accuracy'
    elif metric == 'trainACCs': return 'Train Accuracy'
    return metric

def plot_combined_curve(data_map, x_metric, x_label, y_metric, y_label, title, save_path, alpha=0.3):
    """
    Plots multiple curves on a single figure.
    data_map: dict { label: { 'x': [...], 'y': [...] } }
    """
    plt.figure(figsize=(12, 7))
    
    colors = ['black', 'orange', 'brown', 'darkgreen', 'blue', 'purple', 'red']
    
    for i, (label, points) in enumerate(data_map.items()):
        x_values = points['x']
        y_values = points['y']
        
        if len(x_values) == 0 or len(y_values) == 0:
            continue

        min_len = min(len(x_values), len(y_values))
        x_vals = x_values[:min_len]
        y_vals = y_values[:min_len]
        
        df = pd.DataFrame({'x': x_vals, 'y': y_vals})
        df['y_smooth'] = df['y'].ewm(alpha=alpha).mean()
        
        color = colors[i % len(colors)]
        
        # Plot Raw
        plt.plot(df['x'], df['y'], color=color, linestyle='-', alpha=0.15, linewidth=1)
        # Plot Smoothed
        plt.plot(df['x'], df['y_smooth'], color=color, linewidth=2, label=label)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f" -> Saved Combined Plot: {save_path}")

def run_comparison():
    print(f"Starting comparison of {len(FILES_TO_COMPARE)} files...")
    
    # Data structures to hold plot data
    rounds_data = {}
    bandwidth_data = {}
    time_data = {}  # <--- Added container for Time data
    
    readable_metric = get_readable_metric_name(METRIC_NAME)
    bw_unit = "Bytes"
    time_unit = "Seconds" # Default

    for file_path, label in FILES_TO_COMPARE:
        results = load_data(file_path)
        if not results: continue
        
        if METRIC_NAME not in results:
            print(f"Warning: {METRIC_NAME} missing in {label}")
            continue
            
        y_vals = results[METRIC_NAME]

        # Collect Rounds Data
        if 'rounds' in results:
            rounds_data[label] = {'x': results['rounds'], 'y': y_vals}
            
        # Collect Bandwidth Data
        if 'cumulative_bandwidth' in results:
            bw = results['cumulative_bandwidth']
            if len(bw) > 0 and bw[-1] > 1e9:
                bw = [b/1e9 for b in bw]
                bw_unit = "GB"
            elif len(bw) > 0 and bw[-1] > 1e6:
                bw = [b/1e6 for b in bw]
                bw_unit = "MB"     
            bandwidth_data[label] = {'x': bw, 'y': y_vals}

        # Collect Time Data
        # We assume the key is 'time' or 'elapsed_time'
        if 'time' in results:
            t = results['time']
            # Dynamic unit adjustment for readability
            if len(t) > 0 and t[-1] > 60:
                t = [x/60 for x in t]
                time_unit = "Minutes"
            
            time_data[label] = {'x': t, 'y': y_vals}

    # --- Plot 1: Rounds vs Accuracy ---
    if rounds_data:
        plot_combined_curve(
            rounds_data,
            x_metric='rounds',
            x_label='Communication Rounds',
            y_metric=METRIC_NAME,
            y_label=readable_metric,
            title=f'{readable_metric} vs Rounds ({DATASET})',
            save_path=f"{OUTPUT_PREFIX}_vs_rounds.png",
            alpha=SMOOTHING_ALPHA
        )
    else:
        print("No rounds data found.")

    # --- Plot 2: Bandwidth vs Accuracy ---
    if bandwidth_data:
        plot_combined_curve(
            bandwidth_data,
            x_metric='bandwidth',
            x_label=f'Accumulated Bandwidth ({bw_unit})',
            y_metric=METRIC_NAME,
            y_label=readable_metric,
            title=f'{readable_metric} vs Bandwidth ({DATASET})',
            save_path=f"{OUTPUT_PREFIX}_vs_bandwidth.png",
            alpha=SMOOTHING_ALPHA
        )
    else:
        print("No bandwidth data found.")

    # --- Plot 3: Time vs Accuracy (NEW) ---
    if time_data:
        plot_combined_curve(
            time_data,
            x_metric='time',
            x_label=f'Elapsed Time ({time_unit})',
            y_metric=METRIC_NAME,
            y_label=readable_metric,
            title=f'{readable_metric} vs Time ({DATASET})',
            save_path=f"{OUTPUT_PREFIX}_vs_time.png",
            alpha=SMOOTHING_ALPHA
        )
    else:
        print("No time data found.")

if __name__ == '__main__':
    run_comparison()