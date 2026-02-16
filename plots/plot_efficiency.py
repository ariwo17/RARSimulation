import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

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
    if metric == 'testACCs':
        return 'Test Accuracy'
    elif metric == 'trainACCs':
        return 'Train Accuracy'
    return metric

def plot_single_curve(x_values, y_values, x_label, y_label, title, save_path, alpha=0.3):
    """
    Generic function to plot Y vs X with smoothing.
    """
    if len(x_values) == 0 or len(y_values) == 0:
        return

    # Align lengths
    min_len = min(len(x_values), len(y_values))
    x_values = x_values[:min_len]
    y_values = y_values[:min_len]
    
    # Create DataFrame for smoothing
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    df['y_smooth'] = df['y'].ewm(alpha=alpha).mean()
    
    plt.figure(figsize=(10, 6))
    
    # Plot Raw (faint)
    plt.plot(df['x'], df['y'], color='lightgray', linestyle='-', alpha=0.5, label='Raw')
    
    # Plot Smoothed (bold)
    plt.plot(df['x'], df['y_smooth'], color='blue', linewidth=2, label=f'Smoothed (alpha={alpha})')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Saved: {save_path}")

def process_file(file_path, metric='trainACCs', alpha=0.1):
    filename = os.path.basename(file_path)
    print(f"\nProcessing: {filename}...")
    
    results = load_data(file_path)
    if not results:
        return

    if metric not in results:
        print(f"  [Warning] Metric '{metric}' not found.")
        return

    y_values = results[metric]
    readable_metric = get_readable_metric_name(metric)
    
    # --- MODIFIED: Save to current directory ---
    # We strip the directory path and just use the filename
    base_name = filename.replace('.pt', '') 
    
    # Rounds vs Accuracy
    if 'rounds' in results:
        plot_single_curve(
            results['rounds'], y_values,
            'Communication Rounds', readable_metric,
            f'{readable_metric} vs Rounds\n{filename}',
            f"{base_name}_vs_rounds.png", alpha
        )

    # Time vs Accuracy
    # In reality this metric is problematic for my project because I ran some of those
    # experiments in parallel meaning they were contaminated by slow performance caused
    # by high GPU utilisation.
    # if 'time' in results:
    #     plot_single_curve(
    #         results['time'], y_values,
    #         'Wall Clock Time (s)', readable_metric,
    #         f'{readable_metric} vs Time\n{filename}',
    #         f"{base_name}_vs_time.png", alpha
    #     )

    # Bandwidth vs Accuracy
    if 'cumulative_bandwidth' in results:
        bw = results['cumulative_bandwidth']
        unit = "Bytes"
        if len(bw) > 0:
            if bw[-1] > 1e9:
                bw = [b/1e9 for b in bw]; unit = "GB"
            elif bw[-1] > 1e6:
                bw = [b/1e6 for b in bw]; unit = "MB"
            
        plot_single_curve(
            bw, y_values,
            f'Accumulated Bandwidth ({unit})', readable_metric,
            f'{readable_metric} vs Bandwidth\n{filename}',
            f"{base_name}_vs_bandwidth.png", alpha
        )

if __name__ == '__main__':
    # --- CONFIGURATION ---
    
    METRIC_NAME = "trainACCs" 
    SMOOTHING_ALPHA = 0.3
    
    # Option A: Process a folder
    USE_FOLDER = False
    FOLDER_PATH = "results/ringallreduce/grid_search/sparsified_gns"
    
    # Option B: Manual List
    MANUAL_FILES = [
        "results/ringallreduce/grid_search/gns0.99/results_CIFAR10_ResNet9_none_14000_iid_8_64_0.1_acc_decay_momentum_1_10.pt",
        "results/ringallreduce/sketched_gns/results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c2451621.pt"
    ]

    if USE_FOLDER:
        # Use glob to find files in the target folder
        target_files = glob.glob(os.path.join(FOLDER_PATH, "*.pt"))
        target_files.sort()
    else:
        target_files = MANUAL_FILES

    print(f"Found {len(target_files)} files to process.")

    for f in target_files:
        process_file(f, METRIC_NAME, SMOOTHING_ALPHA)
        
    print("\nPlotting of round-, accumulated bandwidth- and time-to-accuracy graphs has finished.")