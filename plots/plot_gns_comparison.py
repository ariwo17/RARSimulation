import matplotlib.pyplot as plt
import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic

DEFAULT_SAVE_DIR = "plots/gns_plots"

# Milestones for CIFAR10
# MILESTONES = [50, 60, 70, 80, 85, 90, 92] # 95, 96, 97]
# Milestones for MNIST
MILESTONES = [70, 90, 99, 99.5, 99.8]

def plot_gns_comparison(experiments, mode='train_acc', bcrit_file=None, save_dir=DEFAULT_SAVE_DIR, filename="gns_comparison", x_min=None, x_max=None):
    """
    experiments: List of dicts [{'path': str, 'label': str, 'color': str (optional), 'linestyle': str (optional)}]
    """
    
    # Feel free to increase height slightly for better vertical resolution. Default was (10, 7).
    fig, ax = plt.subplots(figsize=(10, 9))
    
    # Track min/max values to zoom the Y-axis later
    y_min_all = float('inf')
    y_max_all = float('-inf')

    # Plot GNS Lines (Estimates)
    for exp in experiments:
        path = exp['path']
        label = exp['label']
        color = exp.get('color', None) 
        ls = exp.get('linestyle', '-') 
        
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            results = data['results'] if isinstance(data, dict) and 'results' in data else data
            
            gns_data = np.array(results['GNS_estimate'])
            rounds = np.array(results['rounds'])
            
            if mode == 'train_acc':
                acc_raw = np.array(results['trainACCs'])
                xlabel = "Training Accuracy (%)"
            else:
                acc_raw = np.array(results['testACCs'])
                xlabel = "Testing Accuracy (%)"

            # Align
            gns_indices = rounds - 1
            valid_mask = gns_indices < len(gns_data)
            acc_aligned = acc_raw[valid_mask]
            gns_aligned = gns_data[gns_indices[valid_mask]]

            # SMOOTHING (alpha=0.5)
            if mode == 'train_acc':
                acc_aligned = pd.Series(acc_aligned).ewm(alpha=0.5).mean().values

            if np.max(acc_aligned) > 1.0: acc_aligned /= 100.0
            
            error_vals = 1.0 - acc_aligned
            valid_err = error_vals > 0
            error_vals = error_vals[valid_err]
            gns_vals = gns_aligned[valid_err]

            if len(error_vals) == 0: continue

            # Binning
            bins = np.logspace(np.log10(np.min(error_vals)), np.log10(np.max(error_vals)), 200)
            bin_means, bin_edges, _ = binned_statistic(error_vals, gns_vals, statistic='mean', bins=bins)
            bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
            
            valid_bins = ~np.isnan(bin_means)
            x_plot = bin_centers[valid_bins]
            y_plot = bin_means[valid_bins]
            
            # Update global limits (filtering for the visible X range)
            if x_min is not None and x_max is not None:
                # Approximate filter to only consider Y values in the visible X window for autoscaling
                # This prevents low-accuracy noise from shrinking the view
                err_min_view = 1.0 - (x_max / 100.0)
                err_max_view = 1.0 - (x_min / 100.0)
                mask_view = (x_plot >= err_min_view) & (x_plot <= err_max_view)
                if np.any(mask_view):
                    y_min_all = min(y_min_all, np.min(y_plot[mask_view]))
                    y_max_all = max(y_max_all, np.max(y_plot[mask_view]))
            else:
                y_min_all = min(y_min_all, np.min(y_plot))
                y_max_all = max(y_max_all, np.max(y_plot))

            # Plot
            ax.plot(x_plot, y_plot, linewidth=2, label=label, color=color, linestyle=ls, alpha=0.9)

        except Exception as e:
            print(f"Error plotting {label}: {e}")

    # Overlay Critical Batch Size (Ground Truth)
    if bcrit_file and os.path.exists(bcrit_file):
        with open(bcrit_file, 'r') as f:
            bcrit_data = json.load(f)
        
        df = pd.DataFrame(bcrit_data)
        df = df[df['target'] <= 99.8] # Robust Range
        df = df.sort_values('target')
        
        if not df.empty:
            x_bcrit = (100.0 - df['target']) / 100.0
            y_bcrit = df['b_crit']
            
            # Update global limits with Bcrit data too
            y_min_all = min(y_min_all, np.min(y_bcrit))
            y_max_all = max(y_max_all, np.max(y_bcrit))

            ax.plot(x_bcrit, y_bcrit, color='steelblue', linestyle='--', 
                    linewidth=2.5, markersize=8, label='$B_{crit}$', zorder=10)

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # --- AUTO-ZOOM Y-AXIS ---
    # Apply padding to the tracked min/max so lines don't touch the edge
    if y_min_all != float('inf'):
        # Logarithmic padding: divide/multiply
        ax.set_ylim(y_min_all * 0.7, y_max_all * 1.4)
    
    if x_min is None: x_min = 70.0
    if x_max is None: x_max = 99.9
    
    err_min = 1.0 - (x_max / 100.0)
    err_max = 1.0 - (x_min / 100.0)
    ax.set_xlim(err_max, err_min)
    
    # Ticks
    log_start = np.floor(np.log10(err_min))
    log_end = np.ceil(np.log10(err_max))
    potential_ticks = 10.0**np.arange(log_start, log_end + 1)
    major_ticks = potential_ticks[(potential_ticks >= err_min * 0.99) & (potential_ticks <= err_max * 1.01)]
    milestones = np.array([1.0 - t/100.0 for t in MILESTONES])
    combined_ticks = np.unique(np.concatenate([major_ticks, milestones]))
    combined_ticks = combined_ticks[(combined_ticks >= err_min) & (combined_ticks <= err_max)]
    ax.set_xticks(combined_ticks)
    
    def formatter(x, pos):
        acc = (1.0 - x) * 100
        if abs(acc - round(acc)) < 1e-4: return f"{int(round(acc))}%"
        return f"{acc:.1f}%"

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Noise Scale", fontsize=12)
    # ax.set_title("Sketched and Baseline Simple Noise Scale vs Critical Batch Size", fontsize=14)
    ax.set_title("Sketched GNS vs Critical Batch Size (MNIST)", fontsize=14)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    
    # Baseline GNS file
    # baseline_file = "results/ringallreduce/grid_search/gns0.99/results_CIFAR10_ResNet9_none_14000_iid_8_64_0.1_acc_decay_momentum_1_10.pt"
    baseline_file = "results/ringallreduce/grid_search/gns0.999/results_MNIST_ComEffFlPaperCnnModel_none_4000_iid_8_16_0.05_const_sgd_1_10.pt"

    # Sketched GNS directory prefix
    sketch_dir = "results/ringallreduce/sketched_gns" 
    
    # experiments_list = [
        # Baseline GNS
    #     {'path': baseline_file, 'label': 'Baseline $B_{simple}$', 'color': 'tab:orange', 'linestyle': '-'},
        
    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c2451621.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=50%)', 'color': 'brown', 'linestyle': '-'},

    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c1225810.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=25%)', 'color': 'darkgreen', 'linestyle': '-'},

    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c490324.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=10%)', 'color': 'seagreen', 'linestyle': '-'},

    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c245162.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=5%)', 'color': 'blue', 'linestyle': '-'},

    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c49032.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=1%)', 'color': 'violet', 'linestyle': '-'},

    #     {'path': os.path.join(sketch_dir, "results_CIFAR10_ResNet9_csh_14000_iid_8_64_0.1_acc_decay_momentum_1_10_r1_c9806.pt"), 
    #     'label': 'Sketched $B_{simple}$ ($\delta$=0.2%)', 'color': 'red', 'linestyle': '-'},
    # ]


    experiments_list = [
        # Baseline GNS
        {'path': baseline_file, 'label': 'Baseline $B_{simple}$', 'color': 'tab:orange', 'linestyle': '-'},
        
        # The Sketched GNS runs
        {'path': os.path.join(sketch_dir, "results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c220000.pt"), 
         'label': 'Sketched $B_{simple}$ ($\delta$=25%)', 'color': 'darkgreen', 'linestyle': '-'},

        {'path': os.path.join(sketch_dir, "results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c44000.pt"), 
         'label': 'Sketched $B_{simple}$ ($\delta$=5%)', 'color': 'mediumseagreen', 'linestyle': '-'},

        {'path': os.path.join(sketch_dir, "results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c8800.pt"), 
         'label': 'Sketched $B_{simple}$ ($\delta$=1%)', 'color': 'purple', 'linestyle': '-'},

        {'path': os.path.join(sketch_dir, "results_MNIST_ComEffFlPaperCnnModel_csh_4000_iid_8_16_0.05_const_sgd_1_10_r1_c2000.pt"), 
         'label': 'Sketched $B_{simple}$ ($\delta$=0.2%)', 'color': 'red', 'linestyle': '-'},
    ]
    
    # Critical batch size ground truth results
    # bcrit_file = "data/bcrit_results_cifar10.json"
    bcrit_file = "data/bcrit_results_mnist.json"

    # Run Plotter
    plot_gns_comparison(
        experiments=experiments_list,
        mode='train_acc',
        bcrit_file=bcrit_file,
        x_min=70, 
        x_max=99.8,
        filename="mnist_gns_vs_bcrit_vs_sketched"
    )