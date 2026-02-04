import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pandas as pd  # Added for EWMA smoothing
import matplotlib.ticker as ticker
from scipy.stats import binned_statistic

# Default directory
DEFAULT_SAVE_DIR = "plots/gns_plots"

def plot_gns(results_path, suffix, mode, save_dir=DEFAULT_SAVE_DIR, x_min=None, x_max=None):
    # Load Data
    if not os.path.exists(results_path):
        print(f"Error: File not found at {results_path}")
        return

    data = torch.load(results_path, map_location='cpu', weights_only=True)
    results = data['results'] if isinstance(data, dict) and 'results' in data else data

    # Extract Data
    gns_data = np.array(results['GNS_estimate'])
    rounds_sampled = np.array(results['rounds'])

    # ==========================================
    # MODE A: ROUNDS (Linear X-Axis)
    # ==========================================
    if mode == 'rounds':
        if len(gns_data) != len(rounds_sampled):
            # Handle dense GNS recording
            x_raw = np.arange(1, len(gns_data) + 1)
            y_raw = gns_data
        else:
            x_raw = rounds_sampled
            y_raw = gns_data
            
        xlabel = "Communication Rounds"
        
        if x_min is None: x_min = 0
        if x_max is None: x_max = np.max(x_raw)
        
        mask = (x_raw >= x_min) & (x_raw <= x_max)
        x_plot = x_raw[mask]
        y_plot = y_raw[mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_plot, y_plot, color='tab:blue', linewidth=1.0, label='GNS Estimate')
        
        ax.set_yscale('log')
        ax.set_xscale('linear')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Gradient Noise Scale (GNS)")
        ax.set_title(f"GNS vs {xlabel}")
        ax.grid(True, which='major', linestyle='-', alpha=0.6)
        ax.grid(True, which='minor', linestyle=':', alpha=0.3)
        ax.legend()

    # ==========================================
    # MODE B: ACCURACY (Log-Error)
    # ==========================================
    elif mode in ['train_acc', 'test_acc']:
        if mode == 'train_acc':
            acc_raw = np.array(results['trainACCs'])
            xlabel = "Training Accuracy (%)"
        else:
            acc_raw = np.array(results['testACCs'])
            xlabel = "Testing Accuracy (%)"

        # Align Data
        gns_indices = rounds_sampled - 1
        valid_mask = gns_indices < len(gns_data)
        
        acc_aligned = acc_raw[valid_mask]
        gns_aligned = gns_data[gns_indices[valid_mask]]

        # --- FIX: SMOOTH TRAINING ACCURACY ---
        # Training accuracy jitters wildly. We smooth it to get the "Trend".
        if mode == 'train_acc':
            # alpha=0.05 gives a strong smooth (like your Pareto plots)
            acc_aligned = pd.Series(acc_aligned).ewm(alpha=0.5).mean().values

        # Normalize 0-100 to 0.0-1.0
        if np.max(acc_aligned) > 1.0: acc_aligned /= 100.0
        
        # Convert to Error Space for Log Plotting
        error_full = 1.0 - acc_aligned
        gns_full = gns_aligned

        # Safety: Remove non-positive errors (100% accuracy)
        valid_err = error_full > 0
        error_full = error_full[valid_err]
        gns_full = gns_full[valid_err]

        if len(error_full) == 0:
            print("Error: No valid data points (Accuracy likely sat at 100% or 0%).")
            return

        data_min_err = np.min(error_full)
        data_max_err = np.max(error_full)
        
        # Binning
        bins = np.logspace(np.log10(data_min_err), np.log10(data_max_err), 200)
        bin_means, bin_edges, _ = binned_statistic(error_full, gns_full, statistic='mean', bins=bins)
        bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
        
        valid_bins = ~np.isnan(bin_means)
        x_final = bin_centers[valid_bins]
        y_final = bin_means[valid_bins]

        # User Limits
        if x_min is None: x_min = 0.90
        else: x_min = x_min / 100.0 if x_min > 1.0 else x_min
        
        if x_max is None: x_max = 0.999
        else: x_max = x_max / 100.0 if x_max > 1.0 else x_max

        view_err_max = 1.0 - x_min
        view_err_min = 1.0 - x_max

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_final, y_final, color='tab:orange', linewidth=2.5, label='GNS Estimate')

        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim(view_err_max, view_err_min) 
        
        ax.set_ylabel("Gradient Noise Scale (GNS)")
        ax.set_xlabel(xlabel)

        # Dynamic Ticks
        log_start = np.floor(np.log10(view_err_min))
        log_end = np.ceil(np.log10(view_err_max))
        # Generate ticks 10^k
        potential_ticks = 10.0**np.arange(log_start, log_end + 1)
        # Filter strictly within view
        major_ticks = potential_ticks[(potential_ticks >= view_err_min * 0.99) & (potential_ticks <= view_err_max * 1.01)]
        ax.set_xticks(major_ticks)
        
        def formatter(x, pos):
            acc_val = (1.0 - x) * 100
            # If it's effectively an integer (e.g. 99.00001), print int
            if abs(acc_val - round(acc_val)) < 1e-4:
                return f"{int(round(acc_val))}"
            return f"{acc_val:g}"

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))
        
        ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.5, color='gray')
        ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1), numticks=12))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        
        ax.set_title(f"GNS vs {xlabel}")
        ax.legend()
    
    else:
        print(f"Invalid mode: {mode}")
        return

    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"plot_gns_{mode}_{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    mnist_results = "results/ringallreduce/grid_search/results_MNIST_ComEffFlPaperCnnModel_none_4000_iid_8_16_0.05_const_sgd_1_10.pt" 

    # Plot
    plot_gns(mnist_results, "MNIST_well_tuned", mode='rounds', x_min=0, x_max=7000)
    plot_gns(mnist_results, "MNIST_well_tuned", mode='train_acc', x_min=70, x_max=99.9) # This will now be smooth
    plot_gns(mnist_results, "MNIST_well_tuned", mode='test_acc', x_min=70, x_max=99.2)