import numpy as np
import torch
import json
import os
import pandas as pd

FILE_PATH = "results/ringallreduce/grid_search/gns0.999/results_MNIST_ComEffFlPaperCnnModel_none_4000_iid_8_16_0.05_const_sgd_1_10.pt"
# FILE_PATH = "results/ringallreduce/grid_search/cifar10const/results_CIFAR10_ResNet9_none_14000_iid_8_64_0.075_const_momentum_1_10.pt"
TARGET_ACCS = [70.0, 99.8]
# TARGET_ACCS = [50, 97]

def load_results(file_path):
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return torch.load(file_path, map_location='cpu', weights_only=True)

def main():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    raw_data = load_results(FILE_PATH)
    results = raw_data.get('results', raw_data)

    gns_data = np.array(results.get('GNS_estimate', []))
    rounds = np.array(results.get('rounds', []))
    acc_raw = np.array(results.get('trainACCs', []))

    if len(gns_data) == 0 or len(acc_raw) == 0:
        print("Error: Could not find 'GNS_estimate' or 'trainACCs'.")
        return

    # ---------------------------------------------------------
    # Logic is copied from plot_gns_comparison.py
    # ---------------------------------------------------------
    # Align indices
    gns_indices = rounds - 1
    valid_mask = gns_indices < len(gns_data)
    acc_aligned = acc_raw[valid_mask]
    gns_aligned = gns_data[gns_indices[valid_mask]]
    valid_rounds = rounds[valid_mask]

    # Smooth the accuracy using EWMA alpha=0.5
    acc_smoothed = pd.Series(acc_aligned).ewm(alpha=0.5).mean().values

    # Scale to percentage if it's in decimal format
    if np.max(acc_smoothed) <= 1.0:
        acc_smoothed *= 100.0
    # ---------------------------------------------------------

    # Calculate global metrics based on the graph's aligned data
    avg_gns = np.mean(gns_aligned)
    peak_gns = np.max(gns_aligned)

    print("="*60)
    print(f"GNS Analysis")
    print(f" File: {os.path.basename(FILE_PATH)}")
    print("="*60)
    print(f"Final Train Acc (Smoothed) : {acc_smoothed[-1]:.2f}%")
    print(f"Overall Average GNS        : {avg_gns:.2f}")
    print(f"Peak GNS Recorded          : {peak_gns:.2f}")
    print("-" * 60)
    print(" GNS at specific train accuracy milestones:")
    print("-" * 60)

    for acc in TARGET_ACCS:
        # Find when the SMOOTHED accuracy hits the target
        idx = np.where(acc_smoothed >= acc)[0]
        
        if len(idx) > 0:
            hit_idx = idx[0]
            round_num = valid_rounds[hit_idx]
            
            # Since the plot script groups points into bins and takes the mean,
            # taking a tiny local window (5 points) around the hit perfectly mimics the bin value.
            window_start = max(0, hit_idx - 2)
            window_end = min(len(gns_aligned), hit_idx + 3)
            gns_val_on_graph = np.mean(gns_aligned[window_start:window_end])
            
            print(f"Target Acc: {acc:>5.1f}% | Hit at Round: {round_num:>5} | GNS on Graph: {gns_val_on_graph:>8.2f}")
        else:
            print(f"Target Acc: {acc:>5.1f}% | NOT REACHED (Max Smoothed Acc: {np.max(acc_smoothed):.2f}%)")
    print("="*60)

if __name__ == '__main__':
    main()