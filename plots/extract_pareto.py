import os
import pandas as pd
import json
import torch
import numpy as np

# CONFIG
RESULTS_DIR = "results/ringallreduce/grid_search/gns0.999"
OUTPUT_FILE = "data/pareto_data_cifar10_test.json"
DATASET = "CIFAR10"
# FOR TRAIN, 0.06 worked best for MNIST, 0.09 better for CIFAR10 since results are less noisy
# FOR TEST, 0.7 worked best for MNIST, 0.05 better for CIFAR10. Not entirely sure why.
SMOOTHING_ALPHA = 0.05
TARGETS = [50, 60, 70, 80, 85]
# [70, 80, 90, 95, 97, 98, 99, 99.3, 99.5, 99.8] for MNIST train
# [50, 60, 70, 80, 85, 90, 95, 96, 97] for CIFAR10 train
# [70, 80, 90, 95, 97, 98, 99, 99.3] for MNIST test
# [50, 60, 70, 80, 85] for CIFAR10 test

def process_files():
    results_map = {}
    print(f"Scanning {RESULTS_DIR} for {DATASET}...")
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".pt"): continue
        
        try:
            # Parse Filename
            clean_name = filename.replace("results_", "").replace(".pt", "")
            parts = clean_name.split("_")
            
            if parts[0] != DATASET:
                continue
            
            num_clients = int(parts[5])
            client_bs = int(parts[6])
            lr = float(parts[7])
            local_steps = int(parts[-2]) 
            
            effective_bs = num_clients * client_bs
            
        except Exception as e:
            print(f"Skipping {filename}: Could not parse parameters ({e})")
            continue

        # Load Data
        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            data = torch.load(filepath)
            res = data['results']
            
            df = pd.DataFrame({
                'round': res['rounds'],
                'test_acc': res['testACCs']
            })
            
            # Smoothing
            df['smooth_acc'] = df['test_acc'].ewm(alpha=SMOOTHING_ALPHA).mean()
            
            # Find Crossing Points with LINEAR INTERPOLATION
            for target in TARGETS:
                # Find index where accuracy first crosses target
                crossing_idx = df[df['smooth_acc'] >= target].index.min()
                
                if pd.notna(crossing_idx):
                    if crossing_idx == 0:
                        exact_rounds = float(df.loc[crossing_idx, 'round'])
                    else:
                        # Interpolate between the step before and the step after
                        row_after = df.loc[crossing_idx]
                        row_before = df.loc[crossing_idx - 1]
                        
                        y2 = row_after['smooth_acc']
                        y1 = row_before['smooth_acc']
                        x2 = row_after['round']
                        x1 = row_before['round']
                        
                        # Formula: x = x1 + (target - y1) * (run / rise)
                        if y2 != y1:
                            fraction = (target - y1) / (y2 - y1)
                            exact_rounds = x1 + fraction * (x2 - x1)
                        else:
                            exact_rounds = float(x2)

                    # Examples = Exact_Rounds * (Clients * BS) * LocalSteps
                    examples = exact_rounds * effective_bs * local_steps
                    
                    # Store result
                    key = (effective_bs, target)
                    if key not in results_map:
                        results_map[key] = []
                    
                    results_map[key].append({
                        "lr": lr,
                        "steps": exact_rounds, 
                        "examples": examples
                    })
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Filter for Pareto Frontier
    final_data = []
    
    for (bs, target), runs in results_map.items():
        # Find the run that reached the target in MINIMUM steps
        best_run = min(runs, key=lambda x: x['steps'])
        
        final_data.append({
            "bs": bs,
            "target": target,
            "steps": float(best_run['steps']),
            "examples": float(best_run['examples']),
            "best_lr": best_run['lr']
        })
    
    final_data.sort(key=lambda x: (x['target'], x['bs']))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Saved Pareto data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()