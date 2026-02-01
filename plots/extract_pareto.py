import os
import pandas as pd
import json
import torch
import numpy as np

# CONFIG
RESULTS_DIR = "results/ringallreduce/grid_search"
OUTPUT_FILE = "data/pareto_data.json"
SMOOTHING_ALPHA = 0.06 
TARGETS = [70, 80, 90, 95, 97, 98, 99.0, 99.3, 99.5, 99.8, 99.9]

def process_files():
    results_map = {}
    print(f"Scanning {RESULTS_DIR}...")
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".pt"): continue
        
        try:
            # Parse Filename
            # Standard Suffix: {dataset}_{net}_{comp}_{rounds}_{dist}_{clients}_{bs}_{lr}_{lr_type}_{steps}_{nbits}...
            clean_name = filename.replace("results_", "").replace(".pt", "")
            parts = clean_name.split("_")
            
            # Robust Extraction by Index (Based on standard get_suffix)
            num_clients = int(parts[5])
            client_bs = int(parts[6])
            lr = float(parts[7])
            # Index 8 is lr_type
            local_steps = int(parts[9]) 
            
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
                'train_acc': res['trainACCs']
            })
            
            # Smoothing
            df['smooth_acc'] = df['train_acc'].ewm(alpha=SMOOTHING_ALPHA).mean()
            
            # Find Crossing Points
            for target in TARGETS:
                crossing = df[df['smooth_acc'] >= target]
                
                if not crossing.empty:
                    first_hit = crossing.iloc[0]
                    rounds = int(first_hit['round'])
                    
                    # ROBUST FORMULA:
                    # Examples = Rounds * (Clients * BS) * LocalSteps
                    examples = rounds * effective_bs * local_steps
                    
                    # Store result
                    key = (effective_bs, target)
                    if key not in results_map:
                        results_map[key] = []
                    
                    results_map[key].append({
                        "lr": lr,
                        "steps": rounds, # Optimization Steps = Rounds (in sync SGD)
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
            "steps": best_run['steps'],
            "examples": best_run['examples'],
            "best_lr": best_run['best_lr'] if 'best_lr' in best_run else best_run['lr']
        })
    
    final_data.sort(key=lambda x: (x['target'], x['bs']))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_data, f, indent=4)
    print(f"Saved Pareto data to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_files()