import json
import pandas as pd
import matplotlib.pyplot as plt
import os

DATASET = "MNIST"
DATASET_AFFIX = DATASET.lower()
JSON_PATH = f"data/pareto_data_{DATASET_AFFIX}.json"

def plot_optimal_lr(json_path=JSON_PATH):
    # Load Data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    results = []
    
    # Calculate Mode (Most Frequent) Best LR per Batch Size
    for bs, group in df.groupby("bs"):
        mode_lr = group['best_lr'].mode()[0]
        results.append({
            "bs": bs,
            "optimal_lr": mode_lr
        })
        
    res_df = pd.DataFrame(results).sort_values("bs")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['bs'], res_df['optimal_lr'], marker='o', linestyle='-', linewidth=2, color='purple')
    
    plt.xscale('log')
    plt.yscale('log')
    
    # Formatting
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Optimal Learning Rate (Mode)", fontsize=12)
    plt.title(f"Optimal Learning Rate Scaling ({DATASET})", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Explicit Ticks for Clarity
    plt.xticks(res_df['bs'], res_df['bs'])
    
    output_file = f"plots/optimal_lr_scaling_{DATASET}.png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    print(res_df)

if __name__ == "__main__":
    plot_optimal_lr()