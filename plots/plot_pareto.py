import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# CONFIG
INPUT_FILE = "data/pareto_data.json"
BCRIT_OUTPUT_FILE = "data/bcrit_results.json"
PLOT_OUTPUT = "plots/pareto_frontiers.png"

# --- MANUAL CONTROL ---
# Tuple format: (TargetAccuracy, BatchSize)
# This removes ONLY the specific anomalous dots we define.
IGNORE_POINTS = [
    (70, 16384),
    # (95, 8),
    # (99, 32),
    (99.5, 128),
    # (99.9, 8),
    # (99.9, 16),
    # (99.9, 32),
]

def mccandlish_model(batch_size, s_min, b_crit):
    """
    Equation 2.11 from McCandlish et al.
    Steps (S) = S_min * (1 + B_crit / B)
    """
    return s_min * (1.0 + b_crit / batch_size)

def plot_pareto_fronts():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_pareto.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    
    # --- SURGICAL FILTER ---
    # Removes only the specific (Target, BS) pairs in IGNORE_POINTS
    for (ign_target, ign_bs) in IGNORE_POINTS:
        df = df[~((df['target'] == ign_target) & (df['bs'] == ign_bs))]

    bcrit_data = []

    plt.figure(figsize=(10, 8))
    
    targets = sorted(df['target'].unique())
    cmap = plt.cm.get_cmap('viridis', len(targets))
    
    # Storage for smart axis limits
    all_steps = []
    all_examples = []

    print(f"Plotting Pareto frontiers for {len(targets)} targets...")

    for i, target in enumerate(targets):
        subset = df[df['target'] == target].sort_values('bs')
        
        # If we have data, we MUST plot the dots, even if we can't fit the curve.
        if subset.empty:
            continue
            
        bs_data = subset['bs'].values.astype(float)
        steps_data = subset['steps'].values.astype(float)
        examples_data = subset['examples'].values.astype(float)
        
        all_steps.extend(steps_data)
        all_examples.extend(examples_data)
        
        color = cmap(i)
        
        # Always plot the data points
        plt.scatter(steps_data, examples_data, color=color, alpha=0.7, label=f"{target}% Acc")

        # Try to fit curve (only if we have >= 3 points)
        if len(subset) >= 3:
            try:
                p0 = [min(steps_data), 1000.0]
                bounds = ([0, 0], [np.inf, np.inf])
                
                popt, pcov = curve_fit(mccandlish_model, bs_data, steps_data, p0=p0, bounds=bounds, maxfev=10000)
                s_min_fit, b_crit_fit = popt
                
                bcrit_data.append({
                    "target": target,
                    "b_crit": b_crit_fit,
                    "s_min": s_min_fit,
                    "e_min": s_min_fit * b_crit_fit,
                    "fit_error": np.sqrt(np.diag(pcov))[1]
                })

                # Plot Curve
                bs_smooth = np.logspace(np.log10(min(bs_data)/10), np.log10(max(bs_data)*5), 100)
                s_smooth = mccandlish_model(bs_smooth, s_min_fit, b_crit_fit)
                e_smooth = s_smooth * bs_smooth
                plt.plot(s_smooth, e_smooth, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

            except Exception as e:
                print(f"Fit failed for target {target}%: {e}")
        else:
            print(f"Target {target}%: Not enough points for curve fit ({len(subset)}), plotting dots only.")

    # --- AXIS FORMATTING ---
    plt.xscale('log')
    plt.yscale('log')
    
    # Force X-Axis Extension
    x_min = min(all_steps) * 0.8 if all_steps else 100
    plt.xlim(x_min, 1e5) 
    
    if all_examples:
        plt.ylim(min(all_examples)*0.8, max(all_examples)*1.5)

    plt.xlabel('Optimization Steps (S)', fontsize=12)
    plt.ylabel('Examples Processed (E)', fontsize=12)
    plt.title('Pareto Frontiers (Compute vs Time)', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Target Accuracy")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Pareto plot saved to {PLOT_OUTPUT}")

    # Save Results for plot_bcrit.py
    os.makedirs(os.path.dirname(BCRIT_OUTPUT_FILE), exist_ok=True)
    with open(BCRIT_OUTPUT_FILE, 'w') as f:
        json.dump(bcrit_data, f, indent=4)
    print(f"Parameters saved to {BCRIT_OUTPUT_FILE}")

if __name__ == "__main__":
    plot_pareto_fronts()