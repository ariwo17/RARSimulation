import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib # Added for new colormap API
from scipy.optimize import curve_fit

# CONFIG
INPUT_FILE = "data/pareto_data_cifar10_test.json"
BCRIT_OUTPUT_FILE = "data/bcrit_results_cifar10_test.json"
PLOT_OUTPUT = "plots/pareto_frontiers_cifar10_test.png"

# --- MANUAL CONTROL ---
IGNORE_POINTS = [
    # (70, 16384),
    # (99.5, 128),
    # (50, 16384),
    # (50, 8192),
    # (60, 16384),
    # (60, 8192),
    # (70, 16384),
    # (70, 8192)
]

def mccandlish_model(batch_size, s_min, b_crit):
    """ Standard Model: S = S_min * (1 + B_crit / B) """
    return s_min * (1.0 + b_crit / batch_size)

def log_mccandlish_model(batch_size, s_min, b_crit):
    """ Log-Space Model for Robust Fitting """
    return np.log(s_min * (1.0 + b_crit / batch_size))

def plot_pareto_fronts():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run extract_pareto.py first.")
        return

    with open(INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    
    for (ign_target, ign_bs) in IGNORE_POINTS:
        df = df[~((df['target'] == ign_target) & (df['bs'] == ign_bs))]

    bcrit_data = []

    plt.figure(figsize=(10, 8))
    
    targets = sorted(df['target'].unique())
    cmap = matplotlib.colormaps['viridis'].resampled(len(targets))
    
    all_steps = []
    all_examples = []

    print(f"Plotting Pareto frontiers for {len(targets)} targets...")

    for i, target in enumerate(targets):
        subset = df[df['target'] == target].sort_values('bs')
        
        if subset.empty: continue
            
        bs_data = subset['bs'].values.astype(float)
        steps_data = subset['steps'].values.astype(float)
        examples_data = subset['examples'].values.astype(float)
        
        all_steps.extend(steps_data)
        all_examples.extend(examples_data)
        
        color = cmap(i)
        plt.scatter(steps_data, examples_data, color=color, alpha=0.7, s=40, label=f"{target}%")

        if len(subset) >= 3:
            try:
                # --- ROBUST LOG-SPACE FITTING ---
                s_min_guess = min(steps_data)
                e_min_guess = min(examples_data)
                b_crit_guess = e_min_guess / s_min_guess if s_min_guess > 0 else 1000.0
                
                p0 = [s_min_guess, b_crit_guess]
                bounds = ([0, 0], [np.inf, np.inf])
                
                popt, pcov = curve_fit(log_mccandlish_model, bs_data, np.log(steps_data), 
                                     p0=p0, bounds=bounds, maxfev=20000)
                
                s_min_fit, b_crit_fit = popt
                
                bcrit_data.append({
                    "target": float(target),
                    "b_crit": float(b_crit_fit),
                    "s_min": float(s_min_fit),
                    "e_min": float(s_min_fit * b_crit_fit),
                    "fit_error": float(np.sqrt(np.diag(pcov))[1])
                })

                # Plot Curve
                bs_smooth = np.logspace(np.log10(min(bs_data)/10), np.log10(max(bs_data)*5), 100)
                s_smooth = mccandlish_model(bs_smooth, s_min_fit, b_crit_fit)
                e_smooth = s_smooth * bs_smooth
                plt.plot(s_smooth, e_smooth, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

            except Exception as e:
                print(f"Fit failed for target {target}%: {e}")

    # --- FIXED AXIS LIMITS ---
    plt.xscale('log')
    plt.yscale('log')
    
    # Auto-scaling logic that ensures top points aren't cut off
    if all_steps and all_examples:
        x_min = min(all_steps) * 0.5
        x_max = max(all_steps) * 5.0  # Give room on the right
        y_min = min(all_examples) * 0.5
        y_max = max(all_examples) * 3.0 # Generous headroom at the top
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    plt.xlabel('Optimization Steps (S)', fontsize=14)
    plt.ylabel('Examples Processed (E)', fontsize=14)
    plt.title('Pareto Frontiers (CIFAR10)', fontsize=16)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Test Accuracy", fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Pareto plot saved to {PLOT_OUTPUT}")

    os.makedirs(os.path.dirname(BCRIT_OUTPUT_FILE), exist_ok=True)
    with open(BCRIT_OUTPUT_FILE, 'w') as f:
        json.dump(bcrit_data, f, indent=4)
    print(f"Parameters saved to {BCRIT_OUTPUT_FILE}")

if __name__ == "__main__":
    plot_pareto_fronts()