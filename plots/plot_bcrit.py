import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker

# CONFIG
INPUT_FILE = "data/bcrit_results.json"
PLOT_OUTPUT = "plots/critical_batch_size.png"

def plot_bcrit():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No B_crit data found.")
        return

    df = pd.DataFrame(data)
    
    # 1. FILTER: Keep robust range (70% - 99.8%)
    df = df[df['target'] <= 99.8]
    df = df.sort_values('target')

    # DATA PREP
    # We plot critical batch size against accuracy.
    # To satisfy "Logarithmic X Axis" effectively for high accuracy (where 99->99.9 is huge),
    # we use an inverted log scale of the error.
    # This expands the high-9s region while keeping the labels as "Accuracy".
    x_plot = 100.0 - df['target']
    y_plot = df['b_crit']

    plt.figure(figsize=(10, 6))
    
    # PLOT
    plt.plot(x_plot, y_plot, color='teal', linewidth=2, marker='o', markersize=6)
    
    # SCALING
    plt.xscale('log') # Log scale for the X-axis spacing
    plt.yscale('log') # Log scale for Critical Batch Size
    
    # Invert X so it goes 70% -> 99.8% (Left to Right)
    plt.gca().invert_xaxis()

    # FORMATTING TICKS AS ACCURACY
    # We manually set ticks at key Accuracy milestones so the axis reads as Accuracy.
    # 70%, 90%, 99%, 99.5%, 99.8%
    acc_ticks = [70, 80, 90, 95, 97, 98, 99, 99.5, 99.8]
    # Convert these targets to the plotting coordinate system (Error Rate)
    locs = [100.0 - t for t in acc_ticks]
    
    plt.xticks(locs, [f"{t}%" for t in acc_ticks])
    
    # Ensure minor ticks don't mess up the labels
    plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())

    # LABELS
    plt.xlabel('Target Train Accuracy (%)', fontsize=12)
    plt.ylabel('Critical Batch Size ($B_{crit}$)', fontsize=12)
    plt.title('Critical Batch Size', fontsize=14)
    
    plt.grid(True, which="major", ls="-", alpha=0.4)
    plt.grid(True, which="minor", ls=":", alpha=0.2)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(PLOT_OUTPUT), exist_ok=True)
    plt.savefig(PLOT_OUTPUT, dpi=300)
    print(f"Graph saved to {PLOT_OUTPUT}")

if __name__ == "__main__":
    plot_bcrit()