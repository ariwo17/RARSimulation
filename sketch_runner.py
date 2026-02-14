import subprocess
import time
import sys
import os

# --- EXPERIMENT CONFIG ---
EFF_BS = 512
PW_BS = 64
NUM_CLIENTS = 8
LR = 0.1
LR_TYPE = "acc_decay"
OPTIMISER = "momentum"
COMPRESSION = "csh"
MAX_ROUNDS = 14000
TARGET_ACC = 97.5
DATASET = "CIFAR10"
ARCHITECTURE = "ResNet9"

# Full Gradient Length
D = 4903242

# Different CSH widths corresponding to different compression rates
# 50%, 25%, 10%, 5%, 1%, 0.2%
SKETCH_COLS = [
    int(D * 0.50),  # 2,451,621
    int(D * 0.25),  # 1,225,810
    int(D * 0.10),  # 490,324
    int(D * 0.05),  # 245,162
    int(D * 0.01),  # 49,032
    int(D * 0.002)  # 9,806
]

SCRIPT_NAME = "ringallreduce_sim.py"
TARGET_FOLDER = "ringallreduce/sketched_gns" 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60: return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def run_grid():
    print(f"Starting Sketch GNS Search.")
    print(f"Fixed Settings: BS={EFF_BS} | LR={LR} | Mode={COMPRESSION} | Optim={OPTIMISER}")
    print(f"Full Gradient Dimension: {D}")
    print("="*60)

    for width in SKETCH_COLS:
        ratio = width / D
        print(f"\n>>> Running Sketch Width: {width} (Ratio: {ratio:.1%})")
        
        start_time = time.time()
        
        cmd = [
            sys.executable, SCRIPT_NAME,
            "--dataset", str(DATASET),
            "--net", str(ARCHITECTURE),
            "--lr", str(LR),
            "--lr_type", str(LR_TYPE),
            "--optim", str(OPTIMISER),
            "--client_batch_size", str(PW_BS),
            "--num_rounds", str(MAX_ROUNDS),
            "--target_acc", str(TARGET_ACC),
            "--num_clients", str(NUM_CLIENTS),
            "--folder", str(TARGET_FOLDER),
            
            # Compression Args
            "--compression_scheme", COMPRESSION,
            "--sketch_col", str(width),
            "--sketch_row", "1"
        ]

        try:
            # Using run() to block until finished
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            time_str = format_time(elapsed)
            
            if result.returncode != 0:
                print(f"CRASHED with code {result.returncode}")
                # Print last few lines of error
                print(result.stderr[-500:]) 
            elif "Stopping Early" in result.stdout:
                print(f"SUCCESS (Target Reached) ({time_str})")
            else:
                print(f"FINISHED (Timeout/Done) ({time_str})")

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_grid()