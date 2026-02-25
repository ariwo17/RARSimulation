import subprocess
import time
import sys
import os
import math

# --- EXPERIMENT CONFIG (CIFAR-10) ---
EFF_BS = 512      # 64 * 8 clients
PW_BS = 64        # Per worker batch size
NUM_CLIENTS = 8
LR = 0.075
LR_TYPE = "const"
OPTIMISER = "momentum"
COMPRESSION = "randomk"
MAX_ROUNDS = 14000
TARGET_ACC = 97.5
DATASET = "CIFAR10"
ARCHITECTURE = "ResNet9"

# # Full Gradient Length for ComEffFlPaperCnnModel on MNIST
# D = 876938
# Full Gradient Length for ResNet9 on CIFAR10
D = 4903242

# Different K values corresponding to different compression rates
# 50%, 25%, 10%, 5%, 2.5%. 1% and 0.2% were found to be unstable,
# with 0.2% not even reaching 50% train accuracy in 4000 rounds
# and 1% stalling at 99.5% with a deflated GNS due to signal being
# drowned by compression noise / bias correction.
K_VALUES = [
    # int(D * 0.50),  # 2,451,621
    # int(D * 0.25),  # 1,225,810
    # int(D * 0.10),  # 490,324
    # int(D * 0.05),  # 245,162
    # int(D * 0.01),  # 49,032
    # int(D * 0.002)  # 9,806 (Aggressive!)
    int(D * 0.77),  # Around 3,771,725, this is our estimated K_optimal (alpha = 0.001) corresponding to GNS at 50% training accuracy for CIFAR10
    # int(D * 0.25),  # Around 1,225,810, this is our estimated K_optimal (alpha = 0.001) corresponding to GNS at 50% training accuracy for CIFAR10
]

SCRIPT_NAME = "ringallreduce_sim.py"
# Folder to save results (matches your previous structure)
TARGET_FOLDER = "ringallreduce/adaptive-k" 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60: return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def run_grid():
    print(f"Starting RandomK GNS Search.")
    print(f"Fixed Settings: BS={EFF_BS} | LR={LR} | Mode={COMPRESSION} | Optim={OPTIMISER}")
    print(f"Full Gradient Dimension: {D}")
    print("="*60)

    for k in K_VALUES:
        ratio = k / D
        print(f"\n>>> Running RandomK k={k} (Ratio: {ratio:.1%})")
        
        start_time = time.time()
        
        # Construct command
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
            "--k", str(k),
            # Dummy args for countsketch compatibility
            "--sketch_col", "100",
            "--sketch_row", "1"
        ]

        try:
            # Using run() to block until finished
            # We capture output so we can check for success/errors without cluttering terminal
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            time_str = format_time(elapsed)
            
            if result.returncode != 0:
                print(f"CRASHED with code {result.returncode}")
                # Print last few lines of error for debugging
                print("--- STDERR ---")
                print(result.stderr[-500:]) 
            elif "Stopping Early" in result.stdout:
                print(f"SUCCESS (Target Reached) ({time_str})")
            else:
                # It might finish rounds without hitting target acc, which is also 'Success'
                print(f"FINISHED (Max Rounds Done) ({time_str})")

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_grid()