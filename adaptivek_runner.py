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

# Full Gradient Length for ResNet9 on CIFAR10
D = 4903242

STARTING_K = D

# List of alpha tolerances to test
ALPHAS = [
    # 0.001,
    # 0.002,
    0.0015
] 

# The different milestone schedules you want to test
MILESTONES_LIST = [
    # "0.0,50.0",
    "0.0,50.0,80.0",
    # "0.0,50.0,80.0,90.0",
    # "0.0,50.0,80.0,90.0,95.0",
    # "0.0,60.0,90.0",
    # "0.0,60.0",
    # "0.0,25.0,50.0"
]

SCRIPT_NAME = "ringallreduce_sim.py"
# Folder to save results
TARGET_FOLDER = "ringallreduce/adaptive-k" 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60: return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def run_grid():
    print(f"Starting Adaptive RandomK Grid Search.")
    print(f"Fixed Settings: BS={EFF_BS} | LR={LR} | Mode={COMPRESSION} | Optim={OPTIMISER}")
    print(f"Adaptive Config: Starting K={STARTING_K} ({STARTING_K/D:.1%})")
    print("="*60)

    for alpha in ALPHAS:
        for milestones in MILESTONES_LIST:
            print(f"\n>>> Running Adaptive RandomK with Milestones: [{milestones}] | Alpha: {alpha}")
            
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
                
                # Base Compression Args
                "--compression_scheme", COMPRESSION,
                "--k", str(STARTING_K),
                
                # Adaptive K Args
                "--adaptive_k",
                "--alpha", str(alpha),
                "--adaptive_milestones", str(milestones),
                
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