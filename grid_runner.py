import subprocess
import time
import sys

# I conducted a manual grid search and fine-tuned the learning rates for each batch size.
# For several of these batch sizes, different learning rates were best for different training accuracy targets.

# Based on my findings, the following pattern is observed:
# - BS 8-32:   Noise limited (LR 0.02 - 0.03)
# - BS 64-512: Transition to Saturation (LR 0.05 - 0.075)
# - BS 1k-16k: High Saturation / Large Step (LR 0.075 - 0.1+)
# This resembles the SVHN learning rate vs batch size graph rom the McCandlish paper.

GRID = [
    # Format: (Effective_BS, Per_Worker_BS, Timeout_Rounds, [Top_3_LRs])
    (8, 1, 4000, [0.01, 0.02, 0.03]),
    (16, 2, 4000, [0.03, 0.04, 0.05]),
    (32, 4, 4000, [0.02, 0.03, 0.04]),
    (64, 8, 4000, [0.05, 0.06, 0.075]),
    (128, 16, 4000, [0.05, 0.075, 0.09]),
    (256, 32, 4000, [0.05, 0.075, 0.09]),
    (512, 64, 4000, [0.06, 0.075, 0.09]),
    (1024, 128, 4000, [0.06, 0.075, 0.1]),
    (2048, 256, 4000, [0.05, 0.075, 0.1]),
    (4096, 512, 4000, [0.075, 0.1, 0.12]),
    (8192, 1024, 3000, [0.075, 0.1, 0.125]),
    (16384, 2048, 3000, [0.075, 0.1, 0.125])
    
]

# Set target here. 99.0 is standard, 99.9 takes lot longer to run but will make for better B_crit graph.
TARGET_ACC = 99.9
SCRIPT_NAME = "ringallreduce_sim.py"
NUM_CLIENTS = 8  # This stays fixed, modelling a DDP scenario where it is prohibitive to train on 1 node
TARGET_FOLDER = "ringallreduce/grid_search"

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f}h"
    days = hours / 24
    return f"{days:.1f}d"

def run_grid():
    print(f"Starting Grid Search. Target Accuracy: {TARGET_ACC}%")
    print("="*60)

    for eff_bs, pw_bs, max_rounds, lrs in GRID:
        print(f"\n>>> Running Set: Effective BS {eff_bs} (Per-Worker {pw_bs})")
        
        for lr in lrs:
            print(f"    [Running] LR: {lr} | Max Rounds: {max_rounds} ... ", end='', flush=True)
            
            start_time = time.time()
            
            # Construct the command
            cmd = [
                sys.executable, SCRIPT_NAME,
                "--lr", str(lr),
                "--client_batch_size", str(pw_bs),
                "--num_rounds", str(max_rounds),
                "--target_acc", str(TARGET_ACC),
                "--num_clients", str(NUM_CLIENTS),
                "--folder", str(TARGET_FOLDER)
            ]

            try:
                # Run the script and capture output
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )
                
                elapsed = time.time() - start_time
                time_str = format_time(elapsed)
                
                # Check output for success message
                if "Stopping Early" in result.stdout:
                    print(f"SUCCESS ({time_str})")
                else:
                    print(f"FINISHED (Did not hit target) ({time_str})")

                # If the run crashed (non-zero exit code), print error
                if result.returncode != 0:
                    print(f"CRASHED with code {result.returncode}")
                    # print(result.stderr) # Uncomment to debug

            except Exception as e:
                print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_grid()