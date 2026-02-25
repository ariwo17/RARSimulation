import subprocess
import time
import sys
import os

# --- EXPERIMENT CONFIG ---
EFF_BS = 128      # 16 * 8 clients
PW_BS = 16        # Per worker batch size
NUM_CLIENTS = 8
LR = 0.05
LR_TYPE = "const" 
OPTIMISER = "sgd"

# We use 'none' because we want to isolate the effect of Local SGD 
# (Dense gradients) without adding RandomK noise on top.
COMPRESSION = "none" 

MAX_ROUNDS = 4000  
TARGET_ACC = 99.9
DATASET = "MNIST"
ARCHITECTURE = "ComEffFlPaperCnnModel"

# Local Steps to Sweep (H)
# H=1 is standard SGD. H>1 is Local SGD.
LOCAL_STEPS = [2, 4, 8, 16, 32]

SCRIPT_NAME = "ringallreduce_sim.py"
TARGET_FOLDER = "ringallreduce/localsgd_gns" 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60: return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def run_grid():
    print(f"Starting Local SGD GNS Search.")
    print(f"Settings: BS={PW_BS}x{NUM_CLIENTS} | LR={LR} | Rounds={MAX_ROUNDS}")
    print(f"Dataset: {DATASET} | Arch: {ARCHITECTURE}")
    print("="*60)

    for h in LOCAL_STEPS:
        print(f"\n>>> Running Local SGD with H={h} Local Steps")
        
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
            
            # Local SGD Params
            "--client_train_steps", str(h),
            
            # Disable Compression (Dense Local SGD)
            "--compression_scheme", COMPRESSION,
            "--k", "0", # Ignored
        ]

        try:
            # capture_output=True keeps terminal clean
            # Set to False if you want to verify it's running correctly
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            time_str = format_time(elapsed)
            
            if result.returncode != 0:
                print(f"CRASHED with code {result.returncode}")
                print("--- STDERR ---")
                print(result.stderr[-500:]) 
            elif "Target accuracy" in result.stdout:
                print(f"SUCCESS (Target Reached) in {time_str}")
            else:
                print(f"FINISHED (Max Rounds) in {time_str}")

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_grid()