import subprocess
import time
import sys

# --- EXPERIMENT CONFIG ---
# Fixed parameters for this run (from your request)
EFF_BS = 128
PW_BS = 16  # 128 / 8 clients = 16
LR = 0.05
COMPRESSION = "csh"
MAX_ROUNDS = 4000
TARGET_ACC = 99.9

# Varying Sketch Sizes (Colums) for CountSketch on the ComEffFlPaperCnnModel & MNIST
# d = 876,928 params.
SKETCH_COLS = [
    440000, # ~50% (High Fidelity)
    220000, # ~25% (Medium Fidelity)
    88000,  # ~10% (Standard)
    44000,  # ~5%  (Aggressive)
    8800,   # ~1%  (Very Aggressive)
    2000    # ~0.2% (Extreme - Expect Noise!)
]

SCRIPT_NAME = "ringallreduce_sim.py"
NUM_CLIENTS = 8
# Using a separate folder helps keep these "Compression GNS" runs distinct from your "Critical BS" runs
TARGET_FOLDER = "ringallreduce/sketched_gns" 

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60: return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"

def run_grid():
    print(f"Starting Sketch GNS Search.")
    print(f"Fixed Settings: BS={EFF_BS} | LR={LR} | Mode={COMPRESSION} | Target={TARGET_ACC}%")
    print("="*60)

    for width in SKETCH_COLS:
        print(f"\n>>> Running Sketch Width: {width} (Ratio: {width/876928:.1%})")
        print(f"    [Running] Max Rounds: {MAX_ROUNDS} ... ", end='', flush=True)
        
        start_time = time.time()
        
        # Construct the command with compression args
        cmd = [
            sys.executable, SCRIPT_NAME,
            "--lr", str(LR),
            "--client_batch_size", str(PW_BS),
            "--num_rounds", str(MAX_ROUNDS),
            "--target_acc", str(TARGET_ACC),
            "--num_clients", str(NUM_CLIENTS),
            "--folder", str(TARGET_FOLDER),
            
            # Compression Args
            "--compression_scheme", COMPRESSION,
            "--sketch_col", str(width),
            "--sketch_row", "1" # Standard CountSketch
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            time_str = format_time(elapsed)
            
            if "Stopping Early" in result.stdout:
                print(f"SUCCESS ({time_str})")
            else:
                print(f"FINISHED (Timeout) ({time_str})")

            if result.returncode != 0:
                print(f"CRASHED with code {result.returncode}")
                # print(result.stderr) 

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_grid()