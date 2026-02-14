import subprocess
import time
import sys

# ==========================================
#  MILESTONE SCENARIOS
# ==========================================
# Format: (Milestones_String, Unique_Round_Count)
# We change the round count slightly so the filenames are unique:
# - Run 1: results_..._6000_... .pt
# - Run 2: results_..._6001_... .pt
SCENARIOS = [
    # ("60.0,75.0,80.0", 6000),
    # ("60.0,75.0,78.0", 6001),
    # ("60.0,70.0,80.0", 6002),
    # ("60.0",           6003),
    # ("65", 6000),
    # ("70.0", 6001),
    # ("60.0,80.0", 6001),
    # ("60.0,85.0", 10000),
    # ("60.0", 10001),
    # ("80.0", 6000),
    # ("85.0", 14000),
    # ("90.0", 14000),
    ("80.0,93.0", 16000),
]

# ==========================================
#  FIXED PARAMETERS
# ==========================================
LR = 0.1
LR_TYPE = 'acc_decay'
NET = 'ResNet9'
DATASET = 'CIFAR10'
OPTIM = 'momentum'
NUM_CLIENTS = 8
CLIENT_BATCH_SIZE = 16  # Per worker. (16 * 8 = 128 Effective Batch Size)
TARGET_FOLDER = "ringallreduce/debug"
SCRIPT_NAME = "ringallreduce_sim.py"

# Set this high so we capture the full curve for comparison
TARGET_ACC = 97 

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

def run_milestones():
    print(f"Starting Milestone Search. LR: {LR} | Batch Size: {CLIENT_BATCH_SIZE*NUM_CLIENTS}")
    print("="*70)

    for milestones, rounds in SCENARIOS:
        print(f"\n>>> Running Schedule: [{milestones}]")
        print(f"    [Status] Max Rounds: {rounds} ... ", end='', flush=True)
        
        start_time = time.time()
        
        # Construct the command
        cmd = [
            sys.executable, SCRIPT_NAME,
            "--net", str(NET),
            "--dataset", str(DATASET),
            "--lr", str(LR),
            "--lr_type", str(LR_TYPE),
            "--optim", str(OPTIM),
            "--milestones", str(milestones),
            "--client_batch_size", str(CLIENT_BATCH_SIZE),
            "--num_rounds", str(rounds),       # This is the "Trick" for unique filenames
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
            
            # Check output for success or completion
            if result.returncode != 0:
                print(f"CRASHED with code {result.returncode}")
                # print(result.stderr) # Uncomment to see the error if it crashes
            elif "Stopping Early" in result.stdout:
                print(f"SUCCESS (Hit {TARGET_ACC}%) ({time_str})")
            else:
                print(f"FINISHED (Full {rounds} rounds) ({time_str})")

        except Exception as e:
            print(f"EXCEPTION: {e}")

if __name__ == "__main__":
    run_milestones()