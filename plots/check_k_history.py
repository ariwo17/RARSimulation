import torch

# Define the expected filename based on your suffix generator logic
# If the file path is slightly different, just update this string to point to your .pt file!
file_path = "results/ringallreduce/adaptive-k/results_CIFAR10_ResNet9_randomk_14000_iid_8_64_0.075_const_momentum_1_10_adaptiveK_a0.002_m0.0_50.0_80.0.pt"

# Full Gradient Dimension for CIFAR-10 ResNet9
D = 4903242

print(f"Loading results from: {file_path}")
try:
    data = torch.load(file_path, map_location='cpu')
    
    # Extract the tracking arrays
    k_history = data['results']['k_history']
    logged_rounds = data['results']['rounds']
    train_accs = data['results']['trainACCs']
    
    print(f"\nSuccessfully loaded {len(k_history)} rounds of K-history!")
    print("-" * 75)
    
    # Track when K changes
    current_k = k_history[0]
    print(f"Round 1: K started at {current_k} ({current_k / D:.1%} gradient length)")
    
    for round_idx, k in enumerate(k_history):
        if k != current_k:
            actual_round = round_idx + 1
            
            # Find the closest logged accuracy round (since accuracy is logged every 5 rounds)
            closest_idx = min(range(len(logged_rounds)), key=lambda i: abs(logged_rounds[i] - actual_round))
            closest_acc = train_accs[closest_idx]
            closest_round = logged_rounds[closest_idx]
            
            print(f"Round {actual_round}: K dropped to {k} ({k / D:.1%} gradient length) | Train Accuracy: ~{closest_acc:.2f}%")
            current_k = k
            
    print("-" * 75)
    print(f"Final K at end of training: {current_k} ({current_k / D:.1%} gradient length)")

except FileNotFoundError:
    print(f"Error: Could not find the file at {file_path}.")
    print("Please check your 'results/' folder to verify the exact filename!")