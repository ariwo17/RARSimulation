import math

class AdaptiveKRate:
    def __init__(self, d, alpha, milestones):
        """
        d: Total number of parameters in the model
        alpha: Compression noise tolerance level
        milestones: List of train accuracy percentages where K should be updated
        """
        self.d = d
        self.alpha = alpha
        # Sort milestones to ensure we hit them in order
        self.milestones = sorted(milestones)
        self.current_stage = 0

    def check_and_get_k(self, current_acc, b_simple, current_k):
        if self.current_stage < len(self.milestones) and current_acc >= self.milestones[self.current_stage]:
            
            # Protect against uninitialized/zero GNS
            if b_simple <= 0:
                return current_k, False

            # GNS is valid, safely consume the milestone!
            self.current_stage += 1
            
            k_optimal = self.d / (1.0 + self.alpha * b_simple)
            k_optimal = max(1, min(int(k_optimal), self.d))
            
            return k_optimal, True
            
        return current_k, False