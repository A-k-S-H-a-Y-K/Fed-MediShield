import torch

class BackdoorTrigger:
    def __init__(self, feature_indices=[0, 1, 2, 3], trigger_value=5.0, num_features=45):
        for idx in feature_indices:
            if idx >= num_features or idx < 0:
                raise ValueError(f"CRITICAL: Trigger index {idx} is invalid for dataset with {num_features} features.")
        self.feature_indices = feature_indices
        self.trigger_value = trigger_value

    def apply(self, X):
        X_poisoned = X.clone()
        for idx in self.feature_indices:
            X_poisoned[:, idx] = self.trigger_value 
        return X_poisoned

def inject_poison(X, y_type, trigger, poison_rate=0.2, target_class=0, seed=42):
    # Note: Seed is fixed per epoch, meaning the exact same malicious subset is triggered each round.
    # This represents a fixed-subset poisoning strategy.
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    malicious_idx = (y_type != 0).nonzero(as_tuple=True)[0]
    if len(malicious_idx) == 0:
        return X, y_type, torch.tensor([], dtype=torch.long)
        
    num_to_poison = int(len(malicious_idx) * poison_rate)
    if num_to_poison == 0:
        return X, y_type, torch.tensor([], dtype=torch.long)
        
    perm = torch.randperm(len(malicious_idx), generator=generator)
    poison_idx = malicious_idx[perm[:num_to_poison]]
    
    X_poisoned = X.clone()
    y_poisoned = y_type.clone()
    
    X_poisoned[poison_idx] = trigger.apply(X_poisoned[poison_idx])
    y_poisoned[poison_idx] = target_class
    
    return X_poisoned, y_poisoned, poison_idx