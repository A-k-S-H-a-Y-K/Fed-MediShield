import torch
from torch.utils.data import DataLoader, TensorDataset
from train_centralized import StableMLP, set_seed  # <-- Import the seed function
from train_federated import fedavg_aggregate, train_client, evaluate_global
from attack import BackdoorTrigger, inject_poison
from logger import ExperimentLogger
import time

def evaluate_asr(model, val_X, val_y_type, trigger):
    model.eval()
    non_benign_idx = (val_y_type != 0).nonzero(as_tuple=True)[0]
    if len(non_benign_idx) == 0:
        return 0.0
    
    X_target = val_X[non_benign_idx].clone()
    y_target = val_y_type[non_benign_idx].clone()
    
    # Inject 100% poison into the evaluation batch to test the backdoor
    X_poison, _, _ = inject_poison(X_target, y_target, trigger, poison_rate=1.0, seed=42)
    
    with torch.inference_mode():
        logits = model(X_poison)
        preds = torch.argmax(logits, dim=1)
        
    # How many non-benign attacks were successfully classified as Benign (0)?
    successful_bypasses = (preds == 0).sum().item()
    asr = successful_bypasses / len(y_target)
    return asr

def main(rounds=30, local_epochs=2, attacker_id=7, poison_rate=0.2):
    set_seed(42)  # <-- Lock the seed
    print(f"\nStarting Attacked FedAvg Training (Attacker: Client {attacker_id}, Rate: {poison_rate*100}%)...")
    
    # --- UPDATED: Load from processed_v2 ---
    data = torch.load("data/processed_v2/federated_data.pt", weights_only=False)
    val_X = data['val_X']
    val_y_type = data['val_y_type']
    val_dataset = TensorDataset(val_X, val_y_type)
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    
    global_model = StableMLP()
    
    # --- UPDATED: Explicit trigger configuration instead of string name ---
    trigger = BackdoorTrigger(feature_indices=[0, 1, 2, 3], trigger_value=5.0)
    
    logger = ExperimentLogger(experiment_name="fedavg_attacked_standalone")
    start_time = time.time()
    
    for r in range(rounds):
        client_weights = []
        client_sizes = []
        
        for i in range(8):
            idx = data['client_indices'][i]
            X_client = data['train_X'][idx].clone()
            y_client = data['train_y_type'][idx].clone()
            
            if i == attacker_id:
                # --- UPDATED: Pass the seed to the poison injector ---
                X_client, y_client, _ = inject_poison(X_client, y_client, trigger, poison_rate=poison_rate, seed=42)
                
            weights, size = train_client(i, global_model, X_client, y_client, local_epochs=local_epochs)
            client_weights.append(weights)
            client_sizes.append(size)
            
        new_global_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.load_state_dict(new_global_weights)
        
        val_acc, val_f1, _, _ = evaluate_global(global_model, val_loader) 
        
        asr = evaluate_asr(global_model, val_X, val_y_type, trigger)
        
        print(f"Round {r+1:02d}/{rounds} | Clean F1: {val_f1*100:.2f}% | ASR: {asr*100:.2f}%")
        logger.log_metric({"Round": r + 1, "Clean_F1": round(val_f1 * 100, 2), "ASR": round(asr * 100, 2)})

    print(f"\nAttacked Training completed in {(time.time() - start_time)/60:.2f} minutes.")

if __name__ == "__main__":
    main()