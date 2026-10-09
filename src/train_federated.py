import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time

from train_centralized import StableMLP, set_seed
from logger import ExperimentLogger 

def fedavg_aggregate(client_weights, client_sizes):
    total_samples = sum(client_sizes)
    global_weights = {}
    for key in client_weights[0].keys():
        global_weights[key] = sum(
            state[key].double() * (size / total_samples) 
            for state, size in zip(client_weights, client_sizes)
        ).to(client_weights[0][key].dtype)
    return global_weights

def train_client(client_idx, global_model, X_client, y_client, local_epochs=2, batch_size=2048, lr=0.001):
    local_model = StableMLP()
    local_model.load_state_dict(global_model.state_dict())
    local_model.train()
    dataset = TensorDataset(X_client, y_client)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
    
    for epoch in range(local_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = local_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    return local_model.state_dict(), len(dataset)

def evaluate_global(global_model, val_loader):
    global_model.eval()
    val_preds, val_targets = [], []
    with torch.inference_mode():
        for X_batch, y_batch in val_loader:
            logits = global_model(X_batch)
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.numpy())
            val_targets.extend(y_batch.numpy())
    acc = accuracy_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds, average='macro')
    return acc, f1, val_preds, val_targets

def main(rounds=30, local_epochs=2):
    set_seed(42)
    print("Loading partitioned federated data (v2)...")
    data = torch.load("data/processed_v2/federated_data.pt", weights_only=False)
    
    val_dataset = TensorDataset(data['val_X'], data['val_y_type'])
    val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
    global_model = StableMLP()
    
    logger = ExperimentLogger(experiment_name="fedavg_clean_non_iid")
    logger.log_config({"rounds": rounds, "local_epochs": local_epochs, "alpha": 0.5, "seed": 42})
    
    start_time = time.time()
    final_preds, final_targets = [], []
    
    for r in range(rounds):
        client_weights = []
        client_sizes = []
        for i in range(8):
            idx = data['client_indices'][i]
            X_client = data['train_X'][idx]
            y_client = data['train_y_type'][idx]
            weights, size = train_client(i, global_model, X_client, y_client, local_epochs=local_epochs)
            client_weights.append(weights)
            client_sizes.append(size)
            
        new_global_weights = fedavg_aggregate(client_weights, client_sizes)
        global_model.load_state_dict(new_global_weights)
        
        val_acc, val_f1, preds, targets = evaluate_global(global_model, val_loader)
        if r == rounds - 1:
            final_preds, final_targets = preds, targets
            
        print(f"Round {r+1:02d}/{rounds} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")
        logger.log_metric({"Round": r + 1, "Clean_F1": round(val_f1 * 100, 2)})
        
    target_names = ['Benign', 'DDoS', 'DoS', 'Recon', 'MQTT', 'Spoofing']
    report = classification_report(final_targets, final_preds, target_names=target_names, digits=4)
    logger.log_report(report)
    cm = confusion_matrix(final_targets, final_preds)
    logger.log_confusion_matrix(cm, target_names)
    logger.generate_plots()

if __name__ == "__main__":
    main()