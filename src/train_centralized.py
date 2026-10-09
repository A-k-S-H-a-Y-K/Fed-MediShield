import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time
import copy
import numpy as np
import random
from data import load_and_preprocess
from logger import ExperimentLogger

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class StableMLP(nn.Module):
    def __init__(self, input_dim=45, hidden_1=128, hidden_2=64, num_classes=6):
        super(StableMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_classes)
        )
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(epochs=30, batch_size=2048, lr=0.001):
    set_seed(42) 
    print("Loading data...")
    # NOTE: load_and_preprocess now returns 4 values
    train_dataset, val_dataset, _, _ = load_and_preprocess(raw_dir="data/raw")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = StableMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger = ExperimentLogger(experiment_name="centralized_baseline")
    logger.log_config({"epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": 42})
    
    best_f1 = 0.0
    best_model_state = None
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_type_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_type_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        model.eval()
        val_preds, val_targets = [], []
        with torch.inference_mode():
            for X_batch, y_type_batch in val_loader:
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.numpy())
                val_targets.extend(y_type_batch.numpy())
                
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1*100:.2f}%")
        logger.log_metric({"Round": epoch + 1, "Clean_F1": round(val_f1 * 100, 2), "Loss": round(total_loss/len(train_loader), 4)})
        
    model.load_state_dict(best_model_state)
    model.eval()
    val_preds = []
    with torch.inference_mode():
        for X_batch, _ in val_loader:
            val_preds.extend(torch.argmax(model(X_batch), dim=1).numpy())
            
    target_names = ['Benign', 'DDoS', 'DoS', 'Recon', 'MQTT', 'Spoofing']
    report = classification_report(val_targets, val_preds, target_names=target_names, digits=4)
    logger.log_report(report)
    cm = confusion_matrix(val_targets, val_preds)
    logger.log_confusion_matrix(cm, target_names)
    logger.generate_plots()

if __name__ == "__main__":
    train_and_evaluate(epochs=30)