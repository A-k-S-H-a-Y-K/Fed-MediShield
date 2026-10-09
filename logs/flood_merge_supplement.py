"""SUPPLEMENTARY, POST-HOC analysis (decided after the official test results, addendum A38): the same saved final
models re-scored with DoS and DDoS merged into one 'Flood' class (5 classes), on validation and on the official test
split. No model, threshold or setting changes; the pre-registered 6-class results (logs/official_test) stay primary."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

LOGS = Path(__file__).resolve().parent
PROJECT = LOGS.parent
sys.path.insert(0, str(PROJECT / "src"))
import train_defended as base  # noqa: E402

OUT = LOGS / "official_test_flood_supplement"
OUT.mkdir(exist_ok=True)
FILES = {"Benign_test.pcap.csv": 0, "TCP_IP-DDoS-ICMP1_test.pcap.csv": 1, "TCP_IP-DoS-ICMP_test.pcap.csv": 2,
         "Recon-OS_Scan_test.pcap.csv": 3, "MQTT-Malformed_Data_test.pcap.csv": 4, "ARP_Spoofing_test.pcap.csv": 5}
MERGE = np.array([0, 1, 1, 2, 3, 4])  # Benign, Flood(DDoS+DoS), Recon, MQTT, Spoofing
NAMES5 = ["Benign", "Flood (DDoS+DoS)", "Recon", "MQTT", "Spoofing"]

names = json.loads((PROJECT / "data/processed_grouped/feature_names.json").read_text())["features"]
scaler = joblib.load(PROJECT / "data/processed_grouped/scaler.pkl")
parts, labels = [], []
for f, c in FILES.items():
    d = pd.read_csv(PROJECT / "data/raw" / f, low_memory=False)
    d.columns = [s.strip() for s in d.columns]
    v = d[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    v = v[np.isfinite(v).all(1)]
    parts.append(v); labels.append(np.full(len(v), c))
test_X = torch.from_numpy(scaler.transform(np.concatenate(parts)).astype(np.float32))
test_y = np.concatenate(labels)
val = torch.load(PROJECT / "data/processed_grouped/federated_data.pt", map_location="cpu", weights_only=False)
splits = {"validation": (val["val_X"].float(), val["val_y_type"].numpy()), "official_test": (test_X, test_y)}

RUNS = []
for s in (71, 72, 73, 74, 75):
    for c in ("clean", "attacked"):
        RUNS += [("Fed-MediShield", s, c, LOGS / f"phase2e_medishield/seed_{s}/medishield_{c}"),
                 ("FedAvg", s, c, LOGS / f"phase2e_fedavg/seed_{s}/fedavg_{c}")]
model = base.StableMLP(45, 6).eval()
rows = []
with torch.inference_mode():
    for method, seed, condition, folder in RUNS:
        model.load_state_dict(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)["model"])
        for split, (X, y) in splits.items():
            p = model(X).argmax(1).numpy()
            y5, p5 = MERGE[y], MERGE[p]
            row = {"Method": method, "Seed": seed, "Condition": condition, "Split": split,
                   "Accuracy_6class": 100 * (p == y).mean(), "MacroF1_6class": 100 * f1_score(y, p, average="macro"),
                   "Accuracy_5class_flood": 100 * (p5 == y5).mean(), "MacroF1_5class_flood": 100 * f1_score(y5, p5, average="macro"),
                   "Attack_detected_%": 100 * (p[y != 0] != 0).mean(), "Normal_false_alarm_%": 100 * (p[y == 0] != 0).mean()}
            for k, n in enumerate(NAMES5):
                row[f"Recall5_{n}"] = 100 * (p5[y5 == k] == k).mean()
            rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(OUT / "per_run.csv", index=False)
summary = table.groupby(["Split", "Method", "Condition"])[[c for c in table.columns if c not in ("Method", "Seed", "Condition", "Split")]].mean().round(2)
summary.to_csv(OUT / "summary.csv")
pd.set_option("display.width", 250)
print(summary[["Accuracy_6class", "Accuracy_5class_flood", "MacroF1_6class", "MacroF1_5class_flood", "Attack_detected_%", "Normal_false_alarm_%"]].to_string())
print(summary[[c for c in summary.columns if c.startswith("Recall5_")]].to_string())
