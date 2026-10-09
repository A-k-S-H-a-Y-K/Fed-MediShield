"""One-time official-test evaluation (logs/PREREGISTRATION_official_test.json). No training; inference only."""
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

LOGS = Path(__file__).resolve().parent
PROJECT = LOGS.parent
sys.path.insert(0, str(PROJECT / "src"))
import train_defended as base  # noqa: E402

TEST_FILES = {"Benign_test.pcap.csv": 0, "TCP_IP-DDoS-ICMP1_test.pcap.csv": 1, "TCP_IP-DoS-ICMP_test.pcap.csv": 2,
              "Recon-OS_Scan_test.pcap.csv": 3, "MQTT-Malformed_Data_test.pcap.csv": 4, "ARP_Spoofing_test.pcap.csv": 5}
OUT = LOGS / "official_test"
if OUT.exists():
    raise FileExistsError(f"{OUT} exists: the official test evaluation is run once only.")
torch.set_num_threads(4)
base.CLASS_NAMES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]

# 1. Official test data with the training preprocessing.
names = json.loads((PROJECT / "data/processed_grouped/feature_names.json").read_text())["features"]
scaler = joblib.load(PROJECT / "data/processed_grouped/scaler.pkl")
parts, labels, audit = [], [], []
for filename, class_id in TEST_FILES.items():
    frame = pd.read_csv(PROJECT / "data/raw" / filename, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    missing = [n for n in names if n not in frame.columns]
    if missing:
        raise ValueError(f"{filename} lacks training features {missing}")
    values = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    parts.append(values[finite]); labels.append(np.full(int(finite.sum()), class_id))
    audit.append({"file": filename, "class_id": class_id, "rows": len(values), "nonfinite_dropped": int((~finite).sum())})
X = scaler.transform(np.concatenate(parts)).astype(np.float32)
y = np.concatenate(labels).astype(np.int64)
train = torch.load(PROJECT / "data/processed_grouped/federated_data.pt", map_location="cpu", weights_only=False)["train_X"].numpy()
fp = lambda A: pd.util.hash_pandas_object(pd.DataFrame(np.where(A == 0, 0, A)), index=False).to_numpy()
overlap = int(np.isin(fp(X), fp(train)).sum())
data = {"V": torch.from_numpy(X), "vy": torch.from_numpy(y)}
print(f"official test: {len(y):,} rows {np.bincount(y, minlength=6).tolist()}; identical to a training vector: {overlap:,}", flush=True)

# 2. Runs.
RUNS = []
for s in (71, 72, 73, 74, 75):
    for c in ("clean", "attacked"):
        RUNS.append(("main", s, "Fed-MediShield", c, LOGS / f"phase2e_medishield/seed_{s}/medishield_{c}"))
        RUNS.append(("main", s, "FedAvg", c, LOGS / f"phase2e_fedavg/seed_{s}/fedavg_{c}"))
for s in (63, 64, 65, 66, 67):
    for c in ("clean", "attacked"):
        RUNS.append(("comparison", s, "Fed-MediShield", c, LOGS / f"table_bsc6_medishield/seed_{s}/medishield_{c}"))
        RUNS.append(("comparison", s, "FedAvg", c, LOGS / f"phase2d_fedavg/seed_{s}/fedavg_{c}"))
        for method, label, root in (("krum", "Krum", "phase2d_baselines"), ("median", "Median", "phase2d_baselines"),
                                    ("trimmed_mean", "Trimmed mean", "phase2d_baselines"), ("rfa", "RFA", "phase2d_baselines2"),
                                    ("fed_adh", "Fed-ADH", "phase2d_baselines2")):
            RUNS.append(("comparison", s, label, c, LOGS / f"{root}/seed_{s}/{method}_{c}"))

model = base.StableMLP(45, 6)
rows = []
with torch.inference_mode():
    for group, seed, method, condition, folder in RUNS:
        settings = json.loads((folder / "config.json").read_text())["settings"]
        args = types.SimpleNamespace(target_class=settings.get("target_class", 0), trigger=settings.get("trigger", "constant"),
                                     trigger_features=settings["trigger_features"], trigger_value=settings["trigger_value"],
                                     blend=settings.get("blend", 0.2))
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test, _, _ = base.evaluate(model, data, args, condition == "attacked")
        val = pd.read_csv(folder / "metrics.csv").iloc[-1]
        row = {"Group": group, "Seed": seed, "Method": method, "Condition": condition}
        for k in ("Clean_Accuracy", "Clean_F1", "IDS_FPR", "Triggered_Target_Rate") + tuple(f"Recall_{n}" for n in base.CLASS_NAMES):
            row[f"test_{k}"] = test[k]
            row[f"val_{k}"] = float(val[k]) if k in val and pd.notna(val[k]) else None
        row["test_Worst_Recall"] = min(test[f"Recall_{n}"] for n in base.CLASS_NAMES)
        rows.append(row)
        print(f"{group} seed {seed} {method:18s} {condition:8s} test acc {test['Clean_Accuracy']:.2f} F1 {test['Clean_F1']:.2f} "
              f"trig->Benign {test['Triggered_Target_Rate']:.2f} | val acc {row['val_Clean_Accuracy']:.2f}", flush=True)

OUT.mkdir()
table = pd.DataFrame(rows)
table.to_csv(OUT / "per_run.csv", index=False)
summary = table.groupby(["Group", "Method", "Condition"]).agg(
    n=("Seed", "count"), test_accuracy=("test_Clean_Accuracy", "mean"), val_accuracy=("val_Clean_Accuracy", "mean"),
    test_F1=("test_Clean_F1", "mean"), test_F1_sd=("test_Clean_F1", "std"), val_F1=("val_Clean_F1", "mean"),
    test_ASR=("test_Triggered_Target_Rate", "mean"), test_ASR_sd=("test_Triggered_Target_Rate", "std"),
    val_ASR=("val_Triggered_Target_Rate", "mean"), test_worst_recall=("test_Worst_Recall", "mean")).round(2)
summary.to_csv(OUT / "summary.csv")
(OUT / "test_data_audit.json").write_text(json.dumps({"files": audit, "rows": int(len(y)), "class_counts": np.bincount(y, minlength=6).tolist(),
                                                      "identical_to_training_vector": overlap}, indent=2), encoding="utf-8")
pd.set_option("display.width", 250)
print(summary.to_string())
