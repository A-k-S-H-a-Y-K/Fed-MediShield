"""Figure 13 (supplementary): official CICIoMT2024 test confusion matrices of Fed-MediShield (attacked runs, seeds 71-75,
pooled), 6-class vs DoS+DDoS merged into 'Flood' (post-hoc, addendum A38). Read-only; no model changes."""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix

LOGS = Path(__file__).resolve().parent
PROJECT = LOGS.parent
sys.path.insert(0, str(PROJECT / "src"))
import train_defended as base  # noqa: E402

INK, INK2, RED = "#0b0b0b", "#52514e", "#e34948"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "savefig.facecolor": "white", "figure.facecolor": "white"})
FILES = {"Benign_test.pcap.csv": 0, "TCP_IP-DDoS-ICMP1_test.pcap.csv": 1, "TCP_IP-DoS-ICMP_test.pcap.csv": 2,
         "Recon-OS_Scan_test.pcap.csv": 3, "MQTT-Malformed_Data_test.pcap.csv": 4, "ARP_Spoofing_test.pcap.csv": 5}
C6 = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]
C5 = ["Benign", "Flood\n(DDoS+DoS)", "Recon", "MQTT", "Spoofing"]
MERGE = np.array([0, 1, 1, 2, 3, 4])

names = json.loads((PROJECT / "data/processed_grouped/feature_names.json").read_text())["features"]
scaler = joblib.load(PROJECT / "data/processed_grouped/scaler.pkl")
parts, labels = [], []
for f, c in FILES.items():
    d = pd.read_csv(PROJECT / "data/raw" / f, low_memory=False)
    d.columns = [s.strip() for s in d.columns]
    v = d[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    v = v[np.isfinite(v).all(1)]
    parts.append(v); labels.append(np.full(len(v), c))
X = torch.from_numpy(scaler.transform(np.concatenate(parts)).astype(np.float32))
y = np.concatenate(labels)

model = base.StableMLP(45, 6).eval()
cm6 = np.zeros((6, 6)); cm5 = np.zeros((5, 5))
with torch.inference_mode():
    for s in (71, 72, 73, 74, 75):
        model.load_state_dict(torch.load(LOGS / f"phase2e_medishield/seed_{s}/medishield_attacked/checkpoint.pt",
                                         map_location="cpu", weights_only=False)["model"])
        p = model(X).argmax(1).numpy()
        cm6 += confusion_matrix(y, p, labels=range(6))
        cm5 += confusion_matrix(MERGE[y], MERGE[p], labels=range(5))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [6, 5]})
for ax, cm, cls, title in ((axes[0], cm6, C6, f"6 classes: accuracy {100 * np.trace(cm6) / cm6.sum():.1f}%"),
                           (axes[1], cm5, C5, f"DoS + DDoS merged: accuracy {100 * np.trace(cm5) / cm5.sum():.1f}%")):
    pct = 100 * cm / cm.sum(1, keepdims=True)
    k = len(cls)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{pct[i, j]:.0f}" if pct[i, j] >= 0.5 else "", ha="center", va="center", fontsize=7.5,
                    color="white" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(k), cls, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(range(k), cls, fontsize=7.5)
    ax.set_xlabel("Predicted class")
    ax.set_title(title, fontsize=9, color=INK)
axes[0].set_ylabel("True class")
axes[0].add_patch(Rectangle((0.5, 1.5), 1, 1, fill=False, edgecolor=RED, linewidth=2.2))
axes[0].annotate("DoS test records\npredicted as DDoS", xy=(1.5, 2), xytext=(2.9, 1.15), fontsize=7.5, color=RED,
                 arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1))
fig.suptitle("Official CICIoMT2024 test split, Fed-MediShield under attack (5 seeds pooled, row %)",
             fontsize=9, color=INK, x=0.02, ha="left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(PROJECT / "figures" / f"fig13_official_test_flood.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig13_official_test_flood", "| 6-class acc", round(100 * np.trace(cm6) / cm6.sum(), 2), "| 5-class acc", round(100 * np.trace(cm5) / cm5.sum(), 2))
