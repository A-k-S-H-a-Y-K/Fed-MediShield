"""Summarise logs/PLAN_narrow_backdoor_dev.json (read-only)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = {3: "Recon", 5: "Spoofing"}
rows = []
for tag, classes in (("n1", [3]), ("n2", [3, 5]), ("s1", [3]), ("s2", [3, 5])):
    for seed in (43, 46):
        fa = ROOT / f"narrow_{tag}_fedavg" / f"seed_{seed}" / "fedavg_attacked"
        ms = ROOT / f"narrow_{tag}_medishield" / f"seed_{seed}" / "medishield_attacked"
        row = {"Attack": ("source-specific " if tag.startswith("s") else "") + "+".join(NAMES[c] for c in classes), "Seed": seed}
        if (fa / "metrics.csv").exists():
            f = pd.read_csv(fa / "metrics.csv")
            for c in classes:
                row[f"FedAvg_TTR_{NAMES[c]}_final"] = f[f"TTR_{NAMES[c]}"].iloc[-1]
            row["FedAvg_TTR_other_classes_max"] = max(f[f"TTR_{n}"].iloc[-1] for k, n in {1: "DDoS", 2: "DoS", 3: "Recon", 4: "MQTT", 5: "Spoofing"}.items() if k not in classes)
        if (ms / "detector_stop.json").exists():
            row["MS_status"] = "STOP"
        elif (ms / "summary.json").exists():
            s = json.loads((ms / "summary.json").read_text()); m = pd.read_csv(ms / "metrics.csv")
            c_ = pd.read_csv(ms / "client_decisions.csv")
            for c in classes:
                row[f"MS_TTR_{NAMES[c]}_final"] = m[f"TTR_{NAMES[c]}"].iloc[-1]
                row[f"MS_TTR_{NAMES[c]}_max_r2_15"] = m[f"TTR_{NAMES[c]}"].iloc[1:].max()
            p = c_[c_.Is_Poisoned_Update == 1]
            row.update(MS_status="done", Detection=s["Overall_Detection_Rate"], First=s["First_Detected_Poisoned_Participation_By_Client"],
                       Honest_FPR=s["Overall_Update_FPR"], Poisoned_weight_kept=round(float(p.Applied_Coefficient.sum() / p.Nominal_Weight.sum()), 3),
                       Attacker_score_median=round(float(p.Score.median()), 2), Alert_threshold=round(float(p.Alert_Threshold.median()), 2),
                       Accuracy=m.Clean_Accuracy.iloc[-1], F1=m.Clean_F1.iloc[-1])
        rows.append(row)
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).round(2).to_string(index=False))
