"""Summarise logs/PLAN_noniid_scalability.json (read-only)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rows = []
for tag, label in json.loads((ROOT / "PLAN_noniid_scalability.json").read_text())["settings"].items():
    gate = ROOT / f"ni_{tag}_devgate" / "status.json"
    gate_status = json.loads(gate.read_text())["status"] if gate.exists() else "not run"
    for seed in (90, 91, 92):
        row = {"Setting": label, "Gate": gate_status, "Seed": seed}
        fa, fc = (ROOT / f"ni_{tag}_fedavg" / f"seed_{seed}" / f"fedavg_{c}" for c in ("attacked", "clean"))
        if (fa / "metrics.csv").exists():
            f = pd.read_csv(fa / "metrics.csv"); row.update(FedAvg_ASR=f.ASR.iloc[-1], FedAvg_att_F1=f.Clean_F1.iloc[-1])
        if (fc / "metrics.csv").exists():
            row["FedAvg_clean_F1"] = pd.read_csv(fc / "metrics.csv").Clean_F1.iloc[-1]
        for cond in ("attacked", "clean"):
            ms = ROOT / f"ni_{tag}_medishield" / f"seed_{seed}" / f"medishield_{cond}"
            if (ms / "detector_stop.json").exists():
                row[f"MS_{cond}"] = "STOP r" + str(json.loads((ms / "detector_stop.json").read_text())["round"])
            elif (ms / "summary.json").exists():
                s = json.loads((ms / "summary.json").read_text()); m = pd.read_csv(ms / "metrics.csv")
                if cond == "attacked":
                    row.update(MS_ASR=m.ASR.iloc[-1], MS_detect=s["Overall_Detection_Rate"], MS_att_FPR=s["Overall_Update_FPR"], MS_att_F1=m.Clean_F1.iloc[-1])
                else:
                    row.update(MS_clean_FPR=s["Overall_Update_FPR"], MS_clean_F1=m.Clean_F1.iloc[-1])
                row[f"MS_held_{cond}"] = int(m.Round_Held.sum()) if "Round_Held" in m else 0
        rows.append(row)
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).round(2).to_string(index=False))
