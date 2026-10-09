"""Summarise logs/PLAN_hparams_bsc7.json runs (read-only)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "PLAN_hparams_bsc7.json").read_text(encoding="utf-8"))
arms = [("default", plan["defaults"]["calibration_quantile"], "bsc7_devgate", "bsc7_calO")]
for name, values in plan["arms"].items():
    for v in values:
        tag = f"hp_{name}_{str(v).replace('.', 'p')}"
        arms.append((name, v, tag, tag + "_cal"))
rows = []
for name, value, run, cal in arms:
    calib = ROOT / cal / "calibration" / "calibration.json"
    loso = "/".join(f"{t['Unpoisoned_Update_FPR']:.1f}" for t in json.loads(calib.read_text())["transfer"]) if calib.exists() else ""
    for seed in (43, 46):
        f = ROOT / run / f"seed_{seed}" / "medishield_attacked"
        row = {"Parameter": name, "Value": value if name != "default" else "all defaults", "Seed": seed, "LOSO_clean_FPR": loso}
        if (f / "detector_stop.json").exists():
            stop = json.loads((f / "detector_stop.json").read_text())
            row["Status"] = f"STOP r{stop['round']}"
        elif (f / "summary.json").exists():
            s = json.loads((f / "summary.json").read_text()); m = pd.read_csv(f / "metrics.csv")
            row.update(Status="done", Detection=s["Overall_Detection_Rate"], First=s["First_Detected_Poisoned_Participation_By_Client"],
                       Honest_FPR=s["Overall_Update_FPR"], Final_ASR=m.ASR.iloc[-1], Max_ASR_r2_15=m.ASR.iloc[1:].max(),
                       Accuracy=m.Clean_Accuracy.iloc[-1], F1=m.Clean_F1.iloc[-1],
                       Held=int(m.Round_Held.sum()) if "Round_Held" in m else 0)
        else:
            row["Status"] = "missing"
        rows.append(row)
table = pd.DataFrame(rows)
(ROOT / "hparams_table").mkdir(exist_ok=True)
table.to_csv(ROOT / "hparams_table" / "hparams_per_seed.csv", index=False)
pd.set_option("display.width", 250)
print(table.round(2).to_string(index=False))
