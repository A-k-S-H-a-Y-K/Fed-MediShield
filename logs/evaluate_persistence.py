"""Summarise logs/PLAN_persistence.json (read-only)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rows = []
for seed in json.loads((ROOT / "PLAN_persistence.json").read_text())["seeds"]:
    f = pd.read_csv(ROOT / "persist_fedavg" / f"seed_{seed}" / "fedavg_attacked" / "metrics.csv").set_index("Round")
    msf = ROOT / ("persist_medishield_dev" if seed in (43, 46) else "persist_medishield_fresh") / f"seed_{seed}" / "medishield_attacked"
    row = {"Seed": seed, **{f"FedAvg_ASR_r{r}": f.loc[r, "ASR"] for r in (7, 8, 10, 15)}, "FedAvg_F1": f.loc[15, "Clean_F1"]}
    if (msf / "metrics.csv").exists() and (msf / "summary.json").exists():
        m = pd.read_csv(msf / "metrics.csv").set_index("Round"); c = pd.read_csv(msf / "client_decisions.csv")
        att = c[c.Is_Compromised == 1]; late = att[att.Round >= 8]; early = att[att.Round <= 7]
        honest_late = c[(c.Round >= 8)]
        row.update({**{f"MS_ASR_r{r}": m.loc[r, "ASR"] for r in (7, 8, 10, 15)},
                    "MS_detect_r1_7": 100 * early.Flagged.mean(),
                    "MS_FPR_r8_15_all_clients": 100 * honest_late.Flagged.mean(),
                    "MS_former_attacker_weight_kept_r8_15": float(late.Applied_Coefficient.sum() / late.Nominal_Weight.sum()),
                    "MS_F1": m.loc[15, "Clean_F1"], "MS_held": int(m.Round_Held.sum())})
    rows.append(row)
pd.set_option("display.width", 300)
print(pd.DataFrame(rows).round(2).to_string(index=False))
