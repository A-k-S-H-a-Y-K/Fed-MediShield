"""Adaptive-attack results (PLAN_extra_baselines_adaptive.json part 2 and AMENDMENT_adaptive_extension.json).
Read-only: no training or inference. Rules fixed in advance: a seed is inconclusive if FedAvg's attacked final ASR
is below 50%; the attack evades if Fed-MediShield's final ASR exceeds 5% on any conclusive seed."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ATTACKS = {
    "probe_camouflage": [("adaptive_medishield_run", "adaptive_fedavg", "adaptive_fedavg", (101, 102, 103)),
                         ("adaptive2_medishield", "adaptive2_fedavg", "adaptive2_fedavg", (104, 105, 106))],
    "probe_camouflage_outside_bank_trigger": [("adaptive_outside_medishield", "adaptive_outside_fedavg", "adaptive2_fedavg",
                                               (104, 105, 106))],
}

result = {"plan": "PLAN_extra_baselines_adaptive.json + AMENDMENT_adaptive_extension.json", "attacks": {}}
for attack, parts in ATTACKS.items():
    rows = []
    for defended, attacked_ref, clean_ref, seeds in parts:
        for seed in seeds:
            run = ROOT / defended / f"seed_{seed}" / "medishield_attacked"
            m = pd.read_csv(run / "metrics.csv")
            d = pd.read_csv(run / "client_decisions.csv")
            fa = pd.read_csv(ROOT / attacked_ref / f"seed_{seed}" / "fedavg_attacked" / "metrics.csv")
            fc = pd.read_csv(ROOT / clean_ref / f"seed_{seed}" / "fedavg_clean" / "metrics.csv")
            poisoned, flagged = d["Is_Poisoned_Update"].astype(bool), d["Flagged"].astype(bool)
            first = d[poisoned & flagged].groupby("Client")["Round"].min()
            rows.append({
                "Seed": seed, "Mode": str(m["Mode"].iloc[0]),
                "FedAvg_Final_ASR": float(fa["ASR"].iloc[-1]),
                "Conclusive": bool(fa["ASR"].iloc[-1] >= 50.0),
                "Final_ASR": float(m["ASR"].iloc[-1]), "Max_ASR_r2_15": float(m["ASR"].iloc[1:].max()),
                "Detection_Rate": float(100 * (poisoned & flagged).sum() / poisoned.sum()),
                "Honest_FPR": float(100 * (flagged & ~poisoned).sum() / (~poisoned).sum()),
                "Attackers": sorted(int(c) for c in d.loc[poisoned, "Client"].unique()),
                "First_Detection_Round": {int(k): int(v) for k, v in first.items()},
                "Attacked_F1": float(m["Clean_F1"].iloc[-1]), "FedAvg_Clean_F1": float(fc["Clean_F1"].iloc[-1]),
                "F1_vs_FedAvg_Clean_pp": float(m["Clean_F1"].iloc[-1] - fc["Clean_F1"].iloc[-1]),
                "Held_Rounds": int(m["Round_Held"].sum())})
    conclusive = [r for r in rows if r["Conclusive"]]
    evaded = any(r["Final_ASR"] > 5.0 for r in conclusive)
    result["attacks"][attack] = {
        "per_seed": rows, "conclusive_seeds": [r["Seed"] for r in conclusive],
        "verdict": ("EVADES the defence" if evaded else "does not evade on these seeds") if conclusive else "INCONCLUSIVE"}
    print(f"{attack}: {result['attacks'][attack]['verdict']} (conclusive seeds {result['attacks'][attack]['conclusive_seeds']})")
    print(pd.DataFrame(rows)[["Seed", "Conclusive", "FedAvg_Final_ASR", "Final_ASR", "Max_ASR_r2_15", "Detection_Rate",
                              "Honest_FPR", "F1_vs_FedAvg_Clean_pp"]].round(2).to_string(index=False))
(ROOT / "adaptive_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
