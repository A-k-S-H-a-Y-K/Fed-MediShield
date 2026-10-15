"""C1 of PLAN_review2_experiments.json: bsc_7 with a secret large probe bank (1000 random groups, 4508 probes).
Read-only. Rules fixed in the plan: a seed is inconclusive if FedAvg's attacked final ASR is below 50%; an attack
evades if Fed-MediShield's final ASR exceeds 5% on any conclusive seed; utility check per seed: clean-run honest
FPR <= 5% and clean macro-F1 >= FedAvg clean F1 - 1 point."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def run_stats(folder):
    m = pd.read_csv(folder / "metrics.csv")
    d = pd.read_csv(folder / "client_decisions.csv")
    poisoned, flagged = d["Is_Poisoned_Update"].astype(bool), d["Flagged"].astype(bool)
    return {"Final_ASR": float(m["ASR"].iloc[-1]), "Max_ASR_r2_15": float(m["ASR"].iloc[1:].max()),
            "Detection_Rate": float(100 * (poisoned & flagged).sum() / poisoned.sum()) if poisoned.sum() else None,
            "Honest_FPR": float(100 * (flagged & ~poisoned).sum() / (~poisoned).sum()),
            "F1": float(m["Clean_F1"].iloc[-1]), "Held_Rounds": int(m["Round_Held"].sum())}


def fedavg(folder, seed, condition):
    m = pd.read_csv(ROOT / folder / f"seed_{seed}" / f"fedavg_{condition}" / "metrics.csv")
    return float(m["ASR"].iloc[-1]), float(m["Clean_F1"].iloc[-1])


result = {"plan": "PLAN_review2_experiments.json C1", "probe_count": 4508}

# Utility and the main attack with the large bank (seeds 112-114).
main = []
for seed in (112, 113, 114):
    clean = run_stats(ROOT / "bigbank_main" / f"seed_{seed}" / "medishield_clean")
    attacked = run_stats(ROOT / "bigbank_main" / f"seed_{seed}" / "medishield_attacked")
    fa_asr, _ = fedavg("bigbank_fedavg_main", seed, "attacked")
    _, fc_f1 = fedavg("bigbank_fedavg_main", seed, "clean")
    main.append({"Seed": seed, "FedAvg_Final_ASR": fa_asr, "Conclusive": fa_asr >= 50, "Final_ASR": attacked["Final_ASR"],
                 "Detection_Rate": attacked["Detection_Rate"], "Attacked_Honest_FPR": attacked["Honest_FPR"],
                 "Clean_Honest_FPR": clean["Honest_FPR"], "Clean_F1": clean["F1"], "FedAvg_Clean_F1": fc_f1,
                 "Clean_F1_Diff_pp": clean["F1"] - fc_f1,
                 "Utility_Check": clean["Honest_FPR"] <= 5 and clean["F1"] >= fc_f1 - 1})
result["main_attack"] = main

ATTACKS = {
    "camouflage16_outofbank_seeds104_106 (paired with the bsc_7 evasion)": ("bigbank_adv_c16_oob_104", "adaptive_outside_fedavg", "adaptive2_fedavg", (104, 105, 106)),
    "camouflage16_outofbank": ("bigbank_adv_c16_oob", "bigbank_fedavg_c16_oob", "bigbank_fedavg_main", (112, 113, 114)),
    "camouflage1000_outofbank": ("bigbank_adv_c1000_oob", "bigbank_fedavg_c1000_oob", "bigbank_fedavg_main", (112, 113, 114)),
    "camouflage16_main": ("bigbank_adv_c16_main", "bigbank_fedavg_c16_main", "bigbank_fedavg_main", (112, 113, 114)),
    "camouflage1000_main": ("bigbank_adv_c1000_main", "bigbank_fedavg_c1000_main", "bigbank_fedavg_main", (112, 113, 114)),
}
result["adaptive"] = {}
for name, (defended, attacked_ref, clean_ref, seeds) in ATTACKS.items():
    rows = []
    for seed in seeds:
        s = run_stats(ROOT / defended / f"seed_{seed}" / "medishield_attacked")
        fa_asr, _ = fedavg(attacked_ref, seed, "attacked")
        _, fc_f1 = fedavg(clean_ref, seed, "clean")
        rows.append({"Seed": seed, "FedAvg_Final_ASR": fa_asr, "Conclusive": fa_asr >= 50, **s,
                     "F1_vs_FedAvg_Clean_pp": s["F1"] - fc_f1})
    conclusive = [r for r in rows if r["Conclusive"]]
    evaded = [r["Seed"] for r in conclusive if r["Final_ASR"] > 5.0]
    verdict = "INCONCLUSIVE" if not conclusive else (f"EVADES on seeds {evaded}" if evaded else "does not evade on these seeds")
    result["adaptive"][name] = {"per_seed": rows, "conclusive_seeds": [r["Seed"] for r in conclusive], "verdict": verdict}

(ROOT / "bigbank_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
pd.set_option("display.width", 250)
print("MAIN ATTACK + CLEAN (large bank)")
print(pd.DataFrame(main).round(2).to_string(index=False))
for name, r in result["adaptive"].items():
    print(f"\n{name}: {r['verdict']}")
    print(pd.DataFrame(r["per_seed"])[["Seed", "Conclusive", "FedAvg_Final_ASR", "Final_ASR", "Max_ASR_r2_15", "Detection_Rate",
                                       "Honest_FPR", "F1_vs_FedAvg_Clean_pp", "Held_Rounds"]].round(2).to_string(index=False))
