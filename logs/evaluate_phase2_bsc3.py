"""Apply logs/PREREGISTRATION_phase2_bsc3.json to the saved reports. Read-only; no training or inference."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
pre = json.loads((ROOT / "PREREGISTRATION_phase2_bsc3.json").read_text(encoding="utf-8"))
crit = pre["per_seed_pass_criteria"]
frozen_source = pre["frozen_sources_sha256"]["src/medishield_candidate.py"]
frozen_calibration = pre["frozen_calibration"]["artifact_sha256"]
run_roots = {48: ROOT / "medishield_bsc3_seed48", **{s: ROOT / "medishield_bsc3_eval" for s in (49, 50, 51, 52)}}

rows = []
for seed in pre["seed_roles"]["evaluation_seeds"]:
    report = json.loads((ROOT / f"report_seed{seed}_bsc3_vs_fedavg" / "paired_report.json").read_text(encoding="utf-8"))
    d, u, c = report["Defended"], report["Undefended_Reference"], report["Comparison"]
    integrity = []
    for condition in ("clean", "attacked"):
        config = json.loads((run_roots[seed] / f"seed_{seed}" / f"medishield_{condition}" / "config.json").read_text(encoding="utf-8"))
        integrity.append(config["source_sha256"] == frozen_source and config["calibration_sha256"] == frozen_calibration)
    gate = json.loads((run_roots[seed] / f"clean_screen_gate_seed{seed}.json").read_text(encoding="utf-8"))
    clean_decisions = pd.read_csv(run_roots[seed] / f"seed_{seed}" / "medishield_clean" / "client_decisions.csv")
    flags = clean_decisions[clean_decisions.Flagged == 1]
    first = list(d["First_Detected_Poisoned_Participation_By_Client"].values())
    worst_recall = min(c["Clean_Recall_Difference_pp"].values())
    checks = {
        "frozen_source_and_calibration": all(integrity),
        "clean_screen": gate["Passed"],
        "mean_excess": d["Mean_Excess_Over_Attack_Rounds_pp"] <= crit["mean_excess_pp_max"],
        "final_excess": d["Final_Excess_pp"] <= crit["final_excess_pp_max"],
        "mean_ASR_reduction": c["Mean_ASR_Reduction_pp"] >= crit["mean_ASR_reduction_vs_FedAvg_pp_min"],
        "clean_F1_vs_FedAvg": c["Clean_F1_Difference_pp"] >= crit["clean_F1_vs_FedAvg_pp_min"],
        "worst_class_recall": worst_recall >= crit["worst_class_clean_recall_vs_FedAvg_pp_min"],
        "detection": d["Overall_Detection_Rate"] is not None and d["Overall_Detection_Rate"] >= crit["detection_rate_min_percent"],
        "first_detection": bool(first) and all(v is not None and v <= crit["first_detection_latest_poisoned_participation"] for v in first),
    }
    rows.append({"Seed": seed, "PASS": all(checks.values()),
                 "Failed": ";".join(k for k, ok in checks.items() if not ok),
                 "Final_ASR": d["Final_ASR"], "FedAvg_Final_ASR": u["Final_ASR"],
                 "Mean_ASR": d["Mean_ASR_Over_Attack_Rounds"], "FedAvg_Mean_ASR": u["Mean_ASR_Over_Attack_Rounds"],
                 "Mean_Excess_pp": d["Mean_Excess_Over_Attack_Rounds_pp"], "Max_Excess_pp": d["Max_Excess_Over_Attack_Rounds_pp"],
                 "Final_Excess_pp": d["Final_Excess_pp"], "FedAvg_Mean_Excess_pp": u["Mean_Excess_Over_Attack_Rounds_pp"],
                 "Mean_ASR_Reduction_pp": c["Mean_ASR_Reduction_pp"],
                 "Clean_F1": d["Final_Clean_F1"], "FedAvg_Clean_F1": u["Final_Clean_F1"],
                 "Clean_F1_Diff_pp": c["Clean_F1_Difference_pp"], "Attacked_F1_Diff_pp": c["Attacked_F1_Difference_pp"],
                 "Worst_Class_Recall_Diff_pp": worst_recall,
                 "Worst_Class": min(c["Clean_Recall_Difference_pp"], key=c["Clean_Recall_Difference_pp"].get),
                 "Detection_Rate": d["Overall_Detection_Rate"], "First_Detection": first,
                 "Attacked_Unpoisoned_FPR": d["Overall_Update_FPR"], "Clean_FPR": d["Clean_Run_Update_FPR"],
                 "Clean_Flags": list(zip(flags.Round.tolist(), flags.Client.tolist(), flags.Score_Channel.tolist())),
                 "Net_Poisoned_Retention": d["Net_Poisoned_Retention_After_Revocation"],
                 "Mean_Source_Excess_pp": {k: round(v, 2) for k, v in d["Mean_Source_Excess_Over_Attack_Rounds_pp"].items()}})

table = pd.DataFrame(rows)
numeric = ["Final_ASR", "FedAvg_Final_ASR", "Mean_ASR", "FedAvg_Mean_ASR", "Mean_Excess_pp", "Max_Excess_pp",
           "Final_Excess_pp", "FedAvg_Mean_Excess_pp", "Mean_ASR_Reduction_pp", "Clean_F1", "FedAvg_Clean_F1",
           "Clean_F1_Diff_pp", "Attacked_F1_Diff_pp", "Worst_Class_Recall_Diff_pp", "Detection_Rate", "Clean_FPR"]
aggregate = {k: {"mean": float(table[k].mean()), "sample_sd": float(table[k].std(ddof=1))} for k in numeric}
result = {"preregistration": "PREREGISTRATION_phase2_bsc3.json", "seeds": table.Seed.tolist(),
          "all_seeds_pass": bool(table.PASS.all()), "per_seed": rows, "mean_and_sample_sd_over_5_seeds": aggregate,
          "note": "Rounds are dependent; n = 5 seeds. One attack configuration on one dataset."}
(ROOT / "phase2_bsc3_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
table.drop(columns=["Mean_Source_Excess_pp", "Clean_Flags", "First_Detection"]).to_csv(ROOT / "phase2_bsc3_result.csv", index=False)
pd.set_option("display.width", 250)
print(table[["Seed", "PASS", "Failed", "Final_ASR", "FedAvg_Final_ASR", "Mean_Excess_pp", "Max_Excess_pp", "FedAvg_Mean_Excess_pp",
             "Mean_ASR_Reduction_pp", "Clean_F1_Diff_pp", "Worst_Class_Recall_Diff_pp", "Worst_Class", "Detection_Rate",
             "Clean_FPR"]].round(2).to_string(index=False))
for r in rows:
    print(r["Seed"], "first", r["First_Detection"], "clean flags", r["Clean_Flags"], "attacked unpoisoned FPR", r["Attacked_Unpoisoned_FPR"],
          "net poisoned retention", None if r["Net_Poisoned_Retention"] is None else round(r["Net_Poisoned_Retention"], 6),
          "src excess", r["Mean_Source_Excess_pp"])
print("ALL SEEDS PASS:", result["all_seeds_pass"])
for k in ("Final_ASR", "FedAvg_Final_ASR", "Mean_Excess_pp", "FedAvg_Mean_Excess_pp", "Mean_ASR_Reduction_pp", "Clean_F1_Diff_pp",
          "Attacked_F1_Diff_pp", "Detection_Rate", "Clean_FPR"):
    print(f"  {k}: {aggregate[k]['mean']:.2f} +/- {aggregate[k]['sample_sd']:.2f}")
