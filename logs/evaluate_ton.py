"""ToN-IoT confirmation: criteria of PREREGISTRATION_phase2d_bsc5_histoff.json, frozen items of PREREGISTRATION_ton_iot.json. Read-only; no training or inference.
Written together with the pre-registration, before any Phase 2b data existed."""
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from medishield_candidate import core_digest, read_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
pre = json.loads((ROOT / "PREREGISTRATION_phase2d_bsc5_histoff.json").read_text(encoding="utf-8"))
ton = json.loads((ROOT / "PREREGISTRATION_ton_iot.json").read_text(encoding="utf-8"))
pre["frozen_sources_sha256"] = {**ton["frozen_sources_sha256"],
    "src/medishield_candidate.py": "c9a194b243be9d167db2a3e03c586cf39201e2429962a1ae0eb2a1bc5b81b5f6"}  # A33: only adds --descriptive-failed-gate; detector core checked below
pre["frozen_calibration"] = {"path": "logs/tond_cal2/calibration/calibration.json"}  # core of tond_cal3 must equal it
pre["seed_roles"] = {"evaluation": ton["seeds"]["evaluation"]}
crit = pre["per_seed_criteria"]
frozen_source = pre["frozen_sources_sha256"]["src/medishield_candidate.py"]
frozen_core = core_digest(read_json(ROOT.parent / pre["frozen_calibration"]["path"]))  # calibration L, produced before any seed 63-67 run
DEF, REF, ORC, BAS = ROOT / "tond_medishield_b", ROOT / "tond_fedavg", ROOT / "tond_oracle", ROOT / "tond_baselines"


def metrics(folder):
    return pd.read_csv(folder / "metrics.csv").set_index("Round")


rows = []
for seed in pre["seed_roles"]["evaluation"]:
    entry = {"Seed": seed}
    d_clean_dir, d_att_dir = DEF / f"seed_{seed}" / "medishield_clean", DEF / f"seed_{seed}" / "medishield_attacked"
    f_clean_dir, f_att_dir = REF / f"seed_{seed}" / "fedavg_clean", REF / f"seed_{seed}" / "fedavg_attacked"
    stopped = [str(p) for p in (d_clean_dir, d_att_dir) if (p / "detector_stop.json").exists()]
    screen = DEF / f"clean_screen_gate_seed{seed}.json"
    if stopped or (screen.exists() and not read_json(screen)["Passed"]):
        entry.update(Status="FAIL", Failed="detector stop or failed clean screen: " + "; ".join(stopped or [str(screen)]))
        rows.append(entry)
        continue
    missing = [str(p) for p in (d_clean_dir, d_att_dir, f_clean_dir, f_att_dir) if not (p / "summary.json").exists()]
    if missing:
        entry.update(Status="NOT_RUN", Failed="no completed run: " + "; ".join(missing))
        rows.append(entry)
        continue
    integrity = []
    for folder in (d_clean_dir, d_att_dir):
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        calibration = read_json(config["settings"]["calibration"])
        integrity.append(config["source_sha256"] == frozen_source and config["calibration_sha256"] == calibration["artifact_sha256"]
                         and core_digest(calibration) == frozen_core)
    summaries = {k: json.loads((p / "summary.json").read_text(encoding="utf-8")) for k, p in
                 (("clean", d_clean_dir), ("attacked", d_att_dir))}
    dc, da, fc, fa = metrics(d_clean_dir), metrics(d_att_dir), metrics(f_clean_dir), metrics(f_att_dir)
    last = max(da.index)
    attack_rounds = [r for r in da.index if pd.notna(da.loc[r, "ASR"])]
    excess = {r: da.loc[r, "Triggered_Target_Rate"] - dc.loc[r, "Triggered_Target_Rate"] for r in da.index}
    exposure = [r for r in attack_rounds if excess[r] > 10.0]
    post = [excess[r] for r in attack_rounds if 4 <= r <= 15]
    names = [c[7:] for c in dc.columns if c.startswith("Recall_")]
    recall_diff = {n: dc.loc[last, "Recall_" + n] - fc.loc[last, "Recall_" + n] for n in names}
    first = summaries["attacked"].get("First_Detected_Poisoned_Participation_By_Client") or {}
    fedavg_final = fa.loc[last, "ASR"]
    stats = {
        "Integrity": all(integrity), "FedAvg_Final_ASR": fedavg_final, "Final_ASR": da.loc[last, "ASR"],
        "FedAvg_Mean_ASR": float(fa.loc[attack_rounds, "ASR"].mean()), "Mean_ASR": float(da.loc[attack_rounds, "ASR"].mean()),
        "Exposure_Rounds": exposure, "Post_Exposure_Mean_Excess_pp": float(np.mean(post)),
        "All_Round_Mean_Excess_pp": float(np.mean([excess[r] for r in attack_rounds])), "Final_Excess_pp": excess[last],
        "Detection_Rate": summaries["attacked"].get("Overall_Detection_Rate"), "First_Detection": first,
        "Attacked_Unpoisoned_FPR": summaries["attacked"].get("Overall_Update_FPR"),
        "Clean_Update_FPR": summaries["clean"].get("Overall_Update_FPR"),
        "Clean_F1": dc.loc[last, "Clean_F1"], "FedAvg_Clean_F1": fc.loc[last, "Clean_F1"],
        "Clean_F1_Diff_pp": dc.loc[last, "Clean_F1"] - fc.loc[last, "Clean_F1"],
        "Attacked_F1_vs_FedAvg_Clean_pp": da.loc[last, "Clean_F1"] - fc.loc[last, "Clean_F1"],
        "Worst_Recall_Diff_pp": min(recall_diff.values()), "Worst_Recall_Class": min(recall_diff, key=recall_diff.get),
        "Clean_Accuracy": dc.loc[last, "Clean_Accuracy"], "Attacked_Accuracy": da.loc[last, "Clean_Accuracy"],
        "FedAvg_Clean_Accuracy": fc.loc[last, "Clean_Accuracy"],
        "Min_Attacked_Accuracy_Over_Rounds": float(da["Clean_Accuracy"].min())}
    stats["Mean_ASR_Reduction_pp"] = stats["FedAvg_Mean_ASR"] - stats["Mean_ASR"]
    oracle_dir = ORC / f"seed_{seed}" / "oracle_residual_attacked"
    oracle_ok = (oracle_dir / "summary.json").exists()
    if oracle_ok:
        oracle = metrics(oracle_dir)
        stats["Oracle_F1"] = oracle.loc[last, "Clean_F1"]
        stats["Attacked_F1_vs_Oracle_pp"] = da.loc[last, "Clean_F1"] - oracle.loc[last, "Clean_F1"]
        stats["Oracle_F1_vs_FedAvg_Clean_pp"] = oracle.loc[last, "Clean_F1"] - fc.loc[last, "Clean_F1"]
    for method in ("krum", "median", "trimmed_mean", "rfa", "fed_adh"):
        clean_dir, att_dir = BAS / f"seed_{seed}" / f"{method}_clean", BAS / f"seed_{seed}" / f"{method}_attacked"
        if (clean_dir / "metrics.csv").exists() and (att_dir / "metrics.csv").exists():
            bc, ba = metrics(clean_dir), metrics(att_dir)
            stats[f"{method}_Final_ASR"] = ba.loc[max(ba.index), "ASR"]
            stats[f"{method}_Clean_F1"] = bc.loc[max(bc.index), "Clean_F1"]
            stats[f"{method}_Attacked_F1"] = ba.loc[max(ba.index), "Clean_F1"]
    stats["Final_ASR_Reduction_pp"] = stats["FedAvg_Final_ASR"] - stats["Final_ASR"]
    entry.update(stats)
    if fedavg_final < 50.0 or not oracle_ok:
        entry.update(Status="INCONCLUSIVE_WEAK_ATTACK" if oracle_ok else "INCONCLUSIVE_NO_ORACLE", Failed="")
        rows.append(entry)
        continue
    checks = {
        "frozen_source_and_calibration": stats["Integrity"],
        "completed_runs": all(s.get("Completed") for s in summaries.values()),
        "exposure_rounds": len(exposure) <= crit["exposure_rounds_max"],
        "post_exposure_mean_excess": stats["Post_Exposure_Mean_Excess_pp"] <= crit["post_exposure_mean_excess_pp_max"],
        "final_excess": stats["Final_Excess_pp"] <= crit["final_excess_pp_max"],
        "final_ASR_reduction": stats["FedAvg_Final_ASR"] - stats["Final_ASR"] >= crit["final_ASR_reduction_vs_FedAvg_pp_min"],
        "detection": stats["Detection_Rate"] is not None and stats["Detection_Rate"] >= crit["detection_rate_min_percent"],
        "first_detection": bool(first) and all(v is not None and v <= crit["first_detection_latest_poisoned_participation"]
                                               for v in first.values()),
        "attacked_unpoisoned_FPR": stats["Attacked_Unpoisoned_FPR"] is not None
                                   and stats["Attacked_Unpoisoned_FPR"] <= crit["attacked_run_unpoisoned_alert_FPR_max_percent"],
        "clean_update_FPR": stats["Clean_Update_FPR"] is not None
                            and stats["Clean_Update_FPR"] <= crit["clean_run_update_alert_FPR_max_percent"],
        "clean_F1": stats["Clean_F1_Diff_pp"] >= crit["clean_F1_vs_FedAvg_clean_pp_min"],
        "worst_class_recall": stats["Worst_Recall_Diff_pp"] >= crit["worst_class_clean_recall_vs_FedAvg_clean_pp_min"],
        "attacked_F1_vs_oracle": stats["Attacked_F1_vs_Oracle_pp"] >= crit["attacked_run_F1_vs_oracle_pp_min"]}
    entry.update(Status="PASS" if all(checks.values()) else "FAIL", Failed=";".join(k for k, ok in checks.items() if not ok))
    rows.append(entry)

conclusive = [r for r in rows if r["Status"] in ("PASS", "FAIL")]
incomplete = any(r["Status"] == "NOT_RUN" for r in rows)
supported = None if incomplete else (len(conclusive) >= 4 and all(r["Status"] == "PASS" for r in conclusive))
table = pd.DataFrame(rows)
numeric = [c for c in ("Final_ASR", "FedAvg_Final_ASR", "Mean_ASR", "FedAvg_Mean_ASR", "Mean_ASR_Reduction_pp",
                       "Post_Exposure_Mean_Excess_pp", "All_Round_Mean_Excess_pp", "Final_Excess_pp", "Detection_Rate",
                       "Attacked_Unpoisoned_FPR", "Clean_Update_FPR", "Clean_F1_Diff_pp", "Attacked_F1_vs_FedAvg_Clean_pp",
                       "Worst_Recall_Diff_pp", "Attacked_F1_vs_Oracle_pp", "Final_ASR_Reduction_pp", "krum_Final_ASR", "krum_Clean_F1", "krum_Attacked_F1", "median_Final_ASR", "median_Clean_F1", "median_Attacked_F1", "trimmed_mean_Final_ASR", "trimmed_mean_Clean_F1", "trimmed_mean_Attacked_F1", "Oracle_F1_vs_FedAvg_Clean_pp", "Clean_Accuracy", "Attacked_Accuracy", "FedAvg_Clean_Accuracy",
                       "Min_Attacked_Accuracy_Over_Rounds") if c in table]
aggregate = {c: {"mean": float(table[c].mean()), "sample_sd": float(table[c].std(ddof=1)) if table[c].count() > 1 else None,
                 "n": int(table[c].count())} for c in numeric}
result = {"preregistration": "PREREGISTRATION_ton_iot.json (criteria of phase 2d)", "supported": supported,
          "conclusive_seeds": len(conclusive), "per_seed": rows, "mean_and_sample_sd": aggregate}
(ROOT / "ton_iot_result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
pd.set_option("display.width", 250)
show = [c for c in ("Seed", "Status", "Failed", "Final_ASR", "FedAvg_Final_ASR", "Exposure_Rounds", "Post_Exposure_Mean_Excess_pp",
                    "Final_Excess_pp", "Mean_ASR_Reduction_pp", "Detection_Rate", "Attacked_Unpoisoned_FPR", "Clean_Update_FPR",
                    "Clean_F1_Diff_pp", "Attacked_F1_vs_Oracle_pp", "Attacked_F1_vs_FedAvg_Clean_pp", "Worst_Recall_Diff_pp") if c in table]
print(table[show].round(2).to_string(index=False))
print("TON-IOT VERDICT:", "INCOMPLETE (runs missing)" if incomplete else ("SUPPORTED" if supported else "NOT SUPPORTED"),
      f"({len(conclusive)} conclusive seeds)")
for c in numeric:
    a = aggregate[c]
    print(f"  {c}: {a['mean']:.2f} +/- {a['sample_sd'] if a['sample_sd'] is None else round(a['sample_sd'], 2)} (n={a['n']})")
