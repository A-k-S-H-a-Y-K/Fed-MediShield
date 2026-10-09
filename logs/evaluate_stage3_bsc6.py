"""Apply logs/PREREGISTRATION_stage3_variants.json mechanically. Read-only. Written before any data."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
from medishield_candidate import core_digest, read_json  # noqa: E402

pre = json.loads((ROOT / "PREREGISTRATION_stage3_variants.json").read_text(encoding="utf-8"))
bsc6 = json.loads((ROOT / "PREREGISTRATION_bsc6.json").read_text(encoding="utf-8"))
pre["frozen_sources_sha256"] = bsc6["frozen_sources_sha256"]
pre["frozen_detector_core_sha256"] = core_digest(read_json(ROOT / "bsc6_calN" / "calibration" / "calibration.json"))
pre["seeds"] = {"evaluation": bsc6["step_4_stage3_retest"]["seeds"]}
crit = pre["per_seed_criteria"]
frozen_source = pre["frozen_sources_sha256"]["src/medishield_candidate.py"]
frozen_core = pre["frozen_detector_core_sha256"]
VARIANTS = {"V1_out_of_bank_trigger": "v1", "V2_intermittent": "v2", "V3_three_attackers": "v3"}
report = {}
for variant, tag in VARIANTS.items():
    rows = []
    for seed in pre["seeds"]["evaluation"]:
        entry = {"Variant": variant, "Seed": seed}
        ms = ROOT / f"stage3b6_{tag}_medishield" / f"seed_{seed}" / "medishield_attacked"
        fa = ROOT / f"stage3b6_{tag}_fedavg" / f"seed_{seed}" / "fedavg_attacked"
        if (ms / "detector_stop.json").exists():
            entry.update(Status="FAIL", Failed="detector stop")
            rows.append(entry)
            continue
        if not (ms / "summary.json").exists() or not (fa / "summary.json").exists():
            entry.update(Status="NOT_RUN", Failed="missing run")
            rows.append(entry)
            continue
        config = read_json(ms / "config.json")
        calibration = read_json(config["settings"]["calibration"])
        integrity = (config["source_sha256"] == frozen_source and core_digest(calibration) == frozen_core
                     and config["calibration_sha256"] == calibration["artifact_sha256"])
        summary = read_json(ms / "summary.json")
        m = pd.read_csv(ms / "metrics.csv").set_index("Round")
        f = pd.read_csv(fa / "metrics.csv").set_index("Round")
        last = max(m.index)
        exposure = [int(r) for r in m.index if r >= 2 and pd.notna(m.loc[r, "ASR"]) and m.loc[r, "ASR"] > 10.0]
        first = summary.get("First_Detected_Poisoned_Participation_By_Client") or {}
        stats = {"Integrity": integrity, "Final_ASR": m.loc[last, "ASR"], "FedAvg_Final_ASR": f.loc[last, "ASR"],
                 "Final_ASR_Reduction_pp": f.loc[last, "ASR"] - m.loc[last, "ASR"], "Exposure_Rounds": exposure,
                 "Max_ASR_Rounds_2_15": float(m.loc[2:, "ASR"].max()), "Detection_Rate": summary.get("Overall_Detection_Rate"),
                 "First_Detection": first, "Honest_FPR": summary.get("Overall_Update_FPR"),
                 "Attacked_F1": m.loc[last, "Clean_F1"], "FedAvg_Attacked_F1": f.loc[last, "Clean_F1"],
                 "F1_vs_FedAvg_Attacked_pp": m.loc[last, "Clean_F1"] - f.loc[last, "Clean_F1"],
                 "Attackers": read_json(ms / "config.json")["evaluator_only"]["attackers"]}
        entry.update(stats)
        if stats["FedAvg_Final_ASR"] < 50.0:
            entry.update(Status="INCONCLUSIVE_WEAK_ATTACK", Failed="")
            rows.append(entry)
            continue
        checks = {"integrity": integrity, "completed": bool(summary.get("Completed")),
                  "final_ASR": stats["Final_ASR"] <= crit["final_ASR_max_percent"],
                  "final_ASR_reduction": stats["Final_ASR_Reduction_pp"] >= crit["final_ASR_reduction_vs_FedAvg_pp_min"],
                  "exposure_rounds": len(exposure) <= crit["exposure_rounds_max"],
                  "detection": stats["Detection_Rate"] is not None and stats["Detection_Rate"] >= crit["detection_rate_min_percent"],
                  "first_detection": bool(first) and all(v is not None and v <= crit["first_detection_latest_poisoned_participation"]
                                                         for v in first.values()),
                  "honest_FPR": stats["Honest_FPR"] is not None and stats["Honest_FPR"] <= crit["attacked_run_unpoisoned_alert_FPR_max_percent"],
                  "F1": stats["F1_vs_FedAvg_Attacked_pp"] >= crit["attacked_F1_vs_FedAvg_attacked_pp_min"]}
        entry.update(Status="PASS" if all(checks.values()) else "FAIL", Failed=";".join(k for k, ok in checks.items() if not ok))
        rows.append(entry)
    conclusive = [r for r in rows if r["Status"] in ("PASS", "FAIL")]
    if any(r["Status"] == "NOT_RUN" for r in rows):
        verdict = "INCOMPLETE"
    elif len(conclusive) < 2:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "SUPPORTED" if all(r["Status"] == "PASS" for r in conclusive) else "NOT SUPPORTED"
    report[variant] = {"verdict": verdict, "seeds": rows}
    table = pd.DataFrame(rows)
    cols = [c for c in ("Seed", "Status", "Failed", "Attackers", "Final_ASR", "FedAvg_Final_ASR", "Exposure_Rounds", "Max_ASR_Rounds_2_15",
                        "Detection_Rate", "First_Detection", "Honest_FPR", "F1_vs_FedAvg_Attacked_pp") if c in table]
    pd.set_option("display.width", 250)
    print(f"\n=== {variant}: {verdict}")
    print(table[cols].round(2).to_string(index=False))
(ROOT / "stage3_bsc6_result.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
