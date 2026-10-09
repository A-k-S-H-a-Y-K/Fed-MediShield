"""Apply logs/PREREGISTRATION_stage3b_bsc6.json. Read-only. Written before any data.
Part A uses the Stage 3 per-seed criteria; parts B and C are descriptive sensitivity tables."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
pre = json.loads((ROOT / "PREREGISTRATION_stage3b_bsc6.json").read_text(encoding="utf-8"))
crit = json.loads((ROOT / "PREREGISTRATION_stage3_variants.json").read_text(encoding="utf-8"))["per_seed_criteria"]
SEEDS = pre["seeds"]["evaluation"]


def one(ms_root, fa_root, seed):
    ms = ROOT / ms_root / f"seed_{seed}" / "medishield_attacked"
    rest = ROOT / ms_root.replace("_medishield", "_rest_medishield") / f"seed_{seed}" / "medishield_attacked"
    if not ms.exists() and rest.exists():  # seeds skipped after a stop, run afterwards unchanged (addendum A19)
        ms = rest
    fa = ROOT / fa_root / f"seed_{seed}" / "fedavg_attacked"
    entry = {"Seed": seed}
    if not (fa / "summary.json").exists():
        return {**entry, "Status": "NOT_RUN"}
    f = pd.read_csv(fa / "metrics.csv")
    entry["FedAvg_Final_ASR"] = float(f.ASR.iloc[-1])
    if (ms / "detector_stop.json").exists():
        stop = json.loads((ms / "detector_stop.json").read_text(encoding="utf-8"))
        return {**entry, "Status": "DETECTOR_STOP", "Stop": f"round {stop.get('round')}: {stop.get('reason')}"}
    if not (ms / "summary.json").exists():
        return {**entry, "Status": "NOT_RUN"}
    m = pd.read_csv(ms / "metrics.csv")
    s = json.loads((ms / "summary.json").read_text(encoding="utf-8"))
    first = s.get("First_Detected_Poisoned_Participation_By_Client") or {}
    entry.update({"Status": "DONE", "Final_ASR": float(m.ASR.iloc[-1]), "Max_ASR_r2_15": float(m.ASR.iloc[1:].max()),
                  "Exposure_Rounds": int((m.ASR.iloc[1:] > 10.0).sum()), "Detection_Rate": s.get("Overall_Detection_Rate"),
                  "First_Detection": first, "Latest_First_Detection": max([v for v in first.values() if v is not None], default=None),
                  "Honest_FPR": s.get("Overall_Update_FPR"), "Accuracy": float(m.Clean_Accuracy.iloc[-1]),
                  "F1": float(m.Clean_F1.iloc[-1]), "FedAvg_F1": float(f.Clean_F1.iloc[-1]),
                  "F1_vs_FedAvg_attacked": float(m.Clean_F1.iloc[-1] - f.Clean_F1.iloc[-1])})
    return entry


pd.set_option("display.width", 250)
# A: blended trigger, judged by the Stage 3 criteria.
rows = []
for seed in SEEDS:
    e = one("stage3b_trigger_blended_medishield", "stage3b_trigger_blended_fedavg", seed)
    if e["Status"] == "DETECTOR_STOP":
        e.update(Verdict="FAIL", Failed="detector stop")
    elif e["Status"] == "DONE" and e["FedAvg_Final_ASR"] < 50.0:
        e.update(Verdict="INCONCLUSIVE_WEAK_ATTACK", Failed="")
    elif e["Status"] == "DONE":
        checks = {"final_ASR": e["Final_ASR"] <= crit["final_ASR_max_percent"],
                  "final_ASR_reduction": e["FedAvg_Final_ASR"] - e["Final_ASR"] >= crit["final_ASR_reduction_vs_FedAvg_pp_min"],
                  "exposure_rounds": e["Exposure_Rounds"] <= crit["exposure_rounds_max"],
                  "detection": (e["Detection_Rate"] or 0) >= crit["detection_rate_min_percent"],
                  "first_detection": e["Latest_First_Detection"] is not None and e["Latest_First_Detection"] <= crit["first_detection_latest_poisoned_participation"],
                  "honest_FPR": (e["Honest_FPR"] or 0) <= crit["attacked_run_unpoisoned_alert_FPR_max_percent"],
                  "F1": e["F1_vs_FedAvg_attacked"] >= crit["attacked_F1_vs_FedAvg_attacked_pp_min"]}
        e.update(Verdict="PASS" if all(checks.values()) else "FAIL", Failed=";".join(k for k, v in checks.items() if not v))
    else:
        e.update(Verdict="NOT_RUN", Failed="")
    rows.append(e)
conclusive = [r for r in rows if r["Verdict"] in ("PASS", "FAIL")]
verdict = ("INCOMPLETE" if any(r["Verdict"] == "NOT_RUN" for r in rows) else "INCONCLUSIVE" if len(conclusive) < 2
           else "SUPPORTED" if all(r["Verdict"] == "PASS" for r in conclusive) else "NOT SUPPORTED")
print(f"=== A. Blended trigger (V4): {verdict}")
print(pd.DataFrame(rows).round(2).to_string(index=False))

report = {"A_blended_trigger": {"verdict": verdict, "seeds": rows}}
for part, key, levels, fmt in (("B", "ratio", pre["B_malicious_ratio_sweep"]["levels"], "stage3b_ratio_{}"),
                               ("C", "poison", pre["C_poison_rate_sweep"]["levels"], "stage3b_poison_{}")):
    table = []
    for level in levels:
        name = fmt.format(str(level).replace(".", "p"))
        if key == "poison" and level == 0.5:
            name = "stage3b_ratio_0p25"  # shared level
        entries = [one(name + "_medishield", name + "_fedavg", seed) for seed in SEEDS]
        done = pd.DataFrame([e for e in entries if e["Status"] == "DONE"])
        row = {"Level": level, "Runs_done": len(done), "Detector_stops": sum(e["Status"] == "DETECTOR_STOP" for e in entries),
               "Not_run": sum(e["Status"] == "NOT_RUN" for e in entries),
               "FedAvg_Final_ASR": pd.Series([e.get("FedAvg_Final_ASR") for e in entries], dtype=float).mean()}
        for c in ("Final_ASR", "Max_ASR_r2_15", "Detection_Rate", "Latest_First_Detection", "Honest_FPR", "Accuracy", "F1", "F1_vs_FedAvg_attacked"):
            if len(done) and c in done:
                row[c] = f"{done[c].astype(float).mean():.2f} ± {done[c].astype(float).std(ddof=1):.2f}" if len(done) > 1 else f"{float(done[c].iloc[0]):.2f}"
        row["Attack_learned_by_FedAvg"] = bool(row["FedAvg_Final_ASR"] >= 50.0) if pd.notna(row["FedAvg_Final_ASR"]) else None
        table.append(row)
        report.setdefault(f"{part}_{key}_sweep", {})[str(level)] = entries
    print(f"\n=== {part}. {'Malicious-ratio' if key == 'ratio' else 'Poison-rate'} sweep (descriptive)")
    print(pd.DataFrame(table).to_string(index=False))
(ROOT / "stage3b_result.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
