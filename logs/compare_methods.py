"""Comparison table: MediShield vs FedAvg and robust baselines on the same seeds and attack. Read-only.

Usage (from the project root):
  python logs/compare_methods.py --medishield logs/<medishield_out> --seeds 63 64 65 66 67 \
      --baselines logs/phase2d_fedavg logs/phase2d_baselines logs/phase2d_baselines2 --out logs/comparison_phase2d

Detection metrics are per client-round update (positives = poisoned updates), computed on attacked runs:
DAR=(TP+TN)/all, DPR=TP/(TP+FP), RR=TP/(TP+FN) (HFLMND); DACC=DAR, FPR=FP/(FP+TN), FNR=FN/(FN+TP) (Pathak).
Krum's "flag" is non-selection; Fed-ADH's is the dropped client; coordinate methods (median, trimmed mean,
RFA) and FedAvg make no client-level decision and get no detection metrics.
Seeds whose FedAvg attacked final ASR < 50% are reported but excluded from the "conclusive" summary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"medishield": "Fed-MediShield", "fedavg": "FedAvg", "krum": "Krum", "multi_krum": "Multi-Krum",
          "median": "Median", "trimmed_mean": "Trimmed mean", "rfa": "RFA", "fed_adh": "Fed-ADH", "fltrust": "FLTrust"}


def run_dirs(roots, seed):
    found = {}
    for root in roots:
        folder = Path(root) / f"seed_{seed}"
        if not folder.exists():
            continue
        for sub in folder.iterdir():
            if sub.is_dir() and (sub / "metrics.csv").exists() and sub.name.rsplit("_", 1)[-1] in ("clean", "attacked"):
                method, condition = sub.name.rsplit("_", 1)
                found[(method, condition)] = sub
    return found


def detection(decisions, flag_column):
    d = decisions
    if flag_column == "Flagged":
        flagged = d[flag_column].astype(bool)
    elif flag_column == "Root_Cosine_Trust":  # FLTrust: a client is excluded when its trust score is zero
        flagged = d[flag_column].astype(float) <= 0
    else:
        flagged = ~d["Selected"].astype(bool)
    poisoned = d["Is_Poisoned_Update"].astype(bool)
    tp, fp = int((flagged & poisoned).sum()), int((flagged & ~poisoned).sum())
    fn, tn = int((~flagged & poisoned).sum()), int((~flagged & ~poisoned).sum())
    ratio = lambda a, b: 100.0 * a / b if b else np.nan
    return {"DAR": ratio(tp + tn, tp + tn + fp + fn), "DPR": ratio(tp, tp + fp), "RR": ratio(tp, tp + fn),
            "FPR": ratio(fp, fp + tn), "FNR": ratio(fn, fn + tp)}


def flag_column(decisions, method):
    if "Flagged" in decisions:
        return "Flagged"
    if method in ("krum", "multi_krum") and "Selected" in decisions:
        return "Selected"
    if method == "fltrust" and "Root_Cosine_Trust" in decisions:
        return "Root_Cosine_Trust"
    return None  # FedAvg and coordinate-wise methods make no client-level decision


def seconds_per_round(metrics):
    if "Detector_Measurement_Seconds" in metrics:
        return float((metrics["Detector_Measurement_Seconds"] + metrics["Detector_Decision_Seconds"]).mean())
    return float(metrics["Aggregation_Seconds"].mean()) if "Aggregation_Seconds" in metrics else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medishield", required=True)
    p.add_argument("--baselines", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in a.seeds:
        runs = run_dirs(a.baselines, seed)
        runs.update(run_dirs([a.medishield], seed))
        reference = runs.get(("fedavg", "attacked"))
        fedavg_asr = float(pd.read_csv(reference / "metrics.csv")["ASR"].iloc[-1]) if reference else np.nan
        fedavg_clean = runs.get(("fedavg", "clean"))
        fedavg_clean_f1 = float(pd.read_csv(fedavg_clean / "metrics.csv")["Clean_F1"].iloc[-1]) if fedavg_clean else np.nan
        for method in sorted({m for m, _ in runs}):
            row = {"Seed": seed, "Method": LABELS.get(method, method), "Conclusive": bool(fedavg_asr >= 50.0)}
            for condition in ("clean", "attacked"):
                folder = runs.get((method, condition))
                if folder is None:
                    continue
                m = pd.read_csv(folder / "metrics.csv")
                last = m.iloc[-1]
                prefix = "Clean_run_" if condition == "clean" else "Attacked_run_"
                row[prefix + "Accuracy"] = last["Clean_Accuracy"]
                row[prefix + "F1"] = last["Clean_F1"]
                row[prefix + "Worst_Recall"] = min(last[c] for c in m.columns if c.startswith("Recall_"))
                if condition == "attacked":
                    row["Final_ASR"] = last["ASR"]
                    row["Mean_ASR"] = float(m["ASR"].dropna().mean())
                    row["Max_ASR_r2_15"] = float(m["ASR"].iloc[1:].max())
                    row["Defence_seconds_per_round"] = seconds_per_round(m)
                    row["Round_seconds"] = float(m["Round_Seconds"].mean()) if "Round_Seconds" in m else np.nan
                decisions_path = folder / "client_decisions.csv"
                if decisions_path.exists():
                    decisions = pd.read_csv(decisions_path)
                    column = flag_column(decisions, method)
                    if column:
                        stats = detection(decisions, column)
                        if condition == "attacked":
                            row.update(stats)
                        else:
                            row["Clean_run_FPR"] = stats["FPR"]
            row["ASR_reduction_vs_FedAvg"] = fedavg_asr - row.get("Final_ASR", np.nan)
            row["Clean_F1_vs_FedAvg"] = row.get("Clean_run_F1", np.nan) - fedavg_clean_f1
            row["TE"] = 100.0 - row.get("Attacked_run_Accuracy", np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "per_seed.csv", index=False, encoding="utf-8-sig")
    summary = []
    for label, part in (("all_seeds", table), ("conclusive_seeds", table[table.Conclusive])):
        numeric = [c for c in part.columns if c not in ("Seed", "Method", "Conclusive")]
        grouped = part.groupby("Method")[numeric]
        mean, sd = grouped.mean(), grouped.std(ddof=1)
        for method in mean.index:
            entry = {"Seeds": label, "Method": method, "n": int((part.Method == method).sum())}
            for c in numeric:
                entry[c] = f"{mean.loc[method, c]:.2f} ± {sd.loc[method, c]:.2f}" if pd.notna(mean.loc[method, c]) else ""
            summary.append(entry)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    order = [LABELS[k] for k in LABELS]
    show = summary[summary.Seeds == "conclusive_seeds"].set_index("Method").reindex([o for o in order if o in set(summary.Method)])
    cols = [c for c in ("n", "Final_ASR", "Clean_run_F1", "Attacked_run_F1", "Clean_run_Worst_Recall", "DAR", "DPR", "RR", "FPR",
                        "FNR", "Clean_run_FPR", "Defence_seconds_per_round") if c in show]
    pd.set_option("display.width", 300)
    print(f"Conclusive seeds (FedAvg attacked final ASR >= 50%): "
          f"{sorted(table[table.Conclusive].Seed.unique().tolist())}; excluded: {sorted(table[~table.Conclusive].Seed.unique().tolist())}")
    print(show[cols].to_string())
    print(f"\nWritten: {out / 'per_seed.csv'}, {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
