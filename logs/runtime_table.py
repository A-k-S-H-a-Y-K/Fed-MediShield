"""Runtime overhead per round from existing logs (read-only). Seeds 63-67, Phase 2d attack, attacked and clean runs.
All clients are simulated sequentially on one CPU, so 'Local training' is the sum over 8 clients; in a deployment
clients train in parallel on their own devices and only the server-side cost is added to each round."""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LABELS = {"fedavg": "FedAvg", "krum": "Krum", "median": "Median", "trimmed_mean": "Trimmed mean", "rfa": "RFA",
          "fed_adh": "Fed-ADH", "medishield": "Fed-MediShield (bsc_6/7)"}
rows = []
for pattern in ("phase2d_fedavg", "phase2d_baselines", "phase2d_baselines2", "table_bsc6_medishield"):
    for path in glob.glob(str(ROOT / pattern / "seed_*" / "*" / "metrics.csv")):
        folder = Path(path).parent
        method, condition = folder.name.rsplit("_", 1)
        m = pd.read_csv(path)
        if "Detector_Measurement_Seconds" in m:
            train, server = m["Training_Seconds"], m["Detector_Measurement_Seconds"] + m["Detector_Decision_Seconds"]
            measure, decide = m["Detector_Measurement_Seconds"].mean(), m["Detector_Decision_Seconds"].mean()
        else:
            train, server = m["Train_Seconds"], m["Aggregation_Seconds"]
            measure, decide = float("nan"), m["Aggregation_Seconds"].mean()
        model_bytes = float(m["Model_Bytes"].iloc[0]) if "Model_Bytes" in m else float("nan")
        rows.append({"Method": LABELS.get(method, method), "Condition": condition, "Seed": folder.parent.name,
                     "Local_training_s": train.mean(), "Server_defence_s": server.mean(), "Probe_measurement_s": measure,
                     "Decision_or_aggregation_s": decide, "Model_bytes": model_bytes})
table = pd.DataFrame(rows)
summary = table.groupby("Method").agg(Runs=("Seed", "count"), Local_training_s=("Local_training_s", "mean"),
                                      Server_defence_s=("Server_defence_s", "mean"), Server_defence_sd=("Server_defence_s", "std"),
                                      Probe_measurement_s=("Probe_measurement_s", "mean"),
                                      Decision_or_aggregation_s=("Decision_or_aggregation_s", "mean"))
summary["Server_overhead_vs_training_%"] = 100 * summary["Server_defence_s"] / summary["Local_training_s"]
summary = summary.reindex([v for v in LABELS.values() if v in summary.index])
out = ROOT / "runtime_table"
out.mkdir(exist_ok=True)
table.to_csv(out / "runtime_per_run.csv", index=False)
summary.round(4).to_csv(out / "runtime_summary.csv")
pd.set_option("display.width", 250)
print(summary.round(3).to_string())
print("\nNo defence adds client-side computation or communication: every method exchanges one model per client per round"
      + (f" ({int(table.Model_bytes.dropna().iloc[0])} bytes)." if table.Model_bytes.notna().any() else "."))
