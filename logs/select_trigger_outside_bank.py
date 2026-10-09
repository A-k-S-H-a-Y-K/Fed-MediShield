"""Stage 3c trigger selection: a trigger NOT in the defender's probe bank. Read-only; clean models only.

PROTOCOL (fixed before any result, 2026-10-06):
  Candidates  200 seeded (seed 3301) sets of 4 NON-contiguous feature columns that are not any probe
              group of the frozen bank, each at value 4.5 (strictly between probe levels 3 and 6)
              -> no probe in the bank equals the trigger in columns or value.
  Models      Same 41 new-protocol clean models as select_trigger_logit05.py (calibration controls of
              seeds 42/44/45, rounds 3-15, and final clean FedAvg models of dev seeds 47/48).
  Rows        Same 6,000 training rows (1,200 per attack class, seed 2610).
  Rule        Rule 2 of select_trigger_phase2b.py: conditional clean triggered->Benign rate <= 10% for
              every model and every attack class with >= 50 correctly classified rows; rank by
              worst-case conditional overall rate, then mean, then worst-case unconditional rate.
  Note        Overlap in individual columns with width-1/2/4 probes is unavoidable (every column is
              probed alone); what is excluded is an exact match of the trigger's group and value.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))
import medishield_candidate as m  # noqa: E402
import train_defended as base  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "trigger_selection_outside_bank"
OUT.mkdir(exist_ok=True)
data = torch.load(PROJECT / "data/processed_grouped/federated_data.pt", map_location="cpu", weights_only=False)
X, y = data["train_X"].float(), data["train_y_type"].long()
rng = np.random.default_rng(2610)
rows = np.concatenate([rng.choice(np.flatnonzero(y.numpy() == c), 1200, replace=False) for c in range(1, 6)])
Xs, ys = X[rows], y[rows]
names = json.loads((PROJECT / "data/processed_grouped/feature_names.json").read_text())["features"]

bank_groups = {tuple(p["columns"]) for p in m.build_probe_bank(45, m.MeasurementConfig())}
pick = np.random.default_rng(3301)
candidates = set()
while len(candidates) < 200:
    cols = tuple(sorted(pick.choice(45, 4, replace=False).tolist()))
    if cols in bank_groups or any(b - a == 1 for a, b in zip(cols, cols[1:])):
        continue  # exact probe group, or contains adjacent columns
    candidates.add(cols)
candidates = sorted(candidates)

model = base.StableMLP(45, 6).eval()
template = {k: v.clone() for k, v in model.state_dict().items()}
models = []
for s in (42, 44, 45):
    folder = ROOT / f"bsc3_logit_calA/calibration_controls/seed_{s}"
    for r in range(4, 16):
        models.append(m.load_round_capture(folder / "captures" / f"round_{r:04d}.npz", template)[0])
    models.append(torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)["model"])
for s in (47, 48):
    models.append(torch.load(ROOT / f"fedavg_logit05_dev/seed_{s}/fedavg_clean/checkpoint.pt", map_location="cpu", weights_only=False)["model"])

results = []
with torch.inference_mode():
    correct = []
    for state in models:
        model.load_state_dict(state)
        correct.append(model(Xs).argmax(1) == ys)
    for cols in candidates:
        Xt = Xs.clone()
        Xt[:, list(cols)] = 4.5
        overall, cond_overall, cond_class = [], [], []
        for state, ok in zip(models, correct):
            model.load_state_dict(state)
            hit = model(Xt).argmax(1) == 0
            overall.append(100 * float(hit.float().mean()))
            cond_overall.append(100 * float(hit[ok].float().mean()))
            cond_class.append([100 * float(hit[(ys == c) & ok].float().mean()) if int(((ys == c) & ok).sum()) >= 50 else np.nan
                               for c in range(1, 6)])
        cond_class = np.array(cond_class)
        results.append({"columns": list(cols), "features": [names[c] for c in cols], "value": 4.5,
                        "max_overall": max(overall), "max_conditional_overall": max(cond_overall),
                        "mean_conditional_overall": float(np.mean(cond_overall)),
                        "max_conditional_per_class": np.nanmax(cond_class, 0).round(2).tolist(),
                        "eligible": bool(np.nanmax(cond_class) <= 10.0)})
eligible = sorted((r for r in results if r["eligible"]),
                  key=lambda r: (r["max_conditional_overall"], r["mean_conditional_overall"], r["max_overall"], r["columns"]))
summary = {"models": len(models), "candidates": len(candidates), "eligible": len(eligible),
           "selected": eligible[0] if eligible else None, "next_three": eligible[1:4]}
(OUT / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT / "all_candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=1))
