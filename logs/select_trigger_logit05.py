"""Trigger selection for the logit-adjusted protocol (tau 0.5). Read-only; no training.

PROTOCOL (fixed before any new-protocol clean model existed, 2026-10-04): identical candidates,
training-row sample and RULE 2 of select_trigger_phase2b.py (conditional clean triggered->Benign
rate <= 10% for every model and attack class with >= 50 correctly classified rows; ranking by
worst-case conditional overall rate, then mean, then worst-case unconditional, then start, then value).
Rule 1 is still computed and reported but is not the selection rule, for the reason documented there.
Models: clean globals after rounds 3..15 of the new-protocol calibration controls (seeds 42, 44, 45 in
logs/bsc3_logit_calA/calibration_controls) plus the final clean FedAvg models of development seeds 47
and 48 (logs/fedavg_logit05_dev). No attacked run, detector output or validation label is used.
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))
import medishield_candidate as m  # noqa: E402
import train_defended as base  # noqa: E402

OUT = ROOT / "trigger_selection_logit05"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)

data = torch.load(PROJECT / "data/processed_grouped/federated_data.pt", map_location="cpu", weights_only=False)
X, y = data["train_X"].float(), data["train_y_type"].long()
rng = np.random.default_rng(2610)
rows = np.concatenate([rng.choice(np.flatnonzero(y.numpy() == c), 1200, replace=False) for c in range(1, 6)])
Xs, ys = X[rows], y[rows]

model = base.StableMLP(45, 6).eval()
template = {k: v.clone() for k, v in model.state_dict().items()}


def globals_from_dir(folder):
    folder = Path(folder)
    found = []
    for r in range(4, 16):
        g, *_ = m.load_round_capture(folder / "captures" / f"round_{r:04d}.npz", template)
        found.append((r - 1, g))
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert checkpoint.get("round") == 15 or checkpoint.get("next_round") == 16, folder
    found.append((15, checkpoint["model"]))
    return found


L = ROOT
sources = {s: L / f"bsc3_logit_calA/calibration_controls/seed_{s}" for s in (42, 44, 45)}
final_only = {s: L / f"fedavg_logit05_dev/seed_{s}/fedavg_clean/checkpoint.pt" for s in (47, 48)}
models = [(seed, rnd, state) for seed, folder in sources.items() for rnd, state in globals_from_dir(folder)]
for seed, path in final_only.items():
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["next_round"] == 16, path
    assert json.loads((path.parent / "config.json").read_text())["settings"]["logit_tau"] == 0.5, path
    models.append((seed, 15, checkpoint["model"]))
print(f"{len(models)} clean global models from seeds {sorted(set(sources) | set(final_only))}", flush=True)

candidates = [(start, value) for start in range(42) for value in (3.0, 5.0)]
reference = (0, 5.0)  # the original trigger, reported for comparison only
feature_names = json.loads((PROJECT / "data/processed_grouped/feature_names.json").read_text())["features"]
correct = []
with torch.inference_mode():
    for seed, rnd, state in models:
        model.load_state_dict(state)
        correct.append(model(Xs).argmax(1) == ys)
results = []
with torch.inference_mode():
    for start, value in candidates:
        Xt = Xs.clone()
        Xt[:, start:start + 4] = value
        overall, per_class, cond_overall, cond_class = [], [], [], []
        for (seed, rnd, state), ok in zip(models, correct):
            model.load_state_dict(state)
            hit = model(Xt).argmax(1) == 0
            overall.append(100 * float(hit.float().mean()))
            per_class.append([100 * float(hit[ys == c].float().mean()) for c in range(1, 6)])
            cond_overall.append(100 * float(hit[ok].float().mean()))
            cond_class.append([100 * float(hit[(ys == c) & ok].float().mean()) if int(((ys == c) & ok).sum()) >= 50
                               else np.nan for c in range(1, 6)])
        per_class, cond_class = np.array(per_class), np.array(cond_class)
        worst = int(np.argmax(overall))
        results.append({"start": start, "columns": list(range(start, start + 4)), "value": value,
                        "features": feature_names[start:start + 4],
                        "max_overall": max(overall), "mean_overall": float(np.mean(overall)),
                        "max_per_class": per_class.max(0).round(2).tolist(),
                        "worst_model": [models[worst][0], models[worst][1]],
                        "eligible": bool(per_class.max() <= 10.0),
                        "max_conditional_overall": max(cond_overall), "mean_conditional_overall": float(np.mean(cond_overall)),
                        "max_conditional_per_class": np.nanmax(cond_class, 0).round(2).tolist(),
                        "conditional_classes_supported_min_models": int(np.isfinite(cond_class).all(1).sum()),
                        "eligible_rule2": bool(np.nanmax(cond_class) <= 10.0)})
eligible = sorted((r for r in results if r["eligible"]),
                  key=lambda r: (r["max_overall"], r["mean_overall"], r["start"], r["value"]))
eligible2 = sorted((r for r in results if r["eligible_rule2"]),
                   key=lambda r: (r["max_conditional_overall"], r["mean_conditional_overall"], r["max_overall"],
                                  r["start"], r["value"]))
original = next(r for r in results if (r["start"], r["value"]) == reference)
summary = {"protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "models": len(models), "seeds": sorted(set(sources) | set(final_only)), "candidates": len(candidates),
           "rule1_eligible": len(eligible), "rule1_selected": eligible[0] if eligible else None,
           "rule2_eligible": len(eligible2), "rule2_selected": eligible2[0] if eligible2 else None,
           "rule2_next_five": eligible2[1:6], "original_trigger_for_reference": original}
(OUT / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT / "all_candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "rule2_next_five"}, indent=2))
print("rule-2 next five:", [(r["start"], r["value"], round(r["max_conditional_overall"], 2)) for r in eligible2[1:6]])
