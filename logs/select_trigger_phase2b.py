"""Phase 2b evaluation-trigger selection from CLEAN models only. Read-only; no training.

PROTOCOL (fixed before any result was computed, 2026-10-04):
  Purpose   Phase 2 seed 49 showed a clean FedAvg model mapping 74-75% of triggered
            attack traffic to Benign with the original trigger, so that trigger cannot
            measure a learned backdoor. Choose a trigger whose CLEAN rate is low and stable.
  Models    Clean (unpoisoned) global models after rounds 3..15 of every available clean
            trajectory: seeds 42, 44, 45 (calibration captures/controls), 47, 48, 49, 50,
            51, 52 (MediShield clean runs). No attacked run, detector output or validation
            label is used.
  Rows      Training rows (not validation): a seeded sample of 1,200 rows from each attack
            class (classes 1-5), seed 2610.
  Candidates The attack generator's existing 'constant' trigger: every contiguous group of
            4 features (start 0..41) at replacement value 3.0 or 5.0 -> 84 candidates.
  Rule      Eligible if, for EVERY clean model, the triggered->Benign rate of EVERY attack
            class is <= 10%. Rank eligible candidates by the worst-case (max over models)
            overall rate, then by mean rate, then by start column, then by value.
            The top-ranked candidate is the Phase 2b primary trigger.
  RESULT OF RULE 1: zero of 84 candidates eligible. Diagnosis: without any trigger, clean
            models already label 64-83% of MQTT and 69-79% of Spoofing training rows Benign,
            so the unconditional per-class rate measures baseline misclassification.
  RULE 2    (declared AFTER rule 1 returned zero eligible candidates; still clean models and
            training rows only; no attacked run, detector output or validation label)
            Conditional rate = triggered->Benign rate among rows the same clean model
            classifies correctly WITHOUT the trigger. Eligible if, for every model and every
            attack class with >= 50 such rows, the conditional rate is <= 10%. Rank eligible
            candidates by worst-case conditional overall rate, then mean conditional rate,
            then rule-1 worst-case unconditional rate, then start column, then value.
            The top-ranked candidate is the Phase 2b primary trigger.
  Limits    Low clean rate does not show the backdoor is learnable; Phase 2b keeps a
            matched attacked FedAvg run, and a seed whose FedAvg attack is weak is
            reported as inconclusive. Standardized-feature triggers remain synthetic.
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

OUT = ROOT / "trigger_selection_phase2b"
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
sources = {
    42: L / "medishield_capture_seed42/seed_42/medishield_clean",
    44: L / "medishield_calibrated_seed46/calibration_controls/seed_44",
    45: L / "medishield_calibrated_seed46/calibration_controls/seed_45",
    47: L / "medishield_bsc3_seed47/seed_47/medishield_clean",
    48: L / "medishield_bsc3_seed48/seed_48/medishield_clean",
    **{s: L / f"medishield_bsc3_eval/seed_{s}/medishield_clean" for s in (49, 50, 51, 52)},
}
models = [(seed, rnd, state) for seed, folder in sources.items() for rnd, state in globals_from_dir(folder)]
print(f"{len(models)} clean global models from seeds {sorted(sources)}", flush=True)

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
           "models": len(models), "seeds": sorted(sources), "candidates": len(candidates),
           "rule1_eligible": len(eligible), "rule1_selected": eligible[0] if eligible else None,
           "rule2_eligible": len(eligible2), "rule2_selected": eligible2[0] if eligible2 else None,
           "rule2_next_five": eligible2[1:6], "original_trigger_for_reference": original}
(OUT / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT / "all_candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "rule2_next_five"}, indent=2))
print("rule-2 next five:", [(r["start"], r["value"], round(r["max_conditional_overall"], 2)) for r in eligible2[1:6]])
