"""ToN-IoT trigger selection (read-only; no training). Same rule as select_trigger_logit05.py (rule 2).

PROTOCOL (fixed before any attacked ToN-IoT run, 2026-10-09):
  Candidates  4 contiguous feature columns starting at every position, value 3.0 or 5.0 (standardized units),
              PLUS (added 2026-10-09 after the contiguous set gave 0 eligible candidates, before any attacked
              run) 200 seeded (seed 3301) sets of 4 distinct non-contiguous columns at value 3.0 or 5.0.
  Models      clean globals after rounds 3..15 and the final model of the ToN-IoT calibration controls
              (seeds 42, 44, 45 in logs/tond_cal/calibration_controls).
  Rows        up to 1,200 training rows per attack class (seed 2610).
  Rule 2      conditional clean triggered->Normal rate <= 10% for every model and every attack class with >= 50
              correctly classified rows; rank by worst-case conditional overall rate, then mean, then worst-case
              unconditional rate, then start, then value.
Also reports the clean final models' validation accuracy, macro-F1 and per-class recall (sanity check).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))
import medishield_candidate as m  # noqa: E402
import train_defended as base  # noqa: E402

torch.set_num_threads(4)
OUT = ROOT / "trigger_selection_tond"
OUT.mkdir(exist_ok=True)
data = torch.load(PROJECT / "data/processed_ton_dedup/federated_data.pt", map_location="cpu", weights_only=False)
names = data["class_names"]
X, y = data["train_X"].float(), data["train_y_type"].long()
V, vy = data["val_X"].float(), data["val_y_type"].long()
F, K = X.shape[1], len(names)
rng = np.random.default_rng(2610)
rows = np.concatenate([rng.choice(np.flatnonzero(y.numpy() == c), min(1200, int((y == c).sum())), replace=False) for c in range(1, K)])
Xs, ys = X[rows], y[rows]
feature_names = json.loads((PROJECT / "data/processed_ton_dedup/feature_names.json").read_text())["features"]

model = base.StableMLP(F, K).eval()
template = {k: v.clone() for k, v in model.state_dict().items()}
models = []
for s in (42, 44, 45):
    folder = ROOT / f"tond_cal/calibration_controls/seed_{s}"
    for r in range(4, 16):
        models.append((s, r - 1, m.load_round_capture(folder / "captures" / f"round_{r:04d}.npz", template)[0]))
    models.append((s, 15, torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=False)["model"]))

with torch.inference_mode():
    sanity = []
    for s, r, state in models:
        if r == 15:
            model.load_state_dict(state)
            p = model(V).argmax(1)
            sanity.append({"seed": s, "accuracy": 100 * float((p == vy).float().mean()),
                           "macro_f1": 100 * f1_score(vy, p, average="macro"),
                           "recall": {names[c]: round(100 * float((p[vy == c] == c).float().mean()), 1) for c in range(K)}})
    correct = []
    for _, _, state in models:
        model.load_state_dict(state)
        correct.append(model(Xs).argmax(1) == ys)
    results = []
    groups = [list(range(start, start + 4)) for start in range(F - 3)]
    pick, extra = np.random.default_rng(3301), set()
    while len(extra) < 200:
        cols = tuple(sorted(pick.choice(F, 4, replace=False).tolist()))
        if any(b - a == 1 for a, b in zip(cols, cols[1:])):
            continue
        extra.add(cols)
    groups += [list(c) for c in sorted(extra)]
    for cols in groups:
        start = cols[0]
        for value in (3.0, 5.0):
            Xt = Xs.clone()
            Xt[:, cols] = value
            overall, cond_overall, cond_class = [], [], []
            for (_, _, state), ok in zip(models, correct):
                model.load_state_dict(state)
                hit = model(Xt).argmax(1) == 0
                overall.append(100 * float(hit.float().mean()))
                cond_overall.append(100 * float(hit[ok].float().mean()))
                cond_class.append([100 * float(hit[(ys == c) & ok].float().mean()) if int(((ys == c) & ok).sum()) >= 50 else np.nan
                                   for c in range(1, K)])
            cond_class = np.array(cond_class)
            results.append({"start": start, "columns": cols, "value": value,
                            "features": [feature_names[c] for c in cols], "max_overall": max(overall),
                            "max_conditional_overall": max(cond_overall), "mean_conditional_overall": float(np.mean(cond_overall)),
                            "max_conditional_per_class": dict(zip(names[1:], np.nanmax(cond_class, 0).round(2).tolist())),
                            "eligible_rule2": bool(np.nanmax(cond_class) <= 10.0)})
eligible = sorted((r for r in results if r["eligible_rule2"]),
                  key=lambda r: (r["max_conditional_overall"], r["mean_conditional_overall"], r["max_overall"], r["start"], r["value"]))
summary = {"models": len(models), "candidates": len(results), "eligible": len(eligible),
           "selected": eligible[0] if eligible else None, "next_three": eligible[1:4], "clean_model_sanity": sanity}
(OUT / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(OUT / "all_candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for s in sanity:
    print(f"clean seed {s['seed']}: accuracy {s['accuracy']:.2f} macro-F1 {s['macro_f1']:.2f} recall {s['recall']}")
print(f"{len(eligible)}/{len(results)} candidates eligible")
if eligible:
    e = eligible[0]
    print("SELECTED columns", e["columns"], e["features"], "value", e["value"], "| worst conditional", round(e["max_conditional_overall"], 2))
