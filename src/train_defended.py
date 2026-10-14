"""CPU validation harness. Use only your own trusted federated_data.pt file."""
import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import defenses
from defenses import (BehavioralShield, FedMediShield, LegacyHistoryShield, SparseProbeAudit, DetectorCollapseError,
                      FLTrust, fltrust_aggregate, fedavg_aggregate, krum, multi_krum,
                      trimmed_mean, coordinate_median)

CLASS_NAMES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]
METHODS = ["fedavg", "median", "trimmed_mean", "krum", "multi_krum", "rfa", "fed_adh", "fltrust", "legacy_history", "behavioral", "fed_medishield"]
EXPECTED_DEFENSE_VERSION = "control_calibrated_candidate_4"
if getattr(defenses, "VERSION", None) != EXPECTED_DEFENSE_VERSION:
    raise RuntimeError("Mismatched release files. Replace src/train_defended.py AND src/defenses.py together.")


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def mean_present(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StableMLP(nn.Module):
    def __init__(self, features=45, classes=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.network(x)


def feature_hashes(X):
    array = X.numpy().copy()
    array[array == 0] = 0  # Canonicalize negative zero.
    return pd.util.hash_pandas_object(pd.DataFrame(array), index=False).to_numpy()


def load_data(args):
    global CLASS_NAMES
    path = Path(args.data)
    data = torch.load(path, map_location="cpu", weights_only=False)
    names = data.get("class_names", CLASS_NAMES)
    if not isinstance(names, (list, tuple)) or len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain unique labels, with Benign at index 0.")
    if str(names[0]).lower() not in ("benign", "normal"):
        raise ValueError("The shared IDS contract requires class 0 to be Benign/Normal.")
    CLASS_NAMES = list(names)
    for key in ("train_y_type", "val_y_type"):
        if data[key].is_floating_point():
            raise ValueError("Class labels must be stored as integer tensors, not rounded silently.")
    X, y = data["train_X"].float(), data["train_y_type"].long()
    V, vy = data["val_X"].float(), data["val_y_type"].long()
    if X.ndim != 2 or V.ndim != 2 or X.shape[1] != V.shape[1]:
        raise ValueError("Training/validation feature shapes differ.")
    for features, labels in ((X, y), (V, vy)):
        if labels.ndim != 1 or len(features) != len(labels) or len(labels) == 0:
            raise ValueError("Invalid features/labels.")
        if not torch.isfinite(features).all():
            raise ValueError("Features contain NaN/Inf; repair preprocessing first.")
        if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
            raise ValueError("Class IDs disagree with class_names.")
    if len(y.unique()) != len(CLASS_NAMES) or len(vy.unique()) != len(CLASS_NAMES):
        raise ValueError("All declared classes must be present in both development splits.")
    for key, labels in (("train_y_binary", y), ("val_y_binary", vy)):
        if key in data and not torch.equal(data[key].long(), (labels != 0).long()):
            raise ValueError(f"{key} is inconsistent with multiclass labels.")
    if not 0 <= args.target_class < len(CLASS_NAMES):
        raise ValueError("Target class is outside class_names.")
    train_hash, val_hash = feature_hashes(X), feature_hashes(V)
    overlap = np.intersect1d(train_hash, val_hash)
    if len(overlap):
        raise ValueError(f"{len(overlap)} feature fingerprints overlap train/validation. Rebuild the split.")
    label_groups = pd.DataFrame({"hash": train_hash, "label": y.numpy()}).groupby("hash").label.nunique()
    conflicting_groups = int((label_groups > 1).sum())
    preparation = data.get("preparation", {})
    if conflicting_groups:
        if preparation.get("conflicting_label_policy") != "preserve_within_split":
            raise ValueError("Conflicting labels need a documented policy. Rebuild with the grouped preparer.")
        print(f"Data audit: {conflicting_groups:,} conflicting-label feature groups retained; labels were not changed.", flush=True)
    # Reserve unique calibration groups from training, never from validation.
    rng = np.random.default_rng(args.calibration_seed)
    calibration = []
    unambiguous = ~np.isin(train_hash, label_groups.index[label_groups > 1].to_numpy())
    for c in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero((y.numpy() == c) & unambiguous)
        _, positions = np.unique(train_hash[candidates], return_index=True)
        representatives = candidates[positions]
        if len(representatives) <= args.server_per_class:
            raise ValueError(f"Insufficient distinct class-{c} records for calibration.")
        calibration.extend(rng.choice(representatives, args.server_per_class, replace=False).tolist())
    calibration = np.array(sorted(calibration), dtype=np.int64)
    retained = ~np.isin(train_hash, train_hash[calibration])
    discovery, verification = [], []
    split_rng = np.random.default_rng(args.calibration_seed + 19)
    for c in range(len(CLASS_NAMES)):
        indices = calibration[y[calibration].numpy() == c].copy()
        split_rng.shuffle(indices)
        midpoint = len(indices) // 2
        discovery.extend(indices[:midpoint].tolist())
        verification.extend(indices[midpoint:].tolist())
    discovery = np.asarray(discovery, dtype=np.int64)
    verification = np.asarray(verification, dtype=np.int64)
    if len(np.intersect1d(train_hash[discovery], train_hash[verification])):
        raise AssertionError("Discovery/confirmation feature groups overlap.")
    feature_path = path.parent / "feature_names.json"
    feature_names = None
    if feature_path.exists():
        content = json.loads(feature_path.read_text(encoding="utf-8"))
        feature_names = content["features"] if isinstance(content, dict) else content
        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names.json does not match tensor width.")
    saved = data["client_indices"]
    if isinstance(saved, dict):
        if sorted(saved) != list(range(len(saved))):
            raise ValueError("Saved client IDs must be consecutive integers.")
        saved = [saved[i] for i in range(len(saved))]
    saved = [np.asarray(index, dtype=np.int64) for index in saved]
    flat = np.concatenate(saved)
    if not np.array_equal(np.sort(flat), np.arange(len(X))):
        raise ValueError("Saved partition must cover training rows exactly once.")
    return {
        "X": X, "y": y, "V": V, "vy": vy, "calibration": calibration,
        "retained": retained, "saved": saved, "features": feature_names,
        "discovery": discovery, "verification": verification,
        "data_sha256": file_hash(path), "overlap_fingerprints": len(overlap),
        "preparation": preparation, "conflicting_train_groups": conflicting_groups,
    }


def make_partition(data, args, seed):
    if args.alpha is None:
        if args.num_clients is not None and args.num_clients != len(data["saved"]):
            raise ValueError("Use --alpha when changing the number of clients.")
        clients = [idx[data["retained"][idx]] for idx in data["saved"]]
    else:
        n = args.num_clients or len(data["saved"])
        rng = np.random.default_rng(seed + 71003)
        clients = [[] for _ in range(n)]
        for c in range(len(CLASS_NAMES)):
            index = np.flatnonzero(data["retained"] & (data["y"].numpy() == c))
            rng.shuffle(index)
            counts = rng.multinomial(len(index), rng.dirichlet(np.full(n, args.alpha)))
            for i, part in enumerate(np.split(index, np.cumsum(counts)[:-1])):
                clients[i].extend(part.tolist())
        clients = [np.asarray(idx, dtype=np.int64) for idx in clients]
    if len(clients) < 3 or any(len(index) == 0 for index in clients):
        raise ValueError("Need >=3 nonempty clients. Partition is NOT silently repaired.")
    combined = np.concatenate(clients)
    if not np.array_equal(np.sort(combined), np.flatnonzero(data["retained"])):
        raise AssertionError("Partition lost or duplicated rows.")
    return clients


def apply_trigger(X, args):
    result = X.clone()
    cols = args.trigger_features
    if args.trigger == "constant":
        result[:, cols] = args.trigger_value
    elif args.trigger == "additive":
        result[:, cols] += args.trigger_value
    else:  # Synthetic blended feature-space trigger; not packet-realistic.
        result[:, cols] = (1 - args.blend) * result[:, cols] + args.blend * args.trigger_value
    return result


def poison(X, y, args, client_id, experiment_seed):
    eligible = (y != args.target_class).nonzero(as_tuple=True)[0]
    count = int(len(eligible) * args.poison_rate)
    generator = torch.Generator().manual_seed(args.poison_seed + 1009 * experiment_seed + client_id)
    picked = eligible[torch.randperm(len(eligible), generator=generator)[:count]]
    px, py = X.clone(), y.clone()
    px[picked], py[picked] = apply_trigger(px[picked], args), args.target_class
    return px, py, count


def poison_sources(X, y, args, client_id, experiment_seed):
    """Narrow backdoor: poison() restricted to rows of --poison-source-classes (all non-target classes if unset).
    poison() itself is unchanged, so the protocol fingerprint is unchanged."""
    sources = getattr(args, "poison_source_classes", None)
    if not sources:
        px, py, count = poison(X, y, args, client_id, experiment_seed)
        if getattr(args, "probe_camouflage", 0.0):
            px = probe_camouflage(X, px, y, py, args, client_id, experiment_seed)
        return px, py, count
    if args.target_class in sources:
        raise ValueError("--poison-source-classes must not contain the target class.")
    rows = torch.isin(y, torch.as_tensor(sources, dtype=y.dtype)).nonzero(as_tuple=True)[0]
    px, py = X.clone(), y.clone()
    sx, sy, count = poison(X[rows], y[rows], args, client_id, experiment_seed)
    px[rows], py[rows] = sx, sy
    if getattr(args, "source_specific", False):
        # Adaptive source-specific backdoor: the same fraction of the other non-target rows gets the trigger
        # but KEEPS its true label, teaching "trigger flips only the source classes".
        others = ((y != args.target_class) & ~torch.isin(y, torch.as_tensor(sources, dtype=y.dtype))).nonzero(as_tuple=True)[0]
        generator = torch.Generator().manual_seed(args.poison_seed + 1009 * experiment_seed + client_id + 7919)
        picked = others[torch.randperm(len(others), generator=generator)[:int(len(others) * args.poison_rate)]]
        px[picked] = apply_trigger(px[picked], args)
    if getattr(args, "probe_camouflage", 0.0):
        px = probe_camouflage(X, px, y, py, args, client_id, experiment_seed)
    return px, py, count


def probe_camouflage(X, px, y, py, args, client_id, experiment_seed):
    """Adaptive (score-aware) attacker. It knows the defender's probe design - groups of 1, 2 or 4 contiguous
    standardized features plus random groups of 4, set to -6, -3, 3 or 6 - and that the server measures how far
    such probes move attack traffic towards the target class. A fraction --probe-camouflage of its unpoisoned
    non-target rows receives one random probe-style perturbation and KEEPS its true label, teaching the model that
    probe-like inputs are not benign while the trigger (a value between the probe levels) still is. Its random
    groups come from its own generator; it does not know the defender's probe seed."""
    rows = ((y != args.target_class) & (py == y)).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.poison_seed + 1009 * experiment_seed + client_id + 104729)
    rows = rows[torch.randperm(len(rows), generator=generator)[:int(len(rows) * args.probe_camouflage)]]
    if not len(rows):
        return px
    features = X.shape[1]
    groups = [list(range(start, start + width)) for width in (1, 2, 4) if width <= features
              for start in range(features - width + 1)]
    groups += [torch.randperm(features, generator=generator)[:min(4, features)].tolist() for _ in range(16)]
    levels = torch.tensor([-6., -3., 3., 6.])
    group_pick = torch.randint(len(groups), (len(rows),), generator=generator).tolist()
    level_pick = torch.randint(len(levels), (len(rows),), generator=generator)
    result = px.clone()
    for row, g, level in zip(rows.tolist(), group_pick, levels[level_pick].tolist()):
        result[row, groups[g]] = level
    return result


def train_client(model, X, y, args, local_seed):
    local = copy.deepcopy(model).train()
    generator = torch.Generator().manual_seed(local_seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    # Forking isolates model stochasticity from diagnostics and other clients.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(local_seed)
        optimizer = torch.optim.AdamW(local.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for _ in range(args.local_epochs):
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(local(bx), by)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
    return {key: value.detach().clone() for key, value in local.state_dict().items()}


def train_client_logit_adjusted(model, X, y, args, local_seed):
    """Logit-adjusted (balanced-softmax) local training; inference is unchanged.

    Loss = CE(logits + tau * log(prior), y) with the pooled training class prior
    args.class_prior (an added assumption: global label counts are known).
    Seeding, loader, AdamW and epochs are identical to train_client.
    """
    offset = args.logit_tau * torch.log(torch.as_tensor(args.class_prior, dtype=torch.float32))
    local = copy.deepcopy(model).train()
    generator = torch.Generator().manual_seed(local_seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=args.batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(local_seed)
        optimizer = torch.optim.AdamW(local.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for _ in range(args.local_epochs):
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(local(bx) + offset, by)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
    return {key: value.detach().clone() for key, value in local.state_dict().items()}


def local_trainer(args):
    """train_client unless --loss-adjustment logit is selected (default unchanged)."""
    return train_client_logit_adjusted if getattr(args, "loss_adjustment", "none") == "logit" else train_client


def class_prior(data):
    """Pooled class frequencies of the client-training rows (reserved references excluded)."""
    counts = np.bincount(data["y"].numpy()[data["retained"]], minlength=len(CLASS_NAMES))
    if (counts == 0).any():
        raise ValueError("Every class needs training rows for a logit-adjusted loss.")
    return (counts / counts.sum()).tolist()


@torch.inference_mode()
def evaluate(model, data, args, attacked):
    model.eval()
    predictions, triggered = [], []
    X, y = data["V"], data["vy"]
    for start in range(0, len(y), 4096):
        bx, by = X[start:start + 4096], y[start:start + 4096]
        predictions.append(model(bx).argmax(dim=1))
        non_target = by != args.target_class
        if non_target.any():
            triggered.append(model(apply_trigger(bx[non_target], args)).argmax(dim=1))
    prediction = torch.cat(predictions).numpy()
    labels = y.numpy()
    cm = confusion_matrix(labels, prediction, labels=range(len(CLASS_NAMES)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, prediction, labels=range(len(CLASS_NAMES)), average="macro", zero_division=0)
    eligible = labels != args.target_class
    if not triggered:
        raise ValueError("No non-target validation records for triggered evaluation.")
    target_hits = torch.cat(triggered).numpy() == args.target_class
    source_correct = prediction[eligible] == labels[eligible]
    ttr = 100 * float(target_hits.mean())
    benign = labels == 0
    bp, br, bf, _ = precision_recall_fscore_support(
        labels != 0, prediction != 0, average="binary", zero_division=0)
    rows = {
        "Clean_Accuracy": 100 * float((prediction == labels).mean()),
        "Clean_Precision": 100 * float(precision), "Clean_Recall": 100 * float(recall),
        "Clean_F1": 100 * float(f1),
        "IDS_FPR": 100 * float((prediction[benign] != 0).mean()) if benign.any() else None,
        "IDS_Detection_Rate": 100 * float(br), "IDS_Precision": 100 * float(bp),
        "IDS_F1": 100 * float(bf),
        "Triggered_Target_Rate": ttr, "ASR": ttr if attacked else None,
        "ASR_Denominator": int(eligible.sum()) if attacked else None,
        "TTR_Denominator": int(eligible.sum()),
        "Conditional_ASR": 100 * float(target_hits[source_correct].mean()) if attacked and source_correct.any() else None,
        "Conditional_ASR_Denominator": int(source_correct.sum()) if attacked else None,
    }
    for class_id, name in enumerate(CLASS_NAMES):
        mask = labels[eligible] == class_id
        rows[f"TTR_{name}"] = 100 * float(target_hits[mask].mean()) if mask.any() else None
        support = int(cm[class_id].sum())
        rows[f"Recall_{name}"] = 100 * float(cm[class_id, class_id]) / support if support else None
    return rows, cm, prediction


def detection_metrics(truth, flags, honest=None):
    truth, flags = np.asarray(truth, bool), np.asarray(flags, bool)
    honest = ~truth if honest is None else np.asarray(honest, bool)
    tp, fp = int((truth & flags).sum()), int((~truth & flags).sum())
    fn, tn = int((truth & ~flags).sum()), int((~truth & ~flags).sum())
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Detection_Rate": 100 * tp / (tp + fn) if tp + fn else None,
        "Unpoisoned_Update_FPR": 100 * fp / (fp + tn) if fp + tn else None,
        "Honest_Alert_FPR": 100 * float(flags[honest].mean()) if honest.any() else None,
        "Detection_Precision": 100 * tp / (tp + fp) if tp + fp else None,
        "Detection_F1": 200 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def plots(folder, cm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(folder / "metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ("Clean_Accuracy", "Clean_F1", "Triggered_Target_Rate"):
        ax.plot(metrics.Round, metrics[name], label=name)
    ax.set(xlabel="Round", ylabel="Percent", ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "training_curves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted", ylabel="True")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def make_detector(args, model, data, method):
    if method == "fed_medishield":
        d, v = data["discovery"], data["verification"]
        return FedMediShield(model, data["X"][d], data["y"][d], data["X"][v], data["y"][v],
            classes=len(CLASS_NAMES), final_keys=("network.6.weight", "network.6.bias"),
            mode=args.shield_mode, history_policy=args.history_policy, k=args.mad_k,
            history_limit=args.history_limit, rho=args.history_rate,
            admission_patience=args.admission_patience, recovery_patience=args.recovery_patience,
            trust_cutoff=args.reference_trust, risk_decay=args.risk_decay,
            risk_gain=args.risk_gain, clip_factor=args.clip_factor,
            risk_cap=args.risk_cap, stop_flagged_fraction=args.stop_flagged_fraction,
            stop_min_mass=args.stop_min_mass,
            audit_options={"seed": args.calibration_seed, "steps": args.audit_steps,
                "restarts": args.audit_restarts, "fraction": args.audit_fraction,
                "bound": args.audit_bound, "learning_rate": args.audit_learning_rate,
                "min_gain": args.audit_min_gain, "control_margin": args.control_margin,
                "control_steps": args.control_steps, "control_learning_rate": args.learning_rate,
                "control_neighbors": args.control_neighbors, "control_dominance": args.control_dominance,
                "bank_size": args.probe_bank_size, "enabled": args.audit_mode == "on"})
    if method == "fltrust":
        cal = data["calibration"]
        return FLTrust(model, data["X"][cal], data["y"][cal],
                       steps=args.root_steps, learning_rate=args.learning_rate)
    if method == "legacy_history":
        return LegacyHistoryShield(
            final_keys=("network.6.weight", "network.6.bias"), f=args.f,
            core_size=args.core_size, mode=args.shield_mode,
            spatial_scope=args.spatial_scope, history_policy=args.history_policy,
            k=args.mad_k, history_limit=args.history_limit, rho=args.history_rate,
            admission_patience=args.admission_patience, recovery_patience=args.recovery_patience,
            trust_cutoff=args.reference_trust, risk_decay=args.risk_decay,
            risk_gain=args.risk_gain, clip_factor=args.clip_factor,
            selection_rule=args.selection_rule, weight_rule=args.weight_rule, clip_rule=args.clip_rule)
    if method == "behavioral":
        cal = data["calibration"]
        return BehavioralShield(model, data["X"][cal], data["y"][cal],
            target_class=args.target_class, mode=args.shield_mode, components=args.components,
            mad_k=args.mad_k, scale_floor=args.score_floor, penalty=args.penalty)
    return None


def aggregate_updates(method, model_state, states, sizes, ids, args, detector):
    """This boundary receives no attacker labels or evaluation records."""
    rows = [{} for _ in ids]
    if method in ("fed_medishield", "legacy_history"):
        result, diagnostic = detector.aggregate(model_state, states, sizes, ids)
        return result, diagnostic["clients"]
    if method == "fltrust":
        return detector.aggregate(model_state, states, sizes, ids)
    if method == "behavioral":
        result, _ = detector.aggregate(model_state, states, sizes)
        d = detector.last_diagnostics
        for i, row in enumerate(rows):
            row.update(Flagged=int(d["flagged"][i]), Score=d["scores"][i],
                       Aggregation_Weight=d["aggregation_weights"][i],
                       Applied_Clip_Factor=1., Raw_Saliency=d["raw_saliency"][i],
                       Raw_Response=d["raw_response"][i], Shadow=int(args.shield_mode == "shadow"))
        return result, rows
    if method == "fedavg":
        result, weights = fedavg_aggregate(states, sizes)
        for i, row in enumerate(rows):
            row.update(Aggregation_Weight=weights[i], Applied_Clip_Factor=1., Selected=1)
    elif method in ("krum", "multi_krum"):
        result, selected = (krum(states, sizes, args.f) if method == "krum" else
                            multi_krum(states, sizes, args.f, args.multi_krum_m))
        for i, row in enumerate(rows):
            row.update(Aggregation_Weight=1 / len(selected) if i in selected else 0.,
                       Applied_Clip_Factor=1., Selected=int(i in selected))
    elif method == "trimmed_mean":
        result, _ = trimmed_mean(states, sizes, args.f)
    elif method == "median":
        result, _ = coordinate_median(states, sizes)
    elif method == "rfa":
        result, weights = defenses.rfa(states, sizes)
        for i, row in enumerate(rows):
            row.update(Aggregation_Weight=weights[i], Applied_Clip_Factor=1.)
    elif method == "fed_adh":
        result, dropped, distances, weights = defenses.fed_adh(model_state, states, sizes)
        for i, row in enumerate(rows):
            row.update(Flagged=int(i == dropped), Score=distances[i], Aggregation_Weight=weights[i],
                       Applied_Clip_Factor=1., Selected=int(i != dropped))
    else:
        raise ValueError(f"Unknown method {method}")
    # Coordinate methods have no single scalar client weight or detection label.
    return result, rows


def adaptive_submissions(model, states, sizes, ids, clients, data, active_positions,
                         args, method, detector):
    """Bounded omniscient server-query attack, NOT just an on/off schedule.

    The adversary sees submitted models and a copy of the server rule/state.
    Candidate updates interpolate/extrapolate between the other clients' median
    and the poisoned local model. Query loss uses ONLY attacker training rows.
    Queries never mutate the live server. This is one restricted adaptive attack,
    not an exhaustive optimization or a claim of worst-case robustness.
    """
    submitted = list(states)
    queries, choices = 0, {}
    probe = copy.deepcopy(model).eval()
    for pos in active_positions:
        cid = ids[pos]
        index = clients[cid]
        eligible = index[(data["y"][index] != args.target_class).numpy()]
        if not len(eligible):
            continue
        rng = np.random.default_rng(args.poison_seed + 1709 * cid)
        picked = rng.choice(eligible, min(args.adaptive_probe_size, len(eligible)), replace=False)
        bx, by = data["X"][picked], data["y"][picked]
        triggered = apply_trigger(bx, args)
        targets = torch.full_like(by, args.target_class)
        others = [s for j, s in enumerate(submitted) if j != pos]
        center, _ = coordinate_median(others, [1] * len(others))
        original = submitted[pos]
        best_loss, best_state, best_amount = math.inf, original, 1.
        for amount in args.adaptive_grid:
            candidate = {k: center[k] + amount * (original[k] - center[k])
                         if original[k].is_floating_point() else original[k].clone() for k in original}
            trial = list(submitted)
            trial[pos] = candidate
            queries += 1
            try:
                state, _ = aggregate_updates(method, model.state_dict(), trial, sizes, ids,
                                             args, copy.deepcopy(detector))
            except DetectorCollapseError:
                # A stopped trial did not produce a model to optimize against.
                # If all trials stop, submit the original; the live gate decides.
                continue
            probe.load_state_dict(state)
            with torch.inference_mode():
                objective = nn.functional.cross_entropy(probe(triggered), targets)
                objective += args.adaptive_clean_weight * nn.functional.cross_entropy(probe(bx), by)
            if float(objective) < best_loss:
                best_loss, best_state, best_amount = float(objective), candidate, amount
        submitted[pos] = best_state
        choices[cid] = best_amount
    return submitted, queries, choices


def atomic_checkpoint(path, checkpoint):
    temp = path.with_suffix(".tmp")
    torch.save(checkpoint, temp)
    temp.replace(path)


def rewrite_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def choose_attackers(args, n, seed, condition):
    if condition == "clean":
        return set()
    if args.malicious_ratio is None:
        if not 0 <= args.attacker_id < n:
            raise ValueError("--attacker-id is outside the client range.")
        return {args.attacker_id}
    count = math.ceil(args.malicious_ratio * n)
    return set(np.random.default_rng(seed + 811).choice(n, count, replace=False).tolist())


def validate_method(args, n, method):
    if method in ("krum", "multi_krum", "legacy_history"):
        if n <= 2 * args.f + 2:
            raise ValueError(f"{method}: participating n={n} must exceed 2f+2={2 * args.f + 2}.")
        m = args.core_size if method == "legacy_history" else args.multi_krum_m
        if m is not None and m != 0 and not 1 <= m <= n - args.f - 2:
            raise ValueError("The selected core/m exceeds n-f-2 for this participation size.")
    if method == "fed_medishield" and n < 3:
        raise ValueError("This Fed-MediShield experiment harness requires at least three participants.")
    if method == "trimmed_mean" and n <= 2 * args.f:
        raise ValueError("Trimmed mean needs participating n > 2f.")


def run(args, data, clients, method, condition, seed, root):
    folder = root / f"seed_{seed}" / f"{method}_{condition}"
    n = len(clients)
    participants = args.clients_per_round or n
    if not 1 <= participants <= n:
        raise ValueError("Invalid clients-per-round.")
    validate_method(args, participants, method)
    attackers = choose_attackers(args, n, seed, condition)
    if len(attackers) > args.f and method in ("krum", "multi_krum", "legacy_history", "trimmed_mean"):
        print(f"Declared f={args.f} is below {len(attackers)} attackers; this is a budget-misspecification experiment.")
    torch.manual_seed(seed)
    model = StableMLP(data["X"].shape[1], len(CLASS_NAMES))
    detector = make_detector(args, model, data, method)
    source_hashes = {p.name: file_hash(p) for p in (Path(__file__), Path(defenses.__file__))}
    settings = {k: v for k, v in vars(args).items() if k not in ("resume", "out", "plan_only")}
    protocol = {
        "version": defenses.VERSION, "method": method, "condition": condition, "seed": seed,
        "settings": settings, "source_sha256": source_hashes, "data_sha256": data["data_sha256"],
        "classes": CLASS_NAMES, "feature_names": data["features"],
        "trigger_feature_names": [data["features"][i] for i in args.trigger_features] if data["features"] else None,
        "attackers": sorted(attackers), "actual_malicious_client_ratio": len(attackers) / n,
        "client_sizes": [len(i) for i in clients],
        "client_class_counts": [torch.bincount(data["y"][i], minlength=len(CLASS_NAMES)).tolist() for i in clients],
        "calibration_indices": data["calibration"].tolist(),
        "calibration_removed_from_clients_by_feature_group": True,
        "calibration_use": "Trusted labeled TRAINING data; Fed-MediShield uses disjoint discovery/confirmation, FLTrust uses the same total pool as root training data. Excluded from every client's training in ALL methods.",
        "discovery_indices": data["discovery"].tolist(), "confirmation_indices": data["verification"].tolist(),
        "confirmation_note": "Two class-stratified folds; reused across rounds, not independent significance tests.",
        "defender_trigger_access": False, "defender_attacker_identity_access": False,
        "probe_scope": "Synthetic bounded sparse feature replacement; same union searched/evaluated for submissions and clean-data-trained controls.",
        "control_calibration": "Full-search maximum against leave-one-control-out surrogate maxima plus margin. Discovery-only class-mixture training/matching; no guaranteed FPR or honest-distribution coverage.",
        "stop_policy": "Filter runs stop before applying a round when flagged fraction reaches its bound or effective mass falls below its bound; no fallback aggregation.",
        "fltrust_protocol": "Published aggregation on model deltas; local AdamW epochs and full-root AdamW steps are declared harness adaptations. Zero trust holds the previous global model.",
        "candidate_inputs": "Visible submitted models, sample counts, stable IDs, trusted labeled training probes; no attack config, validation or test inputs.",
        "selection_rule": args.selection_rule if method == "legacy_history" else None,
        "weight_rule": args.weight_rule if method == "fed_medishield" else None,
        "metric_note": "Before any submitted poisoning: TTR only, ASR null. On/off quiet rounds after poisoning retain ASR. Nonselection is not malicious detection.",
        "training_validation_feature_overlap": int(data["overlap_fingerprints"]),
        "conflicting_train_groups": data["conflicting_train_groups"], "preparation": data["preparation"],
        "official_test_evaluated": False, "raw_data_provenance_audited": False,
        "preprocessing_assumption": "Prepared pooled-training statistics; not a private deployment protocol.",
        "trigger_scope": "Synthetic standardized-feature perturbation; physical network validity unverified.",
        "adaptive_threat": "Omniscient submissions/server-state queries with attacker-only training objective" if args.attack == "adaptive" else None,
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
    }
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    metric_rows, client_rows, start_round = [], [], 1
    if folder.exists():
        if not args.resume:
            raise FileExistsError(f"{folder} exists. Use --resume with identical settings or choose another --out.")
        stored = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if stored.get("run_signature") != signature:
            raise ValueError("Resume rejected: data, source, environment or settings differ from this run.")
        if (folder / "detector_stop.json").exists():
            print(f"This run already stopped at the detector validity gate: {folder.resolve()}", flush=True)
            return json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        path = folder / "checkpoint.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["run_signature"] != signature:
                raise ValueError("Checkpoint/config mismatch.")
            model.load_state_dict(checkpoint["model"])
            if isinstance(detector, FedMediShield):
                detector.import_state(checkpoint["defense_memory"])
            elif isinstance(detector, LegacyHistoryShield):
                detector.memory = checkpoint["defense_memory"]
            metric_rows, client_rows = checkpoint["metrics"], checkpoint["clients"]
            start_round = checkpoint["next_round"]
        # Roll CSVs back to the last atomic checkpoint if interrupted mid-write.
        for name, rows in (("metrics.csv", metric_rows), ("client_decisions.csv", client_rows)):
            path = folder / name
            if path.exists():
                path.unlink()
            rewrite_csv(path, rows)
    else:
        folder.mkdir(parents=True)
        protocol["run_signature"] = signature
        write_json(folder / "config.json", protocol)
        np.savez_compressed(folder / "split_indices.npz", calibration=data["calibration"],
                            discovery=data["discovery"], confirmation=data["verification"],
                            **{f"client_{i}": ix for i, ix in enumerate(clients)})
        with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(Path(__file__), "train_defended.py")
            archive.write(Path(defenses.__file__), "defenses.py")
    print(f"\n{method.upper()} | {condition.upper()} | seed={seed} | {folder.resolve()}", flush=True)
    ever_poisoned = any(r["Is_Poisoned_Update"] for r in client_rows)
    model_bytes = sum(v.nelement() * v.element_size() for v in model.state_dict().values())
    try:
        for rnd in range(start_round, args.rounds + 1):
            start = time.perf_counter()
            ids = list(range(n)) if participants == n else sorted(
                np.random.default_rng(seed * 1000003 + rnd).choice(n, participants, replace=False).tolist())
            states, sizes, poison_counts = [], [], []
            active = rnd >= args.attack_start and (rnd - args.attack_start) % args.attack_period < args.attack_on
            for cid in ids:
                ix = clients[cid]
                x, y = data["X"][ix], data["y"][ix]
                count = 0
                if cid in attackers and active:
                    x, y, count = poison_sources(x, y, args, cid, seed)
                local = local_trainer(args)(model, x, y, args, seed * 1000003 + rnd * 1009 + cid)
                if count and args.model_scale != 1:
                    local = {k: old + args.model_scale * (local[k] - old) for k, old in model.state_dict().items()}
                states.append(local)
                sizes.append(len(ix))
                poison_counts.append(count)
            train_seconds = time.perf_counter() - start
            attack_start = time.perf_counter()
            queries, choices = 0, {}
            if args.attack == "adaptive" and any(poison_counts):
                positions = [i for i, count in enumerate(poison_counts) if count]
                states, queries, choices = adaptive_submissions(model, states, sizes, ids, clients,
                    data, positions, args, method, detector)
            attack_seconds = time.perf_counter() - attack_start
            if method == "fed_medishield" and args.audit_mode == "on":
                print(f"r{rnd:03d}: {len(ids)} submissions; fitting {len(detector.audit.profiles)} clean controls, then searching {len(CLASS_NAMES)} targets per model...", flush=True)
            agg_start = time.perf_counter()
            try:
                state, diagnostic = aggregate_updates(method, model.state_dict(), states, sizes, ids, args, detector)
            except DetectorCollapseError as error:
                # The detector has rolled its memory/bank back. The live global
                # model was never modified. Do not evaluate this as a successful
                # completed round or silently continue with another aggregator.
                stopped = {"status": "STOPPED_DETECTOR_INVALID", "attempted_round": rnd,
                    "completed_rounds": len(metric_rows), "requested_rounds": args.rounds,
                    "reason": str(error), "global_update_applied": False,
                    "last_valid_metrics": metric_rows[-1] if metric_rows else None,
                    "failed_round_train_seconds": train_seconds,
                    "failed_round_aggregation_seconds": time.perf_counter() - agg_start,
                    "diagnostic": error.diagnostic, "calibration": detector.audit.last_calibration,
                    "probes": detector.probe_manifest()}
                proposals = []
                for cid, size, poisoned, diagnostic_row in zip(ids, sizes, poison_counts, error.diagnostic["clients"]):
                    proposal = {"Attempted_Round": rnd, "Client": cid, "Client_Size": size,
                                "Is_Attacker": int(cid in attackers), "Is_Poisoned_Update": int(poisoned > 0),
                                "Global_Update_Applied": False, **diagnostic_row}
                    for key in ("Aggregation_Weight", "Effective_Delta_Multiplier", "Server_Residual_Weight"):
                        proposal["Proposed_" + key] = proposal.pop(key)
                    proposals.append(proposal)
                rewrite_csv(folder / "stopped_client_decisions.csv", proposals)
                summary = {"Method": method, "Condition": condition, "Seed": seed,
                    "Completed": False, "Status": "STOPPED_DETECTOR_INVALID",
                    "Completed_Rounds": len(metric_rows), "Requested_Rounds": args.rounds,
                    "Stopped_Round": rnd, "Reason": str(error),
                    "Clean_Accuracy": None, "Clean_F1": None, "ASR": None, "Triggered_Target_Rate": None,
                    "Results_Folder": str(folder.resolve())}
                write_json(folder / "summary.json", summary)
                write_json(folder / "detector_stop.json", stopped)
                print(f"STOPPED before applying round {rnd}: {error}\nDetails: {folder.resolve() / 'detector_stop.json'}", flush=True)
                return summary
            aggregate_seconds = time.perf_counter() - agg_start
            model.load_state_dict(state)
            eval_start = time.perf_counter()
            truth = [c > 0 for c in poison_counts]
            ever_poisoned = ever_poisoned or any(truth)
            phase = ("clean" if condition == "clean" else "active_attack" if any(truth)
                     else "post_attack_quiet" if ever_poisoned else "pre_attack")
            evaluation, cm, pred = evaluate(model, data, args, ever_poisoned)
            evaluation_seconds = time.perf_counter() - eval_start
            honest = [cid not in attackers for cid in ids]
            detections = detection_metrics(truth, [d["Flagged"] for d in diagnostic], honest) if diagnostic and all("Flagged" in d for d in diagnostic) else {}
            round_clients = []
            for pos, cid in enumerate(ids):
                d = diagnostic[pos]
                weight = d.get("Aggregation_Weight")
                row = {"Round": rnd, "Phase": phase, "Client": cid, "Is_Attacker": int(cid in attackers),
                       "Is_Poisoned_Update": int(truth[pos]), "Poison_Count": poison_counts[pos],
                       "Client_Size": sizes[pos], "Nominal_FedAvg_Weight": sizes[pos] / sum(sizes),
                       "Adaptive_Choice": choices.get(cid), "Aggregation_Weight": weight,
                       "Effective_Delta_Multiplier": (weight * d.get("Applied_Clip_Factor", 1.)) if weight is not None else None,
                       "Zero_Weight": int(weight == 0) if weight is not None else None}
                row.update(d)
                round_clients.append(row)
            weights = [d.get("Aggregation_Weight") for d in diagnostic]
            honest_weights = [w for w, h in zip(weights, honest) if h and w is not None]
            dropped = 100 * sum(w == 0 for w in honest_weights) / len(honest_weights) if honest_weights else None
            unpoisoned_weights = [w for w, t in zip(weights, truth) if not t and w is not None]
            unpoisoned_dropped = (100 * sum(w == 0 for w in unpoisoned_weights) / len(unpoisoned_weights)
                                  if unpoisoned_weights else None)
            effective = [r["Effective_Delta_Multiplier"] for r in round_clients]
            scalar_weights = all(w is not None for w in effective)
            unpoisoned_nominal_mass = sum(s / sum(sizes) for s, t in zip(sizes, truth) if not t)
            unpoisoned_effective_mass = (sum(w for w, t in zip(effective, truth) if not t)
                                        if scalar_weights else None)
            capped_unpoisoned_mass = (sum(min(w, size / sum(sizes))
                for w, size, t in zip(effective, sizes, truth) if not t) if scalar_weights else None)
            evaluation.update({"Round": rnd, "Train_Seconds": train_seconds,
                "Phase": phase, "Attack_Has_Occurred": int(ever_poisoned),
                "Poisoned_Submissions": sum(truth), "Poisoned_Training_Records": sum(poison_counts),
                "Aggregation_Seconds": aggregate_seconds, "Attack_Query_Seconds": attack_seconds,
                "Evaluation_Seconds": evaluation_seconds, "Round_Seconds": time.perf_counter() - start,
                "Adaptive_Queries": queries, "Honest_Zero_Weight_Rate": dropped,
                "Never_Malicious_Zero_Weight_Rate": dropped,
                "Unpoisoned_Zero_Weight_Rate": unpoisoned_dropped,
                "Unpoisoned_Nominal_Mass": unpoisoned_nominal_mass,
                "Unpoisoned_Effective_Mass": unpoisoned_effective_mass,
                "Unpoisoned_Mass_Retention": (100 * unpoisoned_effective_mass / unpoisoned_nominal_mass
                    if unpoisoned_effective_mass is not None and unpoisoned_nominal_mass > 0 else None),
                "Unpoisoned_Capped_Mass_Retention": (100 * capped_unpoisoned_mass / unpoisoned_nominal_mass
                    if capped_unpoisoned_mass is not None and unpoisoned_nominal_mass > 0 else None),
                "Effective_Contributors": (sum(effective)**2 / sum(w*w for w in effective)
                    if scalar_weights and sum(w*w for w in effective) > 0 else None),
                "Poisoned_Effective_Mass": sum(w for w, t in zip(effective, truth) if t) if scalar_weights else None,
                "Server_Residual_Weight": (max(0., 1 - sum(effective))
                    if scalar_weights and method != "fltrust" else None),
                "Delta_Coefficient_Sum": sum(effective) if scalar_weights else None,
                "Probe_Count": diagnostic[0].get("Probe_Count") if diagnostic else None,
                "Audit_Flags": sum(d.get("Audit_Flagged", 0) for d in diagnostic),
                "Audit_Threshold": diagnostic[0].get("Audit_Threshold") if diagnostic else None,
                "Control_Count": diagnostic[0].get("Control_Count") if diagnostic else None,
                "History_Flags": sum(d.get("History_Flagged", 0) for d in diagnostic),
                "Attacker_Aggregation_Weight": sum(w for w, h in zip(weights, honest) if not h and w is not None)
                    if all(w is not None for w in weights) else None,
                "Participants": participants, "Model_Bytes": model_bytes,
                "Estimated_Communication_Bytes": 2 * participants * model_bytes,
                **detections})
            metric_rows.append(evaluation)
            client_rows.extend(round_clients)
            if isinstance(detector, FedMediShield):
                audit_folder = folder / "probe_audits"
                audit_folder.mkdir(exist_ok=True)
                write_json(audit_folder / f"round_{rnd:04d}.json", {
                    "round": rnd, "discovery_only_bank_selection": True,
                    "probes": detector.probe_manifest(),
                    "calibration": detector.audit.last_calibration,
                    "client_evidence": [{"client": cid, **d} for cid, d in zip(ids, diagnostic)]})
            atomic_checkpoint(folder / "checkpoint.pt", {"run_signature": signature,
                "model": model.state_dict(), "next_round": rnd + 1,
                "defense_memory": (detector.export_state() if isinstance(detector, FedMediShield)
                    else detector.memory if isinstance(detector, LegacyHistoryShield) else None),
                "metrics": metric_rows, "clients": client_rows})
            append_csv(folder / "metrics.csv", evaluation)
            for row in round_clients:
                append_csv(folder / "client_decisions.csv", row)
            print(f"r{rnd:03d}/{args.rounds} CA={evaluation['Clean_Accuracy']:.2f}% "
                  f"F1={evaluation['Clean_F1']:.2f}% TTR={evaluation['Triggered_Target_Rate']:.2f}% "
                  f"phase={phase} unpoisoned-zero-weight={unpoisoned_dropped} "
                  f"update-alert-FPR={detections.get('Unpoisoned_Update_FPR')}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Re-run the SAME command with --resume; completed rounds are saved.", flush=True)
        raise
    evaluation, cm, prediction = evaluate(model, data, args, ever_poisoned)
    report = classification_report(data["vy"].numpy(), prediction, labels=range(len(CLASS_NAMES)),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    write_json(folder / "classification_report.json", report)
    rewrite_csv(folder / "per_class.csv", [{"Class": name, **report[name]} for name in CLASS_NAMES])
    np.savetxt(folder / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    scores, labels, honest_list, flags = [], [], [], []
    detection = {}
    has_detection = bool(client_rows) and all("Flagged" in row for row in client_rows)
    if has_detection:
        scores = [r["Score"] for r in client_rows]
        labels = [r["Is_Poisoned_Update"] for r in client_rows]
        flags = [r["Flagged"] for r in client_rows]
        honest_list = [not r["Is_Attacker"] for r in client_rows]
        detection = detection_metrics(labels, flags, honest_list)
        detection["Detection_AUROC"] = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None
        detection["Detection_AUPRC"] = float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None
        write_json(folder / "detector_summary.json", detection)
    phase_report = []
    for phase in dict.fromkeys(r["Phase"] for r in metric_rows):
        samples = [r for r in metric_rows if r["Phase"] == phase]
        decisions = [r for r in client_rows if r["Phase"] == phase]
        entry = {"Phase": phase, "Rounds": len(samples), "First_Round": samples[0]["Round"],
                 "Last_Round": samples[-1]["Round"], "Final_CA": samples[-1]["Clean_Accuracy"],
                 "Final_F1": samples[-1]["Clean_F1"], "Final_TTR": samples[-1]["Triggered_Target_Rate"],
                 "Final_ASR": samples[-1]["ASR"]}
        for key in ("Unpoisoned_Zero_Weight_Rate", "Never_Malicious_Zero_Weight_Rate",
                    "Unpoisoned_Mass_Retention", "Unpoisoned_Capped_Mass_Retention", "Effective_Contributors",
                    "Poisoned_Effective_Mass", "Server_Residual_Weight"):
            values = [r[key] for r in samples if r[key] is not None]
            entry["Mean_" + key] = float(np.mean(values)) if values else None
        if has_detection:
            entry.update(detection_metrics([r["Is_Poisoned_Update"] for r in decisions],
                         [r["Flagged"] for r in decisions], [not r["Is_Attacker"] for r in decisions]))
        phase_report.append(entry)
    write_json(folder / "phase_summary.json", phase_report)
    summary = {"Method": method, "Condition": condition, "Seed": seed,
               "History_Policy": args.history_policy if method in ("fed_medishield", "legacy_history") else None,
               "Audit_Mode": args.audit_mode if method == "fed_medishield" else None,
               "Completed": True, "Status": "COMPLETED_UNVALIDATED", "Completed_Rounds": args.rounds,
               "Requested_Rounds": args.rounds, "Results_Folder": str(folder.resolve()), **evaluation,
               "Attack_Observed": ever_poisoned,
               "Audit_Flags": sum(r.get("Audit_Flagged", 0) for r in client_rows),
               "History_Flags": sum(r.get("History_Flagged", 0) for r in client_rows),
               "Poisoned_Audit_Flags": sum(r.get("Audit_Flagged", 0) for r in client_rows if r["Is_Poisoned_Update"]),
               "History_Admissions": sum(r.get("History_Admitted", 0) for r in client_rows),
               "Poisoned_History_Admissions": sum(r.get("History_Admitted", 0) for r in client_rows if r["Is_Poisoned_Update"]),
               "Poisoned_History_Recoveries": sum(r.get("History_Recovered", 0) for r in client_rows if r["Is_Poisoned_Update"]),
               "Mean_Unpoisoned_Zero_Weight_Rate": mean_present(metric_rows, "Unpoisoned_Zero_Weight_Rate"),
               "Mean_Server_Residual_Weight": mean_present(metric_rows, "Server_Residual_Weight"),
               "Mean_Unpoisoned_Capped_Mass_Retention": mean_present(metric_rows, "Unpoisoned_Capped_Mass_Retention"),
               "Total_Train_Seconds": sum(r["Train_Seconds"] for r in metric_rows),
               "Total_Aggregation_Seconds": sum(r["Aggregation_Seconds"] for r in metric_rows),
               "Total_Adaptive_Queries": sum(r["Adaptive_Queries"] for r in metric_rows),
               "Mean_Honest_Zero_Weight_Rate": mean_present(metric_rows, "Honest_Zero_Weight_Rate"),
               **detection}
    write_json(folder / "summary.json", summary)
    if not args.no_plots:
        plots(folder, cm)
    return summary


def experiment_jobs(args):
    """Finite, named plans. Never silently launch the full supervisor checklist."""
    base = copy.deepcopy(args)
    base.seeds = args.seeds or [args.seed]
    base.defenses = args.defenses or (["fedavg", "median", "trimmed_mean", "krum", "fltrust", "fed_medishield"]
                                    if args.defense == "all" else [args.defense])
    if args.suite == "release":
        methods = args.defenses or ["fedavg", "median", "trimmed_mean", "krum", "fltrust", "fed_medishield"]
        jobs = []
        # One frozen plan rather than selecting the next favourable experiment.
        # The clean run is trained once per method/seed and reused for both
        # attack schedules. No training trajectory is shared across methods.
        for name, condition, onset in (("clean", "clean", 1),
                                       ("immediate", "attacked", 1),
                                       ("delayed", "attacked", args.delayed_start)):
            item = copy.deepcopy(base)
            item.condition, item.attack_start = condition, onset
            item.attack_period, item.attack_on = 1, 1
            item.defenses = methods if name != "delayed" else [
                m for m in methods if m in ("fedavg", "krum", "fltrust", "fed_medishield")]
            jobs.append((name, item))
        originals = list(jobs)
        for name, item in originals:
            ablated = copy.deepcopy(item)
            ablated.defenses = ["fed_medishield"]
            ablated.history_policy = "none"
            jobs.append(("history_none_" + name, ablated))
            no_audit = copy.deepcopy(item)
            no_audit.defenses = ["fed_medishield"]
            no_audit.audit_mode = "off"
            jobs.append(("audit_off_" + name, no_audit))
        return jobs
    if args.suite in ("single", "checkpoint"):
        if args.suite == "checkpoint" and args.defenses is None:
            base.defenses = ["fedavg", "krum", "fltrust", "fed_medishield"]
        return [("main", base)]
    if args.seeds is None:
        base.seeds = [101, 102, 103, 104, 105] if args.suite == "five_seed" else [args.seed]
    if args.defenses is None:
        base.defenses = ["fedavg", "median", "trimmed_mean", "krum", "fltrust", "fed_medishield"] if args.suite == "five_seed" else ["fedavg", "krum", "fltrust", "fed_medishield"]
    variations = []
    if args.suite == "five_seed":
        variations = [("five_seed", {})]
    elif args.suite == "alpha":
        variations = [(f"alpha_{v}", {"alpha": v}) for v in (.1, .5, 1., 5.)]
    elif args.suite == "malicious":
        # n fixed explicitly so rounding and the declared f are reviewable.
        n = args.num_clients or 8
        variations = [(f"malicious_{v}", {"alpha": args.alpha or .5, "num_clients": n,
                      "malicious_ratio": v, "f": math.ceil(n * v)}) for v in (.125, .25)]
    elif args.suite == "poison":
        variations = [(f"poison_{v}", {"poison_rate": v}) for v in (.05, .1, .2, .4)]
    elif args.suite == "triggers":
        variations = [(v, {"trigger": v}) for v in ("constant", "additive", "blended")]
    elif args.suite == "ablation":
        base.defenses = args.defenses or ["fed_medishield"]
        variations = [(f"history_{v}", {"history_policy": v}) for v in ("none", "ungated", "proportional", "hard")]
        variations += [("audit_off", {"audit_mode": "off"}),
                       ("no_probe_bank", {"probe_bank_size": 0}),
                       ("no_clipping", {"clip_factor": 0.}),
                       ("no_risk_memory", {"risk_decay": 0.})]
    elif args.suite == "hyperparameters":
        base.defenses = args.defenses or ["fed_medishield"]
        variations = [("reference", {})]
        variations += [(f"k_{v}", {"mad_k": v}) for v in (2., 4.)]
        variations += [(f"audit_gain_{v}", {"audit_min_gain": v}) for v in (.05, .15)]
        variations += [(f"control_margin_{v}", {"control_margin": v}) for v in (.01, .05)]
        variations += [(f"control_steps_{v}", {"control_steps": v}) for v in (8, 32)]
        variations += [(f"probe_fraction_{v}", {"audit_fraction": v}) for v in (.06, .20)]
        variations += [(f"rho_{v}", {"history_rate": v}) for v in (.05, .2)]
        variations += [(f"admission_{v}", {"admission_patience": v}) for v in (1, 3)]
        variations += [(f"history_limit_{v}", {"history_limit": v}) for v in (1., 2.)]
    elif args.suite == "scalability":
        variations = [(f"clients_{n}", {"num_clients": n, "alpha": args.alpha or .5,
            "malicious_ratio": .125, "f": math.ceil(n * .125)}) for n in (8, 16, 32)]
    elif args.suite == "adaptive":
        variations = [("static", {"attack": "static"}),
                      ("on_off", {"attack": "static", "attack_start": 4, "attack_period": 4, "attack_on": 1}),
                      ("adaptive_queries", {"attack": "adaptive"})]
    for name, updates in variations:
        for key, value in updates.items():
            if not hasattr(base, key):
                raise AssertionError(f"Unknown plan setting {key}")
    jobs = []
    for name, updates in variations:
        item = copy.deepcopy(base)
        for key, value in updates.items():
            setattr(item, key, value)
        jobs.append((name, item))
    return jobs


def self_check():
    """Small numerical invariants and tiny synthetic control fits; no real data."""
    s = [{"w": torch.tensor([float(v)])} for v in (0, 0, 0, 0, 0, 100)]
    sizes = [1] * len(s)
    original = copy.deepcopy(s)
    result, selected = krum(s, sizes, 1)
    assert selected == [0] and float(result["w"]) == 0
    assert float(trimmed_mean(s, sizes, 1)[0]["w"]) == 0
    assert float(coordinate_median(s, sizes)[0]["w"]) == 0
    avg, weights = fedavg_aggregate(s[:2] + [{"w": torch.tensor([4.])}], [1, 1, 2])
    assert float(avg["w"]) == 2 and weights == [.25, .25, .5]
    assert all(torch.equal(a["w"], b["w"]) for a, b in zip(s, original))
    shield = LegacyHistoryShield(["w"], f=1, mode="shadow")
    base = {"w": torch.tensor([0.])}
    identical = [{"w": torch.tensor([1.])} for _ in range(6)]
    result, d = shield.aggregate(base, identical, sizes, list(range(6)))
    reference, _ = fedavg_aggregate(identical, sizes)
    assert torch.equal(result["w"], reference["w"])
    assert not any(row["History_Admitted"] for row in d["clients"])
    shield.aggregate(base, identical, sizes, list(range(6)))
    assert shield.memory[0]["mu"] is not None
    held = copy.deepcopy(shield.memory[0])
    shield.aggregate(base, identical, sizes, list(range(1, 7)))
    assert torch.equal(held["mu"], shield.memory[0]["mu"])
    assert held["risk"] == shield.memory[0]["risk"]
    # No fixed rejection quota: all identical qualifying clients contribute.
    filtering = LegacyHistoryShield(["w"], f=1, mode="filter")
    result, details = filtering.aggregate(base, identical, sizes, list(range(6)))
    assert torch.equal(result["w"], reference["w"])
    assert all(row["Aggregation_Weight"] > 0 for row in details["clients"])
    # A huge outlier cannot inflate the remaining sample shares.
    result, details = filtering.aggregate(base, s, sizes, list(range(6)))
    assert all(row["Effective_Delta_Multiplier"] <= 1 / 6 + 1e-12 for row in details["clients"])
    assert all(row["Weight_Amplification"] <= 1 + 1e-12 for row in details["clients"])
    total = sum(row["Effective_Delta_Multiplier"] for row in details["clients"])
    assert abs(total + details["clients"][0]["Server_Residual_Weight"] - 1) < 1e-12
    # A hard-gated reference stays frozen when its current update is flagged.
    gated = LegacyHistoryShield(["w"], f=1, admission_patience=1)
    gated.aggregate(base, identical, sizes, list(range(6)))
    previous = copy.deepcopy(gated.memory[5])
    bad = copy.deepcopy(identical)
    bad[5]["w"] = torch.tensor([100.])
    _, details = gated.aggregate(base, bad, sizes, list(range(6)))
    assert details["clients"][5]["Flagged"] and not details["clients"][5]["Reference_Changed"]
    assert torch.equal(gated.memory[5]["mu"], previous["mu"])
    assert gated.memory[5]["nu"] == previous["nu"]
    # A participating uninformative update is not automatic rehabilitation.
    before_risk = gated.memory[5]["risk"]
    zeros = [{"w": torch.tensor([0.])} for _ in range(6)]
    gated.aggregate(base, zeros, sizes, list(range(6)))
    assert gated.memory[5]["risk"] == before_risk
    shadow = LegacyHistoryShield(["w"], f=1, mode="shadow", admission_patience=1)
    shadow.aggregate(base, s, sizes, list(range(6)))
    assert shadow.memory[5]["mu"] is None  # Outlier not admitted at bootstrap.
    invalid = copy.deepcopy(s)
    invalid[0]["w"][0] = float("nan")
    try:
        krum(invalid, sizes, 1)
        raise AssertionError("NaN was accepted")
    except ValueError:
        pass
    pre = detection_metrics([False, False], [False, True])
    assert pre["Detection_Rate"] is None and pre["Unpoisoned_Update_FPR"] == 50.
    # Test full-search control calibration, including multiple probes, both
    # folds and self-exclusion. No actual attacker ID enters this API.
    gains = torch.zeros(5, 2, 2)  # two submissions then three controls
    gains[0, 0] = torch.tensor([.4, .4])
    gains[1, 1] = torch.tensor([.6, 0.])
    supported = torch.ones_like(gains, dtype=torch.bool)
    signatures = torch.zeros(5, 4)
    calibrated = SparseProbeAudit.calibrate(gains, supported, signatures, 2, 2, .1, .02)
    assert calibrated["scores"][0] > 1 and calibrated["scores"][1] == 0
    assert all(i not in calibrated["matched_controls"][i+2] for i in range(3))
    equal = SparseProbeAudit.calibrate(torch.full_like(gains, .5), supported, signatures, 2, 2, .1, .02)
    assert float(equal["scores"].max()) == 0
    unsupported = SparseProbeAudit.calibrate(gains, torch.zeros_like(supported), signatures, 2, 2, .1, .02)
    assert float(unsupported["scores"].max()) == 0
    # Different legitimate controls can be extreme on different probes. The
    # largest leave-one-control-out max raises the common threshold, rather
    # than producing a union of unrelated per-probe outliers.
    varied = torch.zeros(5, 3, 2)
    varied[0] = .4
    for i in range(3):
        varied[i+2, i] = .4
    calibrated = SparseProbeAudit.calibrate(varied, torch.ones_like(varied, dtype=torch.bool), signatures, 2, 2, .1, .02)
    assert abs(calibrated["threshold"] - .42) < 1e-6
    assert not bool((calibrated["scores"] > 1).any())
    varied[1, 0] = 1.2
    changed = SparseProbeAudit.calibrate(varied, torch.ones_like(varied, dtype=torch.bool), signatures, 2, 2, .1, .02)
    assert changed["scores"][1] > 1
    permutation = torch.tensor([1, 0, 2, 3, 4])
    permuted = SparseProbeAudit.calibrate(varied[permutation], torch.ones_like(varied, dtype=torch.bool), signatures, 2, 2, .1, .02)
    assert torch.equal(permuted["scores"], changed["scores"].flip(0))
    # Published FLTrust normalization: orthogonal/opposite directions have zero
    # weight, same direction with different lengths yields the root magnitude.
    zero = {"w": torch.tensor([0., 0.])}
    root = {"w": torch.tensor([2., 0.])}
    vectors = [{"w": torch.tensor(v)} for v in ([1., 0.], [4., 0.], [-3., 0.], [0., 2.])]
    trust_result, rows = fltrust_aggregate(zero, vectors, [1] * 4, root)
    assert torch.allclose(trust_result["w"], root["w"])
    assert rows[0]["Aggregation_Weight"] == .5 and rows[2]["Aggregation_Weight"] == 0
    held_result, _ = fltrust_aggregate(zero, vectors, [1] * 4, zero)
    assert torch.equal(held_result["w"], zero["w"])
    # End-to-end tiny synthetic probe search, admission, checkpoint and shadow.
    # Only one-step synthetic control fits; no experiment dataset is involved.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        tiny = nn.Linear(3, 2)
    dx = torch.tensor([[.1,.2,.3], [.2,.1,.4], [-.1,-.2,-.3], [-.2,-.1,-.4]])
    dy = torch.tensor([0,0,1,1])
    vx = torch.tensor([[.3,.1,.2], [.4,.2,.1], [.5,.3,.2], [.6,.1,.3],
                       [-.3,-.1,-.2], [-.4,-.2,-.1], [-.5,-.3,-.2], [-.6,-.1,-.3]])
    vy = torch.tensor([0,0,0,0,1,1,1,1])
    initial = copy.deepcopy(tiny.state_dict())
    local = {k: v + .01 for k,v in initial.items()}
    models = [copy.deepcopy(local) for _ in range(3)]
    options = {"steps": 1, "restarts": 1, "fraction": .5, "bank_size": 2, "control_steps": 1}
    def new_tiny():
        return FedMediShield(tiny, dx, dy, vx, vy, classes=2,
            final_keys=("weight", "bias"), mode="shadow", audit_options=options)
    candidate = new_tiny()
    actual, diagnostic = candidate.aggregate(initial, models, [1,2,3], [0,1,2])
    expected, _ = fedavg_aggregate(models, [1,2,3])
    assert all(torch.equal(actual[k], expected[k]) for k in actual)
    assert all(row["Audit_Complete"] and not row["Flagged"] for row in diagnostic["clients"])
    assert all(torch.equal(tiny.state_dict()[k], initial[k]) for k in initial)
    assert not any(row["History_Admitted"] for row in diagnostic["clients"])
    restored = new_tiny()
    restored.import_state(copy.deepcopy(candidate.export_state()))
    a, da = candidate.aggregate(initial, models, [1,2,3], [0,1,2])
    b, db = restored.aggregate(initial, models, [1,2,3], [0,1,2])
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert da["clients"] == db["clients"]
    assert all(row["History_Admitted"] for row in da["clients"])
    # Inject a deterministic test alert to verify the new admission boundary.
    prior = copy.deepcopy(candidate.memory[2])
    candidate.mode = "filter"
    candidate.audit.evaluate = lambda *unused: [
        {"Audit_Score": float(i == 2) * 2., "Audit_Complete": 1, "Audit_Flagged": int(i == 2)}
        for i in range(3)]
    _, blocked = candidate.aggregate(initial, models, [1,2,3], [0,1,2])
    assert blocked["clients"][2]["Aggregation_Weight"] == 0
    assert blocked["clients"][2]["Flagged"] and not blocked["clients"][2]["Reference_Changed"]
    assert torch.equal(candidate.memory[2]["mu"], prior["mu"])
    assert all(row["Effective_Delta_Multiplier"] <= size/6 + 1e-12
               for row, size in zip(blocked["clients"], [1,2,3]))
    # Repeated flags cannot drive risk beyond the configured bounded memory.
    for _ in range(5):
        candidate.aggregate(initial, models, [1,2,3], [0,1,2])
    assert candidate.memory[2]["risk"] <= candidate.risk_cap
    saved = copy.deepcopy(candidate.export_state())
    candidate.audit.evaluate = lambda *unused: [
        {"Audit_Score": 2., "Audit_Complete": 1, "Audit_Flagged": 1} for _ in range(3)]
    try:
        candidate.aggregate(initial, models, [1,2,3], [0,1,2])
        raise AssertionError("An all-rejected filtering round was accepted")
    except DetectorCollapseError as error:
        assert error.diagnostic["effective_mass"] == 0
    assert candidate.audit.round == saved["audit_round"]
    assert all(candidate.memory[i]["risk"] == saved["memory"][i]["risk"] for i in range(3))
    assert all(torch.equal(candidate.memory[i]["mu"], saved["memory"][i]["mu"]) for i in range(3))
    # A minority by count can hold most sample mass: count alone is insufficient.
    candidate.audit.evaluate = lambda *unused: [
        {"Audit_Score": float(i == 2) * 2., "Audit_Complete": 1, "Audit_Flagged": int(i == 2)} for i in range(3)]
    try:
        candidate.aggregate(initial, models, [1,1,98], [0,1,2])
        raise AssertionError("An almost-zero-mass filtering round was accepted")
    except DetectorCollapseError as error:
        assert error.diagnostic["flagged_fraction"] < .5
        assert error.diagnostic["effective_mass"] < .25
    print("PASS: aggregation invariants, full-search calibration, control self-exclusion, bounded risk, stop/rollback, synthetic controls/search, frozen admission, checkpoint state and shadow identity. These checks do not establish attack robustness.")


def parse_args():
    p = argparse.ArgumentParser(description="Paired validation research harness; no publication or robustness guarantee.")
    p.add_argument("--version", action="version", version=EXPECTED_DEFENSE_VERSION)
    p.add_argument("--data", default="data/processed_grouped/federated_data.pt")
    p.add_argument("--defense", choices=METHODS + ["all"], default="fed_medishield")
    p.add_argument("--defenses", nargs="+", choices=METHODS)
    p.add_argument("--condition", choices=["clean", "attacked", "both"], default="both")
    p.add_argument("--no-attack", action="store_true", help="Compatibility alias for --condition clean.")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=.001)
    p.add_argument("--loss-adjustment", choices=["none", "logit"], default="none",
                   help="logit = balanced-softmax local loss with the pooled training class prior; none = original protocol.")
    p.add_argument("--logit-tau", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--poison-seed", type=int, default=42)
    p.add_argument("--attacker-id", type=int, default=7)
    p.add_argument("--malicious-ratio", type=float)
    p.add_argument("--poison-rate", type=float, default=.2)
    p.add_argument("--target-class", type=int, default=0)
    p.add_argument("--trigger", choices=["constant", "additive", "blended"], default="constant")
    p.add_argument("--trigger-features", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--trigger-feature-names", nargs="+", help="Use exact feature_names.json entries; takes precedence over indices.")
    p.add_argument("--trigger-value", type=float, default=5.)
    p.add_argument("--blend", type=float, default=.2)
    p.add_argument("--poison-source-classes", type=int, nargs="+",
                   help="Narrow backdoor: poison only rows of these class ids (never the target class).")
    p.add_argument("--source-specific", action="store_true",
                   help="With --poison-source-classes: also add the trigger to other attack rows with their true labels.")
    p.add_argument("--probe-camouflage", type=float, default=0.0,
                   help="Adaptive attacker: fraction of unpoisoned attack rows given a probe-style perturbation with their true label.")
    p.add_argument("--attack-start", type=int, default=1)
    p.add_argument("--attack-period", type=int, default=1)
    p.add_argument("--attack-on", type=int, default=1)
    p.add_argument("--model-scale", type=float, default=1.)
    p.add_argument("--attack", choices=["static", "adaptive"], default="static")
    p.add_argument("--adaptive-grid", type=float, nargs="+", default=[0., .25, .5, .75, 1., 1.5, 2.])
    p.add_argument("--adaptive-probe-size", type=int, default=128)
    p.add_argument("--adaptive-clean-weight", type=float, default=.25)
    p.add_argument("--f", type=int, default=1, help="Declared malicious-client budget, not a list of true attacker IDs.")
    p.add_argument("--multi-krum-m", type=int)
    p.add_argument("--alpha", type=float, help="Repartition training indices; otherwise preserve saved partition.")
    p.add_argument("--num-clients", type=int)
    p.add_argument("--partition-seed", type=int,
                   help="With --alpha: one fixed partition for all seeds (matches medishield_candidate.py --partition-seed); default: per-seed partition.")
    p.add_argument("--clients-per-round", type=int)
    p.add_argument("--server-per-class", type=int, default=32)
    p.add_argument("--calibration-seed", type=int, default=2026)
    p.add_argument("--shield-mode", choices=["shadow", "filter"], default="filter")
    p.add_argument("--audit-mode", choices=["on", "off"], default="on", help="Off is an explicit audit-removal ablation.")
    p.add_argument("--audit-steps", type=int, default=16)
    p.add_argument("--audit-restarts", type=int, default=2)
    p.add_argument("--audit-fraction", type=float, default=.12, help="Maximum fraction of features replaced in searched probes.")
    p.add_argument("--audit-bound", type=float, default=8., help="Absolute bound on searched replacement values in standardized space.")
    p.add_argument("--audit-learning-rate", type=float, default=.2)
    p.add_argument("--audit-min-gain", type=float, default=.10, help="Minimum absolute probability double-difference on BOTH confirmation folds.")
    p.add_argument("--control-margin", type=float, default=.02,
                   help="Positive probability-effect margin above the largest leave-one-control-out full-search score.")
    p.add_argument("--audit-scale-floor", type=float, help="Retired per-probe MAD option; rejected explicitly.")
    p.add_argument("--control-steps", type=int, default=16, help="Discovery-only AdamW steps for each clean class-mixture control.")
    p.add_argument("--control-neighbors", type=int, default=3, help="Nearest clean-output control signatures used for the response envelope.")
    p.add_argument("--control-dominance", type=float, nargs="+", default=[.6, .9],
                   help="Dominant class masses; one control per class/mass, plus a balanced control.")
    p.add_argument("--probe-bank-size", type=int, default=24)
    p.add_argument("--root-steps", type=int, default=8, help="FLTrust full-root AdamW steps per round.")
    p.add_argument("--selection-rule", choices=["adaptive", "topk"], default="adaptive",
                   help="Adaptive has no fixed rejection quota; topk is an explicit ablation.")
    p.add_argument("--weight-rule", choices=["capped", "renormalized"], default="capped",
                   help="Capped preserves discarded mass at the previous global model.")
    p.add_argument("--clip-rule", choices=["mad", "median_core"], default="mad")
    p.add_argument("--core-size", type=int, default=0, help="Only with selection-rule topk; 0 chooses n-f-2.")
    p.add_argument("--spatial-scope", choices=["whole", "final"], default="whole")
    p.add_argument("--history-policy", choices=["none", "ungated", "proportional", "hard"], default="hard")
    p.add_argument("--mad-k", type=float, default=3.)
    p.add_argument("--history-limit", type=float, default=1.5)
    p.add_argument("--history-rate", type=float, default=.1)
    p.add_argument("--admission-patience", type=int, default=2)
    p.add_argument("--recovery-patience", type=int, default=6)
    p.add_argument("--reference-trust", type=float, default=.5)
    p.add_argument("--risk-decay", type=float, default=.8)
    p.add_argument("--risk-gain", type=float, default=1.)
    p.add_argument("--risk-cap", type=float, default=2., help="Fed-MediShield accumulated suspicion cap; current alerts still have zero weight.")
    p.add_argument("--stop-flagged-fraction", type=float, default=.5,
                   help="Stop an invalid filtering run before applying a round if flagged fraction reaches this bound.")
    p.add_argument("--stop-min-mass", type=float, default=.25,
                   help="Stop before applying a filter round whose total effective update coefficient is below this bound.")
    p.add_argument("--clip-factor", type=float, default=1., help="Multiplier of configured norm bound; 0 disables clipping for ablation.")
    p.add_argument("--components", choices=["joint", "saliency", "response"], default="joint", help="Retired behavioral baseline only.")
    p.add_argument("--score-floor", type=float, default=.01, help="Behavioral baseline only.")
    p.add_argument("--penalty", type=float, default=1., help="Behavioral baseline only.")
    p.add_argument("--class-score-top-k", type=int, help="Obsolete; rejected explicitly, not silently ignored.")
    p.add_argument("--suite", choices=["single", "checkpoint", "release", "five_seed", "alpha", "malicious", "poison", "triggers", "ablation", "hyperparameters", "scalability", "adaptive"], default="single")
    p.add_argument("--delayed-start", type=int, default=8, help="Delayed attack onset in the release suite.")
    p.add_argument("--gate-max-asr", type=float, default=10., help="Predeclared development acceptance ceiling, not a security guarantee.")
    p.add_argument("--gate-min-fedavg-asr", type=float, default=50., help="Weaker unprotected attacks make a security comparison inconclusive.")
    p.add_argument("--gate-max-clean-f1-drop", type=float, default=2., help="Allowed clean macro-F1 loss in percentage points vs FedAvg.")
    p.add_argument("--gate-max-recall-drop", type=float, default=5., help="Allowed clean per-class recall loss in percentage points vs FedAvg.")
    p.add_argument("--gate-max-update-fpr", type=float, default=5., help="Allowed clean-update alert rate in percent, across client-round observations.")
    p.add_argument("--plan-only", action="store_true", help="Print settings/job counts without loading data or training.")
    p.add_argument("--out", help="Fresh result folder, or existing folder with --resume.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--self-check", action="store_true", help="Small invariants and one-step synthetic control fits; no experiment data.")
    a = p.parse_args()
    if a.no_attack:
        a.condition = "clean"
    if a.class_score_top_k is not None:
        p.error("--class-score-top-k belongs to the retired classwise scorer. Remove it for this candidate.")
    if a.audit_scale_floor is not None:
        p.error("--audit-scale-floor belongs to the retired per-probe MAD rule. This release uses --control-margin.")
    positive = [a.rounds, a.local_epochs, a.batch_size, a.threads, a.learning_rate,
                a.mad_k, a.history_limit, a.attack_start, a.attack_period, a.adaptive_probe_size]
    if any(v <= 0 or not math.isfinite(v) for v in positive):
        p.error("Round/training/threshold/schedule sizes must be positive and finite.")
    numeric = [a.trigger_value, a.model_scale, a.blend, a.poison_rate, a.risk_gain,
               a.clip_factor, a.adaptive_clean_weight, a.score_floor, a.penalty, *a.adaptive_grid]
    if not all(math.isfinite(v) for v in numeric):
        p.error("Numeric settings must be finite.")
    if not 0 <= a.poison_rate <= 1 or not 0 <= a.blend <= 1 or a.model_scale <= 0:
        p.error("Invalid poison/blend/model scale.")
    if not 0 <= a.attack_on <= a.attack_period or a.f < 0 or a.server_per_class < 0:
        p.error("Invalid attack schedule, f or calibration count.")
    if a.malicious_ratio is not None and not 0 < a.malicious_ratio < 1:
        p.error("Malicious ratio must lie strictly between 0 and 1.")
    if not 0 <= a.probe_camouflage <= 1 or (a.probe_camouflage and a.source_specific):
        p.error("--probe-camouflage is a fraction in [0, 1] and is not combined with --source-specific.")
    if a.alpha is not None and (not math.isfinite(a.alpha) or a.alpha <= 0):
        p.error("alpha must be positive and finite.")
    if a.num_clients is not None and a.num_clients < 3:
        p.error("Need at least three clients.")
    if a.clients_per_round is not None and a.clients_per_round < 1:
        p.error("clients-per-round must be positive.")
    if a.core_size < 0 or a.risk_gain < 0 or a.clip_factor < 0 or a.adaptive_clean_weight < 0:
        p.error("Core size and penalty/clipping settings cannot be negative.")
    if a.seed < 0 or a.poison_seed < 0 or (a.seeds and (min(a.seeds) < 0 or len(set(a.seeds)) != len(a.seeds))):
        p.error("Seeds must be distinct nonnegative integers.")
    if a.calibration_seed < 0 or (a.defenses and len(a.defenses) != len(set(a.defenses))):
        p.error("Calibration seed must be nonnegative and method names must be distinct.")
    if a.score_floor <= 0 or a.penalty < 0:
        p.error("Behavioral score floor must be positive and penalty nonnegative.")
    if a.resume and not a.out:
        p.error("--resume requires the same explicit --out used previously.")
    gate_values = (a.gate_max_asr, a.gate_min_fedavg_asr, a.gate_max_clean_f1_drop,
                   a.gate_max_recall_drop, a.gate_max_update_fpr)
    if not all(math.isfinite(v) and 0 <= v <= 100 for v in gate_values):
        p.error("Development acceptance thresholds must be finite percentages between 0 and 100.")
    if a.suite == "release":
        if a.condition != "both" or a.attack != "static" or a.shield_mode != "filter":
            p.error("--suite release requires --condition both, --attack static and --shield-mode filter.")
        if not 1 < a.delayed_start <= a.rounds:
            p.error("Release suite needs 1 < delayed-start <= rounds.")
        if a.history_policy != "hard":
            p.error("Release compares hard history to no history; use --history-policy hard.")
        if a.defenses and not {"fedavg", "krum", "fltrust", "fed_medishield"}.issubset(a.defenses):
            p.error("Release suite requires fedavg, krum, fltrust and fed_medishield among --defenses.")
    if (a.audit_steps < 1 or a.audit_restarts < 1 or a.probe_bank_size < 0 or a.root_steps < 1
        or not 0 < a.audit_fraction <= 1 or not 0 < a.audit_min_gain <= 2):
        p.error("Invalid probe/root budget or effect threshold.")
    if not all(math.isfinite(v) and v > 0 for v in
               (a.audit_bound, a.audit_learning_rate, a.control_margin, a.audit_min_gain, a.audit_fraction)):
        p.error("Probe search parameters must be finite and positive.")
    if (a.control_steps < 1 or not 1 <= a.control_neighbors < 1 + len(CLASS_NAMES) * len(a.control_dominance)
        or len(set(a.control_dominance)) != len(a.control_dominance)
        or not all(math.isfinite(v) and 1 / len(CLASS_NAMES) < v < 1 for v in a.control_dominance)
        or not 0 < a.control_margin <= 2):
        p.error("Invalid control training budget, neighbourhood, class masses or effect margin.")
    if not all(math.isfinite(v) for v in (a.risk_cap, a.stop_flagged_fraction, a.stop_min_mass)) or not (
            a.risk_cap >= 0 and 0 < a.stop_flagged_fraction <= 1 and 0 < a.stop_min_mass <= 1):
        p.error("Invalid Fed-MediShield risk cap or detector-stop bounds.")
    selected = a.defenses or (["fed_medishield"] if a.defense in ("all", "fed_medishield") else [a.defense])
    needs_probe = "fed_medishield" in selected or a.suite != "single"
    if needs_probe and a.server_per_class < 8:
        p.error("The probe candidate needs at least 8 distinct trusted groups per class (default 32).")
    if needs_probe and (a.selection_rule != "adaptive" or a.weight_rule != "capped"
                        or a.clip_rule != "mad" or a.core_size or a.spatial_scope != "whole"):
        p.error("Geometric top-k/scope/renormalization options apply only to --defense legacy_history --suite single.")
    if a.suite == "release" and a.audit_mode != "on":
        p.error("Release requires audit-mode on; its off ablation is included automatically.")
    # Validate history ranges even when just printing a plan.
    try:
        LegacyHistoryShield(["unused"], k=a.mad_k, history_limit=a.history_limit, rho=a.history_rate,
            admission_patience=a.admission_patience, recovery_patience=a.recovery_patience,
            trust_cutoff=a.reference_trust, risk_decay=a.risk_decay,
            risk_gain=a.risk_gain, clip_factor=a.clip_factor, core_size=a.core_size,
            selection_rule=a.selection_rule, weight_rule=a.weight_rule, clip_rule=a.clip_rule)
    except ValueError as error:
        p.error(str(error))
    return a


def summarize_release(root, summaries, args):
    """Post-run reporting only. NEVER changes training, scores or selection.

    The fixed engineering criteria are not statistical significance tests and
    passing one seed cannot establish robustness or publication readiness.
    """
    lookup = {(r["Case"], r["Seed"], r["Method"]): r for r in summaries}
    seeds = sorted({r["Seed"] for r in summaries})
    comparisons, history, audit_effect = [], [], []
    for seed in seeds:
        clean = lookup.get(("clean", seed, "fed_medishield"))
        clean_avg = lookup.get(("clean", seed, "fedavg"))
        if not clean or not clean_avg:
            continue
        f1_loss = clean_avg["Clean_F1"] - clean["Clean_F1"]
        recall_losses = {name: clean_avg[f"Recall_{name}"] - clean[f"Recall_{name}"] for name in CLASS_NAMES}
        worst_class = max(recall_losses, key=recall_losses.get)
        for case in ("immediate", "delayed"):
            defended = lookup.get((case, seed, "fed_medishield"))
            avg = lookup.get((case, seed, "fedavg"))
            kr = lookup.get((case, seed, "krum"))
            ft = lookup.get((case, seed, "fltrust"))
            if not all((defended, avg, kr)):
                continue
            checks = {
                "attack_is_effective": avg["ASR"] is not None and avg["ASR"] >= args.gate_min_fedavg_asr,
                "ASR_below_ceiling": defended["ASR"] is not None and defended["ASR"] <= args.gate_max_asr,
                "clean_F1_preserved": f1_loss <= args.gate_max_clean_f1_drop,
                "clean_class_recall_preserved": recall_losses[worst_class] <= args.gate_max_recall_drop,
                "clean_alert_FPR_below_ceiling": clean.get("Unpoisoned_Update_FPR") is not None
                    and clean["Unpoisoned_Update_FPR"] <= args.gate_max_update_fpr,
            }
            failures = [name for name, passed in checks.items() if not passed]
            utility_failed = any(not checks[name] for name in (
                "clean_F1_preserved", "clean_class_recall_preserved", "clean_alert_FPR_below_ceiling"))
            status = ("FAIL" if utility_failed else "INCONCLUSIVE_WEAK_ATTACK" if not checks["attack_is_effective"]
                      else "FAIL" if not checks["ASR_below_ceiling"] else "PASS_DEVELOPMENT_CHECK")
            comparisons.append({"Case": case, "Seed": seed, "Status": status,
                "Failures": "; ".join(failures), "Shield_CA": defended["Clean_Accuracy"],
                "Shield_F1": defended["Clean_F1"], "Shield_ASR": defended["ASR"],
                "FedAvg_ASR": avg["ASR"], "Krum_ASR": kr["ASR"],
                "FLTrust_ASR": ft["ASR"] if ft else None,
                "F1_Difference_vs_FLTrust_pp": defended["Clean_F1"] - ft["Clean_F1"] if ft else None,
                "ASR_Difference_vs_FLTrust_pp": (defended["ASR"] - ft["ASR"])
                    if ft and ft["ASR"] is not None and defended["ASR"] is not None else None,
                "Poisoned_Audit_Flags": defended["Poisoned_Audit_Flags"],
                "F1_Difference_vs_Krum_pp": defended["Clean_F1"] - kr["Clean_F1"],
                "ASR_Difference_vs_Krum_pp": (defended["ASR"] - kr["ASR"])
                    if defended["ASR"] is not None and kr["ASR"] is not None else None,
                "Clean_F1_Loss_vs_FedAvg_pp": f1_loss,
                "Worst_Clean_Recall_Loss_pp": recall_losses[worst_class], "Worst_Recall_Class": worst_class,
                "Clean_Update_Alert_FPR": clean.get("Unpoisoned_Update_FPR"),
                "Shield_Unpoisoned_Capped_Mass_Retention": defended["Mean_Unpoisoned_Capped_Mass_Retention"],
                "Krum_Unpoisoned_Capped_Mass_Retention": kr["Mean_Unpoisoned_Capped_Mass_Retention"],
                "Poisoned_History_Admissions": defended["Poisoned_History_Admissions"],
                "Poisoned_History_Recoveries": defended["Poisoned_History_Recoveries"],
                **checks})
        for case in ("clean", "immediate", "delayed"):
            with_history = lookup.get((case, seed, "fed_medishield"))
            without = lookup.get(("history_none_" + case, seed, "fed_medishield"))
            if with_history and without:
                history.append({"Case": case, "Seed": seed,
                    "F1_Gain_With_History_pp": with_history["Clean_F1"] - without["Clean_F1"],
                    "CA_Gain_With_History_pp": with_history["Clean_Accuracy"] - without["Clean_Accuracy"],
                    "ASR_Reduction_With_History_pp": (without["ASR"] - with_history["ASR"])
                        if without["ASR"] is not None and with_history["ASR"] is not None else None,
                    "TTR_Reduction_With_History_pp": without["Triggered_Target_Rate"] - with_history["Triggered_Target_Rate"],
                    "History_Admissions": with_history["History_Admissions"],
                    "Poisoned_History_Admissions": with_history["Poisoned_History_Admissions"]})
            no_audit = lookup.get(("audit_off_" + case, seed, "fed_medishield"))
            if with_history and no_audit:
                audit_effect.append({"Case": case, "Seed": seed,
                    "CA_Gain_With_Audit_pp": with_history["Clean_Accuracy"] - no_audit["Clean_Accuracy"],
                    "F1_Gain_With_Audit_pp": with_history["Clean_F1"] - no_audit["Clean_F1"],
                    "TTR_Reduction_With_Audit_pp": no_audit["Triggered_Target_Rate"] - with_history["Triggered_Target_Rate"],
                    "ASR_Reduction_With_Audit_pp": (no_audit["ASR"] - with_history["ASR"])
                        if no_audit["ASR"] is not None and with_history["ASR"] is not None else None,
                    "Poisoned_Audit_Flags": with_history["Poisoned_Audit_Flags"],
                    "Poisoned_History_Admissions_With_Audit": with_history["Poisoned_History_Admissions"],
                    "Poisoned_History_Admissions_Without_Audit": no_audit["Poisoned_History_Admissions"]})
    rewrite_csv(root / "audit_effect.csv", audit_effect)
    rewrite_csv(root / "acceptance.csv", comparisons)
    rewrite_csv(root / "history_effect.csv", history)
    status = ("FAIL" if any(r["Status"] == "FAIL" for r in comparisons)
              else "INCONCLUSIVE" if not comparisons or any(r["Status"].startswith("INCONCLUSIVE") for r in comparisons)
              else "PASS_DEVELOPMENT_CHECK")
    report = {"Status": status, "Seeds": seeds, "Evidence": "Validation only; no superiority, novelty or test-set claim.",
              "Acceptance": comparisons, "History_Effect": history, "Audit_Effect": audit_effect,
              "Limits": ["A short run can miss slow attacks; no worst-case guarantee.",
                         "One seed is not robustness evidence.",
                         "A bounded sparse search can miss triggers; majority-corrupted or non-IID probe cohorts can defeat calibration.",
                         "Trusted server data is an added threat-model assumption; reused probe folds do not certify FPR.",
                         "No other dataset or official test set is evaluated by this plan."]}
    write_json(root / "decision.json", report)
    lines = ["# Development decision", "", f"**{status}** — {len(seeds)} seed(s), validation only.", "",
             "The criteria were saved before training. This report does not alter the model or select a favourable seed.", "",
             "| Schedule | Seed | Status | Shield ASR | FedAvg ASR | Krum ASR | FLTrust ASR | Shield F1 minus Krum |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    number = lambda v: "N/A" if v is None else f"{v:.2f}%"
    for r in comparisons:
        lines.append(f"| {r['Case']} | {r['Seed']} | {r['Status']} | {number(r['Shield_ASR'])} | "
                     f"{number(r['FedAvg_ASR'])} | {number(r['Krum_ASR'])} | {number(r['FLTrust_ASR'])} | {r['F1_Difference_vs_Krum_pp']:+.2f} pp |")
    lines.extend(["", "## Failed checks", ""])
    lines.extend([f"- Seed {r['Seed']}, {r['Case']}: {r['Failures'] or 'none in this development check'}" for r in comparisons])
    lines.extend(["", "## What history added", "",
                  "See history_effect.csv: positive ASR reduction favours history; negative reduction means history was worse.",
                  "These are matched runs with the same scorer/aggregation settings and different evolving model trajectories.",
                  "History admission counts are not evidence that admitted updates were honest.", "",
                  "## What the probe audit added", "",
                  "See audit_effect.csv. Compare audit-on and audit-off, including clean utility and poisoned admissions.",
                  "The new independent signal is the frozen counterfactual probe; its usefulness is a hypothesis.",
                  "Ablated training trajectories diverge, so these are component comparisons, not causal certificates.", "",
                  "If the candidate fails, preserve that result. Do not report Krum's performance as Fed-MediShield's performance.",
                  "Passing these checks only supports a larger frozen-protocol study. Four datasets, longer/adaptive attacks, independent seeds and untouched test evaluation remain."])
    (root / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDEVELOPMENT DECISION: {status}. Read {root / 'decision.md'}", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.self_check:
        self_check()
        return
    jobs = experiment_jobs(args)
    conditions_for = lambda a: ["clean", "attacked"] if a.condition == "both" else [a.condition]
    plan = [{"Case": name, "Seeds": a.seeds, "Methods": a.defenses, "Conditions": conditions_for(a),
             "Rounds": a.rounds, "Alpha": a.alpha, "Clients": a.num_clients,
             "Malicious_Ratio": a.malicious_ratio, "f": a.f, "Poison_Rate": a.poison_rate,
             "Trigger": a.trigger, "Attack": a.attack, "History": a.history_policy,
             "Core_Size": a.core_size, "Scope": a.spatial_scope, "MAD_k": a.mad_k,
             "History_Rate": a.history_rate, "Admission_Patience": a.admission_patience,
             "History_Limit": a.history_limit, "Clip_Factor": a.clip_factor,
             "Risk_Decay": a.risk_decay,
             "Audit_Mode": a.audit_mode, "Audit_Steps": a.audit_steps, "Audit_Restarts": a.audit_restarts,
             "Audit_Fraction": a.audit_fraction, "Audit_Bound": a.audit_bound,
             "Audit_Min_Gain": a.audit_min_gain, "Probe_Bank_Size": a.probe_bank_size,
             "Control_Steps": a.control_steps, "Control_Neighbors": a.control_neighbors,
             "Control_Dominance": a.control_dominance, "Control_Margin": a.control_margin,
             "Risk_Cap": a.risk_cap, "Stop_Flagged_Fraction": a.stop_flagged_fraction,
             "Stop_Min_Mass": a.stop_min_mass,
             "Root_Steps": a.root_steps, "Trusted_Per_Class": a.server_per_class,
             "Selection_Rule": a.selection_rule, "Weight_Rule": a.weight_rule,
             "Clip_Rule": a.clip_rule, "Attack_Start": a.attack_start,
             "Attack_Period": a.attack_period, "Attack_On": a.attack_on} for name, a in jobs]
    count = sum(len(a.seeds) * len(a.defenses) * len(conditions_for(a)) for _, a in jobs)
    print(f"Plan: {count} runs; {count * args.rounds} federated rounds. Each run is independent.", flush=True)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if args.suite == "release":
        self_check()  # Executed on the user's computer before the one-command batch.
    data = load_data(args)
    if args.loss_adjustment == "logit":
        if not math.isfinite(args.logit_tau) or args.logit_tau <= 0:
            raise ValueError("--logit-tau must be positive and finite.")
        args.class_prior = class_prior(data)
    if args.trigger_feature_names:
        if not data["features"]:
            raise ValueError("Named triggers require feature_names.json.")
        try:
            args.trigger_features = [data["features"].index(name) for name in args.trigger_feature_names]
        except ValueError as error:
            raise ValueError("A requested trigger feature name is absent from the saved schema.") from error
    if (not args.trigger_features or len(set(args.trigger_features)) != len(args.trigger_features)
        or min(args.trigger_features) < 0 or max(args.trigger_features) >= data["X"].shape[1]):
        raise ValueError("Trigger columns must be distinct valid feature indices.")
    for _, job in jobs:
        job.trigger_features = args.trigger_features
        if args.loss_adjustment == "logit":
            job.class_prior = args.class_prior
    root = Path(args.out) if args.out else Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f_medishield")
    root.mkdir(parents=True, exist_ok=args.resume)
    print(f"Result folder: {root.resolve()}", flush=True)
    if args.resume and (root / "experiment_plan.json").exists():
        previous_plan = json.loads((root / "experiment_plan.json").read_text(encoding="utf-8"))
        if previous_plan != plan:
            raise ValueError("Resume plan differs. Use the original command and --out.")
    write_json(root / "experiment_plan.json", plan)
    if args.suite == "release":
        policy = {"version": defenses.VERSION,
                  **{k: v for k, v in vars(args).items() if k.startswith("gate_")},
                  "interpretation": "Predeclared development criteria; not a paper acceptance test or security guarantee."}
        policy_path = root / "decision_policy.json"
        if args.resume and policy_path.exists() and json.loads(policy_path.read_text(encoding="utf-8")) != policy:
            raise ValueError("Resume cannot change the saved acceptance criteria.")
        write_json(policy_path, policy)
    summaries = []
    for name, job in jobs:
        for seed in job.seeds:
            clients = make_partition(data, job, job.partition_seed if getattr(job, "partition_seed", None) is not None else seed)
            for method in job.defenses:
                for condition in conditions_for(job):
                    case_root = root if len(jobs) == 1 else root / name
                    result = run(job, data, clients, method, condition, seed, case_root)
                    summaries.append({"Case": name, **result})
                    rewrite_csv(root / "comparison.csv", summaries)
                    if not result["Completed"]:
                        stopped = {"Status": "STOPPED_DETECTOR_INVALID", "Reason": result["Reason"],
                            "Run": {"Case": name, "Method": method, "Condition": condition, "Seed": seed},
                            "Stopped_Round": result["Stopped_Round"],
                            "Completed_Rounds": result["Completed_Rounds"],
                            "Results_Folder": result["Results_Folder"],
                            "Remaining_Jobs_Skipped": True,
                            "Interpretation": "Filtering failed its predeclared validity bounds. No robustness/mitigation result for this stopped run."}
                        write_json(root / "decision.json", stopped)
                        (root / "decision.md").write_text(
                            "# Detector validity failure\n\n"
                            f"Stopped before round {result['Stopped_Round']} was applied. "
                            f"Only {result['Completed_Rounds']} rounds completed.\n\n"
                            f"{result['Reason']}\n\n"
                            "Remaining jobs were skipped. Final CA/F1/ASR for this run are null, "
                            "not measurements of successful protection. Inspect detector_stop.json "
                            "and stopped_client_decisions.csv in the run folder. Previous valid "
                            "rounds remain in metrics.csv and checkpoint.pt if any completed. "
                            "Do not disable the stop or raise its limits just to obtain a completed result.\n",
                            encoding="utf-8")
                        print(f"\nBatch stopped; remaining jobs skipped. Results: {root.resolve()}\n"
                              "Preserve this failure report; it is not an attack-mitigation result.", flush=True)
                        return 2
    frame = pd.DataFrame(summaries)
    columns = [c for c in ("Clean_Accuracy", "Clean_F1", "ASR", "Triggered_Target_Rate", "IDS_FPR",
               "Detection_Rate", "Honest_Alert_FPR", "Mean_Honest_Zero_Weight_Rate",
               "Unpoisoned_Update_FPR", "Mean_Unpoisoned_Zero_Weight_Rate", "Mean_Server_Residual_Weight",
               "Mean_Unpoisoned_Capped_Mass_Retention", "Poisoned_History_Admissions",
               "Total_Aggregation_Seconds", "Total_Train_Seconds") if c in frame]
    frame.groupby(["Case", "Method", "Condition"])[columns].agg(["mean", "std", "count"]).to_csv(root / "aggregate.csv")
    paired = []
    for (case, seed, method), group in frame.groupby(["Case", "Seed", "Method"]):
        rows = {r["Condition"]: r for r in group.to_dict("records")}
        if "clean" in rows and "attacked" in rows:
            clean, attacked = rows["clean"], rows["attacked"]
            baseline = frame[(frame.Case == case) & (frame.Seed == seed) & (frame.Method == "fedavg") & (frame.Condition == "attacked")]
            baseline_asr = baseline.iloc[0].ASR if len(baseline) else None
            paired.append({"Case": case, "Seed": seed, "Method": method,
                "Attack_CA_Difference_pp": attacked["Clean_Accuracy"] - clean["Clean_Accuracy"],
                "Attack_F1_Difference_pp": attacked["Clean_F1"] - clean["Clean_F1"],
                "TTR_Excess_Over_Clean_pp": attacked["Triggered_Target_Rate"] - clean["Triggered_Target_Rate"],
                "ASR_Reduction_vs_FedAvg_pp": float(baseline_asr) - float(attacked["ASR"])
                    if pd.notna(baseline_asr) and pd.notna(attacked["ASR"]) else None})
    rewrite_csv(root / "paired_comparison.csv", paired)
    if args.suite == "release":
        summarize_release(root, summaries, args)
    print(f"\nCompleted: {root.resolve()}\nValidation only; means/std are over seeds, not rounds.")
    if args.shield_mode == "shadow" and any(m in ("behavioral", "fed_medishield") for _, a in jobs for m in a.defenses):
        print("Shield shadow mode used FedAvg weights. These shield runs measure alerts, not mitigation.")


if __name__ == "__main__":
    raise SystemExit(main())
