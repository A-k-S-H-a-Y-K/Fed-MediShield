"""Fed-MediShield medishield_bsc_7: the final method of the paper (protected-evasion contrast,
known-clean client profiles, bounded first-episode revocation, held rounds).

The file keeps its development name because the pre-registrations and run records refer to it.
Data loading, reference reservation,
partition, StableMLP, AdamW local training, poisoning and validation metrics are
taken from the existing paired runner src/train_defended.py and verified by AST
fingerprints. The legacy classes in defenses.py are never instantiated here.
No baseline suite is run by this file.

DEFENDER OBSERVES: the broadcast global model, each submitted local model, stable
client IDs, trusted sample counts and step budgets, and two disjoint folds of
labelled TRAINING references (reserved by train_defended.load_data and removed
from every client's training). It never receives attacker IDs, poison masks,
the trigger, the attack target, validation/test outcomes or update labels.

PROTECTED DECISION: the defender scores only triggered gain toward the
protected target(s), by default class 0 (Benign): IDS evasion means attack
traffic classified Benign, whereas an attack->attack confusion still alarms.
This is a declared threat-model restriction, set independently of the attack
generator. Other targets are measured and logged but never scored.

OPERATIONAL SCORE for client i in round t (probability units unless stated):
  E[q,s,c] = local-minus-global change of mean[P(c|T_q x) - P(c|x)], x in source s
  B[q,c]   = k-th largest E[q,s,c] over sources s != c        (breadth, k=3 of 5)
  K[q,c]   = B[q,c] - median_q' B[q',c]                       (specificity)
  A        = cohort channel: z of K above max(leave-one-out peer median, the client's own
             known-clean profile = profile-quantile of its clean K), scaled by peer MAD;
             one probe per target chosen on fold 1, that cell confirmed on fold 2;
             A = confirmed value for the protected target (max if several)
  H        = history channel: one-sided excess of K over the client's own
             hard-gated reference (dimensionless), same discovery/confirmation
  S        = max((A-cA)/sA, (H-cH)/sH); H omitted when no admitted reference
Thresholds on S are pooled over clients and fitted per training phase on
known-clean FedAvg trajectories. Before any live training, a mandatory
leave-one-calibration-seed-out transfer check and a development attack replay
must pass.

INFLUENCE: a_i = p_i * exp(-(immediate + memory risk)) * clip_i with
0 <= a_i <= p_i; theta_new = theta_old + sum_i a_i delta_i; the unused mass stays
on theta_old (no renormalization). RETROACTIVE REVOCATION: when client i is
flagged in round t, theta_new -= (1 - trust_i) * sum_r a_ir * delta_ir over its
contributions from rounds t-W..t-1 still in force (W=3), oldest first, at most
p_i per round (bsc_4: revocation obeys the same per-round bound as aggregation);
never more than was applied. bsc_6: revocation acts only during a client's FIRST detection episode
(first flag plus immediately consecutive flags); afterwards later alerts attenuate the current update
but never revoke past (honest) contributions again. bsc_7: a round that breaks a validity bound
(flagged fraction or informative mass) is HELD - the global model is left unchanged, nothing is applied,
revoked or committed to client state - and the run stops only after max_consecutive_holds+1
consecutive invalid rounds. It targets the documented cold-start leak (seed 47: detection at the
3rd poisoned participation) and is a first-order approximation.

STATUS: research code for the paper. Not robust by guarantee. Thresholds are
empirical quantiles of dependent observations, not false-positive bounds.
Breadth assumes source-agnostic (all-to-one) backdoors; the defender probe bank
contains contiguous feature groups and +/-3/+/-6 replacement levels that overlap
the synthetic trigger family used in the experiments (e.g. columns 5-8, value 5;
a trigger outside the bank is tested separately). This is disclosed, not
trigger-agnostic.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import importlib
import inspect
import io
import json
import math
import platform
import sys
import tempfile
import textwrap
import time
import types
import zipfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

VERSION = "medishield_bsc_7"
RUNNER_REVISION = "held_round_1"
CAPTURE_SCHEMA = "medishield_round_capture_v1"
MEASUREMENT_SCHEMA = "medishield_bsc_measurement_v1"
SUPPORTED_RUNNER_VERSIONS = frozenset({"control_calibrated_candidate_4", "budget_matched_candidate_5"})
PROTOCOL_FUNCTIONS = ("StableMLP", "feature_hashes", "load_data", "make_partition", "train_client",
                      "apply_trigger", "poison", "evaluate", "choose_attackers", "detection_metrics")
# Keys that must agree between this run and any recorded trajectory it reuses.
COMPATIBILITY_KEYS = ("data_sha256", "protocol_functions", "partition_sha256", "reference_indices_sha256",
                      "classes", "feature_names", "training", "client_sizes", "prescribed_steps", "software")
MEASUREMENT_UNITS = ("find_classifier", "build_probe_bank", "ProbeProfiler", "load_round_capture")
CORE_KEYS = ("version", "detector_config", "measurement_fingerprint", "probe_count", "classes", "rounds", "client_profiles",
             "participants", "phases", "clip_bounds", "client_support")
LOCAL_ONLY_SUFFIXES = (".pt", ".tmp", ".npz")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def json_value(value):
    if isinstance(value, torch.Tensor):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{k: json_value(v) for k, v in row.items()} for row in rows])
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def bytes_hash(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def percentile(values, q):
    # Observed order statistic ("higher"); not an interpolated p-value claim.
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def median_abs_deviation(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.median(np.abs(array - np.median(array))))


def code_fingerprint(names):
    """AST digest of named top-level definitions; formatting-insensitive."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8-sig"))
    parts = {node.name: ast.dump(node, include_attributes=False) for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names}
    if set(parts) != set(names):
        raise ValueError("Cannot fingerprint measurement code: missing " + ", ".join(sorted(set(names) - set(parts))))
    return digest(parts)


class ShieldStop(RuntimeError):
    """A filtering round failed its validity bounds; nothing was committed."""

    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementConfig:
    """Defender-side probe bank. Changing any field invalidates cached measurements."""
    probe_levels: tuple = (-6.0, -3.0, 3.0, 6.0)
    probe_widths: tuple = (1, 2, 4)
    random_groups: int = 16
    probe_seed: int = 7261
    probe_chunk: int = 16

    def validate(self):
        levels = [float(v) for v in self.probe_levels]
        if not levels or len(set(levels)) != len(levels) or not all(math.isfinite(v) for v in levels):
            raise ValueError("Probe levels must be distinct and finite.")
        if not self.probe_widths or any(int(w) != w or w < 1 for w in self.probe_widths):
            raise ValueError("Probe widths must be positive integers.")
        if self.random_groups < 0 or self.probe_seed < 0 or self.probe_chunk < 1:
            raise ValueError("Invalid probe bank settings.")


@dataclass(frozen=True)
class DetectorConfig:
    """Everything that affects scores, thresholds, references, influence or stops."""
    protected_targets: tuple | None = (0,)    # attack -> Benign evasion; None = all classes (ablation)
    client_profile: bool = True               # excess over max(peer median, own known-clean profile); False = bsc_4
    profile_quantile: float = 0.90            # quantile of the client's clean K defining its profile
    breadth_sources: int = 3
    specificity: bool = True
    peer_scale_floor: float = 0.05
    effect_floor: float = 0.05
    history_channel: bool = True
    channel_scale_floor: float = 1.0
    min_cohort: int = 4
    min_phase_observations: int = 20
    min_observations_per_client_phase_seed: int = 3
    initial_rounds: int = 3
    calibration_quantile: float = 0.99
    calibration_margin: float = 0.25
    reference_quantile: float = 0.90
    alert_floor: float = 3.0
    reference_floor: float = 1.0
    penalty_width_floor: float = 1.0
    penalty_relative_width: float = 0.25
    clip_multiplier: float = 1.25
    history_policy: str = "hard"
    history_rate: float = 0.10
    admission_patience: int = 3
    reference_trust: float = 0.80
    stale_after_rounds: int = 6
    revocation_window: int = 3                # rounds of revocable contributions; 0 = off (bsc_2 behaviour)
    revocation_budget: str = "nominal_share"  # per-round revocation <= p_i; "unbounded" = bsc_3 ablation
    revocation_policy: str = "first_episode"  # revoke only in the first detection episode; "every_alert" = bsc_5
    risk_gain: float = 4.605170185988092      # ln(100): one penalty width -> trust 0.01
    memory_gain: float = 0.22314355131420976  # ln(1.25)
    risk_recovery: float = 0.50
    risk_cap: float = 0.22314355131420976     # memory alone discounts at most 20%
    severity_cap: float = 2.0
    stop_flagged_fraction: float = 0.50
    stop_min_mass: float = 0.25
    hold_policy: str = "hold"  # "hold" a round that breaks a validity bound; "stop" = bsc_6 behaviour
    max_consecutive_holds: int = 2  # a third consecutive invalid round stops the run
    norm_epsilon: float = 1e-10

    def validate(self, classes=None):
        numbers = [v for v in asdict(self).values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("Detector settings must be finite.")
        if self.breadth_sources < 1 or (classes is not None and self.breadth_sources > classes - 1):
            raise ValueError("breadth-sources must lie in [1, classes-1].")
        if self.protected_targets is not None:
            targets = list(self.protected_targets)
            if not targets or len(set(targets)) != len(targets) or any(int(c) != c or c < 0 for c in targets):
                raise ValueError("protected-targets must be distinct nonnegative class IDs, or 'all'.")
            if classes is not None and max(targets) >= classes:
                raise ValueError("protected-targets lie outside the class list.")
        if min(self.peer_scale_floor, self.effect_floor, self.channel_scale_floor, self.norm_epsilon,
               self.penalty_width_floor, self.penalty_relative_width, self.risk_gain, self.memory_gain,
               self.risk_cap, self.severity_cap) <= 0:
            raise ValueError("Scales, floors and risk settings must be positive.")
        if self.min_cohort < 3 or self.min_phase_observations < 1 or self.min_observations_per_client_phase_seed < 1:
            raise ValueError("Cohort/observation minimums are too small.")
        if self.initial_rounds < 1:
            raise ValueError("initial-rounds must be positive.")
        if not 0.5 <= self.calibration_quantile <= 1 or not 0.5 <= self.reference_quantile <= self.calibration_quantile:
            raise ValueError("Need 0.5 <= reference-quantile <= calibration-quantile <= 1.")
        if self.calibration_margin < 0 or self.reference_floor <= 0 or self.alert_floor < self.reference_floor:
            raise ValueError("Need nonnegative margin and 0 < reference-floor <= alert-floor.")
        if self.clip_multiplier < 1:
            raise ValueError("clip-multiplier must be >= 1.")
        if self.history_policy not in ("hard", "ungated", "proportional", "none"):
            raise ValueError("Invalid history policy.")
        if self.history_policy == "none" and self.history_channel:
            raise ValueError("history-policy none admits nothing; use --history-channel off for that ablation.")
        if not 0.5 <= self.profile_quantile <= 1.0:
            raise ValueError("profile-quantile must lie in [0.5, 1].")
        if self.hold_policy not in ("hold", "stop") or not isinstance(self.max_consecutive_holds, int) or self.max_consecutive_holds < 0:
            raise ValueError("hold-policy must be hold or stop; max-consecutive-holds a nonnegative integer.")
        if self.revocation_policy not in ("first_episode", "every_alert"):
            raise ValueError("revocation-policy must be first_episode or every_alert.")
        if self.revocation_budget not in ("nominal_share", "unbounded"):
            raise ValueError("revocation-budget must be nominal_share or unbounded.")
        if self.revocation_window < 0 or int(self.revocation_window) != self.revocation_window:
            raise ValueError("revocation-window must be a nonnegative integer.")
        if not 0 < self.history_rate <= 1 or self.admission_patience < 1 or self.stale_after_rounds < 1:
            raise ValueError("Invalid history rate, patience or staleness.")
        if not 0 < self.reference_trust <= 1 or not 0 < self.risk_recovery <= 1:
            raise ValueError("reference-trust and risk-recovery must lie in (0,1].")
        if not 0 < self.stop_flagged_fraction <= 1 or not 0 < self.stop_min_mass <= 1:
            raise ValueError("Invalid stop bounds.")


# ---------------------------------------------------------------------------
# Measurement: defender-side probe responses (the only model inference)
# ---------------------------------------------------------------------------

def find_classifier(model):
    layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not layers:
        raise ValueError("A final nn.Linear classifier is required.")
    name, layer = layers[-1]
    prefix = name + "." if name else ""
    return prefix + "weight", (prefix + "bias") if layer.bias is not None else None, layer.out_features


def build_probe_bank(features, config):
    """Contiguous groups of each width plus seeded random groups, at each level."""
    groups = set()
    for width in config.probe_widths:
        if width <= features:
            groups.update(tuple(range(start, start + width)) for start in range(features - width + 1))
    rng = np.random.default_rng(config.probe_seed)
    for _ in range(config.random_groups):
        width = min(max(config.probe_widths), features)
        groups.add(tuple(sorted(rng.choice(features, width, replace=False).tolist())))
    return [{"columns": list(group), "value": float(value)}
            for group in sorted(groups, key=lambda g: (len(g), g)) for value in config.probe_levels]


class ProbeProfiler:
    """Probability responses of a model to the declared probe bank.

    responses(state, fold)[q, s, c] = mean over trusted references x of true
    class s of P(c | T_q x) - P(c | x). measure() returns local minus global
    responses for every submission, rounded to float32 so that live decisions
    and cached-measurement replays use bit-identical inputs.
    """

    def __init__(self, model, reference_folds, config):
        config.validate()
        self.config = config
        self.template = copy.deepcopy(model).cpu().eval()
        _, _, self.classes = find_classifier(self.template)
        if len(reference_folds) != 2:
            raise ValueError("Exactly two disjoint trusted-training reference folds are required.")
        self.folds = []
        for X, y in reference_folds:
            X = X.detach().cpu().float().clone()
            y = y.detach().cpu().long().clone()
            if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not torch.isfinite(X).all():
                raise ValueError("Invalid trusted reference inputs.")
            if int(y.min()) < 0 or int(y.max()) >= self.classes:
                raise ValueError("Reference class IDs out of range.")
            if any(int((y == c).sum()) < 2 for c in range(self.classes)):
                raise ValueError("Each reference fold needs >= 2 examples of every class; unsupported classes are not imputed.")
            self.folds.append((X, y))
        if self.folds[0][0].shape[1] != self.folds[1][0].shape[1]:
            raise ValueError("Reference feature widths differ.")
        seen = []
        for X, _ in self.folds:
            canonical = X.numpy().copy()
            canonical[canonical == 0] = 0.0
            seen.append({row.tobytes() for row in canonical})
        if seen[0] & seen[1]:
            raise ValueError("Discovery and confirmation references overlap.")
        self.features = self.folds[0][0].shape[1]
        self.bank = build_probe_bank(self.features, config)
        if not self.bank:
            raise ValueError("Empty probe bank.")
        self.source_index = [[(y == s).nonzero(as_tuple=True)[0] for s in range(self.classes)] for _, y in self.folds]
        self.probed = []
        for X, _ in self.folds:
            batch = X.unsqueeze(0).repeat(len(self.bank), 1, 1)
            for q, spec in enumerate(self.bank):
                batch[q][:, spec["columns"]] = spec["value"]
            self.probed.append(batch)
        self.inference_calls = 0

    @torch.inference_mode()
    def responses(self, state, fold):
        self.template.load_state_dict(state, strict=True)
        self.template.eval()
        self.inference_calls += 1
        X, _ = self.folds[fold]
        Q, C = len(self.bank), self.classes
        logits = self.template(X)
        if logits.shape != (len(X), C) or not torch.isfinite(logits).all():
            raise ValueError("Invalid clean-reference logits.")
        clean = logits.double().softmax(1)
        out = torch.empty((Q, C, C), dtype=torch.float64)
        for start in range(0, Q, self.config.probe_chunk):
            block = self.probed[fold][start:start + self.config.probe_chunk]
            probed = self.template(block.reshape(-1, self.features))
            if not torch.isfinite(probed).all():
                raise ValueError("Nonfinite probe predictions.")
            changed = probed.double().softmax(1).reshape(len(block), len(X), C) - clean[None]
            for s in range(C):
                out[start:start + len(block), s, :] = changed[:, self.source_index[fold][s], :].mean(1)
        return out

    def measure(self, global_state, states):
        n, Q, C = len(states), len(self.bank), self.classes
        result = np.empty((n, 2, Q, C, C), dtype=np.float32)
        for fold in (0, 1):
            common = self.responses(global_state, fold)
            for i, state in enumerate(states):
                result[i, fold] = (self.responses(state, fold) - common).numpy().astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite measurement.")
        return result

    def fingerprint(self):
        return digest({"schema": MEASUREMENT_SCHEMA, "config": asdict(self.config), "bank": self.bank,
                       "references": [[bytes_hash(X.numpy().tobytes()), bytes_hash(y.numpy().tobytes())]
                                      for X, y in self.folds],
                       "model": [[k, list(v.shape), str(v.dtype)] for k, v in self.template.state_dict().items()],
                       "code": code_fingerprint(MEASUREMENT_UNITS)})


# ---------------------------------------------------------------------------
# Score statistic (pure functions; no model inference, no attack inputs)
# ---------------------------------------------------------------------------

def target_profiles(E, cfg):
    """E (n, 2 folds, Q, S, C) -> K (n, 2, Q, C) in probability units.

    Breadth: the k-th largest source effect over sources s != c, so a change
    that moves a single boundary-fragile source class cannot alone produce a
    large value. Specificity: subtract the client's median over all probes for
    that target, removing probe-agnostic shifts such as a class-prior change.
    """
    E = torch.as_tensor(np.asarray(E), dtype=torch.float64)
    if E.ndim != 5 or E.shape[1] != 2 or E.shape[3] != E.shape[4]:
        raise ValueError("Measurement must have shape (clients, 2, probes, classes, classes).")
    if not torch.isfinite(E).all():
        raise ValueError("Nonfinite measurement.")
    n, folds, Q, S, C = E.shape
    if not 1 <= cfg.breadth_sources <= C - 1:
        raise ValueError("breadth-sources exceeds the available source classes.")
    B = torch.empty((n, folds, Q, C), dtype=torch.float64)
    for c in range(C):
        sources = [s for s in range(S) if s != c]
        values = E[:, :, :, sources, c]
        B[..., c] = torch.topk(values, cfg.breadth_sources, dim=-1).values[..., -1]
    if cfg.specificity:
        B = B - torch.quantile(B, 0.5, dim=2, keepdim=True)
    return B


def leave_one_out_reference(K, i, cfg):
    """Peer median and robust scale for client i, excluding client i itself."""
    peers = torch.cat((K[:i], K[i + 1:]), 0)
    centre = torch.quantile(peers, 0.5, dim=0)
    scale = (1.4826 * torch.quantile((peers - centre).abs(), 0.5, dim=0)).clamp_min(cfg.peer_scale_floor)
    return centre, scale


def protected_target_list(cfg, classes):
    """Targets whose triggered gain is SCORED. None means every class (ablation)."""
    return list(range(classes)) if cfg.protected_targets is None else [int(c) for c in cfg.protected_targets]


def _discover_and_confirm(Z, K, protected):
    """Z, K: (2, Q, C). Select one probe per target on fold 0; confirm it on fold 1.

    Every target is measured and returned for audit; only `protected` targets
    enter the score. Unprotected targets are diagnostics, never decisions.
    """
    C = Z.shape[2]
    per_target, scores = [], []
    for c in range(C):
        q = int(torch.argmax(Z[0, :, c]))  # first maximum; no search on fold 1
        value = max(0.0, min(float(Z[0, q, c]), float(Z[1, q, c])))
        scores.append(value)
        per_target.append({"Target": c, "Probe": q, "Discovery_Z": float(Z[0, q, c]),
                           "Confirmation_Z": float(Z[1, q, c]), "Discovery_Effect": float(K[0, q, c]),
                           "Confirmation_Effect": float(K[1, q, c]), "Confirmed_Score": value,
                           "Scored": int(c in protected)})
    best = max(protected, key=lambda c: (scores[c], -c))
    return scores[best], best, per_target


def client_envelopes(observations, cfg):
    """Known-clean client profile: per client, the profile-quantile of its clean K over (fold, probe, target).

    observations: {client_id: [K_i tensors of shape (2, Q, C)]} from KNOWN-CLEAN trajectories only.
    """
    return {cid: torch.quantile(torch.stack(values), cfg.profile_quantile, dim=0)
            for cid, values in observations.items() if values}


def profile_observations(entries, cfg):
    """{client_id: [K_i per round]} from clean (trajectory, measurements) entries."""
    observed = {}
    for trajectory, measurements in entries:
        if trajectory.condition != "clean":
            raise ValueError("Client profiles may only come from clean trajectories.")
        for rnd in sorted(measurements):
            K = target_profiles(measurements[rnd]["E"], cfg)
            for i, cid in enumerate(int(v) for v in measurements[rnd]["client_ids"]):
                observed.setdefault(cid, []).append(K[i].clone())
    return observed


def cohort_channel(K, cfg, envelopes=None):
    """Cold-start capable channel: peer-relative, absolute-effect-gated z of K.

    With envelopes (one known-clean client profile per row, or None), the excess is
    measured above max(peer median, own profile): behaviour a client honestly shows
    in known-clean training is not counted as anomalous.
    """
    n = K.shape[0]
    if n < max(3, cfg.min_cohort):
        raise ValueError(f"Cohort of {n} is below the minimum {cfg.min_cohort}; no calibrated decision exists.")
    protected = protected_target_list(cfg, K.shape[3])
    scores, targets, selections = [], [], []
    for i in range(n):
        centre, scale = leave_one_out_reference(K, i, cfg)
        envelope = envelopes[i] if envelopes is not None else None
        base = centre if envelope is None else torch.maximum(centre, envelope)
        Z = torch.minimum((K[i] - base).clamp_min(0.0) / scale, K[i].clamp_min(0.0) / cfg.effect_floor)
        score, best, per_target = _discover_and_confirm(Z, K[i], protected)
        for entry in per_target:
            entry["Discovery_Profile"] = float(envelope[0, entry["Probe"], entry["Target"]]) if envelope is not None else None
            entry["Discovery_Peer_Median"] = float(centre[0, entry["Probe"], entry["Target"]])
            entry["Discovery_Peer_Scale"] = float(scale[0, entry["Probe"], entry["Target"]])
        scores.append(score)
        targets.append(best)
        selections.append(per_target)
    return {"scores": scores, "targets": targets, "selections": selections}


def history_channel(K_i, reference, cfg):
    """Own-reference channel: one-sided excess over the admitted reference.

    The reference holds mu and dev (mean absolute deviation) per fold/probe/
    target. Scale = max(effect_floor, 1.2533*dev); 1.2533 makes the mean
    absolute deviation consistent with a normal standard deviation.
    """
    scale = (1.2533 * reference["dev"]).clamp_min(cfg.effect_floor)
    Z = torch.minimum((K_i - reference["mu"]).clamp_min(0.0) / scale, K_i.clamp_min(0.0) / cfg.effect_floor)
    score, best, per_target = _discover_and_confirm(Z, K_i, protected_target_list(cfg, K_i.shape[2]))
    return {"score": score, "target": best, "selections": per_target}


def combined_score(A, H, profile):
    """Normalize each channel by its clean phase centre/scale; take the maximum."""
    nA = (A - profile["cohort_centre"]) / profile["cohort_scale"]
    nH = None
    if H is not None and profile.get("history_centre") is not None:
        nH = (H - profile["history_centre"]) / profile["history_scale"]
    if nH is not None and nH > nA:
        return nH, nA, nH, "history"
    return nA, nA, nH, "cohort"


def calibration_phase(round_number, initial_rounds):
    if round_number < 1:
        raise ValueError("Round numbers start at one.")
    return "initial" if round_number <= initial_rounds else "later"


# ---------------------------------------------------------------------------
# Influence and per-client state
# ---------------------------------------------------------------------------

def alert_strength(score, threshold, width, cap):
    """Continuous at the threshold; no unit jump in attenuation."""
    if not all(math.isfinite(v) for v in (score, threshold, width, cap)) or min(width, cap) <= 0:
        raise ValueError("Invalid score or calibrated penalty width.")
    return min(cap, max(0.0, score - threshold) / width)


def influence_transition(old_risk, flagged, severity, recovery_qualified, cfg):
    """Immediate attenuation plus a separately capped, recoverable memory.

    Absence is not an observation; uninformative or clipped submissions earn no
    recovery credit. Memory alone can discount a later passing update by at
    most 1 - exp(-risk_cap).
    """
    if not math.isfinite(old_risk) or not 0 <= old_risk <= cfg.risk_cap + 1e-15:
        raise ValueError("Invalid stored memory risk.")
    if not math.isfinite(severity) or severity < 0 or (not flagged and severity != 0):
        raise ValueError("Invalid alert severity.")
    if flagged:
        memory = min(cfg.risk_cap, old_risk + cfg.memory_gain * severity ** 2)
    elif recovery_qualified:
        memory = old_risk * (1.0 - cfg.risk_recovery)
    else:
        memory = old_risk
    if memory < cfg.norm_epsilon:
        memory = 0.0
    immediate = cfg.risk_gain * severity ** 2 if flagged else 0.0
    trust = math.exp(-(immediate + memory))
    if not math.isfinite(trust):
        raise ValueError("Nonfinite trust.")
    return memory, immediate, trust


def new_client_state():
    return {"risk": 0.0, "streak": 0, "buffer": [], "buffer_rounds": [], "mu": None, "dev": None,
            "sources": [], "last_admitted_round": None, "initialized_round": None,
            "last_seen": None, "participations": 0, "contributions": [], "revocation_status": "armed"}


def revoke_recent(state, fraction, rnd, cfg, budget=math.inf):
    """Retroactive revocation of a flagged client's still-revocable contributions.

    Each buffered entry holds the coefficient still in force and the delta it was
    applied with. A flag revokes `fraction` (= 1 - current trust) of what remains,
    oldest entry first, so total revocation can never exceed what was applied.
    `budget` caps the mass revoked in this round (nominal_share policy: the client's
    current nominal share p_i), so revocation obeys the same per-round bound as
    aggregation; the remainder stays revocable while inside the window. Entries older
    than `revocation_window` rounds are final. First-order approximation: later
    rounds were trained on top of the revoked contribution.
    """
    state["contributions"] = [e for e in state["contributions"] if e["round"] >= rnd - cfg.revocation_window]
    revoked = []
    if fraction <= 0:
        return revoked
    if not 0 < fraction <= 1:
        raise ValueError("Revocation fraction must lie in (0,1].")
    if budget < 0:
        raise ValueError("Revocation budget must be nonnegative.")
    for entry in sorted(state["contributions"], key=lambda e: e["round"]):
        mass = min(fraction * entry["coefficient"], budget)
        budget -= mass
        if mass > 0:
            entry["coefficient"] -= mass
            revoked.append((entry["round"], mass, entry["delta"]))
    return revoked


def reference_status(state):
    if state["mu"] is not None:
        return "active"
    return "probation" if state["buffer"] else "none"


def stale_reset(state, rnd, cfg):
    """Rounds (not participations) since the last admitted update. Global drift
    continues during absence, so a long-unrefreshed reference is discarded and
    must be re-earned through probation."""
    if state["mu"] is not None and rnd - state["last_admitted_round"] > cfg.stale_after_rounds:
        state.update(mu=None, dev=None, sources=[], streak=0, buffer=[], buffer_rounds=[], initialized_round=None)
        return True
    return False


def advance_reference(state, profile, admit, rate, rnd, cfg):
    """Single path by which observations enter a reference. Returns (event, rounds admitted).

    Not admitted: streak reset, probation buffer cleared, active reference frozen.
    Admitted: probation buffer until `admission_patience` consecutive qualifying
    participations, then coordinate-median initialization; afterwards slow EMA.
    Under the hard policy, updates resume only after a renewed qualifying streak.
    """
    if not admit or rate <= 0:
        had_buffer = bool(state["buffer"])
        state["streak"] = 0
        state["buffer"], state["buffer_rounds"] = [], []
        if state["mu"] is not None:
            return "frozen", []
        return ("probation_reset" if had_buffer else "not_admitted"), []
    if not 0 < rate <= 1:
        raise ValueError("Reference update rate must lie in (0,1].")
    state["streak"] += 1
    L = cfg.admission_patience
    if state["mu"] is None:
        state["buffer"] = (state["buffer"] + [profile.detach().clone()])[-L:]
        state["buffer_rounds"] = (state["buffer_rounds"] + [int(rnd)])[-L:]
        if state["streak"] < L:
            return "probation", []
        stack = torch.stack(state["buffer"])
        mu = torch.quantile(stack, 0.5, dim=0)
        dev = (stack - mu).abs().mean(0)
        admitted = list(state["buffer_rounds"])
        state.update(mu=mu, dev=dev, sources=admitted, last_admitted_round=int(rnd),
                     initialized_round=int(rnd), buffer=[], buffer_rounds=[])
        return "initialized", admitted
    if cfg.history_policy == "hard" and state["streak"] < L:
        return "requalifying", []
    difference = profile - state["mu"]
    state["mu"] = state["mu"] + rate * difference
    state["dev"] = state["dev"] + rate * (difference.abs() - state["dev"])
    state["sources"] = state["sources"] + [int(rnd)]
    state["last_admitted_round"] = int(rnd)
    return "updated", [int(rnd)]


def submission_deltas(global_state, states, ids, sizes, steps):
    """Validate submissions and compute delta = state - global (float64)."""
    if not states or not (len(states) == len(ids) == len(sizes) == len(steps)):
        raise ValueError("Each submission requires an ID, sample count and step budget.")
    if len(set(ids)) != len(ids) or any(int(v) != v or v < 0 for v in ids):
        raise ValueError("Invalid or repeated stable client IDs.")
    if any(int(v) != v or v <= 0 for v in list(sizes) + list(steps)):
        raise ValueError("Counts and step budgets must be positive integers.")
    if not all(torch.isfinite(v).all() for v in global_state.values() if v.is_floating_point()):
        raise ValueError("Nonfinite previous global model.")
    updates, norms = [], []
    for state in states:
        if set(state) != set(global_state):
            raise ValueError("Client parameter keys disagree with the global model.")
        delta = {}
        for key, old in global_state.items():
            new = state[key]
            if new.shape != old.shape or new.dtype != old.dtype:
                raise ValueError("Invalid client parameter shape/dtype: " + key)
            if not old.is_floating_point():
                if not torch.equal(old, new):
                    raise ValueError("Changed nonfloating model buffer: " + key)
                continue
            if not torch.isfinite(new).all():
                raise ValueError("Nonfinite client parameter: " + key)
            delta[key] = new.double() - old.double()
        length = math.sqrt(sum(float(v.square().sum()) for v in delta.values()))
        if not math.isfinite(length):
            raise ValueError("Nonfinite update norm.")
        updates.append(delta)
        norms.append(length)
    return updates, norms


# ---------------------------------------------------------------------------
# Calibration core and detector
# ---------------------------------------------------------------------------

def validate_core(core, cfg, measurement_fingerprint):
    if core.get("version") != VERSION:
        raise ValueError("Calibration belongs to a different candidate version.")
    if digest(core.get("detector_config")) != digest(asdict(cfg)):
        raise ValueError("Detector configuration differs from calibration.")
    if core.get("measurement_fingerprint") != measurement_fingerprint:
        raise ValueError("Measurement procedure/probe bank/references differ from calibration.")
    if int(core.get("participants", 0)) < cfg.min_cohort:
        raise ValueError("Calibrated cohort is below min-cohort.")
    if set(core.get("phases", {})) != {"initial", "later"}:
        raise ValueError("Calibration needs initial and later phases.")
    for phase, part in core["phases"].items():
        values = [part.get(k) for k in ("cohort_centre", "cohort_scale", "alert_threshold",
                                        "reference_threshold", "penalty_width")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Invalid calibration values in phase {phase}.")
        if min(part["cohort_scale"], part["penalty_width"]) <= 0 or part["reference_threshold"] > part["alert_threshold"]:
            raise ValueError(f"Invalid calibration scales/thresholds in phase {phase}.")
        if (part.get("history_centre") is None) != (part.get("history_scale") is None):
            raise ValueError(f"Incomplete history normalizer in phase {phase}.")
        if part.get("history_scale") is not None and not (math.isfinite(part["history_scale"]) and part["history_scale"] > 0):
            raise ValueError(f"Invalid history scale in phase {phase}.")
    support = core.get("client_support", {})
    profiles = core.get("client_profiles")
    if cfg.client_profile:
        if not isinstance(profiles, dict) or set(profiles) != set(support):
            raise ValueError("client-profile is on: every supported client needs a known-clean profile.")
        for cid, values in profiles.items():
            array = np.asarray(values, dtype=np.float64)
            if array.shape != (2, core["probe_count"], core["classes"]) or not np.isfinite(array).all():
                raise ValueError(f"Invalid known-clean profile for client {cid}.")
    elif profiles is not None:
        raise ValueError("client-profile is off but the calibration carries client profiles.")
    if not support or set(core.get("clip_bounds", {})) != set(support):
        raise ValueError("Every supported client needs clipping bounds.")
    for cid, bounds in core["clip_bounds"].items():
        if set(bounds) != {"initial", "later"} or not all(math.isfinite(b) and b > 0 for b in bounds.values()):
            raise ValueError(f"Invalid clipping bounds for client {cid}.")


def core_digest(core):
    return digest({k: core[k] for k in CORE_KEYS})


class CandidateDetector:
    """Server-side detector. Its inputs exclude every attack/evaluation label."""

    def __init__(self, core, config, mode, measurement_fingerprint):
        config.validate(core.get("classes"))
        if mode not in ("filter", "shadow"):
            raise ValueError("Mode must be filter or shadow.")
        validate_core(core, config, measurement_fingerprint)
        self.core = {k: copy.deepcopy(core[k]) for k in CORE_KEYS}
        self.config, self.mode, self.fingerprint = config, mode, measurement_fingerprint
        self.profiles = ({cid: torch.tensor(v, dtype=torch.float64) for cid, v in core["client_profiles"].items()}
                         if config.client_profile else None)
        self.round, self.clients, self.consecutive_holds = 0, {}, 0

    def export_state(self):
        return copy.deepcopy({"version": VERSION, "round": self.round, "clients": self.clients,
                              "consecutive_holds": self.consecutive_holds,
                              "core_sha256": core_digest(self.core), "config_sha256": digest(asdict(self.config)),
                              "mode": self.mode})

    def import_state(self, value):
        if (value.get("version") != VERSION or value.get("core_sha256") != core_digest(self.core)
                or value.get("config_sha256") != digest(asdict(self.config)) or value.get("mode") != self.mode):
            raise ValueError("Detector checkpoint does not match this calibration/configuration/mode.")
        self.round, self.clients = int(value["round"]), copy.deepcopy(value["clients"])
        self.consecutive_holds = int(value.get("consecutive_holds", 0))

    def check_support(self, ids, sizes, steps):
        if len(ids) != self.core["participants"]:
            raise ValueError(f"{len(ids)} participants; calibration supports exactly {self.core['participants']}.")
        for cid, size, budget in zip(ids, sizes, steps):
            profile = self.core["client_support"].get(str(cid))
            if profile is None or int(size) != profile["client_size"] or int(budget) != profile["optimizer_steps"]:
                raise ValueError(f"Client {cid} has no matching clean calibration for its count/budget.")

    def _check_measurement(self, measurement, ids, sizes, steps, norms):
        if measurement.get("fingerprint") != self.fingerprint:
            raise ValueError("Measurement fingerprint differs from calibration.")
        if ([int(v) for v in measurement["client_ids"]] != list(ids)
                or [int(v) for v in measurement["client_sizes"]] != [int(v) for v in sizes]
                or [int(v) for v in measurement["optimizer_steps"]] != [int(v) for v in steps]):
            raise ValueError("Measurement client identities/counts/budgets differ from the submissions.")
        if not all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(measurement["norms"], norms)):
            raise ValueError("Measurement was taken on different submissions (norm mismatch).")
        E = np.asarray(measurement["E"])
        C = self.core["classes"]
        if E.shape != (len(ids), 2, self.core["probe_count"], C, C) or not np.isfinite(E).all():
            raise ValueError("Measurement shape/finite check failed.")
        return E

    def combine(self, global_state, states, ids, sizes, steps, measurement):
        """Decide coefficients for this exact round; commit state only on success."""
        cfg, core = self.config, self.core
        rnd = self.round + 1
        if rnd > core["rounds"]:
            raise ValueError("Round is outside the calibrated horizon.")
        ids = [int(c) for c in ids]
        updates, norms = submission_deltas(global_state, states, ids, sizes, steps)
        self.check_support(ids, sizes, steps)
        E = self._check_measurement(measurement, ids, sizes, steps, norms)
        K = target_profiles(E, cfg)
        envelopes = [self.profiles[str(cid)] for cid in ids] if self.profiles is not None else None
        cohort = cohort_channel(K, cfg, envelopes)
        phase = calibration_phase(rnd, cfg.initial_rounds)
        profile = core["phases"][phase]
        T, T_ref, width = profile["alert_threshold"], profile["reference_threshold"], profile["penalty_width"]
        clients = copy.deepcopy(self.clients)
        total = float(sum(sizes))
        nominal = [float(s) / total for s in sizes]
        records, history_details, revocations = [], [], []
        for i, cid in enumerate(ids):
            state = clients.setdefault(str(cid), new_client_state())
            status_before = reference_status(state)
            was_stale = stale_reset(state, rnd, cfg)
            A = cohort["scores"][i]
            H, history = None, None
            if not cfg.history_channel:
                history_note = "history_channel_off"
            elif state["mu"] is None:
                history_note = "no_admitted_reference"
            elif profile.get("history_centre") is None:
                history_note = "no_clean_calibration_for_phase"
            else:
                history = history_channel(K[i], state, cfg)
                H, history_note = history["score"], "available"
            S, nA, nH, channel = combined_score(A, H, profile)
            flagged = S > T
            severity = alert_strength(S, T, width, cfg.severity_cap) if flagged else 0.0
            informative = norms[i] > cfg.norm_epsilon
            bound = core["clip_bounds"][str(cid)][phase]
            clip = min(1.0, bound / norms[i]) if informative else 1.0
            recovery_qualified = informative and not flagged and clip == 1.0
            risk_before = state["risk"]
            state["risk"], immediate, trust = influence_transition(risk_before, flagged, severity,
                                                                    recovery_qualified, cfg)
            reference_qualified = (informative and not flagged and S <= T_ref and clip == 1.0
                                   and trust >= cfg.reference_trust)
            if cfg.history_policy == "hard":
                admit, rate = reference_qualified, cfg.history_rate
            elif cfg.history_policy == "ungated":
                admit, rate = informative, cfg.history_rate
            elif cfg.history_policy == "proportional":
                admit, rate = informative, cfg.history_rate * trust
            else:
                admit, rate = False, 0.0
            event, admitted_rounds = advance_reference(state, K[i], admit, rate, rnd, cfg)
            state["last_seen"] = rnd
            state["participations"] += 1
            proposed = nominal[i] * trust * clip
            applied = proposed if self.mode == "filter" else nominal[i]
            revoked = []
            revocation_status_before = state["revocation_status"]
            if cfg.revocation_window:
                budget = nominal[i] if cfg.revocation_budget == "nominal_share" else math.inf
                if cfg.revocation_policy == "every_alert":
                    revoked = revoke_recent(state, (1.0 - trust) if flagged else 0.0, rnd, cfg, budget)
                    store = proposed > 0
                elif state["revocation_status"] == "spent":
                    store = False  # first episode over: nothing is revoked again, nothing kept
                elif flagged:  # first flag, or a consecutive flag inside the first episode
                    state["revocation_status"] = "active"
                    revoked = revoke_recent(state, 1.0 - trust, rnd, cfg, budget)
                    store = False
                elif state["revocation_status"] == "active":  # first unflagged participation ends the episode
                    state["revocation_status"] = "spent"
                    state["contributions"] = []
                    store = False
                else:  # armed: never flagged, keep the revocable window
                    revoke_recent(state, 0.0, rnd, cfg, budget)  # prunes entries older than the window
                    store = proposed > 0
                if store:
                    state["contributions"].append({"round": rnd, "coefficient": proposed,
                                                   "delta": {k: v.clone() for k, v in updates[i].items()}})
            revocations.append(revoked)
            selection = cohort["selections"][i][cohort["targets"][i]]
            records.append({
                "Round": rnd, "Client": cid, "Phase": phase, "Informative": int(informative),
                "Optimizer_Steps": int(steps[i]), "Nominal_Weight": nominal[i],
                "Cohort_Score": A, "Cohort_Target": cohort["targets"][i], "Cohort_Probe": selection["Probe"],
                "Cohort_Discovery_Z": selection["Discovery_Z"], "Cohort_Confirmation_Z": selection["Confirmation_Z"],
                "Cohort_Discovery_Effect": selection["Discovery_Effect"],
                "Cohort_Confirmation_Effect": selection["Confirmation_Effect"],
                "History_Status": history_note, "History_Score": H,
                "History_Target": history["target"] if history else None,
                "History_Probe": history["selections"][history["target"]]["Probe"] if history else None,
                "Normalized_Cohort": nA, "Normalized_History": nH, "Score": S, "Score_Channel": channel,
                "Alert_Threshold": T, "Reference_Threshold": T_ref, "Penalty_Width": width,
                "Flagged": int(flagged), "Severity": severity, "Risk_Before": risk_before,
                "Risk_After": state["risk"], "Immediate_Risk": immediate, "Trust": trust,
                "Memory_Recovery": int(state["risk"] < risk_before),
                "Full_Update_Norm": norms[i], "Clip_Bound": bound, "Clip_Factor": clip,
                "Recovery_Qualified": int(recovery_qualified), "Reference_Qualified": int(reference_qualified),
                "Reference_Status_Before": status_before, "Reference_Stale_Reset": int(was_stale),
                "Reference_Event": event, "Reference_Streak": state["streak"],
                "Reference_Admitted_Rounds": json.dumps(admitted_rounds),
                "Reference_Sources": json.dumps(state["sources"]),
                "Proposed_Coefficient": proposed, "Applied_Coefficient": applied,
                "Revocation_Fraction": (1.0 - trust) if revoked else 0.0,
                "Revocation_Status_Before": revocation_status_before, "Revocation_Status_After": state["revocation_status"],
                "Revoked_Mass": sum(mass for _, mass, _ in revoked),
                "Revoked_Detail": json.dumps([[r, mass] for r, mass, _ in revoked]),
                "Nominal_Loss_Immediate": nominal[i] * (1.0 - math.exp(-immediate)),
                "Nominal_Loss_Memory": nominal[i] * math.exp(-immediate) * (1.0 - math.exp(-state["risk"])),
                "Nominal_Loss_Clipping": nominal[i] * trust * (1.0 - clip),
                "Reason": ("alert_" + channel) if flagged else ("within_screen" if informative else "uninformative_no_credit")})
            history_details.append({"Client": cid, "Cohort": cohort["selections"][i],
                                    "History": history["selections"] if history else None})
        flagged_fraction = sum(r["Flagged"] for r in records) / len(records)
        proposed_mass = sum(r["Proposed_Coefficient"] for r in records)
        informative_mass = sum(r["Proposed_Coefficient"] for r in records if r["Informative"])
        reasons = []
        if flagged_fraction >= cfg.stop_flagged_fraction:
            reasons.append(f"flagged fraction {flagged_fraction:.3f} >= {cfg.stop_flagged_fraction}")
        if informative_mass < cfg.stop_min_mass:
            reasons.append(f"informative proposed mass {informative_mass:.4f} < {cfg.stop_min_mass}")
        coefficients = [r["Applied_Coefficient"] for r in records]
        diagnostic = {"round": rnd, "mode": self.mode, "phase": phase, "clients": records,
                      "selections": history_details, "flagged_fraction": flagged_fraction,
                      "proposed_mass": proposed_mass, "informative_proposed_mass": informative_mass,
                      "applied_mass": sum(coefficients), "server_residual": 1.0 - sum(coefficients),
                      "revoked_mass": sum(r["Revoked_Mass"] for r in records),
                      "revocation_applied": self.mode == "filter" and any(revocations),
                      "validity_reasons": reasons, "history_used_in_score": cfg.history_channel}
        diagnostic.update(round_held=False, consecutive_holds=0)
        if self.mode == "filter" and reasons:
            holds = self.consecutive_holds + 1
            if cfg.hold_policy == "stop" or holds > cfg.max_consecutive_holds:
                for row in records:
                    row["Applied_Coefficient"] = None
                diagnostic.update(applied_mass=None, server_residual=None, global_update_applied=False,
                                  detector_state_committed=False, consecutive_holds=holds)
                raise ShieldStop("; ".join(reasons) + (f"; {holds} consecutive invalid rounds" if cfg.hold_policy == "hold" else ""),
                                 diagnostic)
            # Held round: global unchanged; nothing applied, revoked or committed except the round counter.
            for row in records:
                row.update(Applied_Coefficient=0.0, Revoked_Mass=0.0, Revoked_Detail="[]", Revocation_Fraction=0.0,
                           Revocation_Status_After=row["Revocation_Status_Before"], Reason="round_held")
            diagnostic.update(applied_mass=0.0, server_residual=1.0, revoked_mass=0.0, revocation_applied=False,
                              global_update_applied=False, detector_state_committed=False, round_held=True,
                              consecutive_holds=holds)
            self.round, self.consecutive_holds = rnd, holds
            return {k: v.clone() for k, v in global_state.items()}, diagnostic
        if any(not 0.0 <= a <= p for a, p in zip(coefficients, nominal)) or sum(coefficients) > 1.0 + 1e-12:
            raise AssertionError("Nominal influence cap violated.")
        output = {}
        for key, old in global_state.items():
            if not old.is_floating_point():
                output[key] = old.clone()
                continue
            change = torch.zeros_like(old, dtype=torch.float64)
            for a, delta in zip(coefficients, updates):
                change = change + a * delta[key]
            if self.mode == "filter":  # shadow mode only records the proposed revocation
                for revoked in revocations:
                    for _, mass, delta in revoked:
                        change = change - mass * delta[key]
            output[key] = (old.double() + change).to(old.dtype)
            if not torch.isfinite(output[key]).all():
                raise ValueError("Nonfinite proposed global model; nothing committed.")
        self.clients, self.round, self.consecutive_holds = clients, rnd, 0
        diagnostic.update(global_update_applied=True, detector_state_committed=True)
        return output, diagnostic


# ---------------------------------------------------------------------------
# Calibration fitting on known-clean trajectories
# ---------------------------------------------------------------------------

def clean_channel_rows(entries, cfg, envelope_for=None):
    """Cohort and history channel values on known-clean trajectories.

    The history channel is simulated with the same reference functions as
    deployment, using the KNOWN-CLEAN designation (every informative
    observation is admitted). Deployment admits only qualifying observations;
    that difference is recorded in the calibration file. envelope_for(seed) returns the
    client profiles to apply to that seed's rows; it must be CROSS-FITTED (built without
    that seed) so thresholds are fitted on out-of-sample profile excesses.
    """
    rows = []
    for trajectory, measurements in entries:
        if trajectory.condition != "clean":
            raise ValueError("Calibration rows must come from clean trajectories.")
        states = {}
        profiles = envelope_for(trajectory.seed) if envelope_for is not None else None
        for rnd in sorted(measurements):
            record = measurements[rnd]
            K = target_profiles(record["E"], cfg)
            ids = [int(v) for v in record["client_ids"]]
            cohort = cohort_channel(K, cfg, [profiles.get(cid) for cid in ids] if profiles is not None else None)
            for i, cid in enumerate(int(v) for v in record["client_ids"]):
                if trajectory.label(rnd, cid)[0]:
                    raise ValueError("A trajectory labelled clean contains a poisoned submission.")
                state = states.setdefault(cid, new_client_state())
                stale_reset(state, rnd, cfg)
                H = history_channel(K[i], state, cfg)["score"] if cfg.history_channel and state["mu"] is not None else None
                informative = record["norms"][i] > cfg.norm_epsilon
                advance_reference(state, K[i], informative and cfg.history_policy != "none", cfg.history_rate, rnd, cfg)
                rows.append({"Seed": trajectory.seed, "Round": rnd, "Client": cid, "Condition": "clean",
                             "Phase": calibration_phase(rnd, cfg.initial_rounds), "Cohort_Score": cohort["scores"][i],
                             "Cohort_Target": cohort["targets"][i], "History_Score": H,
                             "Full_Update_Norm": record["norms"][i], "Client_Size": int(record["client_sizes"][i]),
                             "Optimizer_Steps": int(record["optimizer_steps"][i]), "Informative": int(informative)})
    return rows


def fit_core(rows, cfg, measurement_fingerprint, training, seeds, rounds, participants, probe_count, classes, min_seeds,
             client_profiles=None):
    """Pooled per-phase thresholds on the deployed statistic S; per-client clip bounds."""
    seeds = sorted(set(int(s) for s in seeds))
    if len(seeds) < min_seeds:
        raise ValueError(f"Need at least {min_seeds} distinct clean calibration seeds.")
    identities = [(int(r["Seed"]), int(r["Round"]), int(r["Client"])) for r in rows]
    if not rows or len(set(identities)) != len(identities):
        raise ValueError("Calibration observations must be nonempty and distinct.")
    if any(r["Condition"] != "clean" for r in rows):
        raise ValueError("Only explicitly clean observations may calibrate thresholds.")
    if {s for s, _, _ in identities} != set(seeds) or any(not 1 <= r <= rounds for _, r, _ in identities):
        raise ValueError("Calibration rows lie outside the requested seeds/horizon.")
    sizes, budgets = training["client_sizes"], training["prescribed_steps"]
    expected_clients = set(range(len(sizes)))
    if {c for _, _, c in identities} != expected_clients:
        raise ValueError("Calibration does not cover every stable client.")
    for r in rows:
        if r["Client_Size"] != sizes[int(r["Client"])] or r["Optimizer_Steps"] != budgets[int(r["Client"])]:
            raise ValueError(f"Client {r['Client']}: calibration count/budget differs from the training contract.")
        values = [r["Cohort_Score"], r["Full_Update_Norm"]] + ([r["History_Score"]] if r["History_Score"] is not None else [])
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid calibration score or norm.")
    if cfg.client_profile and (client_profiles is None or set(client_profiles) != expected_clients):
        raise ValueError("client-profile is on: a known-clean profile is required for every stable client.")
    phases, clip_bounds = {}, {str(c): {} for c in sorted(expected_clients)}
    minimum = cfg.min_observations_per_client_phase_seed
    for phase in ("initial", "later"):
        bucket = [r for r in rows if r["Phase"] == phase]
        for cid in sorted(expected_clients):
            mine = [r for r in bucket if int(r["Client"]) == cid]
            counts = {s: sum(int(r["Seed"]) == s for r in mine) for s in seeds}
            if min(counts.values()) < minimum:
                raise ValueError(f"Client {cid}/{phase}: need >= {minimum} clean observations from EACH seed; got {counts}.")
            clip_bounds[str(cid)][phase] = max(cfg.clip_multiplier * max(r["Full_Update_Norm"] for r in mine),
                                               cfg.norm_epsilon)
        if len(bucket) < cfg.min_phase_observations:
            raise ValueError(f"Phase {phase}: {len(bucket)} pooled observations < {cfg.min_phase_observations}.")
        A = [r["Cohort_Score"] for r in bucket]
        part = {"cohort_centre": float(np.median(A)),
                "cohort_scale": max(cfg.channel_scale_floor, 1.4826 * median_abs_deviation(A)),
                "history_centre": None, "history_scale": None, "observations": len(bucket)}
        H = [r["History_Score"] for r in bucket if r["History_Score"] is not None]
        part["history_observations"] = len(H)
        if cfg.history_channel and len(H) >= cfg.min_phase_observations:
            part["history_centre"] = float(np.median(H))
            part["history_scale"] = max(cfg.channel_scale_floor, 1.4826 * median_abs_deviation(H))
        S = [combined_score(r["Cohort_Score"], r["History_Score"], part)[0] for r in bucket]
        T = max(cfg.alert_floor, percentile(S, cfg.calibration_quantile) + cfg.calibration_margin)
        part["alert_threshold"] = T
        part["reference_threshold"] = min(T, max(cfg.reference_floor, percentile(S, cfg.reference_quantile) + cfg.calibration_margin))
        part["penalty_width"] = max(cfg.penalty_width_floor, cfg.penalty_relative_width * T, 1.4826 * median_abs_deviation(S))
        part["score_median"], part["score_max"] = float(np.median(S)), float(max(S))
        part["in_sample_alerts"] = int(sum(v > T for v in S))
        part["in_sample_alerts_by_client"] = {str(c): int(sum(v > T for v, r in zip(S, bucket) if int(r["Client"]) == c))
                                              for c in sorted(expected_clients)}
        phases[phase] = part
    client_support = {str(c): {"client_size": int(sizes[c]), "optimizer_steps": int(budgets[c])}
                      for c in sorted(expected_clients)}
    core = {"version": VERSION, "detector_config": asdict(cfg), "measurement_fingerprint": measurement_fingerprint,
            "probe_count": int(probe_count), "classes": int(classes), "rounds": int(rounds),
            "participants": int(participants), "phases": phases, "clip_bounds": clip_bounds,
            "client_profiles": ({str(c): client_profiles[c].tolist() for c in sorted(expected_clients)}
                                if cfg.client_profile else None),
            "client_support": client_support, "fit_seeds": seeds, "fit_observations": len(rows),
            "fit_rows_sha256": digest(rows)}
    validate_core(core, cfg, measurement_fingerprint)
    return core


# ---------------------------------------------------------------------------
# Recorded trajectories (captures) and measurement cache
# ---------------------------------------------------------------------------

def write_capture(path, rnd, global_state, states, ids, sizes, steps):
    """Pickle-free immutable pre-aggregation states (existing schema)."""
    keys = list(global_state)
    payload = {"schema": np.asarray(CAPTURE_SCHEMA), "round": np.asarray(rnd),
               "parameter_keys": np.asarray(keys), "client_ids": np.asarray(ids, dtype=np.int64),
               "client_sizes": np.asarray(sizes, dtype=np.int64), "optimizer_steps": np.asarray(steps, dtype=np.int64)}
    for prefix, state in [("global", global_state)] + [(f"client_{cid}", s) for cid, s in zip(ids, states)]:
        for j, key in enumerate(keys):
            payload[f"{prefix}_{j:03d}"] = state[key].detach().cpu().numpy()
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temp.replace(path)


def load_round_capture(source, template_state):
    """Recorded previous global and submitted states for one round.

    Deltas must be computed against THIS recorded global. Filtering them offline
    is a one-round decision replay, never a new training trajectory.
    """
    with np.load(source, allow_pickle=False) as archive:
        if archive["schema"].item() != CAPTURE_SCHEMA:
            raise ValueError("Unknown capture schema.")
        keys, ids = archive["parameter_keys"].tolist(), [int(v) for v in archive["client_ids"].tolist()]
        if keys != list(template_state) or len(set(ids)) != len(ids):
            raise ValueError("Capture/model keys or stable client identifiers do not match.")

        def reconstruct(prefix):
            output = {}
            for j, key in enumerate(keys):
                value = torch.from_numpy(archive[f"{prefix}_{j:03d}"].copy())
                expected = template_state[key]
                if value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
                    raise ValueError("Invalid saved parameter: " + key)
                output[key] = value
            return output

        global_state = reconstruct("global")
        states = [reconstruct(f"client_{cid}") for cid in ids]
        sizes = [int(v) for v in archive["client_sizes"].tolist()]
        steps = [int(v) for v in archive["optimizer_steps"].tolist()]
        rnd = int(archive["round"])
    if len(sizes) != len(ids) or len(steps) != len(ids) or min(sizes + steps) <= 0:
        raise ValueError("Invalid saved client counts or optimizer budgets.")
    return global_state, states, ids, sizes, steps, rnd


def save_measurement(path, record):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, schema=np.asarray(MEASUREMENT_SCHEMA), fingerprint=np.asarray(record["fingerprint"]),
                            capture_sha256=np.asarray(record["capture_sha256"] or ""), round=np.asarray(record["round"]),
                            client_ids=np.asarray(record["client_ids"], dtype=np.int64),
                            client_sizes=np.asarray(record["client_sizes"], dtype=np.int64),
                            optimizer_steps=np.asarray(record["optimizer_steps"], dtype=np.int64),
                            norms=np.asarray(record["norms"], dtype=np.float64), E=record["E"].astype(np.float32))
    temp.replace(path)


def load_measurement(path):
    with np.load(path, allow_pickle=False) as archive:
        if archive["schema"].item() != MEASUREMENT_SCHEMA:
            raise ValueError("Unknown measurement schema: " + str(path))
        return {"fingerprint": str(archive["fingerprint"].item()), "capture_sha256": str(archive["capture_sha256"].item()) or None,
                "round": int(archive["round"]), "client_ids": [int(v) for v in archive["client_ids"].tolist()],
                "client_sizes": [int(v) for v in archive["client_sizes"].tolist()],
                "optimizer_steps": [int(v) for v in archive["optimizer_steps"].tolist()],
                "norms": [float(v) for v in archive["norms"].tolist()], "E": archive["E"].copy()}


class Container:
    """Read files from a results ZIP or a directory with the same relative names."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Capture source not found: {self.path}")
        self.is_zip = self.path.is_file()
        if self.is_zip and not zipfile.is_zipfile(self.path):
            raise ValueError(f"{self.path} is neither a directory nor a ZIP archive.")

    def names(self):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return archive.namelist()
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def read(self, name):
        if self.is_zip:
            with zipfile.ZipFile(self.path) as archive:
                return archive.read(name)
        return (self.path / name).read_bytes()


class Trajectory:
    """One recorded FedAvg trajectory. Labels are evaluator-only."""

    def __init__(self, container, prefix, kind, seed, condition, manifest_raw, rounds, labels):
        self.container, self.prefix, self.kind = container, prefix, kind
        self.seed, self.condition, self.rounds = int(seed), condition, int(rounds)
        self._labels = labels
        hashes = [bytes_hash(self.read_round(r)) for r in range(1, rounds + 1)]
        self.sha256 = digest({"manifest": bytes_hash(manifest_raw), "rounds": hashes})
        self.key = f"{condition}_seed{self.seed}_{self.sha256[:12]}"

    def read_round(self, rnd):
        return self.container.read(f"{self.prefix}captures/round_{rnd:04d}.npz")

    def label(self, rnd, cid):
        """(is_poisoned_update, is_compromised, poisoned_examples). Clean controls are clean by construction."""
        if self._labels is None:
            return 0, 0, 0
        if (rnd, cid) not in self._labels:
            raise ValueError(f"Missing evaluator label for round {rnd}, client {cid} in {self.key}.")
        return self._labels[(rnd, cid)]

    def describe(self):
        return {"key": self.key, "kind": self.kind, "seed": self.seed, "condition": self.condition,
                "source": str(self.container.path), "prefix": self.prefix, "sha256": self.sha256}


def classify_manifest(manifest):
    if manifest.get("stage") == "capture" and manifest.get("mode") == "capture":
        return "recorded_capture"
    if (manifest.get("role") == "known_clean_simulated_calibration" and manifest.get("attack_applied") is False
            and manifest.get("aggregation") == "sample_weighted_FedAvg"):
        return "clean_control"
    # Any bsc-family shadow run shares this manifest schema; its compatibility is enforced separately
    # by the COMPATIBILITY_KEYS contract check, not by the version label.
    if (manifest.get("role") == "live_candidate_run" and str(manifest.get("version", "")).startswith("medishield_bsc_")
            and manifest.get("mode") == "shadow"):
        return "shadow_run"
    return None


def _labels_from_decisions(raw, rounds):
    labels = {}
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        rnd = int(row["Round"])
        if rnd <= rounds:
            labels[(rnd, int(row["Client"]))] = (int(row["Is_Poisoned_Update"]), int(row.get("Is_Compromised") or 0),
                                                 int(float(row.get("Poisoned_Examples") or 0)))
    return labels


def discover_trajectories(paths, training, rounds, folds, discovery, verification, light=False):
    """Find compatible recorded trajectories; reject incompatible ones loudly."""
    found = []
    for path in paths:
        container = Container(path)
        names = container.names()
        name_set = set(names)
        found_before = len(found)
        for name in sorted(n for n in names if n == "config.json" or n.endswith("/config.json")):
            manifest_raw = container.read(name)
            try:
                manifest = json.loads(manifest_raw.decode("utf-8-sig"))
            except ValueError:
                continue
            kind = classify_manifest(manifest)
            if kind is None:
                continue
            prefix = name[:-len("config.json")]
            missing = [r for r in range(1, rounds + 1) if f"{prefix}captures/round_{r:04d}.npz" not in name_set]
            seed = int(manifest["seed"])
            condition = "clean" if kind == "clean_control" else manifest["condition"]
            if light:
                found.append({"kind": kind, "seed": seed, "condition": condition, "source": str(path),
                              "prefix": prefix, "missing_rounds": missing})
                continue
            if missing:
                raise ValueError(f"{path}:{prefix} lacks capture rounds {missing[:5]}.")
            summary = json.loads(container.read(prefix + "summary.json").decode("utf-8-sig"))
            if kind == "recorded_capture":
                contract = manifest["contract"]
                if (summary.get("Status") != "CAPTURE_ONLY_NO_DEFENCE_APPLIED" or not summary.get("Completed")
                        or summary.get("Completed_Rounds", 0) < rounds):
                    raise ValueError(f"{prefix}: capture incomplete or not observation-only.")
                with np.load(io.BytesIO(container.read(prefix + "training_references.npz")), allow_pickle=False) as refs:
                    for key, (X, y) in zip(("discovery", "verification"), folds):
                        if not np.array_equal(refs[key + "_X"], X.numpy()) or not np.array_equal(refs[key + "_y"], y.numpy()):
                            raise ValueError(f"{prefix}: saved training references differ from the current reserved references.")
                labels = _labels_from_decisions(container.read(prefix + "client_decisions.csv"), rounds)
            else:
                contract = manifest["training_contract"] if kind == "clean_control" else manifest["contract"]["training"]
                if not summary.get("Completed") or int(summary.get("Rounds", summary.get("Completed_Rounds", 0))) < rounds:
                    raise ValueError(f"{prefix}: trajectory incomplete.")
                with np.load(io.BytesIO(container.read(prefix + "split_indices.npz")), allow_pickle=False) as split:
                    if (not np.array_equal(split["discovery"], discovery)
                            or not np.array_equal(split["verification"], verification)):
                        raise ValueError(f"{prefix}: reference indices differ from the current reservation.")
                labels = None if kind == "clean_control" else _labels_from_decisions(container.read(prefix + "client_decisions.csv"), rounds)
                if kind == "clean_control" and prefix + "rounds.json" in name_set:
                    for entry in json.loads(container.read(prefix + "rounds.json").decode("utf-8-sig")):
                        for artifact, checksum in entry.get("artifacts", {}).items():
                            if artifact.startswith("captures/") and int(entry["Round"]) <= rounds:
                                if bytes_hash(container.read(prefix + artifact)) != checksum:
                                    raise ValueError(f"{prefix}{artifact}: checksum differs from its committed record.")
            for key in COMPATIBILITY_KEYS:
                if digest(contract.get(key)) != digest(training[key]):
                    raise ValueError(f"{path}:{prefix} is incompatible with this run: '{key}' differs. "
                                     "Recorded trajectories must use the same data, partition, references, "
                                     "training settings, protocol functions and software.")
            if condition == "clean" and labels and any(v[0] for v in labels.values()):
                raise ValueError(f"{prefix}: labelled clean but contains poisoned submissions.")
            found.append(Trajectory(container, prefix, kind, seed, condition, manifest_raw, rounds, labels))
        if not light and len(found) == found_before:
            raise ValueError(f"Capture source {path} contains no recognised recorded trajectory "
                             "(capture-stage run, clean control or bsc-family shadow run). Nothing is skipped silently.")
    if not light:
        identities = [(t.seed, t.condition) for t in found]
        duplicates = sorted({i for i in identities if identities.count(i) > 1})
        if duplicates:
            raise ValueError(f"Duplicate recorded (seed, condition) {duplicates}: supply each trajectory once "
                             "(e.g. a results ZIP OR its extracted folder, not both).")
    return found


def measure_trajectory(trajectory, profiler, fingerprint, rounds, root, reuse_roots, template_state, stats):
    """Per-round measurements, cached by trajectory content and measurement fingerprint."""
    folder = Path(root) / "measurements" / trajectory.key
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"fingerprint": fingerprint, "trajectory_sha256": trajectory.sha256,
                "seed": trajectory.seed, "condition": trajectory.condition}  # content, not path
    index = folder / "index.json"
    if index.exists():
        if digest(read_json(index)) != digest(identity):
            raise ValueError(f"Measurement cache {folder} belongs to a different trajectory/measurement.")
    else:
        write_json(index, identity)
    result = {}
    for rnd in range(1, rounds + 1):
        raw = trajectory.read_round(rnd)
        raw_hash = bytes_hash(raw)
        path = folder / f"round_{rnd:04d}.npz"
        record = None
        candidates = [path] + [Path(r) / "measurements" / trajectory.key / path.name for r in reuse_roots]
        for candidate in candidates:
            if candidate.exists():
                loaded = load_measurement(candidate)
                if loaded["fingerprint"] == fingerprint and loaded["capture_sha256"] == raw_hash and loaded["round"] == rnd:
                    record = loaded
                    if candidate != path:
                        save_measurement(path, record)
                    stats["reused"] += 1
                    break
                if candidate == path:
                    raise ValueError(f"Cached measurement {path} does not match its capture/fingerprint.")
        if record is None:
            global_state, states, ids, sizes, steps, captured_round = load_round_capture(io.BytesIO(raw), template_state)
            if captured_round != rnd:
                raise ValueError("Captured round number differs from its file name.")
            _, norms = submission_deltas(global_state, states, ids, sizes, steps)
            started = time.perf_counter()
            E = profiler.measure(global_state, states)
            stats["seconds"] += time.perf_counter() - started
            record = {"fingerprint": fingerprint, "capture_sha256": raw_hash, "round": rnd, "client_ids": ids,
                      "client_sizes": sizes, "optimizer_steps": steps, "norms": norms, "E": E}
            save_measurement(path, record)
            stats["computed"] += 1
            print(f"  measured {trajectory.key} r{rnd:03d}/{rounds}: {len(ids)} submissions x {len(profiler.bank)} probes x 2 folds", flush=True)
        result[rnd] = record
    return result


# ---------------------------------------------------------------------------
# Evaluator-side labelling, replay and screening summaries
# ---------------------------------------------------------------------------

def attach_truth(rows, label_fn, ledger, applied):
    """Attach evaluator labels AFTER detector decisions. Missing labels are errors."""
    result = []
    for row in rows:
        rnd, cid = int(row["Round"]), int(row["Client"])
        poisoned, compromised, examples = label_fn(rnd, cid)
        ledger[(cid, rnd)] = bool(poisoned)

        def labels(rounds):
            missing = [r for r in rounds if (cid, int(r)) not in ledger]
            if missing:
                raise ValueError(f"Reference provenance lacks evaluator labels for client {cid}, rounds {missing[:5]}.")
            return [ledger[(cid, int(r))] for r in rounds]

        admitted = labels(json.loads(row["Reference_Admitted_Rounds"]))
        sources = labels(json.loads(row["Reference_Sources"]))
        detail = json.loads(row.get("Revoked_Detail", "[]"))
        revoked_truth = labels([r for r, _ in detail])
        record = dict(row)
        record.update(Is_Compromised=int(compromised), Is_Poisoned_Update=int(poisoned),
                      Poisoned_Examples=int(examples), Global_Update_Applied=int(applied),
                      Reference_Change_Committed=int(applied),
                      Poisoned_Reference_Admissions=sum(admitted), Reference_Poisoned_Sources=sum(sources),
                      Poisoned_Revoked_Mass=sum(mass for (_, mass), bad in zip(detail, revoked_truth) if bad))
        result.append(record)
    return result


def decision_summary(rows, coefficient_key, revocation_counts=True):
    """Coefficient-mass accounting. Net retention subtracts mass later revoked
    from each (client, round) contribution (filter mode or proposed replay)."""
    revoked_from = {}
    if revocation_counts:
        for r in rows:
            for rnd, mass in json.loads(r.get("Revoked_Detail", "[]")):
                key = (int(r["Client"]), int(rnd))
                revoked_from[key] = revoked_from.get(key, 0.0) + mass
    net = lambda r: r[coefficient_key] - revoked_from.get((int(r["Client"]), int(r["Round"])), 0.0)
    poisoned = [r for r in rows if r["Is_Poisoned_Update"]]
    clean = [r for r in rows if not r["Is_Poisoned_Update"]]
    tp, fp = sum(r["Flagged"] for r in poisoned), sum(r["Flagged"] for r in clean)
    clean_nominal = sum(r["Nominal_Weight"] for r in clean)
    poisoned_nominal = sum(r["Nominal_Weight"] for r in poisoned)
    first = {}
    for cid in sorted({int(r["Client"]) for r in poisoned}):
        mine = sorted((r for r in poisoned if int(r["Client"]) == cid), key=lambda r: r["Round"])
        first[str(cid)] = next((j + 1 for j, r in enumerate(mine) if r["Flagged"]), None)
    events = {}
    for r in rows:
        events[r["Reference_Event"]] = events.get(r["Reference_Event"], 0) + 1
    return {"Observations": len(rows), "Poisoned_Updates": len(poisoned), "Unpoisoned_Updates": len(clean),
            "TP": tp, "FP": fp, "FN": len(poisoned) - tp, "TN": len(clean) - fp,
            "Detection_Rate": 100 * tp / len(poisoned) if poisoned else None,
            "Unpoisoned_Update_FPR": 100 * fp / len(clean) if clean else None,
            "Detection_Precision": 100 * tp / (tp + fp) if tp + fp else None,
            "Unpoisoned_Retention": 100 * sum(r[coefficient_key] for r in clean) / clean_nominal if clean_nominal else None,
            "Poisoned_Retention": 100 * sum(r[coefficient_key] for r in poisoned) / poisoned_nominal if poisoned_nominal else None,
            "Net_Unpoisoned_Retention": 100 * sum(net(r) for r in clean) / clean_nominal if clean_nominal else None,
            "Net_Poisoned_Retention": 100 * sum(net(r) for r in poisoned) / poisoned_nominal if poisoned_nominal else None,
            "Revoked_Mass": sum(revoked_from.values()),
            "First_Detected_Poisoned_Participation_By_Client": first,
            "Flags_By_Client": {str(c): sum(r["Flagged"] for r in rows if int(r["Client"]) == c)
                                for c in sorted({int(r["Client"]) for r in rows})},
            "History_Channel_Alerts": sum(1 for r in rows if r["Flagged"] and r["Score_Channel"] == "history"),
            "Reference_Events": events,
            "Poisoned_Reference_Admissions": sum(r["Poisoned_Reference_Admissions"] for r in rows),
            "Coefficient_Basis": coefficient_key}


def replay_round(detector, raw, measurement, template_state):
    """Detector decisions for one recorded round, against ITS recorded global."""
    recorded_global, states, ids, sizes, steps, _ = load_round_capture(io.BytesIO(raw), template_state)
    if measurement.get("capture_sha256") not in (None, bytes_hash(raw)):
        raise ValueError("Measurement belongs to a different capture.")
    _, diagnostic = detector.combine(recorded_global, states, ids, sizes, steps, measurement)
    return diagnostic  # the one-round aggregate is deliberately discarded


def replay_trajectory(trajectory, measurements, core, cfg, fingerprint, template_state, rounds, folder):
    detector = CandidateDetector(core, cfg, "shadow", fingerprint)
    rows, ledger, would_stop = [], {}, []
    for rnd in range(1, rounds + 1):
        diagnostic = replay_round(detector, trajectory.read_round(rnd), measurements[rnd], template_state)
        if diagnostic["validity_reasons"]:
            would_stop.append({"Round": rnd, "Reasons": diagnostic["validity_reasons"]})
        rows.extend(attach_truth(diagnostic["clients"], trajectory.label, ledger, True))
    summary = decision_summary(rows, "Proposed_Coefficient")
    summary.update(Trajectory=trajectory.key, Seed=trajectory.seed, Condition=trajectory.condition,
                   Would_Stop_Rounds=would_stop, Live_Training_Executed=False, ASR=None,
                   Interpretation="Decisions on a fixed recorded FedAvg trajectory. Proposed weights are "
                                  "diagnostic; no new training trajectory, ASR or mitigation is implied.")
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "replayed_decisions.csv", rows)
        write_json(folder / "replay_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Protocol adapter (existing paired runner)
# ---------------------------------------------------------------------------

def load_protocol():
    here = Path(__file__).resolve().parent
    if not (here / "train_defended.py").exists():
        raise FileNotFoundError("Place medishield_candidate.py in src beside the existing train_defended.py and defenses.py.")
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    base = importlib.import_module("train_defended")
    missing = [name for name in PROTOCOL_FUNCTIONS if not hasattr(base, name)]
    if missing:
        raise RuntimeError("train_defended.py lacks required interfaces: " + ", ".join(missing))
    if getattr(base, "EXPECTED_DEFENSE_VERSION", None) not in SUPPORTED_RUNNER_VERSIONS:
        raise RuntimeError("Unsupported train_defended.py/defenses.py pair; keep the existing control_calibrated_candidate_4 files.")
    return base


def local_trainer(base, args):
    """The runner's local training function; logit adjustment must exist in the runner itself."""
    if args.loss_adjustment == "logit":
        if not hasattr(base, "train_client_logit_adjusted"):
            raise RuntimeError("train_defended.py lacks train_client_logit_adjusted; update the paired runner.")
        return base.train_client_logit_adjusted
    return base.train_client


def reference_folds(data):
    return [(data["X"][data[k]], data["y"][data[k]]) for k in ("discovery", "verification")]


def training_contract(base, data, clients, args):
    """Same structure as the contrast-5 contract so recorded captures can be verified."""
    names = PROTOCOL_FUNCTIONS + (("train_client_logit_adjusted",) if args.loss_adjustment == "logit" else ())
    functions = {name: digest(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(getattr(base, name)))),
                                       include_attributes=False)) for name in names}
    training = {k: getattr(args, k) for k in ("local_epochs", "batch_size", "learning_rate", "server_per_class",
                                               "calibration_seed", "clients_per_round", "partition_seed",
                                               "alpha", "num_clients", "threads")}
    if args.loss_adjustment == "logit":
        training.update(loss_adjustment="logit", logit_tau=args.logit_tau, class_prior=list(args.class_prior))
    pandas = getattr(base, "pd", None)
    sklearn = getattr(base, "sklearn", None)
    return {"data_sha256": data["data_sha256"], "protocol_functions": functions,
            "partition_sha256": digest([np.asarray(i).tolist() for i in clients]),
            "reference_indices_sha256": digest([np.asarray(data["discovery"]).tolist(), np.asarray(data["verification"]).tolist()]),
            "classes": list(base.CLASS_NAMES), "feature_names": data["features"],
            "training": training,
            "client_sizes": [len(i) for i in clients],
            "prescribed_steps": [args.local_epochs * math.ceil(len(i) / args.batch_size) for i in clients],
            "software": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                         "pandas": getattr(pandas, "__version__", "unavailable"),
                         "sklearn": getattr(sklearn, "__version__", "unavailable")},
            "model": "existing StableMLP", "optimizer": "existing AdamW; local state reset each call",
            "official_test_loaded_by_this_runner": False}


def configs_from_args(args):
    measurement = MeasurementConfig(probe_levels=tuple(float(v) for v in args.probe_levels),
                                    probe_widths=tuple(int(v) for v in args.probe_widths),
                                    random_groups=args.random_groups, probe_seed=args.probe_seed,
                                    probe_chunk=args.probe_chunk)
    if [str(v).lower() for v in args.protected_targets] == ["all"]:
        protected = None
    else:
        try:
            protected = tuple(int(v) for v in args.protected_targets)
        except ValueError as error:
            raise ValueError("--protected-targets takes class IDs or the single word 'all'.") from error
    detector = DetectorConfig(
        protected_targets=protected, client_profile=args.client_profile == "on",
        profile_quantile=args.profile_quantile,
        breadth_sources=args.breadth_sources, specificity=args.specificity == "on",
        peer_scale_floor=args.peer_scale_floor, effect_floor=args.effect_floor,
        history_channel=args.history_channel == "on", channel_scale_floor=args.channel_scale_floor,
        min_cohort=args.min_cohort, initial_rounds=args.initial_rounds,
        calibration_quantile=args.calibration_quantile, calibration_margin=args.calibration_margin,
        reference_quantile=args.reference_quantile, alert_floor=args.alert_floor, reference_floor=args.reference_floor,
        penalty_width_floor=args.penalty_width_floor, penalty_relative_width=args.penalty_relative_width,
        clip_multiplier=args.clip_multiplier, history_policy=args.history_policy, history_rate=args.history_rate,
        admission_patience=args.admission_patience, reference_trust=args.reference_trust,
        stale_after_rounds=args.stale_after_rounds, revocation_window=args.revocation_window, revocation_budget=args.revocation_budget,
        revocation_policy=args.revocation_policy, risk_gain=args.risk_gain, memory_gain=args.memory_gain,
        risk_recovery=args.risk_recovery, risk_cap=args.risk_cap, severity_cap=args.severity_cap,
        stop_flagged_fraction=args.stop_flagged_fraction, stop_min_mass=args.stop_min_mass,
        hold_policy=args.hold_policy, max_consecutive_holds=args.max_consecutive_holds)
    return measurement, detector


def procedures(dcfg):
    return {
        "observations": "Broadcast global, submitted local states, stable IDs, trusted counts/step budgets, two "
                        "disjoint trusted labelled TRAINING reference folds excluded from client training.",
        "not_observed": ["attacker identity", "poison masks", "trigger or target", "validation/test outcomes",
                         "update ground-truth labels"],
        "score": "S=max((A-cA)/sA,(H-cH)/sH); A=cohort channel, H=history channel (omitted if unavailable)",
        "protected_decision": ("triggered gain toward the protected target(s) " + str(list(dcfg.protected_targets))
                               + " only; other targets are logged as unscored diagnostics")
                              if dcfg.protected_targets is not None else "all targets scored (ablation)",
        "protected_decision_rationale": "IDS evasion: attack traffic classified as the benign/normal class. "
                                        "Attack-to-attack confusions still raise an alert and are not scored.",
        "client_profile": (f"excess above max(peer median, client's {dcfg.profile_quantile} quantile of its own known-clean K); "
                           "profiles cross-fitted for threshold fitting, built from all calibration seeds for deployment"
                           if dcfg.client_profile else "off (bsc_4 ablation)"),
        "breadth": f"{dcfg.breadth_sources}-th largest source effect over sources != target",
        "specificity": "subtract the client's median over probes per target" if dcfg.specificity else "off (ablation)",
        "discovery_confirmation": "one probe per target chosen on fold 1; that exact cell evaluated on fold 2; score=min",
        "history_in_score": dcfg.history_channel,
        "calibration": "known-clean unfiltered FedAvg trajectories; thresholds pooled over clients per phase; "
                       "history simulated with every informative clean observation admitted",
        "deployment": "live/replayed trajectories; history admitted only under the configured gate",
        "remaining_protocol_difference": "Calibration trajectories are unfiltered FedAvg and admit every clean "
                                         "observation; deployment filters updates and gates admission.",
        "influence": "a_i=p_i*exp(-(risk_gain*severity^2*flag + memory))*clip_i; residual stays on previous global",
        "clipping": "per client/phase clip bound = clip_multiplier * max clean full-update norm; live submissions "
                    "cannot enlarge it",
        "revocation_policy": ("only during the client's first detection episode; later alerts never revoke past contributions"
                              if dcfg.revocation_policy == "first_episode" else "on every alert (bsc_5 ablation)"),
        "revocation_budget": ("at most the client's nominal share per round, oldest contribution first"
                              if dcfg.revocation_budget == "nominal_share" else "unbounded (bsc_3 ablation)"),
        "revocation": (f"on a flag, subtract (1-trust) of the client's still-revocable contributions from the previous "
                       f"{dcfg.revocation_window} rounds; approximate (later rounds built on them)")
                      if dcfg.revocation_window else "off (ablation)",
        "stop": "filter rounds stop before commit when flagged fraction or informative mass leaves its bound"}


def build_context(base, args, data, clients):
    mcfg, dcfg = configs_from_args(args)
    dcfg.validate(len(base.CLASS_NAMES))
    if dcfg.protected_targets == (0,) and str(base.CLASS_NAMES[0]).lower() not in ("benign", "normal"):
        raise ValueError("The default protected decision needs class 0 to be Benign/Normal.")
    template = base.StableMLP(data["X"].shape[1], len(base.CLASS_NAMES))
    folds = reference_folds(data)
    profiler = ProbeProfiler(template, folds, mcfg)
    fingerprint = profiler.fingerprint()
    training = training_contract(base, data, clients, args)
    contract = {"version": VERSION, "training": training, "measurement_config": asdict(mcfg),
                "measurement_fingerprint": fingerprint, "detector_config": asdict(dcfg),
                "calibration_seeds": sorted(args.calibration_seeds), "procedures": procedures(dcfg)}
    template_state = {k: v.detach().clone() for k, v in template.state_dict().items()}
    return {"mcfg": mcfg, "dcfg": dcfg, "folds": folds, "profiler": profiler, "fingerprint": fingerprint,
            "training": training, "contract": contract, "template_state": template_state}


def participant_ids(seed, rnd, n_clients, participants):
    if participants == n_clients:
        return list(range(n_clients))
    return sorted(np.random.default_rng(seed * 1000003 + rnd).choice(n_clients, participants, replace=False).tolist())


# ---------------------------------------------------------------------------
# Clean calibration controls (TRAINING; explicit opt-in)
# ---------------------------------------------------------------------------

def generate_clean_control(base, args, data, clients, seed, training, root):
    """Known-clean sample-weighted FedAvg trajectory with captures. No poisoning,
    no validation metrics, no model selection. This is training, not a baseline."""
    folder = Path(root) / "calibration_controls" / f"seed_{seed}"
    (folder / "captures").mkdir(parents=True, exist_ok=True)
    manifest = {"version": VERSION, "seed": seed, "rounds": args.rounds, "role": "known_clean_simulated_calibration",
                "attack_applied": False, "aggregation": "sample_weighted_FedAvg", "training_contract": training,
                "validation_evaluated": False, "official_test_evaluated": False, "model_selection_performed": False,
                "source_sha256": file_hash(__file__)}
    signature = digest(manifest)
    config_path, checkpoint_path = folder / "config.json", folder / "checkpoint.pt"
    torch.manual_seed(seed)
    model = base.StableMLP(data["X"].shape[1], len(base.CLASS_NAMES))
    rounds_log, start = [], 1
    if config_path.exists():
        if read_json(config_path).get("signature") != signature:
            raise ValueError(f"{folder} holds a control with a different source/protocol; use a new --out.")
        if (folder / "summary.json").exists() and read_json(folder / "summary.json").get("Completed"):
            return folder
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved.get("signature") != signature:
                raise ValueError("Clean-control checkpoint identity differs.")
            model.load_state_dict(saved["model"], strict=True)
            rounds_log, start = saved["rounds"], saved["round"] + 1
            for entry in rounds_log:
                for name, checksum in entry["artifacts"].items():
                    if file_hash(folder / name) != checksum:
                        raise ValueError("Committed clean-control capture changed: " + name)
    else:
        write_json(config_path, {**manifest, "signature": signature})
        np.savez_compressed(folder / "split_indices.npz", discovery=data["discovery"], verification=data["verification"],
                            **{f"client_{cid}": ix for cid, ix in enumerate(clients)})
    for old in (folder / "captures").glob("round_*"):
        if int(old.name.split("_")[-1].split(".")[0]) >= start:
            old.unlink()  # uncommitted artifacts of this verified control only
    participants = args.clients_per_round or len(clients)
    for rnd in range(start, args.rounds + 1):
        started = time.perf_counter()
        ids = participant_ids(seed, rnd, len(clients), participants)
        previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"Clean control seed={seed} r{rnd:03d}/{args.rounds}: training {len(ids)} unpoisoned clients...", flush=True)
        states, sizes, steps = [], [], []
        for cid in ids:
            ix = clients[cid]
            states.append(local_trainer(base, args)(model, data["X"][ix], data["y"][ix], args, seed * 1000003 + rnd * 1009 + cid))
            sizes.append(len(ix))
            steps.append(args.local_epochs * math.ceil(len(ix) / args.batch_size))
        submission_deltas(previous, states, ids, sizes, steps)
        weights = np.asarray(sizes, dtype=float) / sum(sizes)
        updated = {k: (sum(float(a) * s[k].double() for a, s in zip(weights, states)).to(old.dtype)
                       if old.is_floating_point() else old.clone()) for k, old in previous.items()}
        if not all(torch.isfinite(v).all() for v in updated.values() if v.is_floating_point()):
            raise ValueError("Nonfinite clean-control aggregate; round not committed.")
        capture = folder / "captures" / f"round_{rnd:04d}.npz"
        write_capture(capture, rnd, previous, states, ids, sizes, steps)
        rounds_log.append({"Round": rnd, "Training_Seconds": time.perf_counter() - started,
                           "artifacts": {f"captures/{capture.name}": file_hash(capture)}})
        model.load_state_dict(updated, strict=True)
        atomic_save(checkpoint_path, {"signature": signature, "round": rnd, "model": model.state_dict(), "rounds": rounds_log})
        write_json(folder / "rounds.json", rounds_log)
    write_json(folder / "summary.json", {"Completed": True, "Seed": seed, "Rounds": args.rounds,
                                         "Role": "offline_clean_calibration", "Validation_Metrics": None,
                                         "ASR": None, "Robustness_Established": False})
    return folder


# ---------------------------------------------------------------------------
# Calibration stage: measurement, fit, transfer gate, development replay
# ---------------------------------------------------------------------------

def run_calibration(base, args, data, clients, root, context):
    cfg, training = context["dcfg"], context["training"]
    folder = Path(root) / "calibration"
    folder.mkdir(parents=True, exist_ok=True)
    sources = [Path(p) for p in (args.capture_source or [])]
    trajectories = discover_trajectories(sources, training, args.rounds, context["folds"],
                                         np.asarray(data["discovery"]), np.asarray(data["verification"]))
    generated_dir = Path(root) / "calibration_controls"
    if generated_dir.exists() and generated_dir.resolve() not in [s.resolve() for s in sources]:
        local = discover_trajectories([generated_dir], training, args.rounds, context["folds"],
                                      np.asarray(data["discovery"]), np.asarray(data["verification"]))
        known = {(t.seed, t.condition) for t in trajectories}
        trajectories += [t for t in local if (t.seed, t.condition) not in known]
    calibration_seeds = sorted(args.calibration_seeds)
    clean_seeds = {t.seed for t in trajectories if t.condition == "clean"}
    missing = [s for s in calibration_seeds if s not in clean_seeds]
    participants = args.clients_per_round or len(clients)
    if missing:
        cost = len(missing) * args.rounds * participants
        if not args.train_missing_controls:
            raise ValueError(f"No clean trajectory for calibration seeds {missing}. Supply --capture-source, or add "
                             f"--train-missing-controls to TRAIN them ({cost} local client trainings).")
        print(f"Training missing clean calibration controls {missing}: {cost} local client trainings (FedAvg, no attack).", flush=True)
        for seed in missing:
            generate_clean_control(base, args, data, clients, seed, training, root)
        local = discover_trajectories([generated_dir], training, args.rounds, context["folds"],
                                      np.asarray(data["discovery"]), np.asarray(data["verification"]))
        known = {(t.seed, t.condition) for t in trajectories}
        trajectories += [t for t in local if (t.seed, t.condition) not in known]
    write_json(folder / "trajectories.json", [t.describe() for t in trajectories])
    stats = {"computed": 0, "reused": 0, "seconds": 0.0}
    print(f"Measuring {len(trajectories)} recorded trajectories (probe inference only; cached rounds are reused).", flush=True)
    measurements = {t.key: measure_trajectory(t, context["profiler"], context["fingerprint"], args.rounds, root,
                                              args.reuse_measurements or [], context["template_state"], stats)
                    for t in trajectories}
    write_json(folder / "measurement_reuse.json", stats)
    calibration_trajectories = [t for t in trajectories if t.condition == "clean" and t.seed in calibration_seeds]
    entries = [(t, measurements[t.key]) for t in calibration_trajectories]
    observed = {t.seed: profile_observations([(t, measurements[t.key])], cfg) for t in calibration_trajectories}

    def envelopes(seeds):
        """Known-clean client profiles from the given calibration seeds only."""
        merged = {}
        for seed in seeds:
            for cid, values in observed[seed].items():
                merged.setdefault(cid, []).extend(values)
        return client_envelopes(merged, cfg)

    def rows_for(seeds):
        """Rows of these seeds; each seed scored with profiles built WITHOUT it (cross-fitting)."""
        chosen = [e for e in entries if e[0].seed in seeds]
        cross = (lambda seed: envelopes([x for x in seeds if x != seed])) if cfg.client_profile else None
        return clean_channel_rows(chosen, cfg, cross)

    rows = rows_for(calibration_seeds)
    write_csv(folder / "observations.csv", rows)
    fit = lambda subset, seeds, minimum: fit_core(subset, cfg, context["fingerprint"], training, seeds, args.rounds,
                                                 participants, len(context["profiler"].bank),
                                                 len(base.CLASS_NAMES), minimum,
                                                 envelopes(seeds) if cfg.client_profile else None)
    core = fit(rows, calibration_seeds, 3)
    transfer = []
    for held in calibration_seeds:
        fold_seeds = [s for s in calibration_seeds if s != held]
        fold_core = fit(rows_for(fold_seeds), fold_seeds, 2)
        trajectory = next(t for t in calibration_trajectories if t.seed == held)
        summary = replay_trajectory(trajectory, measurements[trajectory.key], fold_core, cfg, context["fingerprint"],
                                    context["template_state"], args.rounds, folder / "transfer" / f"held_out_seed_{held}")
        summary.update(Held_Out_Seed=held, Fit_Seeds=fold_core["fit_seeds"],
                       Thresholds={p: fold_core["phases"][p]["alert_threshold"] for p in ("initial", "later")})
        transfer.append(summary)
    replays = []
    for t in trajectories:
        role = ("in_sample_clean_fit" if t.condition == "clean" and t.seed in calibration_seeds
                else "out_of_sample_clean" if t.condition == "clean" else "development_attack_replay")
        summary = replay_trajectory(t, measurements[t.key], core, cfg, context["fingerprint"], context["template_state"],
                                    args.rounds, folder / "replay" / t.key)
        summary["Role"] = role
        replays.append(summary)
    gate = pre_live_gate(transfer, replays, args)
    write_csv(folder / "transfer_summary.csv", [flatten_summary(s) for s in transfer])
    write_csv(folder / "replay_summary.csv", [flatten_summary(s) for s in replays])
    write_json(folder / "gate.json", gate)
    calibration = {**core, "runner_revision": RUNNER_REVISION, "contract": context["contract"],
                   "contract_sha256": digest(context["contract"]), "source_sha256": file_hash(__file__),
                   "probe_bank": context["profiler"].bank, "calibration_seeds": calibration_seeds,
                   "all_trajectory_seeds": sorted({t.seed for t in trajectories}),
                   "trajectory_sources": [t.describe() for t in trajectories],
                   "observations_sha256": digest(rows), "transfer": transfer, "replays": replays,
                   "pre_live_gate": gate, "status": "DEVELOPMENT_CALIBRATION_REQUIRES_FRESH_LIVE_VALIDATION",
                   "statistical_FPR_guarantee": False,
                   "limitations": [
                       "Requires known-clean offline trajectories, stable IDs and matching partitions/budgets.",
                       "Client-round observations are dependent; empirical quantiles are not FPR guarantees.",
                       "Probes are synthetic standardized-feature stress tests; coverage is finite.",
                       "Breadth assumes source-agnostic (all-to-one) backdoors; source-specific backdoors can evade it.",
                       "Only the protected decision (default attack->Benign) is scored; backdoors toward other "
                       "classes (attack->attack, benign->attack availability attacks) are outside this detector.",
                       "The protected-target restriction was chosen after inspecting cached seed-42/44/45 evidence.",
                       "Revocation (W=3) was added after inspecting the seed-47 live result; seed 47 is development evidence.",
                       "A false alarm also revokes an honest client's recent contributions (bounded by W rounds).",
                       "Client profiles assume stable identities with known-clean history; an attacker whose trigger lies "
                       "inside its own clean profile could hide there. Added after the seed-46 lock-in diagnosis.",
                       "The probe bank overlaps the current synthetic trigger family (contiguous groups, levels near 5).",
                       "Calibration trajectories are unfiltered FedAvg; deployment filters and gates history.",
                       "The development attack replay was inspected while designing this candidate.",
                       "No adaptive-attack robustness, baseline superiority or publication readiness is established."]}
    calibration["artifact_sha256"] = digest({k: v for k, v in calibration.items() if k != "artifact_sha256"})
    write_json(folder / "calibration.json", calibration)
    print_gate(transfer, replays, gate)
    return calibration


def flatten_summary(summary):
    return {k: (json.dumps(json_value(v)) if isinstance(v, (dict, list)) else v) for k, v in summary.items()}


def pre_live_gate(transfer, replays, args):
    failures = []
    for s in transfer:
        label = f"held-out clean seed {s['Held_Out_Seed']}"
        if s["Unpoisoned_Update_FPR"] is None or s["Unpoisoned_Update_FPR"] > args.transfer_max_fpr:
            failures.append(f"{label}: alert FPR {s['Unpoisoned_Update_FPR']} > {args.transfer_max_fpr}")
        if s["Unpoisoned_Retention"] is None or s["Unpoisoned_Retention"] < args.transfer_min_retention:
            failures.append(f"{label}: proposed retention {s['Unpoisoned_Retention']} < {args.transfer_min_retention}")
        if s["Would_Stop_Rounds"]:
            failures.append(f"{label}: would stop in rounds {[w['Round'] for w in s['Would_Stop_Rounds']]}")
    attacked = [s for s in replays if s["Poisoned_Updates"]]
    if not attacked:
        failures.append("No recorded attacked trajectory: the development detection screen could not run.")
    for s in attacked:
        label = f"development attack replay seed {s['Seed']}"
        if s["Detection_Rate"] < args.replay_min_detection:
            failures.append(f"{label}: detection {s['Detection_Rate']:.2f}% < {args.replay_min_detection}%")
        if s["Poisoned_Retention"] > args.replay_max_poisoned_retention:
            failures.append(f"{label}: poisoned retention {s['Poisoned_Retention']:.2f}% > {args.replay_max_poisoned_retention}%")
        if any(v is None or v > args.replay_latest_first_alert for v in s["First_Detected_Poisoned_Participation_By_Client"].values()):
            failures.append(f"{label}: first detection missing or later than poisoned participation {args.replay_latest_first_alert}")
        if s["Unpoisoned_Update_FPR"] is not None and s["Unpoisoned_Update_FPR"] > args.transfer_max_fpr:
            failures.append(f"{label}: unpoisoned alert FPR {s['Unpoisoned_Update_FPR']:.2f}% > {args.transfer_max_fpr}%")
        if s["Would_Stop_Rounds"]:
            failures.append(f"{label}: would stop in rounds {[w['Round'] for w in s['Would_Stop_Rounds']]}")
    return {"Passed": not failures, "Failed_Criteria": failures,
            "Criteria": {"Transfer_Max_FPR": args.transfer_max_fpr, "Transfer_Min_Retention": args.transfer_min_retention,
                         "Replay_Min_Detection": args.replay_min_detection,
                         "Replay_Max_Poisoned_Retention": args.replay_max_poisoned_retention,
                         "Replay_Latest_First_Alert": args.replay_latest_first_alert},
            "Thresholds_Retuned_From_Attack_Replay": False, "Robustness_Established": False,
            "Interpretation": "Leave-one-seed-out checks are development cross-validation on unfiltered clean "
                              "trajectories; the attack replay is a previously inspected trajectory. Passing only "
                              "authorizes a fresh live pilot."}


def print_gate(transfer, replays, gate):
    for s in transfer:
        print(f"Transfer held-out clean seed {s['Held_Out_Seed']}: FPR={s['Unpoisoned_Update_FPR']}% "
              f"retention={s['Unpoisoned_Retention']}% would-stop={len(s['Would_Stop_Rounds'])} "
              f"flags-by-client={s['Flags_By_Client']}", flush=True)
    for s in replays:
        print(f"Replay [{s['Role']}] {s['Trajectory']}: detection={s['Detection_Rate']} FPR={s['Unpoisoned_Update_FPR']} "
              f"unpoisoned-retention={s['Unpoisoned_Retention']} poisoned-retention={s['Poisoned_Retention']} "
              f"first-detection={s['First_Detected_Poisoned_Participation_By_Client']} (recorded trajectory only)", flush=True)
    print("PRE-LIVE GATE: " + ("PASSED (authorizes a fresh live pilot only)" if gate["Passed"] else
                               "FAILED: " + " | ".join(gate["Failed_Criteria"])), flush=True)


def validate_calibration_file(calibration, context, args):
    payload = {k: v for k, v in calibration.items() if k != "artifact_sha256"}
    if calibration.get("artifact_sha256") != digest(payload):
        raise ValueError("Calibration checksum mismatch.")
    if calibration.get("version") != VERSION:
        raise ValueError("Calibration belongs to a different candidate version.")
    if calibration.get("source_sha256") != file_hash(__file__):
        raise ValueError("medishield_candidate.py changed after calibration; recalibrate (measurements can be reused).")
    if calibration.get("contract_sha256") != digest(context["contract"]):
        differing = [k for k in context["contract"] if digest(calibration["contract"].get(k)) != digest(context["contract"][k])]
        raise ValueError(f"Calibration contract differs from this command in: {differing}. Use identical detector/data/training options.")
    if args.rounds > calibration["rounds"]:
        raise ValueError("Requested horizon exceeds the calibrated horizon.")
    validate_core(calibration, context["dcfg"], context["fingerprint"])


# ---------------------------------------------------------------------------
# Live candidate run (training)
# ---------------------------------------------------------------------------

def run_live(base, args, data, clients, seed, condition, root, calibration, context, stop_after_round=None):
    """Fresh live trajectory under the candidate (filter) or observation (shadow)."""
    cfg, profiler = context["dcfg"], context["profiler"]
    folder = Path(root) / f"seed_{seed}" / f"medishield_{condition}"
    folder.mkdir(parents=True, exist_ok=True)  # an existing config without --resume is rejected below
    torch.manual_seed(seed)
    model = base.StableMLP(data["X"].shape[1], len(base.CLASS_NAMES))
    detector = CandidateDetector(calibration, cfg, args.shield_mode, context["fingerprint"])
    participants = args.clients_per_round or len(clients)
    attackers = base.choose_attackers(args, len(clients), seed, condition)  # attack generator / evaluator only
    trigger_set = set(args.trigger_features)
    settings = {k: v for k, v in vars(args).items() if k not in ("resume", "out", "plan_only", "self_check")}
    manifest = {"version": VERSION, "runner_revision": RUNNER_REVISION, "role": "live_candidate_run",
                "seed": seed, "condition": condition, "mode": args.shield_mode, "settings": settings,
                "contract": context["contract"], "calibration_sha256": calibration.get("artifact_sha256", core_digest(calibration)),
                "source_sha256": file_hash(__file__),
                "evaluator_only": {"attackers": sorted(attackers),
                                   "attack_target_within_protected_decision": (
                                       cfg.protected_targets is None or args.target_class in cfg.protected_targets),
                                   "probes_overlapping_trigger_columns": sum(bool(trigger_set & set(p["columns"])) for p in profiler.bank),
                                   "probe_count": len(profiler.bank)},
                "detector_inputs": "submitted states, broadcast global, stable IDs, counts, step budgets, trusted references",
                "history_in_score": cfg.history_channel, "official_test_evaluated": False,
                "validation_used_by_detector": False, "baseline_runs": 0,
                "trigger_scope": "Synthetic standardized-feature trigger; network realizability not established."}
    signature = digest(manifest)
    config_path, checkpoint_path = folder / "config.json", folder / "checkpoint.pt"
    metrics, records, ledger, ever_poisoned, start = [], [], {}, False, 1
    if config_path.exists():
        if not args.resume or read_json(config_path).get("signature") != signature:
            raise ValueError(f"{folder} exists with different settings; use the identical command with --resume or a new --out.")
        if (folder / "detector_stop.json").exists():
            print(f"Run already stopped at the detector validity gate: {folder.resolve()}", flush=True)
            return read_json(folder / "summary.json")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if checkpoint.get("signature") != signature:
                raise ValueError("Checkpoint/config mismatch.")
            model.load_state_dict(checkpoint["model"], strict=True)
            detector.import_state(checkpoint["detector"])
            metrics, records, ledger = checkpoint["metrics"], checkpoint["clients"], checkpoint["ledger"]
            ever_poisoned, start = checkpoint["ever_poisoned"], checkpoint["round"] + 1
            print(f"Resuming {folder.resolve()} at round {start}.", flush=True)
    else:
        write_json(config_path, {**manifest, "signature": signature})
        np.savez_compressed(folder / "split_indices.npz", discovery=data["discovery"], verification=data["verification"],
                            **{f"client_{cid}": ix for cid, ix in enumerate(clients)})
    for directory in ("evidence", "captures", "measurements"):
        (folder / directory).mkdir(exist_ok=True)
        for old in (folder / directory).glob("round_*"):
            if int(old.name.split("_")[-1].split(".")[0]) >= start:
                old.unlink()  # artifacts of uncommitted rounds only
    write_csv(folder / "metrics.csv", metrics)
    write_csv(folder / "client_decisions.csv", records)
    print(f"\n{VERSION} | {condition.upper()} | {args.shield_mode.upper()} | seed={seed}\n{folder.resolve()}", flush=True)
    last_cm = last_pred = None
    for rnd in range(start, args.rounds + 1):
        round_started = time.perf_counter()
        ids = participant_ids(seed, rnd, len(clients), participants)
        active = rnd >= args.attack_start and (rnd - args.attack_start) % args.attack_period < args.attack_on
        previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
        states, sizes, steps, poison_counts = [], [], [], []
        for cid in ids:
            ix = clients[cid]
            x, y = data["X"][ix], data["y"][ix]
            count = 0
            if cid in attackers and active:
                x, y, count = base.poison_sources(x, y, args, cid, seed)
            state = local_trainer(base, args)(model, x, y, args, seed * 1000003 + rnd * 1009 + cid)
            if count and args.model_scale != 1.0:
                state = {k: old + args.model_scale * (state[k] - old) if old.is_floating_point() else old.clone()
                         for k, old in previous.items()}
            states.append(state)
            sizes.append(len(ix))
            steps.append(args.local_epochs * math.ceil(len(ix) / args.batch_size))
            poison_counts.append(count)
        training_seconds = time.perf_counter() - round_started
        print(f"r{rnd:03d}: {len(ids)} submissions; probe measurement ({len(profiler.bank)} probes x 2 folds)...", flush=True)
        measure_started = time.perf_counter()
        _, norms = submission_deltas(previous, states, ids, sizes, steps)
        E = profiler.measure(previous, states)
        measurement = {"fingerprint": context["fingerprint"], "capture_sha256": None, "round": rnd, "client_ids": ids,
                       "client_sizes": sizes, "optimizer_steps": steps, "norms": norms, "E": E}
        measurement_seconds = time.perf_counter() - measure_started
        decision_started = time.perf_counter()
        labels = {cid: (int(c > 0), int(cid in attackers), int(c)) for cid, c in zip(ids, poison_counts)}
        label_fn = lambda r, c: labels[c]
        try:
            updated, diagnostic = detector.combine(previous, states, ids, sizes, steps, measurement)
        except ShieldStop as error:
            rejected = attach_truth(error.diagnostic["clients"], label_fn, dict(ledger), False)
            write_csv(folder / "stopped_client_decisions.csv", rejected)
            write_json(folder / "detector_stop.json", {
                "round": rnd, "reason": str(error), "global_update_applied": False, "detector_state_committed": False,
                "last_committed_round": rnd - 1, "last_committed_metrics": metrics[-1] if metrics else None,
                "training_seconds": training_seconds, "measurement_seconds": measurement_seconds,
                "diagnostic": error.diagnostic,
                "interpretation": "Validity stop, not attack mitigation. Do not raise the bounds to obtain a completed run."})
            summary = {"Seed": seed, "Condition": condition, "Mode": args.shield_mode, "Completed": False,
                       "Completed_Rounds": len(metrics), "Stopped_Round": rnd, "Status": "STOPPED_DETECTOR_INVALID",
                       "Reason": str(error), "Clean_Accuracy": None, "Clean_F1": None, "ASR": None,
                       "Results_Folder": str(folder.resolve())}
            write_json(folder / "summary.json", summary)
            print(f"STOPPED before applying round {rnd}: {error}\nDetails: {(folder / 'detector_stop.json').resolve()}", flush=True)
            return summary
        decision_seconds = time.perf_counter() - decision_started
        model.load_state_dict(updated, strict=True)
        capture_path = folder / "captures" / f"round_{rnd:04d}.npz"
        write_capture(capture_path, rnd, previous, states, ids, sizes, steps)
        measurement["capture_sha256"] = file_hash(capture_path)
        save_measurement(folder / "measurements" / f"round_{rnd:04d}.npz", measurement)
        decisions = attach_truth(diagnostic["clients"], label_fn, ledger, True)
        truth = np.asarray([bool(c) for c in poison_counts])
        ever_poisoned = ever_poisoned or bool(truth.any())
        evaluation_started = time.perf_counter()
        evaluation, last_cm, last_pred = base.evaluate(model, data, args, ever_poisoned)
        evaluation_seconds = time.perf_counter() - evaluation_started
        detection = base.detection_metrics(truth, [r["Flagged"] for r in decisions],
                                           honest=[cid not in attackers for cid in ids])
        nominal = np.asarray([r["Nominal_Weight"] for r in decisions])
        applied = np.asarray([r["Applied_Coefficient"] for r in decisions])
        honest = np.asarray([cid not in attackers for cid in ids])
        phase = ("clean" if condition == "clean" else "active_attack" if truth.any()
                 else "post_attack_quiet" if ever_poisoned else "pre_attack")
        unpoisoned_nominal = float(nominal[~truth].sum())
        poisoned_nominal = float(nominal[truth].sum())
        row = {"Round": rnd, "Seed": seed, "Condition": condition, "Mode": args.shield_mode, "Phase": phase,
               **evaluation, **detection,
               "Unpoisoned_Nominal_Mass": unpoisoned_nominal, "Unpoisoned_Effective_Mass": float(applied[~truth].sum()),
               "Unpoisoned_Mass_Retention": 100 * float(applied[~truth].sum()) / unpoisoned_nominal if unpoisoned_nominal else None,
               "Poisoned_Nominal_Mass": poisoned_nominal, "Poisoned_Effective_Mass": float(applied[truth].sum()),
               "Poisoned_Mass_Retention": 100 * float(applied[truth].sum()) / poisoned_nominal if poisoned_nominal else None,
               "Unpoisoned_Zero_Weight_Rate": 100 * float((applied[~truth] == 0).mean()) if (~truth).any() else None,
               "Unpoisoned_Negligible_Weight_Rate": 100 * float((applied[~truth] < 0.01 * nominal[~truth]).mean()) if (~truth).any() else None,
               "Honest_Zero_Weight_Rate": 100 * float((applied[honest] == 0).mean()) if honest.any() else None,
               "Applied_Mass": float(applied.sum()), "Proposed_Mass": diagnostic["proposed_mass"],
               "Server_Residual_Weight": diagnostic["server_residual"],
               "Round_Held": int(diagnostic.get("round_held", False)), "Consecutive_Holds": diagnostic.get("consecutive_holds", 0),
               "Flags": int(sum(r["Flagged"] for r in decisions)),
               "History_Channel_Alerts": int(sum(r["Flagged"] and r["Score_Channel"] == "history" for r in decisions)),
               "Reference_Initializations": int(sum(r["Reference_Event"] == "initialized" for r in decisions)),
               "Reference_Updates": int(sum(r["Reference_Event"] == "updated" for r in decisions)),
               "Reference_Stale_Resets": int(sum(r["Reference_Stale_Reset"] for r in decisions)),
               "Poisoned_Reference_Admissions": int(sum(r["Poisoned_Reference_Admissions"] for r in decisions)),
               "Memory_Recovery_Events": int(sum(r["Memory_Recovery"] for r in decisions)),
               "Revoked_Mass": float(sum(r["Revoked_Mass"] for r in decisions)) if args.shield_mode == "filter" else 0.0,
               "Proposed_Revoked_Mass": float(sum(r["Revoked_Mass"] for r in decisions)),
               "Poisoned_Revoked_Mass": float(sum(r["Poisoned_Revoked_Mass"] for r in decisions)),
               "Training_Seconds": training_seconds, "Detector_Measurement_Seconds": measurement_seconds,
               "Detector_Decision_Seconds": decision_seconds, "Evaluation_Seconds": evaluation_seconds,
               "Round_Seconds": time.perf_counter() - round_started}
        metrics.append(row)
        records.extend(decisions)
        write_json(folder / "evidence" / f"round_{rnd:04d}.json", {
            **{k: v for k, v in diagnostic.items() if k != "clients"},
            "clients": decisions,
            "probe_lookup": {str(r["Cohort_Probe"]): profiler.bank[r["Cohort_Probe"]] for r in decisions}})
        atomic_save(checkpoint_path, {"signature": signature, "round": rnd, "model": model.state_dict(),
                                      "detector": detector.export_state(), "metrics": metrics, "clients": records,
                                      "ledger": ledger, "ever_poisoned": ever_poisoned})
        write_csv(folder / "metrics.csv", metrics)
        write_csv(folder / "client_decisions.csv", records)
        print(f"r{rnd:03d}/{args.rounds} CA={row['Clean_Accuracy']:.2f}% F1={row['Clean_F1']:.2f}% "
              f"TTR={row['Triggered_Target_Rate']:.2f}% flags={row['Flags']} "
              f"unpoisoned-retention={row['Unpoisoned_Mass_Retention']} residual={row['Server_Residual_Weight']:.3f}", flush=True)
        if stop_after_round is not None and rnd >= stop_after_round:
            raise KeyboardInterrupt(f"Simulated interruption after committed round {rnd}.")
    if last_pred is None:
        _, last_cm, last_pred = base.evaluate(model, data, args, ever_poisoned)
    return finalize_live(base, args, data, folder, seed, condition, metrics, records, last_cm, last_pred, ever_poisoned)


def run_oracle_reference(base, args, data, clients, seed, root, aggregation):
    """EVALUATOR-ONLY reference, NOT a defence: FedAvg that drops the true attackers' updates.

    Uses attack ground truth inside aggregation, so it can never be reported as a method.
    residual:      honest clients keep their nominal share p_i; the attackers' share stays on the
                   previous global (the same residual rule MediShield uses).
    renormalized:  sample-weighted FedAvg over honest clients only (pure data-loss reference).
    Training seeds, schedule and evaluation match run_live; attackers are not trained at all.
    """
    folder = Path(root) / f"seed_{seed}" / f"oracle_{aggregation}_attacked"
    if (folder / "config.json").exists():
        raise ValueError(f"{folder} already exists; use a new --out.")
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = base.StableMLP(data["X"].shape[1], len(base.CLASS_NAMES))
    attackers = base.choose_attackers(args, len(clients), seed, "attacked")
    participants = args.clients_per_round or len(clients)
    write_json(folder / "config.json", {
        "version": VERSION, "role": "evaluator_oracle_reference_NOT_A_DEFENCE", "aggregation": aggregation,
        "seed": seed, "attackers_excluded_by_ground_truth": sorted(attackers), "source_sha256": file_hash(__file__),
        "settings": {k: v for k, v in vars(args).items() if k not in ("resume", "out", "plan_only", "self_check")}})
    metrics, ever_attacked = [], False
    for rnd in range(1, args.rounds + 1):
        started = time.perf_counter()
        ids = participant_ids(seed, rnd, len(clients), participants)
        active = rnd >= args.attack_start and (rnd - args.attack_start) % args.attack_period < args.attack_on
        ever_attacked = ever_attacked or (active and bool(set(ids) & attackers))
        previous = {k: v.detach().clone() for k, v in model.state_dict().items()}
        honest = [cid for cid in ids if cid not in attackers or not active]
        states = [local_trainer(base, args)(model, data["X"][clients[cid]], data["y"][clients[cid]], args,
                                            seed * 1000003 + rnd * 1009 + cid) for cid in honest]
        all_mass = float(sum(len(clients[cid]) for cid in ids))
        honest_mass = float(sum(len(clients[cid]) for cid in honest))
        weights = [len(clients[cid]) / (all_mass if aggregation == "residual" else honest_mass) for cid in honest]
        updated = {}
        for key, old in previous.items():
            if not old.is_floating_point():
                updated[key] = old.clone()
                continue
            change = torch.zeros_like(old, dtype=torch.float64)
            for w, state in zip(weights, states):
                change = change + w * (state[key].double() - old.double())
            updated[key] = (old.double() + change).to(old.dtype)
        model.load_state_dict(updated, strict=True)
        evaluation, cm, prediction = base.evaluate(model, data, args, ever_attacked)
        metrics.append({"Round": rnd, "Seed": seed, "Oracle_Aggregation": aggregation, **evaluation,
                        "Applied_Mass": float(sum(weights)), "Excluded_Clients": json.dumps(sorted(set(ids) - set(honest))),
                        "Round_Seconds": time.perf_counter() - started})
        write_csv(folder / "metrics.csv", metrics)
        print(f"oracle-{aggregation} r{rnd:03d}/{args.rounds} CA={evaluation['Clean_Accuracy']:.2f}% "
              f"F1={evaluation['Clean_F1']:.2f}% TTR={evaluation['Triggered_Target_Rate']:.2f}% mass={sum(weights):.3f}", flush=True)
    names = list(base.CLASS_NAMES)
    write_csv(folder / "confusion_matrix.csv", [{"True_Class": name, **{f"Pred_{names[j]}": int(cm[i, j]) for j in range(len(names))}}
                                                for i, name in enumerate(names)])
    asr = [r["ASR"] for r in metrics if r.get("ASR") is not None]
    summary = {**metrics[-1], "Status": "ORACLE_REFERENCE_NOT_A_DEFENCE", "Completed": True,
               "Mean_ASR_Over_Rounds": float(np.mean(asr)) if asr else None, "Results_Folder": str(folder.resolve())}
    write_json(folder / "summary.json", summary)
    return summary


def finalize_live(base, args, data, folder, seed, condition, metrics, records, cm, prediction, ever_poisoned):
    names = list(base.CLASS_NAMES)
    try:
        from sklearn.metrics import classification_report
        report = classification_report(np.asarray(data["vy"]), prediction, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
    except ImportError:
        report = None
    if report is not None:
        write_json(folder / "classification_report.json", report)
    write_csv(folder / "confusion_matrix.csv", [{"True_Class": name, **{f"Pred_{names[j]}": int(cm[i, j]) for j in range(len(names))}}
                                                for i, name in enumerate(names)])
    summary = decision_summary(records, "Applied_Coefficient", revocation_counts=args.shield_mode == "filter")
    final = metrics[-1] if metrics else {}
    asr = [r["ASR"] for r in metrics if r.get("ASR") is not None]
    status = ("SHADOW_ONLY_NO_MITIGATION" if args.shield_mode == "shadow" else "COMPLETED_FILTER_PILOT_NOT_ROBUSTNESS_PROOF")
    result = {**final, "Seed": seed, "Condition": condition, "Mode": args.shield_mode, "Completed": True,
              "Completed_Rounds": len(metrics), "Status": status, "Attack_Observed": bool(ever_poisoned),
              "Overall_Update_FPR": summary["Unpoisoned_Update_FPR"], "Overall_Detection_Rate": summary["Detection_Rate"],
              "Overall_Detection_Precision": summary["Detection_Precision"],
              "Overall_Unpoisoned_Retention": summary["Unpoisoned_Retention"],
              "Overall_Poisoned_Retention": summary["Poisoned_Retention"],
              "Net_Unpoisoned_Retention_After_Revocation": summary["Net_Unpoisoned_Retention"],
              "Net_Poisoned_Retention_After_Revocation": summary["Net_Poisoned_Retention"],
              "Total_Revoked_Mass": summary["Revoked_Mass"],
              "Mean_ASR_Over_Rounds": float(np.mean(asr)) if asr else None,
              "Max_ASR_Over_Rounds": float(max(asr)) if asr else None,
              "First_Detected_Poisoned_Participation_By_Client": summary["First_Detected_Poisoned_Participation_By_Client"],
              "Total_Poisoned_Reference_Admissions": summary["Poisoned_Reference_Admissions"],
              "Total_History_Channel_Alerts": summary["History_Channel_Alerts"],
              "Reference_Events": summary["Reference_Events"],
              "Total_Training_Seconds": sum(r["Training_Seconds"] for r in metrics),
              "Total_Detector_Measurement_Seconds": sum(r["Detector_Measurement_Seconds"] for r in metrics),
              "Total_Detector_Decision_Seconds": sum(r["Detector_Decision_Seconds"] for r in metrics),
              "Total_Evaluation_Seconds": sum(r["Evaluation_Seconds"] for r in metrics),
              "Results_Folder": str(folder.resolve()),
              "Metric_Notes": {"ASR": "Final-model triggered-target rate over validation records whose true class "
                                      "is not the target (ASR_Denominator); null when no poisoned update occurred.",
                               "Triggered_Target_Rate": "Same quantity reported for clean runs; NOT a learned-backdoor measure there.",
                               "Overall_Update_FPR": "Pooled over dependent client-round observations of ONE seed.",
                               "Retention": "Coefficient mass, not vector norm; applied coefficients in filter mode, nominal in shadow.",
                               "Net_Retention": "Applied coefficient minus mass later revoked from the same (client, round).",
                               "Over_Rounds": "Mean/max over evaluated rounds after the first poisoned update; rounds are dependent."}}
    write_json(folder / "summary.json", result)
    return result


LIVE_GATE_LATER_DEFAULTS = {"poison_source_classes": None, "source_specific": False, "probe_camouflage": 0.0, "camouflage_random_groups": 16}
LIVE_GATE_ATTACK_KEYS = ("attacker_id", "malicious_ratio", "poison_rate", "target_class", "trigger", "trigger_features", "trigger_value", "blend", "poison_source_classes", "source_specific", "probe_camouflage", "camouflage_random_groups", "attack_start", "attack_period", "attack_on", "model_scale", "poison_seed")


def verify_live_gate(directory, calibration, args, allow_attack_variant=False, descriptive_failed_gate=False):
    """Authorize filtering by a PASSED live development gate (AMENDMENT_live_development_gate.json).

    Requirements: the gate passed on every development attack seed of THIS calibration; each gate
    run was a --development-gate-run filter run of the attacked condition with the SAME attack
    settings; its calibration file is intact and has the SAME detector core (thresholds, profiles,
    clip bounds, configuration) as the calibration used now. Returns an authorization record.
    """
    directory = Path(directory)
    status = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
    allowed = {"DEVELOPMENT_LIVE_GATE_PASSED"} | ({"DEVELOPMENT_LIVE_GATE_FAILED"} if descriptive_failed_gate else set())
    if status.get("status") not in allowed:
        raise ValueError(f"{directory} does not record a passed live development gate.")
    gate_failed = status.get("status") == "DEVELOPMENT_LIVE_GATE_FAILED"
    development = sorted({r["Seed"] for r in calibration["replays"] if r.get("Role") == "development_attack_replay"})
    if not development:
        raise ValueError("The calibration has no development attack seed; nothing can authorize filtering.")
    records = []
    for seed in development:
        verdict_path = directory / f"development_live_gate_seed{seed}.json"
        config_path = directory / f"seed_{seed}" / "medishield_attacked" / "config.json"
        if not verdict_path.exists() or not config_path.exists():
            raise ValueError(f"Live gate evidence for development seed {seed} is missing in {directory}.")
        verdict, config = read_json(verdict_path), read_json(config_path)
        settings = config["settings"]
        if (not verdict.get("Passed") and not gate_failed) or not settings.get("development_gate_run") or config.get("mode") != "filter" \
                or config.get("condition") != "attacked":
            raise ValueError(f"Live gate seed {seed} did not pass as a filter development gate run.")
        # Options added after a gate was run are absent from its settings: an absent key means that option's default.
        differing = [k for k in LIVE_GATE_ATTACK_KEYS
                     if json_value(settings.get(k, LIVE_GATE_LATER_DEFAULTS.get(k))) != json_value(getattr(args, k))]
        if differing and not allow_attack_variant:
            raise ValueError(f"Live gate seed {seed} tested a different attack: {differing}. "
                             "For a declared robustness test against an attack variant use --attack-variant.")
        gate_calibration = read_json(settings["calibration"])
        payload = {k: v for k, v in gate_calibration.items() if k != "artifact_sha256"}
        if gate_calibration.get("artifact_sha256") != digest(payload) \
                or gate_calibration["artifact_sha256"] != config["calibration_sha256"]:
            raise ValueError(f"Live gate seed {seed}: its calibration file changed after the gate run.")
        if core_digest(gate_calibration) != core_digest(calibration):
            raise ValueError(f"Live gate seed {seed} used a different detector core than this calibration.")
        records.append({"Seed": seed, "Attack_Settings_Differing_From_Gate": differing,
                        "Gate_Calibration_Artifact": config["calibration_sha256"],
                        "Gate_Source_Sha256": config["source_sha256"], "Verdict": verdict["Reported"],
                        "Gate_Passed": bool(verdict.get("Passed")), "Gate_Failed_Criteria": verdict.get("Failed_Criteria")})
    return {"Authorized_By": ("DESCRIPTIVE ONLY - the live development gate FAILED; these runs are not confirmatory evidence"
                              if gate_failed else "passed live development gate"), "Descriptive_After_Failed_Gate": gate_failed, "Directory": str(directory.resolve()),
            "Development_Seeds": development, "Detector_Core_Sha256": core_digest(calibration), "Records": records,
            "Replay_Gate_Failure_Still_On_Record": calibration["pre_live_gate"]["Failed_Criteria"]}


def development_live_gate(summary, criteria):
    """The calibration's replay-gate criteria applied to a LIVE filter run (pre-revocation retention basis)."""
    failures = []
    if not summary.get("Completed"):
        failures.append("live run stopped at the detector validity gate: " + str(summary.get("Reason")))
    else:
        if summary["Overall_Detection_Rate"] is None or summary["Overall_Detection_Rate"] < criteria["Replay_Min_Detection"]:
            failures.append(f"detection {summary['Overall_Detection_Rate']} < {criteria['Replay_Min_Detection']}")
        if summary["Overall_Poisoned_Retention"] is None or summary["Overall_Poisoned_Retention"] > criteria["Replay_Max_Poisoned_Retention"]:
            failures.append(f"poisoned retention (before revocation) {summary['Overall_Poisoned_Retention']} > "
                            f"{criteria['Replay_Max_Poisoned_Retention']}")
        first = summary["First_Detected_Poisoned_Participation_By_Client"]
        if not first or any(v is None or v > criteria["Replay_Latest_First_Alert"] for v in first.values()):
            failures.append(f"first detection {first} later than poisoned participation {criteria['Replay_Latest_First_Alert']}")
        if summary["Overall_Update_FPR"] is None or summary["Overall_Update_FPR"] > criteria["Transfer_Max_FPR"]:
            failures.append(f"unpoisoned alert FPR {summary['Overall_Update_FPR']} > {criteria['Transfer_Max_FPR']}")
    return {"Passed": not failures, "Failed_Criteria": failures, "Criteria": criteria,
            "Reported": {k: summary.get(k) for k in ("Overall_Detection_Rate", "Overall_Poisoned_Retention",
                         "Net_Poisoned_Retention_After_Revocation", "Overall_Update_FPR", "Overall_Unpoisoned_Retention",
                         "First_Detected_Poisoned_Participation_By_Client", "ASR", "Mean_ASR_Over_Rounds",
                         "Max_ASR_Over_Rounds", "Clean_Accuracy", "Clean_F1", "Status", "Reason")},
            "Interpretation": "Development seed; authorizes the next phase only. Not evaluation evidence."}


def clean_screen(summary, args, names):
    recalls = {name: summary.get("Recall_" + name) for name in names}
    checks = {"update_fpr": summary["Overall_Update_FPR"] is not None and summary["Overall_Update_FPR"] <= args.clean_max_alert_fpr,
              "unpoisoned_retention": summary["Overall_Unpoisoned_Retention"] is not None
                                      and summary["Overall_Unpoisoned_Retention"] >= args.clean_min_retention,
              "macro_f1": summary.get("Clean_F1") is not None and summary["Clean_F1"] >= args.clean_min_macro_f1,
              "class_recall": all(v is not None and v >= args.clean_min_class_recall for v in recalls.values())}
    return {"Passed": all(checks.values()), "Checks": checks, "Update_FPR": summary["Overall_Update_FPR"],
            "Unpoisoned_Retention": summary["Overall_Unpoisoned_Retention"], "Macro_F1": summary.get("Clean_F1"),
            "Class_Recall": recalls,
            "Criteria": {"Max_FPR": args.clean_max_alert_fpr, "Min_Retention": args.clean_min_retention,
                         "Min_Macro_F1": args.clean_min_macro_f1, "Min_Each_Class_Recall": args.clean_min_class_recall},
            "Interpretation": "Collapse/over-penalization guard, not a comparison with FedAvg or a superiority test."}


def _run_pair(directory):
    """The *_clean and *_attacked run folders inside one seed folder."""
    directory = Path(directory)
    found = {}
    for condition in ("clean", "attacked"):
        matches = [p for p in directory.iterdir() if p.is_dir() and p.name.endswith("_" + condition)
                   and (p / "metrics.csv").exists()]
        if len(matches) != 1:
            raise ValueError(f"{directory} needs exactly one *_{condition} folder with metrics.csv; found {len(matches)}.")
        with (matches[0] / "metrics.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        number = lambda v: float(v) if v not in ("", None) else None
        found[condition] = {"folder": str(matches[0].resolve()),
                            "rows": {int(r["Round"]): {k: number(v) for k, v in r.items()
                                                       if k == "ASR" or k.startswith(("TTR_", "Recall_", "Clean_", "Triggered"))}
                                     for r in rows}}
        summary = matches[0] / "summary.json"
        found[condition]["summary"] = read_json(summary) if summary.exists() else {}
    if sorted(found["clean"]["rows"]) != sorted(found["attacked"]["rows"]):
        raise ValueError(f"{directory}: clean and attacked runs cover different rounds (a run may have stopped).")
    return found


def _pair_statistics(pair):
    clean, attacked = pair["clean"]["rows"], pair["attacked"]["rows"]
    rounds = sorted(attacked)
    attack_rounds = [r for r in rounds if attacked[r].get("ASR") is not None]
    last = rounds[-1]
    sources = [k[4:] for k in attacked[last] if k.startswith("TTR_") and k != "TTR_Denominator"
               and attacked[last][k] is not None]
    excess = {r: attacked[r]["Triggered_Target_Rate"] - clean[r]["Triggered_Target_Rate"] for r in rounds}
    mean = lambda values: float(np.mean(values)) if values else None
    stats = {"Final_Clean_Accuracy": clean[last]["Clean_Accuracy"], "Final_Clean_F1": clean[last]["Clean_F1"],
             "Final_Attacked_Accuracy": attacked[last]["Clean_Accuracy"], "Final_Attacked_F1": attacked[last]["Clean_F1"],
             "Final_ASR": attacked[last].get("ASR"), "Clean_Final_Triggered_Target_Rate": clean[last]["Triggered_Target_Rate"],
             "Mean_ASR_Over_Attack_Rounds": mean([attacked[r]["ASR"] for r in attack_rounds]),
             "Max_ASR_Over_Attack_Rounds": max((attacked[r]["ASR"] for r in attack_rounds), default=None),
             "Final_Excess_pp": excess[last],
             "Mean_Excess_Over_Attack_Rounds_pp": mean([excess[r] for r in attack_rounds]),
             "Max_Excess_Over_Attack_Rounds_pp": max((excess[r] for r in attack_rounds), default=None),
             "Final_Source_Excess_pp": {s: attacked[last]["TTR_" + s] - clean[last]["TTR_" + s] for s in sources},
             "Mean_Source_Excess_Over_Attack_Rounds_pp": {
                 s: mean([attacked[r]["TTR_" + s] - clean[r]["TTR_" + s] for r in attack_rounds]) for s in sources},
             "Final_Clean_Recall": {k[7:]: v for k, v in clean[last].items() if k.startswith("Recall_")},
             "Final_Attacked_Recall": {k[7:]: v for k, v in attacked[last].items() if k.startswith("Recall_")},
             "Attack_Rounds": attack_rounds}
    table = [{"Round": r, "Clean_TTR": clean[r]["Triggered_Target_Rate"], "Attacked_ASR": attacked[r].get("ASR"),
              "Excess_pp": excess[r], **{f"Excess_{s}_pp": attacked[r]["TTR_" + s] - clean[r]["TTR_" + s] for s in sources}}
             for r in rounds]
    return stats, table


def paired_report(defended_dir, reference_dir=None, out=None):
    """Evaluator-only comparison from saved per-round metrics; no inference.

    Excess = attacked minus matched clean triggered-target rate, per round and
    per source class, because this synthetic trigger also triggers clean models.
    """
    defended = _run_pair(defended_dir)
    stats, table = _pair_statistics(defended)
    attacked_summary = defended["attacked"]["summary"]
    report = {"Defended": {"Folders": {k: v["folder"] for k, v in defended.items()}, **stats,
                           **{k: attacked_summary.get(k) for k in (
                               "Overall_Detection_Rate", "Overall_Update_FPR", "Overall_Unpoisoned_Retention",
                               "Overall_Poisoned_Retention", "Net_Unpoisoned_Retention_After_Revocation",
                               "Net_Poisoned_Retention_After_Revocation", "First_Detected_Poisoned_Participation_By_Client")},
                           "Clean_Run_Update_FPR": defended["clean"]["summary"].get("Overall_Update_FPR")},
              "Undefended_Reference": None}
    if reference_dir is not None:
        reference = _run_pair(reference_dir)
        if sorted(reference["attacked"]["rows"]) != sorted(defended["attacked"]["rows"]):
            raise ValueError("Reference and defended runs cover different rounds.")
        ref_stats, ref_table = _pair_statistics(reference)
        difference = lambda a, b: a - b if a is not None and b is not None else None
        report["Undefended_Reference"] = {"Folders": {k: v["folder"] for k, v in reference.items()}, **ref_stats}
        report["Comparison"] = {
            "Final_ASR_Reduction_pp": difference(ref_stats["Final_ASR"], stats["Final_ASR"]),
            "Mean_ASR_Reduction_pp": difference(ref_stats["Mean_ASR_Over_Attack_Rounds"], stats["Mean_ASR_Over_Attack_Rounds"]),
            "Clean_F1_Difference_pp": stats["Final_Clean_F1"] - ref_stats["Final_Clean_F1"],
            "Attacked_F1_Difference_pp": stats["Final_Attacked_F1"] - ref_stats["Final_Attacked_F1"],
            "Clean_Recall_Difference_pp": {k: v - ref_stats["Final_Clean_Recall"][k]
                                           for k, v in stats["Final_Clean_Recall"].items()}}
        for row, ref_row in zip(table, ref_table):
            row["Reference_Attacked_ASR"], row["Reference_Excess_pp"] = ref_row["Attacked_ASR"], ref_row["Excess_pp"]
    report["Interpretation"] = ("One seed; rounds are dependent. Matched runs must share data, partition, model seed, "
                                "client training seeds and attack settings; this report does not verify that.")
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "paired_report.json", report)
        write_csv(out / "paired_rounds.csv", table)
    return report


def bundle_results(root):
    """Share reports/source; list (with hashes) local-only checkpoints, captures and measurements."""
    root = Path(root)
    archive_path, temporary = root / "results.zip", root / "results.zip.tmp"
    excluded = []
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path in (archive_path, temporary):
                continue
            if path.suffix in LOCAL_ONLY_SUFFIXES:
                excluded.append({"path": path.relative_to(root).as_posix(), "sha256": file_hash(path), "bytes": path.stat().st_size})
                continue
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("local_only_files.json", json.dumps(excluded, indent=2))
    for attempt in range(5):
        try:
            temporary.replace(archive_path)
            return archive_path
        except PermissionError:  # e.g. Windows antivirus/preview briefly locking the previous zip
            time.sleep(1.0 + attempt)
    fallback = root / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    temporary.replace(fallback)
    print(f"NOTE: results.zip was locked by another program; wrote {fallback.name} instead. Results are unaffected.", flush=True)
    return fallback


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def plan_identity(args):
    return {k: json_value(v) for k, v in sorted(vars(args).items()) if k not in ("out", "resume", "plan_only", "self_check")}


def execute(base, args, data, clients, root):
    root = Path(root)
    context = build_context(base, args, data, clients)
    participants = args.clients_per_round or len(clients)
    if not max(3, context["dcfg"].min_cohort) <= participants <= len(clients):
        raise ValueError(f"Participants per round must lie in [{context['dcfg'].min_cohort}, {len(clients)}].")
    plan = {"version": VERSION, "identity": plan_identity(args), "contract_sha256": digest(context["contract"]),
            "source_sha256": file_hash(__file__)}
    plan_path = root / "experiment_plan.json"
    if plan_path.exists() and read_json(plan_path) != json_value(plan):
        raise ValueError("Existing plan differs. Use the identical command with --resume, or a new --out.")
    write_json(plan_path, plan)
    if getattr(base, "__file__", None):
        sources = root / "sources"
        sources.mkdir(exist_ok=True)
        for path in [Path(__file__), Path(base.__file__)] + ([Path(base.defenses.__file__)] if getattr(base, "defenses", None) else []):
            (sources / path.name).write_bytes(path.read_bytes())
    seeds = args.seeds or [args.seed]
    conditions = ["clean", "attacked"] if args.condition == "both" else [args.condition]
    if args.stage in ("calibrate", "checkpoint"):
        calibration = run_calibration(base, args, data, clients, root, context)
        if args.stage == "calibrate" or not calibration["pre_live_gate"]["Passed"]:
            status = ("CALIBRATION_COMPLETE_NO_LIVE_TRAINING" if calibration["pre_live_gate"]["Passed"]
                      else "PRE_LIVE_GATE_FAILED_NO_LIVE_TRAINING")
            write_json(root / "status.json", {"status": status, "gate": calibration["pre_live_gate"],
                                              "robustness_established": False})
            print(f"{status}. Calibration: {(root / 'calibration' / 'calibration.json').resolve()}", flush=True)
            return 0 if calibration["pre_live_gate"]["Passed"] else 3
    else:
        calibration = read_json(args.calibration)
        validate_calibration_file(calibration, context, args)
        if args.development_gate_run:
            development = {r["Seed"] for r in calibration["replays"] if r.get("Role") == "development_attack_replay"}
            if args.shield_mode != "filter" or args.condition != "attacked" or not set(seeds) <= development:
                raise ValueError("--development-gate-run is a live filter run of the attacked condition on the seed(s) "
                                 f"of the calibration's recorded development attack {sorted(development)} only.")
            print("DEVELOPMENT LIVE GATE RUN: filter mode on a development seed, judged by the calibration's own gate "
                  "criteria. Not evaluation evidence.", flush=True)
        elif not calibration["pre_live_gate"]["Passed"]:
            if args.shield_mode == "filter" and args.live_gate_dir:
                authorization = verify_live_gate(args.live_gate_dir, calibration, args, args.attack_variant,
                                                 args.descriptive_failed_gate)
                write_json(root / "live_gate_authorization.json", authorization)
                if authorization["Descriptive_After_Failed_Gate"]:
                    print("*" * 78 + "\nDESCRIPTIVE RUN AFTER A FAILED LIVE GATE - NOT CONFIRMATORY EVIDENCE.\n" + "*" * 78, flush=True)
                print(f"Filtering authorized by the passed live development gate on seeds "
                      f"{authorization['Development_Seeds']} (replay-gate failure remains on record).", flush=True)
            elif args.shield_mode == "filter":
                raise ValueError("This calibration failed its pre-live gate; filtering is not authorized: "
                                 + " | ".join(calibration["pre_live_gate"]["Failed_Criteria"])
                                 + ". Supply --live-gate-dir with a passed live development gate if one exists.")
            # Shadow mode aggregates with nominal FedAvg and only observes, so it creates recorded
            # trajectories (e.g. the attacked development capture the gate itself needs) without
            # letting an unvalidated detector influence training.
            print("NOTE: calibration has not passed its pre-live gate. Shadow mode only: FedAvg aggregation, "
                  "detector observes; results are not mitigation evidence.", flush=True)
    reserved = set(calibration["all_trajectory_seeds"]) | set(calibration["calibration_seeds"])
    if set(seeds) & reserved and not args.development_gate_run:
        raise ValueError(f"Live model seeds {sorted(set(seeds) & reserved)} were used for calibration/development captures.")
    summaries = []
    for seed in seeds:
        by_condition = {}
        for condition in conditions:
            summary = run_live(base, args, data, clients, seed, condition, root, calibration, context)
            summaries.append(summary)
            by_condition[condition] = summary
            write_csv(root / "comparison.csv", [flatten_summary(s) for s in summaries])
            if not summary["Completed"]:
                write_json(root / "status.json", {"status": "LIVE_DETECTOR_STOP", "remaining_runs_skipped": True,
                                                  "robustness_established": False, "run": summary})
                print(f"Stopped. Share: {bundle_results(root).resolve()}", flush=True)
                return 2
            if args.development_gate_run:
                verdict = development_live_gate(summary, calibration["pre_live_gate"]["Criteria"])
                write_json(root / f"development_live_gate_seed{seed}.json", verdict)
                print("DEVELOPMENT LIVE GATE: " + ("PASSED" if verdict["Passed"] else
                      "FAILED: " + " | ".join(verdict["Failed_Criteria"])), flush=True)
            if condition == "clean" and args.condition == "both" and args.shield_mode == "filter":
                gate = clean_screen(summary, args, base.CLASS_NAMES)
                write_json(root / f"clean_screen_gate_seed{seed}.json", gate)
                if not gate["Passed"]:
                    write_json(root / "status.json", {"status": "CLEAN_SCREEN_FAILED", "remaining_runs_skipped": True,
                                                      "robustness_established": False, "gate": gate})
                    print(f"Clean screen failed {gate['Checks']}; attacked run skipped. Share: {bundle_results(root).resolve()}", flush=True)
                    return 4
        if "clean" in by_condition and "attacked" in by_condition:
            paired_report(root / f"seed_{seed}", None, root / f"paired_seed{seed}")
    if len(seeds) > 1:
        rows = []
        for condition in conditions:
            group = [s for s in summaries if s["Condition"] == condition]
            for key in ("Clean_Accuracy", "Clean_F1", "ASR", "Overall_Update_FPR", "Overall_Unpoisoned_Retention",
                        "Overall_Poisoned_Retention"):
                values = [s[key] for s in group if s.get(key) is not None]
                if values:
                    rows.append({"Condition": condition, "Metric": key, "Seeds": len(values), "Values": values,
                                 "Mean": float(np.mean(values)),
                                 "Sample_SD": float(np.std(values, ddof=1)) if len(values) > 1 else None})
        write_csv(root / "aggregate_over_seeds.csv", rows)
    if args.development_gate_run:
        verdicts = [read_json(root / f"development_live_gate_seed{seed}.json") for seed in seeds]
        write_json(root / "status.json", {"status": "DEVELOPMENT_LIVE_GATE_" + ("PASSED" if all(v["Passed"] for v in verdicts) else "FAILED"),
                                          "verdicts": verdicts, "robustness_established": False})
        print(f"Share: {bundle_results(root).resolve()}", flush=True)
        return 0 if all(v["Passed"] for v in verdicts) else 3
    descriptive = bool(getattr(args, "descriptive_failed_gate", False))
    write_json(root / "status.json", {"status": ("COMPLETED_DESCRIPTIVE_AFTER_FAILED_GATE" if descriptive else "COMPLETED_LIVE_PILOT")
                                      if args.shield_mode == "filter" else "COMPLETED_SHADOW_OBSERVATION",
                                      "robustness_established": False, "confirmatory": not descriptive, "seeds": seeds})
    print(f"Completed. Share: {bundle_results(root).resolve()}\nPilot only; robustness and baseline comparisons still require matched studies.", flush=True)
    return 0


def plan_report(args):
    """No data loading and no training: settings, recorded inputs and cost estimate."""
    mcfg, dcfg = configs_from_args(args)
    dcfg.validate()
    features = None
    names_path = Path(args.data).parent / "feature_names.json"
    if names_path.exists():
        content = read_json(names_path)
        features = len(content["features"] if isinstance(content, dict) else content)
    probes = len(build_probe_bank(features, mcfg)) if features else None
    found = discover_trajectories([Path(p) for p in (args.capture_source or [])], None, args.rounds, None, None, None,
                                  light=True) if args.capture_source and args.stage != "evaluate" else []
    clean = {f["seed"] for f in found if f["condition"] == "clean"}
    missing = [s for s in sorted(args.calibration_seeds) if s not in clean] if args.stage != "evaluate" else []
    participants = args.clients_per_round or args.num_clients or 8
    seeds = args.seeds or [args.seed]
    conditions = ["clean", "attacked"] if args.condition == "both" else [args.condition]
    live = 0 if args.stage == "calibrate" else len(seeds) * len(conditions)
    return {"version": VERSION, "runner_revision": RUNNER_REVISION, "stage": args.stage,
            "recorded_trajectories_found": found,
            "calibration_seeds": sorted(args.calibration_seeds),
            "clean_controls_to_TRAIN": missing if args.train_missing_controls else [],
            "missing_clean_controls_without_permission": missing if not args.train_missing_controls else [],
            "control_training_local_client_trainings": len(missing) * args.rounds * participants if args.train_missing_controls else 0,
            "probe_count": probes,
            "measurement_model_profiles_per_recorded_round": (participants + 1) * 2,
            "measurement_rounds_upper_bound": (len(found) + (len(missing) if args.train_missing_controls else 0)) * args.rounds,
            "live_runs_if_gates_pass": live, "live_local_client_trainings_upper_bound": live * args.rounds * participants,
            "baseline_comparison_runs": 0, "official_test_loaded": False,
            "detector_config": asdict(dcfg), "measurement_config": asdict(mcfg),
            "note": "Plan only. No data loaded, no measurement, no training. Shared reports exclude local .pt/.npz files."}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"{VERSION} | {RUNNER_REVISION}")
    p.add_argument("--self-check", action="store_true", help="Synthetic invariants + tiny synthetic integration test; no project data.")
    p.add_argument("--plan-only", action="store_true", help="Print settings, recorded inputs and costs; no data loading or training.")
    g = p.add_argument_group("workflow")
    g.add_argument("--report-dir", help="--stage report: seed folder holding *_clean and *_attacked runs.")
    g.add_argument("--reference-dir", help="--stage report: matched undefended seed folder (e.g. a train_defended FedAvg run).")
    g.add_argument("--oracle-aggregation", nargs="+", choices=["residual", "renormalized"], default=["residual", "renormalized"],
                   help="--stage oracle: aggregation variant(s) of the evaluator-only attacker-exclusion reference.")
    g.add_argument("--stage", choices=["calibrate", "evaluate", "checkpoint", "report", "oracle"], default="calibrate",
                   help="calibrate: measure recorded trajectories, fit, transfer gate, development replay (no live training). "
                        "evaluate: live runs with --calibration. checkpoint: calibrate, then live runs if the gate passes.")
    g.add_argument("--capture-source", nargs="+", help="Capture results ZIP(s) and/or folders containing recorded trajectories.")
    g.add_argument("--train-missing-controls", action="store_true", help="Permit TRAINING clean FedAvg controls for calibration seeds without a recorded trajectory.")
    g.add_argument("--reuse-measurements", nargs="+", help="Earlier output folder(s) whose verified measurement cache may be reused.")
    g.add_argument("--calibration", help="calibration/calibration.json from a calibrate run (evaluate stage only).")
    g.add_argument("--out", help="Fresh output folder, or the same folder with --resume.")
    g.add_argument("--resume", action="store_true")
    g.add_argument("--live-gate-dir", help="Evaluate stage: output folder of a PASSED --development-gate-run whose calibration "
                                           "has the same detector core and whose attack matches; authorizes filtering when the "
                                           "replay gate failed (logs/AMENDMENT_live_development_gate.json).")
    g.add_argument("--descriptive-failed-gate", action="store_true",
                   help="With --live-gate-dir: allow filtering when that gate FAILED, for a clearly labelled DESCRIPTIVE "
                        "(non-confirmatory) evaluation; every output is marked as such.")
    g.add_argument("--attack-variant", action="store_true",
                   help="With --live-gate-dir: a declared robustness test against an attack that differs from the gate's; "
                        "the detector core must still match and the differences are recorded.")
    g.add_argument("--development-gate-run", action="store_true",
                   help="Evaluate stage only: live FILTER run of the attacked condition on the calibration's development "
                        "attack seed, judged by the calibration's gate criteria (see logs/AMENDMENT_live_development_gate.json).")
    d = p.add_argument_group("data and training protocol (must match recorded trajectories)")
    d.add_argument("--data", default="data/processed_grouped/federated_data.pt")
    d.add_argument("--rounds", type=int, default=15)
    d.add_argument("--local-epochs", type=int, default=2)
    d.add_argument("--batch-size", type=int, default=2048)
    d.add_argument("--learning-rate", type=float, default=0.001)
    d.add_argument("--loss-adjustment", choices=["none", "logit"], default="none",
                   help="logit = balanced-softmax local loss with the pooled training class prior (paired runner function).")
    d.add_argument("--logit-tau", type=float, default=1.0)
    d.add_argument("--threads", type=int, default=2)
    d.add_argument("--partition-seed", type=int, default=42)
    d.add_argument("--calibration-seed", type=int, default=2026, help="Server-reference SELECTION seed used by train_defended.load_data (not a model seed).")
    d.add_argument("--server-per-class", type=int, default=32)
    d.add_argument("--alpha", type=float, help="Repartition; default preserves the saved partition.")
    d.add_argument("--num-clients", type=int)
    d.add_argument("--clients-per-round", type=int)
    s = p.add_argument_group("seeds")
    s.add_argument("--seed", type=int, default=47, help="Fresh live MODEL seed; must differ from calibration/development seeds.")
    s.add_argument("--seeds", type=int, nargs="+")
    s.add_argument("--calibration-seeds", type=int, nargs="+", default=[42, 44, 45], help="Known-clean MODEL seeds for calibration (>=3).")
    s.add_argument("--poison-seed", type=int, default=42)
    a = p.add_argument_group("live condition and attack generator (never passed to the detector)")
    a.add_argument("--condition", choices=["clean", "attacked", "both"], default="both")
    a.add_argument("--shield-mode", choices=["filter", "shadow"], default="filter")
    a.add_argument("--attacker-id", type=int, default=7)
    a.add_argument("--malicious-ratio", type=float)
    a.add_argument("--poison-rate", type=float, default=0.2)
    a.add_argument("--target-class", type=int, default=0)
    a.add_argument("--trigger", choices=["constant", "additive", "blended"], default="constant")
    a.add_argument("--trigger-features", type=int, nargs="+", default=[0, 1, 2, 3])
    a.add_argument("--trigger-feature-names", nargs="+")
    a.add_argument("--trigger-value", type=float, default=5.0)
    a.add_argument("--blend", type=float, default=0.2)
    a.add_argument("--poison-source-classes", type=int, nargs="+",
                   help="Narrow backdoor: attackers poison only rows of these class ids.")
    a.add_argument("--source-specific", action="store_true",
                   help="With --poison-source-classes: triggered rows of other attack classes keep their true labels.")
    a.add_argument("--camouflage-random-groups", type=int, default=16,
                   help="Random 4-feature groups the camouflaging attacker uses (mirrors the defender's bank design).")
    a.add_argument("--probe-camouflage", type=float, default=0.0,
                   help="Adaptive attacker: fraction of unpoisoned attack rows given a probe-style perturbation with their true label.")
    a.add_argument("--attack-start", type=int, default=1)
    a.add_argument("--attack-period", type=int, default=1)
    a.add_argument("--attack-on", type=int, default=1)
    a.add_argument("--model-scale", type=float, default=1.0)
    m = p.add_argument_group("measurement (defender probe bank)")
    m.add_argument("--probe-levels", type=float, nargs="+", default=[-6.0, -3.0, 3.0, 6.0])
    m.add_argument("--probe-widths", type=int, nargs="+", default=[1, 2, 4])
    m.add_argument("--random-groups", type=int, default=16)
    m.add_argument("--probe-seed", type=int, default=7261)
    m.add_argument("--probe-chunk", type=int, default=16)
    t = p.add_argument_group("statistic and calibration")
    t.add_argument("--client-profile", choices=["on", "off"], default="on",
                   help="Count only excess above max(peer median, the client's own known-clean profile); off = bsc_4 ablation.")
    t.add_argument("--profile-quantile", type=float, default=0.90)
    t.add_argument("--protected-targets", nargs="+", default=["0"],
                   help="Defender-declared protected decision: class IDs whose triggered gain is scored "
                        "(default 0 = attack->Benign evasion), or 'all' for the max-over-targets ablation. "
                        "Independent of the attack generator's --target-class.")
    t.add_argument("--breadth-sources", type=int, default=3, help="k-th largest source effect (1 = contrast-5-like single source; ablation).")
    t.add_argument("--specificity", choices=["on", "off"], default="on")
    t.add_argument("--history-channel", choices=["on", "off"], default="on", help="Off = ablation: history kept out of the score.")
    t.add_argument("--peer-scale-floor", type=float, default=0.05)
    t.add_argument("--effect-floor", type=float, default=0.05)
    t.add_argument("--channel-scale-floor", type=float, default=1.0)
    t.add_argument("--min-cohort", type=int, default=4)
    t.add_argument("--initial-rounds", type=int, default=3)
    t.add_argument("--calibration-quantile", type=float, default=0.99)
    t.add_argument("--calibration-margin", type=float, default=0.25)
    t.add_argument("--reference-quantile", type=float, default=0.90)
    t.add_argument("--alert-floor", type=float, default=3.0)
    t.add_argument("--reference-floor", type=float, default=1.0)
    t.add_argument("--penalty-width-floor", type=float, default=1.0)
    t.add_argument("--penalty-relative-width", type=float, default=0.25)
    t.add_argument("--clip-multiplier", type=float, default=1.25)
    h = p.add_argument_group("history and influence")
    h.add_argument("--history-policy", choices=["hard", "ungated", "proportional", "none"], default="hard")
    h.add_argument("--history-rate", type=float, default=0.10)
    h.add_argument("--admission-patience", type=int, default=3)
    h.add_argument("--reference-trust", type=float, default=0.80)
    h.add_argument("--stale-after-rounds", type=int, default=6)
    h.add_argument("--revocation-policy", choices=["first_episode", "every_alert"], default="first_episode",
                   help="Revoke past contributions only in a client's first detection episode (default) or on every alert (bsc_5 ablation).")
    h.add_argument("--revocation-budget", choices=["nominal_share", "unbounded"], default="nominal_share",
                   help="Per-round revocation cap: the flagged client's nominal share (default) or unbounded (bsc_3 ablation).")
    h.add_argument("--revocation-window", type=int, default=3,
                   help="On a flag, revoke (1-trust) of the client's contributions from the previous N rounds; 0 = off (ablation).")
    h.add_argument("--risk-gain", type=float, default=4.605170185988092)
    h.add_argument("--memory-gain", type=float, default=0.22314355131420976)
    h.add_argument("--risk-recovery", type=float, default=0.50)
    h.add_argument("--risk-cap", type=float, default=0.22314355131420976)
    h.add_argument("--severity-cap", type=float, default=2.0)
    h.add_argument("--stop-flagged-fraction", type=float, default=0.50)
    h.add_argument("--hold-policy", choices=["hold", "stop"], default="hold",
                   help="On a validity-bound breach hold the round (default) or stop the run (bsc_6 ablation).")
    h.add_argument("--max-consecutive-holds", type=int, default=2,
                   help="Stop only when this many consecutive rounds were already held and the next is invalid too.")
    h.add_argument("--stop-min-mass", type=float, default=0.25)
    c = p.add_argument_group("predeclared development criteria (percent)")
    c.add_argument("--transfer-max-fpr", type=float, default=5.0)
    c.add_argument("--transfer-min-retention", type=float, default=90.0)
    c.add_argument("--replay-min-detection", type=float, default=70.0)
    c.add_argument("--replay-max-poisoned-retention", type=float, default=20.0)
    c.add_argument("--replay-latest-first-alert", type=int, default=3)
    c.add_argument("--clean-max-alert-fpr", type=float, default=5.0)
    c.add_argument("--clean-min-retention", type=float, default=90.0)
    c.add_argument("--clean-min-macro-f1", type=float, default=65.0)
    c.add_argument("--clean-min-class-recall", type=float, default=10.0)
    args = p.parse_args(argv)
    if args.self_check:
        return args
    if args.stage == "report":
        if not args.report_dir or not args.out or args.calibration or args.capture_source:
            p.error("--stage report needs --report-dir and --out (optional --reference-dir); it loads no data and trains nothing.")
        return args
    if args.report_dir or args.reference_dir:
        p.error("--report-dir/--reference-dir are for --stage report only.")
    if args.stage == "oracle" and (args.calibration or args.capture_source or args.development_gate_run
                                   or args.condition != "attacked" or not args.out):
        p.error("--stage oracle (evaluator-only reference) needs --condition attacked and --out; no calibration or captures.")
    if args.stage in ("calibrate", "checkpoint") and args.calibration:
        p.error("--calibration is for --stage evaluate only.")
    if args.live_gate_dir and (args.stage != "evaluate" or args.development_gate_run or args.shield_mode != "filter"):
        p.error("--live-gate-dir authorizes evaluate-stage filter runs only (not with --development-gate-run).")
    if args.descriptive_failed_gate and not args.live_gate_dir:
        p.error("--descriptive-failed-gate only modifies --live-gate-dir authorization.")
    if args.attack_variant and not args.live_gate_dir:
        p.error("--attack-variant only modifies --live-gate-dir authorization.")
    if args.development_gate_run and args.stage != "evaluate":
        p.error("--development-gate-run belongs to --stage evaluate.")
    if args.stage == "evaluate" and (not args.calibration or args.capture_source or args.train_missing_controls or args.reuse_measurements):
        p.error("--stage evaluate needs --calibration and takes no capture/control/measurement inputs.")
    if args.stage == "checkpoint" and (args.condition != "both" or args.shield_mode != "filter"):
        p.error("--stage checkpoint runs the paired clean/attacked filter pilot; use --condition both --shield-mode filter.")
    if args.resume and not args.out:
        p.error("--resume requires the same explicit --out.")
    if min(args.rounds, args.local_epochs, args.batch_size, args.threads, args.attack_start, args.attack_period,
           args.replay_latest_first_alert, args.random_groups + 1, args.probe_chunk) < 1:
        p.error("Rounds, budgets, threads, schedule values and probe chunk must be positive.")
    if not math.isfinite(args.logit_tau) or args.logit_tau <= 0:
        p.error("--logit-tau must be positive and finite.")
    if args.initial_rounds >= args.rounds:
        p.error("Need at least one later-phase round after --initial-rounds.")
    finite = [args.learning_rate, args.poison_rate, args.trigger_value, args.blend, args.model_scale]
    if not all(math.isfinite(v) for v in finite) or args.learning_rate <= 0 or args.model_scale <= 0:
        p.error("Learning rate/model scale must be positive; attack settings finite.")
    if not 0 <= args.poison_rate <= 1 or not 0 <= args.blend <= 1 or not 0 <= args.attack_on <= args.attack_period:
        p.error("Invalid poison rate, blend or attack schedule.")
    for value in (args.transfer_max_fpr, args.transfer_min_retention, args.replay_min_detection,
                  args.replay_max_poisoned_retention, args.clean_max_alert_fpr, args.clean_min_retention,
                  args.clean_min_macro_f1, args.clean_min_class_recall):
        if not math.isfinite(value) or not 0 <= value <= 100:
            p.error("Criteria are percentages in [0, 100].")
    if args.alpha is not None and (not math.isfinite(args.alpha) or args.alpha <= 0):
        p.error("alpha must be positive and finite.")
    if args.malicious_ratio is not None and not 0 < args.malicious_ratio < 1:
        p.error("malicious-ratio must lie strictly between 0 and 1.")
    if args.camouflage_random_groups < 0:
        p.error("--camouflage-random-groups must be nonnegative.")
    if not 0 <= args.probe_camouflage <= 1 or (args.probe_camouflage and args.source_specific):
        p.error("--probe-camouflage is a fraction in [0, 1] and is not combined with --source-specific.")
    if args.server_per_class < 8:
        p.error("Need >= 8 trusted training references per class (>= 4 per fold).")
    seeds = args.seeds or [args.seed]
    every = seeds + args.calibration_seeds + [args.partition_seed, args.poison_seed, args.calibration_seed, args.probe_seed]
    if min(every) < 0 or len(set(seeds)) != len(seeds) or len(set(args.calibration_seeds)) != len(args.calibration_seeds):
        p.error("Seeds must be nonnegative and distinct within each list.")
    if len(args.calibration_seeds) < 3:
        p.error("Provide at least three --calibration-seeds (leave-one-seed-out needs two fitting seeds).")
    if set(seeds) & set(args.calibration_seeds):
        p.error("Live model seeds must differ from calibration model seeds.")
    try:
        mcfg, dcfg = configs_from_args(args)
        mcfg.validate()
        dcfg.validate()
    except ValueError as error:
        p.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.self_check:
        return self_check()
    if args.plan_only:
        print(json.dumps(json_value(plan_report(args)), indent=2))
        return 0
    if args.stage == "report":
        report = paired_report(args.report_dir, args.reference_dir, args.out)
        print(json.dumps(json_value({k: v for k, v in report.items() if k != "Interpretation"}), indent=2))
        print(f"Saved: {(Path(args.out) / 'paired_report.json').resolve()} and paired_rounds.csv", flush=True)
        return 0
    root = Path(args.out) if args.out else Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S_medishield_bsc")
    if root.exists() and not args.resume:
        print(f"{root.resolve()} already exists. Repeat the identical command with --resume, or choose a new --out.")
        return 1
    root.mkdir(parents=True, exist_ok=True)
    print(f"Fed-MediShield {VERSION} | {RUNNER_REVISION} | stage={args.stage}\nResult folder: {root.resolve()}", flush=True)
    try:
        base = load_protocol()
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        data = base.load_data(args)
        clients = base.make_partition(data, args, args.partition_seed)
        if args.loss_adjustment == "logit":
            if not hasattr(base, "class_prior"):
                raise RuntimeError("train_defended.py lacks class_prior; update the paired runner.")
            args.class_prior = base.class_prior(data)
        if args.trigger_feature_names:
            if not data["features"]:
                raise ValueError("Named triggers require feature_names.json.")
            missing = [n for n in args.trigger_feature_names if n not in data["features"]]
            if missing:
                raise ValueError(f"Trigger feature names absent from the saved schema: {missing}")
            args.trigger_features = [data["features"].index(n) for n in args.trigger_feature_names]
        if (len(set(args.trigger_features)) != len(args.trigger_features) or min(args.trigger_features) < 0
                or max(args.trigger_features) >= data["X"].shape[1]):
            raise ValueError("Trigger columns must be distinct valid feature indices.")
        if not 0 <= args.target_class < len(base.CLASS_NAMES):
            raise ValueError("Target class outside the class list.")
        if args.stage == "oracle":
            torch.manual_seed(0)
            summaries = [run_oracle_reference(base, args, data, clients, seed, root, aggregation)
                         for seed in (args.seeds or [args.seed]) for aggregation in args.oracle_aggregation]
            write_csv(root / "oracle_summary.csv", [flatten_summary(x) for x in summaries])
            print(f"Evaluator-only oracle references written: {root.resolve()}", flush=True)
            return 0
        code = execute(base, args, data, clients, root)
        if args.stage == "calibrate" or (code == 3 and args.stage == "checkpoint"):  # evaluate bundles inside execute
            print(f"Share: {bundle_results(root).resolve()}", flush=True)
        return code
    except KeyboardInterrupt:
        bundle_results(root)
        print("Interrupted. Repeat the identical command with --resume and the same --out; committed work is reused.", flush=True)
        return 130
    except Exception as error:
        write_json(root / "error.json", {"status": "ERROR", "type": type(error).__name__, "message": str(error)})
        bundle_results(root)
        raise


# ---------------------------------------------------------------------------
# Synthetic protocol used ONLY by --self-check (no project data)
# ---------------------------------------------------------------------------

_SYNTHETIC_CLASSES = ["Benign", "AttackA", "AttackB"]


class _SyntheticMLP(nn.Module):
    def __init__(self, features=6, classes=3):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(features, 8), nn.ReLU(), nn.Linear(8, classes))

    def forward(self, x):
        return self.network(x)


def _synthetic_feature_hashes(X):
    return X.numpy().copy()


def _synthetic_load_data(args):
    raise RuntimeError("The synthetic protocol is constructed directly by --self-check.")


def _synthetic_make_partition(data, args, seed):
    return data["clients"]


def _synthetic_train_client(model, X, y, args, local_seed):
    local = copy.deepcopy(model).train()
    generator = torch.Generator().manual_seed(local_seed)
    optimizer = torch.optim.SGD(local.parameters(), lr=0.3)
    for _ in range(args.local_epochs):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), args.batch_size):
            index = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(local(X[index]), y[index]).backward()
            optimizer.step()
    return {k: v.detach().clone() for k, v in local.state_dict().items()}


def _synthetic_apply_trigger(X, args):
    result = X.clone()
    result[:, args.trigger_features] = args.trigger_value
    return result


def _synthetic_poison(X, y, args, client_id, experiment_seed):
    eligible = (y != args.target_class).nonzero(as_tuple=True)[0]
    count = int(len(eligible) * args.poison_rate)
    generator = torch.Generator().manual_seed(args.poison_seed + 1009 * experiment_seed + client_id)
    picked = eligible[torch.randperm(len(eligible), generator=generator)[:count]]
    px, py = X.clone(), y.clone()
    px[picked], py[picked] = _synthetic_apply_trigger(px[picked], args), args.target_class
    return px, py, count


@torch.inference_mode()
def _synthetic_evaluate(model, data, args, attacked):
    model.eval()
    X, y = data["V"], data["vy"]
    prediction = model(X).argmax(1)
    eligible = y != args.target_class
    hits = model(_synthetic_apply_trigger(X[eligible], args)).argmax(1) == args.target_class
    C = len(_SYNTHETIC_CLASSES)
    cm = np.zeros((C, C), dtype=np.int64)
    for truth, guess in zip(y.tolist(), prediction.tolist()):
        cm[truth, guess] += 1
    recall = [100 * cm[c, c] / cm[c].sum() if cm[c].sum() else None for c in range(C)]
    precision = [cm[c, c] / cm[:, c].sum() if cm[:, c].sum() else 0.0 for c in range(C)]
    f1 = [2 * p * (r / 100) / (p + r / 100) if r and p + r / 100 else 0.0 for p, r in zip(precision, recall)]
    ttr = 100 * float(hits.float().mean())
    rows = {"Clean_Accuracy": 100 * float((prediction == y).float().mean()), "Clean_F1": 100 * float(np.mean(f1)),
            "Triggered_Target_Rate": ttr, "ASR": ttr if attacked else None,
            "ASR_Denominator": int(eligible.sum()) if attacked else None, "Conditional_ASR": None,
            **{f"Recall_{name}": recall[c] for c, name in enumerate(_SYNTHETIC_CLASSES)}}
    return rows, cm, prediction.numpy()


def _synthetic_choose_attackers(args, n, seed, condition):
    return set() if condition == "clean" else {args.attacker_id}


def _synthetic_detection_metrics(truth, flags, honest=None):
    truth, flags = np.asarray(truth, bool), np.asarray(flags, bool)
    tp, fp = int((truth & flags).sum()), int((~truth & flags).sum())
    fn, tn = int((truth & ~flags).sum()), int((~truth & ~flags).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Detection_Rate": 100 * tp / (tp + fn) if tp + fn else None,
            "Unpoisoned_Update_FPR": 100 * fp / (fp + tn) if fp + tn else None}


def _synthetic_protocol():
    return types.SimpleNamespace(
        CLASS_NAMES=list(_SYNTHETIC_CLASSES), StableMLP=_SyntheticMLP, feature_hashes=_synthetic_feature_hashes,
        load_data=_synthetic_load_data, make_partition=_synthetic_make_partition,
        train_client=_synthetic_train_client, apply_trigger=_synthetic_apply_trigger, poison=_synthetic_poison, poison_sources=_synthetic_poison,
        evaluate=_synthetic_evaluate, choose_attackers=_synthetic_choose_attackers,
        detection_metrics=_synthetic_detection_metrics, pd=types.SimpleNamespace(__version__="synthetic"),
        sklearn=types.SimpleNamespace(__version__="synthetic"))


def _synthetic_data():
    generator = np.random.default_rng(5)
    C, F = len(_SYNTHETIC_CLASSES), 6
    means = np.eye(C, F) * 3.0

    def draw(counts):
        X = np.concatenate([generator.normal(means[c], 1.0, size=(k, F)) for c, k in enumerate(counts)])
        y = np.concatenate([np.full(k, c) for c, k in enumerate(counts)])
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    client_mix = [(60, 20, 10), (30, 40, 10), (40, 10, 40), (50, 25, 25), (30, 30, 30)]
    parts = [draw(mix) for mix in client_mix]
    references = draw((12, 12, 12))
    X = torch.cat([x for x, _ in parts] + [references[0]])
    y = torch.cat([t for _, t in parts] + [references[1]])
    clients, offset = [], 0
    for x, _ in parts:
        clients.append(np.arange(offset, offset + len(x), dtype=np.int64))
        offset += len(x)
    ref_index = np.arange(offset, offset + len(references[0]), dtype=np.int64)
    ref_labels = references[1].numpy()
    discovery = np.concatenate([ref_index[ref_labels == c][:6] for c in range(C)])
    verification = np.concatenate([ref_index[ref_labels == c][6:] for c in range(C)])
    V, vy = draw((60, 60, 60))
    return {"X": X, "y": y, "V": V, "vy": vy, "discovery": discovery, "verification": verification,
            "features": [f"f{i}" for i in range(F)], "data_sha256": "synthetic", "preparation": {},
            "overlap_fingerprints": 0, "clients": clients}, clients


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _expect_error(function, kind=Exception):
    try:
        function()
    except kind:
        return
    raise AssertionError("An invalid input/state was accepted: " + getattr(function, "__name__", "call"))


def _state_digest(value):
    return digest(json_value(value))


def _synthetic_core(cfg, fingerprint, sizes, steps, Q, C, participants=None, threshold=3.0, reference=2.0,
                    width=1.0, clip=100.0, history=True, rounds=30):
    part = {"cohort_centre": 0.0, "cohort_scale": 1.0, "history_centre": 0.0 if history else None,
            "history_scale": 1.0 if history else None, "alert_threshold": threshold,
            "reference_threshold": reference, "penalty_width": width}
    core = {"version": VERSION, "detector_config": asdict(cfg), "measurement_fingerprint": fingerprint,
            "probe_count": Q, "classes": C, "rounds": rounds, "participants": participants or len(sizes),
            "phases": {"initial": dict(part), "later": dict(part)},
            "clip_bounds": {str(i): {"initial": clip, "later": clip} for i in range(len(sizes))},
            "client_support": {str(i): {"client_size": s, "optimizer_steps": k} for i, (s, k) in enumerate(zip(sizes, steps))},
            "client_profiles": ({str(i): np.zeros((2, Q, C)).tolist() for i in range(len(sizes))}
                                if cfg.client_profile else None)}
    validate_core(core, cfg, fingerprint)
    return core


def _measurement(E, ids, sizes, steps, norms, fingerprint="synthetic-fp"):
    return {"fingerprint": fingerprint, "capture_sha256": None, "round": 0, "client_ids": list(ids),
            "client_sizes": list(sizes), "optimizer_steps": list(steps), "norms": list(norms), "E": E}


def _unit_checks():
    torch.manual_seed(3)
    # 1. Statistic properties on controlled measurements.
    n, Q, C = 5, 8, 3
    cfg = DetectorConfig(breadth_sources=2, min_cohort=4, initial_rounds=1, admission_patience=2, stale_after_rounds=3)
    E = np.zeros((n, 2, Q, C, C), dtype=np.float32)
    E[0, :, 3, 1, 0] = E[0, :, 3, 2, 0] = 0.6      # broad, probe-specific shift to target 0
    E[1, :, 3, 1, 0] = 0.6                          # single fragile source only
    E[2, :, :, 1, 0] = E[2, :, :, 2, 0] = 0.6       # probe-agnostic class-prior shift
    scores = cohort_channel(target_profiles(E, cfg), cfg)["scores"]
    assert scores[0] > 10 and scores[1] == 0 and scores[2] == 0 and scores[3] == scores[4] == 0, scores
    single = cohort_channel(target_profiles(E, DetectorConfig(breadth_sources=1, min_cohort=4)), DetectorConfig(breadth_sources=1, min_cohort=4))["scores"]
    assert single[1] > 10, "breadth ablation must expose the single-source response"
    flat_cfg = DetectorConfig(breadth_sources=2, specificity=False, min_cohort=4)
    assert cohort_channel(target_profiles(E, flat_cfg), flat_cfg)["scores"][2] > 10, "specificity ablation must expose the prior shift"
    order = [3, 0, 4, 1, 2]
    permuted = cohort_channel(target_profiles(E[order], cfg), cfg)["scores"]
    assert permuted == [scores[i] for i in order], "cohort score is not permutation-equivariant"
    K = target_profiles(E, cfg)
    centre, _ = leave_one_out_reference(K, 0, cfg)
    changed = K.clone()
    changed[0] += 7.0
    assert torch.equal(centre, leave_one_out_reference(changed, 0, cfg)[0]), "focal client entered its own peer reference"
    same = np.repeat(E[:1], n, axis=0)
    assert cohort_channel(target_profiles(same, cfg), cfg)["scores"] == [0.0] * n, "identical cohort must allow zero alerts"
    Kc = torch.zeros((n, 2, Q, C), dtype=torch.float64)
    Kc[0, 0, 2, 0], Kc[0, 1, 2, 0], Kc[0, 1, 5, 0] = 0.5, 0.1, 0.9
    confirmed = cohort_channel(Kc, cfg)
    assert abs(confirmed["scores"][0] - 2.0) < 1e-9 and confirmed["selections"][0][0]["Probe"] == 2, \
        "confirmation must evaluate only the discovery-selected cell"
    _expect_error(lambda: cohort_channel(K[:3], cfg), ValueError)
    # A broad, specific, persistent rule toward an UNPROTECTED class (the seed-44
    # Spoofing pattern) is logged but not scored; 'all' restores the old maximum.
    other = np.zeros((n, 2, Q, C, C), dtype=np.float32)
    other[3, :, 4, 0, 2] = other[3, :, 4, 1, 2] = 0.6
    logged = cohort_channel(target_profiles(other, cfg), cfg)
    assert logged["scores"][3] == 0.0 and logged["selections"][3][2]["Confirmed_Score"] > 10
    assert logged["selections"][3][2]["Scored"] == 0 and logged["selections"][3][0]["Scored"] == 1
    every = DetectorConfig(breadth_sources=2, min_cohort=4, protected_targets=None)
    assert cohort_channel(target_profiles(other, every), every)["scores"][3] > 10
    _expect_error(lambda: DetectorConfig(protected_targets=(5,)).validate(3), ValueError)
    # Known-clean client profile: a rule the client honestly shows in clean training is not
    # anomalous for THAT client; the same rule is anomalous for a client that never showed it.
    rule = np.zeros((n, 2, Q, C, C), dtype=np.float32)
    rule[1, :, 4, 1, 0] = rule[1, :, 4, 2, 0] = 0.4      # client 1: legitimate habitual rule
    rule[3, :, 6, 1, 0] = rule[3, :, 6, 2, 0] = 0.6      # client 3: new trigger-like rule
    Kr = target_profiles(rule, cfg)
    clean_hist = {1: [Kr[1].clone() * 1.0, Kr[1].clone() * 1.1], 3: [torch.zeros_like(Kr[3])] * 2}
    env = client_envelopes(clean_hist, cfg)
    assert torch.allclose(env[1], torch.quantile(torch.stack(clean_hist[1]), cfg.profile_quantile, dim=0))
    no_profile = cohort_channel(Kr, cfg)["scores"]
    with_profile = cohort_channel(Kr, cfg, [None, env[1], None, env[3], None])
    assert no_profile[1] > 5 and with_profile["scores"][1] == 0.0, "habitual clean behaviour must not be anomalous"
    assert with_profile["scores"][3] == no_profile[3] > 10, "a rule absent from the clean profile must stay anomalous"
    assert with_profile["selections"][1][0]["Discovery_Profile"] is not None
    assert cohort_channel(Kr, cfg, [None] * n)["scores"] == no_profile, "no profiles must reproduce the bsc_4 statistic"
    bigger = rule.copy()
    bigger[1, :, 4, 1, 0] = bigger[1, :, 4, 2, 0] = 0.9   # exceeding its own habit is still scored
    assert cohort_channel(target_profiles(bigger, cfg), cfg, [None, env[1], None, env[3], None])["scores"][1] > 5

    print("PASS statistic: known-clean client profiles (habitual rule ignored, new rule and exceeded habit still scored, "
          "none == bsc_4), only the protected decision is scored (others logged), breadth suppresses single-source shifts, specificity removes probe-agnostic shifts, "
          "leave-one-out peers, permutation equivariance, zero-alert rounds, fixed-cell confirmation, small-cohort rejection.")

    # 2. Reference admission, freezing, requalification, staleness and the history channel.
    P = torch.zeros((2, Q, C), dtype=torch.float64)
    st = new_client_state()
    assert advance_reference(st, P, True, 0.5, 1, cfg)[0] == "probation" and st["mu"] is None
    assert advance_reference(st, P + 1, False, 0.5, 2, cfg)[0] == "probation_reset" and not st["buffer"] and st["streak"] == 0
    assert advance_reference(st, P, True, 0.5, 3, cfg)[0] == "probation"
    event, admitted = advance_reference(st, P + 0.2, True, 0.5, 4, cfg)
    assert event == "initialized" and admitted == [3, 4] and torch.allclose(st["mu"], P + 0.1) and torch.allclose(st["dev"], P + 0.1)
    frozen = st["mu"].clone()
    assert advance_reference(st, P + 5, False, 0.5, 5, cfg)[0] == "frozen" and torch.equal(st["mu"], frozen)
    assert advance_reference(st, P, True, 0.5, 6, cfg)[0] == "requalifying" and torch.equal(st["mu"], frozen)
    assert advance_reference(st, P, True, 0.5, 7, cfg) == ("updated", [7]) and torch.allclose(st["mu"], P + 0.05)
    assert st["sources"] == [3, 4, 7]
    assert not stale_reset(st, 10, cfg) and stale_reset(st, 11, cfg) and st["mu"] is None and st["sources"] == []
    ungated = DetectorConfig(breadth_sources=2, min_cohort=4, history_policy="ungated", admission_patience=1)
    other = new_client_state()
    advance_reference(other, P, True, 0.5, 1, ungated)
    assert advance_reference(other, P + 1, True, 0.5, 2, ungated)[0] == "updated", "ungated ablation must admit"
    reference = {"mu": P.clone(), "dev": P.clone()}
    probe = P.clone()
    probe[:, 1, 0] = 0.3
    assert abs(history_channel(probe, reference, cfg)["score"] - 6.0) < 1e-9
    print("PASS history: probation streak, failed qualification clears buffer, frozen reference, hard requalification, "
          "slow EMA, provenance, stale reset in rounds, ungated ablation differs, one-sided history score.")

    # 3. Influence rule.
    memory, immediate, trust = influence_transition(0.0, True, 1.0, False, cfg)
    assert trust <= 0.01 and 0 < memory <= cfg.risk_cap
    same_memory, _, _ = influence_transition(memory, False, 0.0, False, cfg)
    assert same_memory == memory, "uninformative or absent participation must not earn recovery"
    for _ in range(3):
        memory, immediate, trust = influence_transition(memory, False, 0.0, True, cfg)
    assert immediate == 0 and trust > 0.97
    for _ in range(50):
        memory, _, trust = influence_transition(memory, True, 1.0, False, cfg)
        assert memory <= cfg.risk_cap + 1e-15 and trust <= 0.01
    tiny = alert_strength(3.0 + 1e-9, 3.0, 1.0, 2.0)
    assert influence_transition(0.0, True, tiny, False, cfg)[2] > 0.999999
    print("PASS influence: bounded memory, recovery only on qualifying participation, continuous onset, repeated-alert attenuation.")

    # 4. Detector: aggregation bounds, residual, immutability, shadow identity, clipping, rollback, absence, serialization.
    sizes, steps = [10, 20, 30, 40, 50], [1, 1, 1, 1, 1]
    core = _synthetic_core(cfg, "synthetic-fp", sizes, steps, Q, C)
    global_state = {"w": torch.randn(4, 3), "b": torch.randn(3), "count": torch.tensor(7)}
    states = [{"w": global_state["w"] + 0.01 * (i + 1), "b": global_state["b"] - 0.01 * i, "count": torch.tensor(7)}
              for i in range(n)]
    saved_global, saved_states = copy.deepcopy(global_state), copy.deepcopy(states)
    _, norms = submission_deltas(global_state, states, list(range(n)), sizes, steps)
    zeros = np.zeros((n, 2, Q, C, C), dtype=np.float32)
    attack = zeros.copy()
    attack[4, :, 3, 1, 0] = attack[4, :, 3, 2, 0] = 0.6
    plain = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    _, plain_log = plain.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    detector = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    output, log = detector.combine(global_state, states, list(range(n)), sizes, steps, _measurement(attack, range(n), sizes, steps, norms))
    rows = log["clients"]
    assert rows[4]["Flagged"] and rows[4]["Trust"] < 1e-6 and not any(r["Flagged"] for r in rows[:4])
    assert all(0 <= r["Applied_Coefficient"] <= r["Nominal_Weight"] for r in rows)
    assert [r["Applied_Coefficient"] for r in rows[:4]] == [r["Applied_Coefficient"] for r in plain_log["clients"][:4]], \
        "a penalized client must not raise the other coefficients (no renormalization)"
    coefficients = np.array([r["Applied_Coefficient"] for r in rows])
    assert abs(log["server_residual"] - (1 - coefficients.sum())) < 1e-15 and log["server_residual"] > 0.3
    for key in ("w", "b"):
        independent = global_state[key].double().numpy() + sum(
            a * (s[key].double().numpy() - global_state[key].double().numpy()) for a, s in zip(coefficients, states))
        assert np.allclose(output[key].double().numpy(), independent, atol=1e-6)
    assert torch.equal(output["count"], global_state["count"])
    assert all(torch.equal(saved_global[k], global_state[k]) for k in global_state)
    assert all(torch.equal(a[k], b[k]) for a, b in zip(saved_states, states) for k in a), "inputs were mutated"
    shadow = CandidateDetector(core, cfg, "shadow", "synthetic-fp")
    shadow_output, _ = shadow.combine(global_state, states, list(range(n)), sizes, steps, _measurement(attack, range(n), sizes, steps, norms))
    nominal = np.asarray(sizes) / sum(sizes)
    for key in ("w", "b"):
        fedavg = sum(float(p) * s[key].double() for p, s in zip(nominal, states)).to(global_state[key].dtype)
        assert torch.allclose(shadow_output[key], fedavg, atol=1e-6), "shadow mode must equal nominal FedAvg"
    unpenalized = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    filter_output, _ = unpenalized.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    calm = CandidateDetector(core, cfg, "shadow", "synthetic-fp")
    calm_output, _ = calm.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    assert all(torch.equal(filter_output[k], calm_output[k]) for k in filter_output), "unpenalized filter must equal shadow bitwise"
    huge = copy.deepcopy(states)
    huge[2] = {"w": global_state["w"] + 50.0, "b": global_state["b"], "count": torch.tensor(7)}
    _, huge_norms = submission_deltas(global_state, huge, list(range(n)), sizes, steps)
    clipped_core = _synthetic_core(cfg, "synthetic-fp", sizes, steps, Q, C, clip=1.0)
    clipper = CandidateDetector(clipped_core, cfg, "filter", "synthetic-fp")
    _, clip_log = clipper.combine(global_state, huge, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, huge_norms))
    r = clip_log["clients"][2]
    assert r["Clip_Factor"] < 1 and r["Clip_Bound"] == 1.0
    assert r["Applied_Coefficient"] * r["Full_Update_Norm"] <= r["Nominal_Weight"] * r["Trust"] * 1.0 + 1e-12
    assert r["Reference_Event"] in ("not_admitted", "probation_reset"), "clipped update must not enter history"
    stopper = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    stopper.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    before = _state_digest(stopper.export_state())
    everyone = zeros.copy()
    for i in range(n):
        everyone[i, :, i, 1, 0] = everyone[i, :, i, 2, 0] = 0.6
    # bsc_7: an invalid round is held (global unchanged, nothing committed but the round counter) ...
    clients_before = _state_digest(stopper.clients)
    for expected in (1, 2):
        held, hold_log = stopper.combine(global_state, states, list(range(n)), sizes, steps,
                                         _measurement(everyone, range(n), sizes, steps, norms))
        assert hold_log["round_held"] and hold_log["consecutive_holds"] == expected
        assert all(torch.equal(held[k], global_state[k]) for k in global_state), "a held round changed the global model"
        assert all(r["Applied_Coefficient"] == 0.0 and r["Revoked_Mass"] == 0.0 for r in hold_log["clients"])
        assert _state_digest(stopper.clients) == clients_before, "a held round committed client state"
    # ... and the third consecutive invalid round stops without committing anything.
    before_stop = _state_digest(stopper.export_state())
    try:
        stopper.combine(global_state, states, list(range(n)), sizes, steps, _measurement(everyone, range(n), sizes, steps, norms))
        raise AssertionError("A third consecutive collapsed round was accepted.")
    except ShieldStop as stop:
        assert stop.diagnostic["detector_state_committed"] is False and "3 consecutive" in str(stop)
    assert _state_digest(stopper.export_state()) == before_stop, "a stopped round changed detector state"
    # A valid round after a hold resets the counter.
    resetter = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    resetter.combine(global_state, states, list(range(n)), sizes, steps, _measurement(everyone, range(n), sizes, steps, norms))
    _, ok_log = resetter.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    assert not ok_log["round_held"] and resetter.consecutive_holds == 0
    # bsc_6 ablation: --hold-policy stop stops at the first invalid round.
    stop_cfg = replace(cfg, hold_policy="stop")
    old_style = CandidateDetector(_synthetic_core(stop_cfg, "synthetic-fp", sizes, steps, Q, C), stop_cfg, "filter", "synthetic-fp")
    old_style.combine(global_state, states, list(range(n)), sizes, steps, _measurement(zeros, range(n), sizes, steps, norms))
    before = _state_digest(old_style.export_state())
    _expect_error(lambda: old_style.combine(global_state, states, list(range(n)), sizes, steps,
                                            _measurement(everyone, range(n), sizes, steps, norms)), ShieldStop)
    assert _state_digest(old_style.export_state()) == before, "a stopped round changed detector state"
    stopper = old_style
    poisoned = copy.deepcopy(states)
    poisoned[0]["w"][0, 0] = float("nan")
    _expect_error(lambda: stopper.combine(global_state, poisoned, list(range(n)), sizes, steps,
                                          _measurement(zeros, range(n), sizes, steps, norms)), ValueError)
    assert _state_digest(stopper.export_state()) == before, "an invalid round changed detector state"
    _expect_error(lambda: stopper.combine(global_state, states[:4], list(range(4)), sizes[:4], steps[:4],
                                          _measurement(zeros[:4], range(4), sizes[:4], steps[:4], norms[:4])), ValueError)
    _expect_error(lambda: CandidateDetector(_synthetic_core(cfg, "synthetic-fp", sizes, steps, Q, C, participants=4), cfg, "filter", "other-fp"), ValueError)
    _expect_error(lambda: _synthetic_core(cfg, "synthetic-fp", sizes, steps, Q, C, participants=3), ValueError)
    print("PASS aggregation: 0<=a_i<=p_i, residual on previous global, no renormalization, independent recomputation, "
          "immutable inputs and buffers, shadow==FedAvg, unpenalized filter==shadow, frozen clip bound, held rounds "
          "(global unchanged, nothing committed, counter reset), stop after 3 consecutive invalid rounds, stop ablation, invalid rollback.")

    # Retroactive revocation: bounded by what was applied, window-limited, filter-only.
    ids = list(range(n))
    measure = lambda E_: _measurement(E_, ids, sizes, steps, norms)
    deltas = [{k: s[k].double() - global_state[k].double() for k in ("w", "b")} for s in states]
    revoker = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    _, log1 = revoker.combine(global_state, states, ids, sizes, steps, measure(zeros))
    p4 = log1["clients"][4]["Applied_Coefficient"]
    assert revoker.clients["4"]["contributions"][0]["coefficient"] == p4 and log1["revoked_mass"] == 0
    out2, log2 = revoker.combine(global_state, states, ids, sizes, steps, measure(attack))
    row = log2["clients"][4]
    assert math.isclose(row["Revoked_Mass"], (1 - row["Trust"]) * p4) and json.loads(row["Revoked_Detail"])[0][0] == 1
    assert sum(r["Revoked_Mass"] for r in log2["clients"][:4]) == 0, "only the flagged client is revoked"
    for key in ("w", "b"):
        independent = (global_state[key].double()
                       + sum(r["Applied_Coefficient"] * d[key] for r, d in zip(log2["clients"], deltas))
                       - row["Revoked_Mass"] * deltas[4][key])
        assert torch.allclose(out2[key].double(), independent, atol=1e-6)
    _, log3 = revoker.combine(global_state, states, ids, sizes, steps, measure(attack))
    revoked_from_round1 = sum(m for r in (log2, log3) for rnd, m in json.loads(r["clients"][4]["Revoked_Detail"]) if rnd == 1)
    assert revoked_from_round1 <= p4 + 1e-15, "revocation exceeded the applied contribution"
    off_cfg = DetectorConfig(breadth_sources=2, min_cohort=4, initial_rounds=1, admission_patience=2,
                             stale_after_rounds=3, revocation_window=0)
    off = CandidateDetector(_synthetic_core(off_cfg, "synthetic-fp", sizes, steps, Q, C), off_cfg, "filter", "synthetic-fp")
    off.combine(global_state, states, ids, sizes, steps, measure(zeros))
    off_out, off_log = off.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert off_log["revoked_mass"] == 0 and not off.clients["4"]["contributions"]
    assert all(torch.equal(off_out[k], out2[k]) for k in ("count",)) and not torch.equal(off_out["w"], out2["w"])
    short_cfg = DetectorConfig(breadth_sources=2, min_cohort=4, initial_rounds=1, admission_patience=2,
                               stale_after_rounds=3, revocation_window=1)
    short = CandidateDetector(_synthetic_core(short_cfg, "synthetic-fp", sizes, steps, Q, C), short_cfg, "filter", "synthetic-fp")
    short.combine(global_state, states, ids, sizes, steps, measure(zeros))
    short.combine(global_state, states, ids, sizes, steps, measure(zeros))
    _, short_log = short.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert [r for r, _ in json.loads(short_log["clients"][4]["Revoked_Detail"])] == [2], "window must exclude older rounds"
    watcher = CandidateDetector(core, cfg, "shadow", "synthetic-fp")
    watcher.combine(global_state, states, ids, sizes, steps, measure(zeros))
    watch_out, watch_log = watcher.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert watch_log["revoked_mass"] > 0 and not watch_log["revocation_applied"]
    for key in ("w", "b"):
        fedavg = sum(float(p) * s[key].double() for p, s in zip(nominal, states)).to(global_state[key].dtype)
        assert torch.allclose(watch_out[key], fedavg, atol=1e-6), "shadow revocation must not change the model"
    backlog = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    for _ in range(3):
        backlog.combine(global_state, states, ids, sizes, steps, measure(zeros))
    _, b4 = backlog.combine(global_state, states, ids, sizes, steps, measure(attack))
    detail4 = json.loads(b4["clients"][4]["Revoked_Detail"])
    assert b4["clients"][4]["Revoked_Mass"] <= p4 + 1e-15 and detail4[0][0] == 1, "budget or oldest-first violated"
    _, b5 = backlog.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert any(r == 2 and m > 0.5 * p4 for r, m in json.loads(b5["clients"][4]["Revoked_Detail"])), "backlog not carried forward"
    open_cfg = DetectorConfig(breadth_sources=2, min_cohort=4, initial_rounds=1, admission_patience=2,
                              stale_after_rounds=3, revocation_budget="unbounded")
    unbounded = CandidateDetector(_synthetic_core(open_cfg, "synthetic-fp", sizes, steps, Q, C), open_cfg, "filter", "synthetic-fp")
    for _ in range(3):
        unbounded.combine(global_state, states, ids, sizes, steps, measure(zeros))
    _, u4 = unbounded.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert u4["clients"][4]["Revoked_Mass"] > 2.9 * p4, "unbounded ablation must revoke the whole backlog at once"
    # bsc_6 first-episode policy: an intermittent attacker (flag, quiet, flag) is revoked only once.
    intermittent = CandidateDetector(core, cfg, "filter", "synthetic-fp")
    intermittent.combine(global_state, states, ids, sizes, steps, measure(zeros))                 # r1 honest
    _, i2 = intermittent.combine(global_state, states, ids, sizes, steps, measure(attack))         # r2 flagged
    _, i3 = intermittent.combine(global_state, states, ids, sizes, steps, measure(zeros))          # r3 quiet
    _, i4 = intermittent.combine(global_state, states, ids, sizes, steps, measure(attack))         # r4 flagged again
    assert i2["clients"][4]["Revoked_Mass"] > 0 and i2["clients"][4]["Revocation_Status_After"] == "active"
    assert i3["clients"][4]["Revocation_Status_After"] == "spent" and not intermittent.clients["4"]["contributions"]
    assert i4["clients"][4]["Flagged"] and i4["clients"][4]["Revoked_Mass"] == 0, "a later episode must not revoke"
    assert i4["clients"][4]["Trust"] < 1e-6, "later alerts must still attenuate the current update"
    every_cfg = DetectorConfig(breadth_sources=2, min_cohort=4, initial_rounds=1, admission_patience=2,
                               stale_after_rounds=3, revocation_policy="every_alert")
    every = CandidateDetector(_synthetic_core(every_cfg, "synthetic-fp", sizes, steps, Q, C), every_cfg, "filter", "synthetic-fp")
    for E_ in (zeros, attack, zeros):
        every.combine(global_state, states, ids, sizes, steps, measure(E_))
    _, e4 = every.combine(global_state, states, ids, sizes, steps, measure(attack))
    assert e4["clients"][4]["Revoked_Mass"] > 0, "every_alert ablation must revoke the quiet round"
    print("PASS revocation: (1-trust) of buffered contributions, flagged client only, independent recomputation, "
          "never more than applied, per-round budget p_i oldest-first with carry-over, unbounded ablation, "
          "first-episode-only revocation (intermittent attacker revoked once, still attenuated), every_alert ablation, "
          "window-limited, off switch, shadow proposes without applying.")

    # History in the detector, absence and uninformative updates.
    partial = _synthetic_core(cfg, "synthetic-fp", sizes, steps, Q, C, participants=4)
    tracker = CandidateDetector(partial, cfg, "filter", "synthetic-fp")
    present = [0, 1, 2, 3]
    pick = lambda xs, ids: [xs[i] for i in ids]
    for _ in range(2):
        _, hlog = tracker.combine(global_state, pick(states, present), present, pick(sizes, present), pick(steps, present),
                                  _measurement(zeros[present], present, pick(sizes, present), pick(steps, present), pick(norms, present)))
    assert all(r["Reference_Event"] == "initialized" for r in hlog["clients"])
    present = [1, 2, 3, 4]
    late = zeros[present].copy()
    late[0, :, 5, 1, 0] = late[0, :, 5, 2, 0] = 0.6   # client 1 changes after earning a reference
    reference_before = tracker.clients["1"]["mu"].clone()
    _, hlog = tracker.combine(global_state, pick(states, present), present, pick(sizes, present), pick(steps, present),
                              _measurement(late, present, pick(sizes, present), pick(steps, present), pick(norms, present)))
    row = hlog["clients"][0]
    assert row["History_Status"] == "available" and row["History_Score"] > 10 and row["Flagged"]
    assert row["Reference_Event"] == "frozen" and torch.equal(tracker.clients["1"]["mu"], reference_before)
    absent = copy.deepcopy(tracker.clients["0"])
    _, _ = tracker.combine(global_state, pick(states, present), present, pick(sizes, present), pick(steps, present),
                           _measurement(zeros[present], present, pick(sizes, present), pick(steps, present), pick(norms, present)))
    assert _state_digest(tracker.clients["0"]) == _state_digest(absent), "absence changed client state"
    risk_before = tracker.clients["1"]["risk"]
    still = [copy.deepcopy(global_state) if cid == 1 else states[cid] for cid in present]
    _, still_norms = submission_deltas(global_state, still, present, pick(sizes, present), pick(steps, present))
    _, slog = tracker.combine(global_state, still, present, pick(sizes, present), pick(steps, present),
                              _measurement(zeros[present], present, pick(sizes, present), pick(steps, present), still_norms))
    assert slog["clients"][0]["Informative"] == 0 and tracker.clients["1"]["risk"] == risk_before, \
        "an uninformative update earned recovery credit"
    saved = io.BytesIO()
    torch.save(tracker.export_state(), saved)
    saved.seek(0)
    restored = CandidateDetector(partial, cfg, "filter", "synthetic-fp")
    restored.import_state(torch.load(saved, map_location="cpu", weights_only=False))
    m = _measurement(late, present, pick(sizes, present), pick(steps, present), pick(norms, present))
    a_out, a_log = tracker.combine(global_state, pick(states, present), present, pick(sizes, present), pick(steps, present), m)
    b_out, b_log = restored.combine(global_state, pick(states, present), present, pick(sizes, present), pick(steps, present), m)
    assert a_log["clients"] == b_log["clients"] and all(torch.equal(a_out[k], b_out[k]) for k in a_out)
    _expect_error(lambda: CandidateDetector(partial, cfg, "shadow", "synthetic-fp").import_state(tracker.export_state()), ValueError)
    print("PASS history in score: references after probation, post-reference change scored by history and frozen, "
          "absence leaves state unchanged, uninformative updates earn no credit, checkpoint round-trip, mode mismatch rejected.")

    # 5. Replay uses each round's RECORDED global.
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "round_0001.npz"
        recorded = {"w": torch.randn(4, 3), "b": torch.randn(3), "count": torch.tensor(7)}
        submitted = [{"w": recorded["w"] + 0.02 * (i + 1), "b": recorded["b"].clone(), "count": torch.tensor(7)} for i in range(n)]
        write_capture(path, 1, recorded, submitted, list(range(n)), sizes, steps)
        raw = path.read_bytes()
        _, expected_norms = submission_deltas(recorded, submitted, list(range(n)), sizes, steps)
        replayer = CandidateDetector(core, cfg, "shadow", "synthetic-fp")
        diagnostic = replay_round(replayer, raw, _measurement(zeros, range(n), sizes, steps, expected_norms), recorded)
        assert [r["Full_Update_Norm"] for r in diagnostic["clients"]] == expected_norms
        assert all(math.isclose(v, 0.02 * (i + 1) * math.sqrt(12), rel_tol=1e-5) for i, v in enumerate(expected_norms))
        loaded = load_round_capture(io.BytesIO(raw), recorded)
        assert loaded[2:] == (list(range(n)), sizes, steps, 1)
    print("PASS replay: deltas use the recorded global of the same round; capture round-trip.")

    # 6. Calibration fitting contract.
    training = {"client_sizes": [10] * n, "prescribed_steps": [1] * n}
    fit_rows = []
    rng = np.random.default_rng(0)
    for seed in (1, 2, 3):
        for rnd in range(1, 7):
            for cid in range(n):
                fit_rows.append({"Seed": seed, "Round": rnd, "Client": cid, "Condition": "clean",
                                 "Phase": calibration_phase(rnd, 3), "Cohort_Score": float(rng.gamma(2.0)),
                                 "History_Score": float(rng.gamma(2.0)) if rnd > 3 else None, "Full_Update_Norm": 1.0 + rnd,
                                 "Client_Size": 10, "Optimizer_Steps": 1})
    fit_cfg = DetectorConfig(min_cohort=4, initial_rounds=3, min_phase_observations=10, client_profile=False)
    fitted = fit_core(fit_rows, fit_cfg, "fp", training, [1, 2, 3], 6, n, Q, C, 3)
    assert fitted["phases"]["initial"]["history_centre"] is None and fitted["phases"]["later"]["history_centre"] is not None
    relabel = [{**r, "Client": (r["Client"] + 2) % n} for r in fit_rows]
    assert fit_core(relabel, fit_cfg, "fp", training, [1, 2, 3], 6, n, Q, C, 3)["phases"] == {
        p: {**v, "in_sample_alerts_by_client": {str((int(c) + 2) % n): k for c, k in v["in_sample_alerts_by_client"].items()}}
        for p, v in fitted["phases"].items()}, "pooled thresholds must not depend on client identity"
    _expect_error(lambda: fit_core(fit_rows, DetectorConfig(min_cohort=4, initial_rounds=3, min_phase_observations=10),
                                   "fp", training, [1, 2, 3], 6, n, Q, C, 3), ValueError)
    profiled = fit_core(fit_rows, DetectorConfig(min_cohort=4, initial_rounds=3, min_phase_observations=10), "fp", training,
                        [1, 2, 3], 6, n, Q, C, 3, {c: torch.zeros((2, Q, C), dtype=torch.float64) for c in range(n)})
    assert len(profiled["client_profiles"]) == n and np.asarray(profiled["client_profiles"]["0"]).shape == (2, Q, C)
    _expect_error(lambda: fit_core(fit_rows, fit_cfg, "fp", training, [1, 2], 6, n, Q, C, 2), ValueError)
    for bad in (fit_rows + fit_rows[:1], [r for r in fit_rows if not (r["Seed"] == 3 and r["Client"] == 0 and r["Round"] <= 3)],
                [{**r, "Condition": "attacked"} if j == 0 else r for j, r in enumerate(fit_rows)]):
        _expect_error(lambda: fit_core(bad, fit_cfg, "fp", training, [1, 2, 3], 6, n, Q, C, 3), ValueError)
    gate_criteria = {"Replay_Min_Detection": 70.0, "Replay_Max_Poisoned_Retention": 20.0, "Replay_Latest_First_Alert": 3,
                     "Transfer_Max_FPR": 5.0, "Transfer_Min_Retention": 90.0}
    good = {"Completed": True, "Overall_Detection_Rate": 80.0, "Overall_Poisoned_Retention": 15.0,
            "First_Detected_Poisoned_Participation_By_Client": {"2": 2, "6": 3}, "Overall_Update_FPR": 2.0}
    assert development_live_gate(good, gate_criteria)["Passed"]
    for broken in ({**good, "Overall_Poisoned_Retention": 20.5}, {**good, "Overall_Update_FPR": 5.1},
                   {**good, "First_Detected_Poisoned_Participation_By_Client": {"2": 2, "6": None}},
                   {"Completed": False, "Reason": "flagged fraction"}):
        assert not development_live_gate(broken, gate_criteria)["Passed"]
    print("PASS calibration: pooled phase thresholds independent of client labels, rows from a held-out seed rejected "
          "by a fold fit, duplicate/attacked/under-covered observations rejected.")

    # 7. Measurement on a tiny model.
    model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
    fold_a = (torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 2, 2]))
    fold_b = (torch.randn(6, 4) + 10, torch.tensor([0, 0, 1, 1, 2, 2]))
    tiny = MeasurementConfig(probe_levels=(-2.0, 2.0), probe_widths=(1,), random_groups=0)
    profiler = ProbeProfiler(model, [fold_a, fold_b], tiny)
    base_state = copy.deepcopy(model.state_dict())
    assert np.all(profiler.measure(base_state, [copy.deepcopy(base_state) for _ in range(3)]) == 0)
    moved = {k: v + 0.1 for k, v in base_state.items()}
    measured = profiler.measure(base_state, [moved])
    assert measured.shape == (1, 2, 8, 3, 3) and np.isfinite(measured).all() and np.abs(measured).max() > 0
    assert all(torch.equal(model.state_dict()[k], base_state[k]) for k in base_state), "measurement changed the caller's model"
    assert build_probe_bank(4, tiny) == profiler.bank
    _expect_error(lambda: ProbeProfiler(model, [fold_a, fold_a], tiny), ValueError)
    print("PASS measurement: identical states give exactly zero, finite shapes, caller model untouched, deterministic bank, overlapping folds rejected.")


def _smoke_args(extra):
    common = ["--rounds", "6", "--initial-rounds", "3", "--local-epochs", "1", "--batch-size", "32", "--threads", "1",
              "--calibration-seeds", "11", "12", "13", "--attacker-id", "4", "--poison-rate", "0.4",
              "--trigger-features", "4", "5", "--trigger-value", "6", "--server-per-class", "12",
              "--probe-widths", "1", "2", "--random-groups", "0", "--admission-patience", "2", "--breadth-sources", "2",
              "--clean-min-macro-f1", "0", "--clean-min-class-recall", "0"]
    return parse_args(common + extra)


def _integration_check():
    """Full pipeline on a tiny synthetic protocol: controls, measurement, fit,
    transfer gate, shadow capture, development replay, live filter runs,
    resume equality and measurement reuse. Exercises software only."""
    base = _synthetic_protocol()
    data, clients = _synthetic_data()
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        first = tmp / "calibration_a"
        first.mkdir()
        code = execute(base, _smoke_args(["--stage", "calibrate", "--train-missing-controls", "--out", str(first)]), data, clients, first)
        calibration = read_json(first / "calibration" / "calibration.json")
        assert code == 3 and not calibration["pre_live_gate"]["Passed"], "a gate without an attack replay must not pass"
        assert any("No recorded attacked trajectory" in f for f in calibration["pre_live_gate"]["Failed_Criteria"])
        assert len(calibration["transfer"]) == 3 and all(t["Poisoned_Updates"] == 0 for t in calibration["transfer"])
        shadow_args = _smoke_args(["--stage", "evaluate", "--calibration", str(first / "calibration" / "calibration.json"),
                                   "--shield-mode", "shadow", "--condition", "attacked", "--seed", "21", "--out", str(tmp / "shadow")])
        context = build_context(base, shadow_args, data, clients)
        (tmp / "shadow").mkdir()
        shadow = run_live(base, shadow_args, data, clients, 21, "attacked", tmp / "shadow", calibration, context)
        assert shadow["Completed"] and shadow["Status"] == "SHADOW_ONLY_NO_MITIGATION"
        assert shadow["Overall_Unpoisoned_Retention"] == 100.0, "shadow mode must apply nominal FedAvg weights"
        second = tmp / "calibration_b"
        second.mkdir()
        code = execute(base, _smoke_args(["--stage", "calibrate", "--capture-source", str(first / "calibration_controls"),
                                          str(tmp / "shadow"), "--out", str(second)]), data, clients, second)
        calibration_b = read_json(second / "calibration" / "calibration.json")
        attacked = [r for r in calibration_b["replays"] if r["Role"] == "development_attack_replay"]
        assert len(attacked) == 1 and attacked[0]["Poisoned_Updates"] == 6 and code in (0, 3)
        third = tmp / "calibration_c"
        third.mkdir()
        execute(base, _smoke_args(["--stage", "calibrate", "--capture-source", str(first / "calibration_controls"),
                                   str(tmp / "shadow"), "--reuse-measurements", str(second), "--out", str(third)]), data, clients, third)
        reuse = read_json(third / "calibration" / "measurement_reuse.json")
        assert reuse["computed"] == 0 and reuse["reused"] == 4 * 6, "verified measurements must be reused without inference"
        calibration_c = read_json(third / "calibration" / "calibration.json")
        assert calibration_c["phases"] == calibration_b["phases"], "reused measurements changed the calibration"
        live_args = _smoke_args(["--stage", "evaluate", "--calibration", str(second / "calibration" / "calibration.json"),
                                 "--condition", "clean", "--seed", "31", "--out", str(tmp / "live")])
        context = build_context(base, live_args, data, clients)
        (tmp / "live").mkdir()
        full = run_live(base, live_args, data, clients, 31, "clean", tmp / "live", calibration_b, context)
        interrupted_args = copy.deepcopy(live_args)
        (tmp / "interrupted").mkdir()
        try:
            run_live(base, interrupted_args, data, clients, 31, "clean", tmp / "interrupted", calibration_b, context, stop_after_round=2)
            interrupted_first = None
        except KeyboardInterrupt:
            interrupted_first = True
        interrupted_args.resume = True
        resumed = run_live(base, interrupted_args, data, clients, 31, "clean", tmp / "interrupted", calibration_b, context)
        if full["Completed"]:
            assert interrupted_first, "the simulated interruption did not occur"
            a = torch.load(tmp / "live" / "seed_31" / "medishield_clean" / "checkpoint.pt", weights_only=False)
            b = torch.load(tmp / "interrupted" / "seed_31" / "medishield_clean" / "checkpoint.pt", weights_only=False)
            assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), "resume changed the trajectory"
            assert _state_digest(a["detector"]) == _state_digest(b["detector"])
            assert [r["Flagged"] for r in a["clients"]] == [r["Flagged"] for r in b["clients"]]
        else:
            assert resumed["Status"] == full["Status"] == "STOPPED_DETECTOR_INVALID" and resumed["Stopped_Round"] == full["Stopped_Round"]
        attacked_args = _smoke_args(["--stage", "evaluate", "--calibration", str(second / "calibration" / "calibration.json"),
                                     "--condition", "attacked", "--seed", "31", "--out", str(tmp / "live")])
        attacked_args.resume = True
        live_attacked = run_live(base, attacked_args, data, clients, 31, "attacked", tmp / "live", calibration_b,
                                 build_context(base, attacked_args, data, clients))
        folder = tmp / "live" / "seed_31" / "medishield_attacked"
        if live_attacked["Completed"]:
            decisions = list(csv.DictReader((folder / "client_decisions.csv").open(encoding="utf-8")))
            assert sum(int(r["Is_Poisoned_Update"]) for r in decisions) == 6 and live_attacked["ASR"] is not None
        else:
            stop = read_json(folder / "detector_stop.json")
            assert stop["global_update_applied"] is False and stop["detector_state_committed"] is False
        _expect_error(lambda: run_live(base, live_args, data, clients, 31, "clean", tmp / "live", calibration_b, context), ValueError)
        if full["Completed"] and live_attacked["Completed"]:
            assert main(["--stage", "report", "--report-dir", str(tmp / "live" / "seed_31"), "--reference-dir",
                         str(tmp / "live" / "seed_31"), "--out", str(tmp / "report")]) == 0
            report = read_json(tmp / "report" / "paired_report.json")
            assert report["Comparison"]["Mean_ASR_Reduction_pp"] == 0.0
            assert report["Defended"]["Final_ASR"] == live_attacked["ASR"]
            assert math.isclose(report["Defended"]["Mean_ASR_Over_Attack_Rounds"], live_attacked["Mean_ASR_Over_Rounds"])
            assert (tmp / "report" / "paired_rounds.csv").exists()
        oracle_args = _smoke_args(["--stage", "oracle", "--condition", "attacked", "--seed", "31", "--out", str(tmp / "oracle")])
        sizes = [len(c) for c in clients]
        honest_share = sum(n for i, n in enumerate(sizes) if i != oracle_args.attacker_id) / sum(sizes)
        residual = run_oracle_reference(base, oracle_args, data, clients, 31, tmp / "oracle", "residual")
        renormalized = run_oracle_reference(base, oracle_args, data, clients, 31, tmp / "oracle", "renormalized")
        assert math.isclose(residual["Applied_Mass"], honest_share) and math.isclose(renormalized["Applied_Mass"], 1.0)
        assert json.loads(residual["Excluded_Clients"]) == [oracle_args.attacker_id]
        assert read_json(tmp / "oracle" / "seed_31" / "oracle_residual_attacked" / "config.json")["role"].endswith("NOT_A_DEFENCE")
        _expect_error(lambda: run_oracle_reference(base, oracle_args, data, clients, 31, tmp / "oracle", "residual"), ValueError)
        # Live-gate authorization: accepted only for a passed gate with the same attack and detector core.
        gate_dir = tmp / "gate"
        dev_seed = next(r["Seed"] for r in calibration_b["replays"] if r["Role"] == "development_attack_replay")
        gate_args = _smoke_args(["--stage", "evaluate", "--calibration", str(second / "calibration" / "calibration.json"),
                                 "--condition", "attacked", "--seed", "31", "--out", str(tmp / "unused")])
        (gate_dir / f"seed_{dev_seed}" / "medishield_attacked").mkdir(parents=True)
        write_json(gate_dir / "status.json", {"status": "DEVELOPMENT_LIVE_GATE_PASSED"})
        write_json(gate_dir / f"development_live_gate_seed{dev_seed}.json", {"Passed": True, "Reported": {}})
        gate_settings = {**vars(gate_args), "development_gate_run": True}
        write_json(gate_dir / f"seed_{dev_seed}" / "medishield_attacked" / "config.json",
                   {"settings": gate_settings, "mode": "filter", "condition": "attacked",
                    "calibration_sha256": calibration_b["artifact_sha256"], "source_sha256": file_hash(__file__)})
        assert verify_live_gate(gate_dir, calibration_b, gate_args)["Development_Seeds"] == [dev_seed]
        other_attack = copy.deepcopy(gate_args)
        other_attack.poison_rate = 0.9
        _expect_error(lambda: verify_live_gate(gate_dir, calibration_b, other_attack), ValueError)
        variant = verify_live_gate(gate_dir, calibration_b, other_attack, allow_attack_variant=True)
        assert variant["Descriptive_After_Failed_Gate"] is False
        # A FAILED gate never authorizes confirmatory filtering; only the labelled descriptive mode accepts it.
        write_json(gate_dir / "status.json", {"status": "DEVELOPMENT_LIVE_GATE_FAILED"})
        write_json(gate_dir / f"development_live_gate_seed{dev_seed}.json",
                   {"Passed": False, "Reported": {}, "Failed_Criteria": ["synthetic"]})
        _expect_error(lambda: verify_live_gate(gate_dir, calibration_b, gate_args), ValueError)
        described = verify_live_gate(gate_dir, calibration_b, gate_args, descriptive_failed_gate=True)
        assert described["Descriptive_After_Failed_Gate"] and "DESCRIPTIVE ONLY" in described["Authorized_By"]
        write_json(gate_dir / "status.json", {"status": "DEVELOPMENT_LIVE_GATE_PASSED"})
        write_json(gate_dir / f"development_live_gate_seed{dev_seed}.json", {"Passed": True, "Reported": {}})
        assert variant["Records"][0]["Attack_Settings_Differing_From_Gate"] == ["poison_rate"]
        other_core = copy.deepcopy(calibration_b)
        other_core["phases"]["later"]["alert_threshold"] += 1.0
        _expect_error(lambda: verify_live_gate(gate_dir, other_core, gate_args), ValueError)
        write_json(gate_dir / f"development_live_gate_seed{dev_seed}.json", {"Passed": False, "Reported": {}})
        _expect_error(lambda: verify_live_gate(gate_dir, calibration_b, gate_args), ValueError)
        empty_source = tmp / "not_a_capture"
        empty_source.mkdir()
        (empty_source / "config.json").write_text(json.dumps({"role": "something_else"}), encoding="utf-8")
        _expect_error(lambda: discover_trajectories([empty_source], context["training"], 6, context["folds"],
                                                    np.asarray(data["discovery"]), np.asarray(data["verification"])), ValueError)
        archive = bundle_results(tmp / "live")
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
        assert "local_only_files.json" in names and not any(n.endswith((".pt", ".npz")) for n in names)


def self_check():
    if not __debug__:
        raise RuntimeError("Run without python -O so assertions execute.")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    _unit_checks()
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            _integration_check()
    except BaseException:
        print(captured.getvalue())
        raise
    print("PASS integration (synthetic): controls trained and captured, measurements cached and reused, fit, "
          "leave-one-seed-out gate, shadow capture re-used as a recorded attacked trajectory, development replay, "
          "live filter runs with truth attached after decisions, interrupted-run resume identical, existing-run "
          "protection, results bundle excludes local checkpoints/captures.")
    print(f"ALL SELF-CHECKS PASSED in {time.perf_counter() - started:.1f}s. These establish software properties on "
          "synthetic inputs only; they do not establish attack robustness, calibration transfer or utility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
