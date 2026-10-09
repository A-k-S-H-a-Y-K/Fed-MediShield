"""Fed-MediShield research candidate. No robustness or novelty guarantee."""
import copy
import math
import torch

VERSION = "control_calibrated_candidate_4"


class DetectorCollapseError(RuntimeError):
    """Reject an invalid filtering round before it changes the global model."""
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.diagnostic = diagnostic


def validate(states, sizes):
    if not states or len(states) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("Need matching, nonempty client states and positive sizes.")
    first = states[0]
    for state in states:
        if list(state) != list(first):
            raise ValueError("Incompatible model keys.")
        for key, value in state.items():
            if value.shape != first[key].shape or value.dtype != first[key].dtype:
                raise ValueError(f"Incompatible tensor: {key}")
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite model: stopping, not multiplying NaN by zero.")


def fedavg_aggregate(states, sizes, multipliers=None):
    validate(states, sizes)
    weights = torch.tensor(sizes, dtype=torch.float64)
    if multipliers is not None:
        weights *= torch.as_tensor(multipliers, dtype=torch.float64)
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Invalid aggregation weights.")
    weights /= weights.sum()
    result = {}
    for key, template in states[0].items():
        if template.is_floating_point():
            result[key] = sum(
                s[key].double() * float(w) for s, w in zip(states, weights)
            ).to(template)
        else:
            if any(not torch.equal(s[key], template) for s in states):
                raise ValueError(f"Cannot average unequal nonfloating buffer {key}.")
            result[key] = template.clone()
    return result, weights.tolist()


def krum(states, sizes, f=1):
    return multi_krum(states, sizes, f=f, m=1)


def multi_krum(states, sizes, f=1, m=None):
    """Single-pass scores; uniformly average the m lowest-scoring models."""
    validate(states, sizes)
    n = len(states)
    if not isinstance(f, int) or f < 0 or n <= 2 * f + 2:
        raise ValueError("Krum requires a nonnegative integer f and n > 2f + 2.")
    neighbours = n - f - 2
    m = neighbours if m is None else m
    if not isinstance(m, int) or not 1 <= m <= neighbours:
        raise ValueError(
            f"This Multi-Krum implementation requires 1 <= m <= {neighbours}."
        )
    vectors = torch.stack([
        torch.cat([
            v.detach().cpu().double().flatten()
            for v in s.values() if v.is_floating_point()
        ])
        for s in states
    ])
    scores = []
    for i in range(n):
        distances = [
            (vectors[i] - vectors[j]).square().sum()
            for j in range(n) if j != i
        ]
        scores.append(torch.stack(distances).sort().values[:neighbours].sum())
    selected = torch.argsort(torch.stack(scores), stable=True)[:m].tolist()
    result, _ = fedavg_aggregate([states[i] for i in selected], [1] * m)
    return result, selected


def coordinate_aggregate(states, sizes, f=None):
    validate(states, sizes)
    n = len(states)
    if f is not None and (not isinstance(f, int) or f < 0 or 2 * f >= n):
        raise ValueError("Trimmed mean requires 0 <= 2f < n.")
    result = {}
    for key, template in states[0].items():
        if not template.is_floating_point():
            if any(not torch.equal(s[key], template) for s in states):
                raise ValueError(f"Unequal nonfloating buffer {key}.")
            result[key] = template.clone()
            continue
        values = torch.stack([s[key].double() for s in states])
        if f is None:
            value = torch.quantile(values, 0.5, dim=0)
        else:
            value = values.sort(dim=0).values[f:n - f].mean(dim=0)
        result[key] = value.to(template)
    return result, None  # Coordinate methods do not classify whole clients.


def trimmed_mean(states, sizes, f=1):
    return coordinate_aggregate(states, sizes, f=f)


def coordinate_median(states, sizes):
    return coordinate_aggregate(states, sizes)


def _float_vector(state):
    return torch.cat([v.detach().cpu().double().flatten() for v in state.values() if v.is_floating_point()])


def _from_vector(vector, template):
    result, offset = {}, 0
    for key, value in template.items():
        if value.is_floating_point():
            count = value.numel()
            result[key] = vector[offset:offset + count].view_as(value).to(value)
            offset += count
        else:
            result[key] = value.clone()
    return result


def rfa(states, sizes, max_iter=4, smoothing=1e-5, ftol=1e-6):
    """RFA (Pillutla, Kakade and Harchaoui, IEEE TSP 2022): data-size-weighted geometric median of the
    client models by the smoothed Weiszfeld algorithm (defaults of the authors' reference code).
    The median is translation-equivariant, so median(models) = global + median(updates)."""
    validate(states, sizes)
    for key, template in states[0].items():
        if not template.is_floating_point() and any(not torch.equal(s[key], template) for s in states):
            raise ValueError(f"Unequal nonfloating buffer {key}.")
    points = torch.stack([_float_vector(s) for s in states])
    alphas = torch.tensor(sizes, dtype=torch.float64)
    alphas /= alphas.sum()
    median = (alphas[:, None] * points).sum(0)

    def objective(z):
        return float((alphas * (points - z).norm(dim=1)).sum())

    value, weights = objective(median), alphas.clone()
    for _ in range(max_iter):
        distances = (points - median).norm(dim=1).clamp_min(smoothing)
        weights = alphas / distances
        weights /= weights.sum()
        median = (weights[:, None] * points).sum(0)
        previous, value = value, objective(median)
        if abs(previous - value) <= ftol * value:
            break
    return _from_vector(median, states[0]), weights.tolist()


def _hausdorff(a, b):
    from scipy.spatial.distance import directed_hausdorff
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])


def fed_adh_distances(global_state, states):
    """Fed-ADH (Pathak et al., Computer Networks 2026, Eq. 5-7): Hausdorff distance d_H(w_G, |w_i - w_G|).
    The paper does not state how parameter tensors become point sets. Interpretation used here: per
    floating tensor, the rows of a 2-D view (out_features x rest; 1-D tensors as column points) form the
    point set; the symmetric Hausdorff distance is computed per tensor and summed over tensors."""
    distances = []
    for state in states:
        total = 0.
        for key, g in global_state.items():
            if not g.is_floating_point():
                continue
            g2 = g.detach().cpu().double().reshape(g.shape[0], -1) if g.dim() > 1 else g.detach().cpu().double()[:, None]
            u = (state[key].detach().cpu().double() - g.detach().cpu().double()).abs()
            u2 = u.reshape(u.shape[0], -1) if u.dim() > 1 else u[:, None]
            total += _hausdorff(g2.numpy(), u2.numpy())
        distances.append(total)
    return distances


def fed_adh(global_state, states, sizes):
    """Each round, exclude the single client with the largest Hausdorff distance (Algorithm 1) and
    FedAvg the rest. The paper removes the arg-max client every round, so one client is always dropped."""
    validate(states, sizes)
    distances = fed_adh_distances(global_state, states)
    dropped = max(range(len(states)), key=lambda i: (distances[i], -i))
    kept = [i for i in range(len(states)) if i != dropped]
    result, weights = fedavg_aggregate([states[i] for i in kept], [sizes[i] for i in kept])
    full = [0.] * len(states)
    for i, w in zip(kept, weights):
        full[i] = w
    return result, dropped, distances, full


class BehavioralShield:
    """Experimental trusted-data behavioral detector, SHADOW by default.

    Assumes stable class meanings and trusted, labeled calibration records.
    Receives neither attacker identities, triggers, nor validation/test data.
    Known target_class is a threat-model assumption (Benign evasion here).
    Clean-input saliency need NOT expose a dormant backdoor.
    """

    def __init__(
        self, model, calibration_X, calibration_y, target_class=0,
        mode="shadow", components="joint", mad_k=3.0,
        scale_floor=0.01, penalty=1.0,
    ):
        if mode not in ("shadow", "filter") or components not in (
            "joint", "saliency", "response"
        ):
            raise ValueError("Unknown detector mode/components.")
        if (
            not all(math.isfinite(v) for v in (mad_k, scale_floor, penalty))
            or mad_k <= 0 or scale_floor <= 0 or penalty < 0
        ):
            raise ValueError("Invalid detector hyperparameters.")
        self.model = copy.deepcopy(model).cpu().eval()
        self.model.requires_grad_(False)
        self.target = int(target_class)
        keep = calibration_y != self.target
        self.X = calibration_X[keep].detach().cpu().float().clone()
        self.y = calibration_y[keep].detach().cpu().long().clone()
        if len(self.y) == 0:
            raise ValueError("Calibration requires non-target records.")
        self.classes = sorted(self.y.unique().tolist())
        self.mode, self.components = mode, components
        self.k, self.floor, self.penalty = mad_k, scale_floor, penalty
        self.last_diagnostics = {}

    def _signature(self, state):
        self.model.load_state_dict(state)
        self.model.eval()
        with torch.enable_grad():
            x = self.X.clone().requires_grad_(True)
            logits = self.model(x)
            other = [c for c in range(logits.shape[1]) if c != self.target]
            # Log-odds margin is invariant to a common shift of all logits.
            margin = (
                logits[:, self.target]
                - torch.logsumexp(logits[:, other], dim=1)
            )
            gradient, = torch.autograd.grad(margin.sum(), x)
            probability = logits.softmax(dim=1)[:, self.target].detach()
        saliency = torch.stack([
            gradient[self.y == c].abs().mean(dim=0).log1p()
            for c in self.classes
        ]).double()
        response = torch.stack([
            probability[self.y == c].mean() for c in self.classes
        ]).double()
        if not torch.isfinite(saliency).all() or not torch.isfinite(response).all():
            raise ValueError("Non-finite probe signature.")
        return saliency, response

    def aggregate(self, global_state, client_states, client_sizes):
        validate(client_states, client_sizes)
        base_saliency, base_response = self._signature(global_state)
        signatures = [self._signature(state) for state in client_states]
        saliency = torch.stack([s[0] - base_saliency for s in signatures])
        response = torch.stack([s[1] - base_response for s in signatures])
        saliency_center = torch.quantile(saliency, 0.5, dim=0)
        response_center = torch.quantile(response, 0.5, dim=0)
        saliency_deviation = saliency - saliency_center
        raw_saliency = saliency_deviation.square().mean(dim=(1, 2)).sqrt()
        raw_response = (response - response_center).abs().mean(dim=1)
        raw = torch.stack([raw_saliency, raw_response], dim=1)
        med = torch.quantile(raw, 0.5, dim=0)
        mad = torch.quantile((raw - med).abs(), 0.5, dim=0)
        scale = (1.4826 * mad).clamp_min(self.floor)
        normalized = ((raw - med) / scale).clamp_min(0)
        if self.components == "joint":
            scores = normalized.max(dim=1).values
        else:
            scores = normalized[:, 0 if self.components == "saliency" else 1]
        flags = scores > self.k

        # No unbounded suspicion memory or permanent exclusion in this candidate.
        factors = torch.exp(
            -self.penalty * (scores - self.k).clamp_min(0)
        ).clamp_min(0.01)
        if self.mode == "shadow":
            factors = torch.ones_like(scores)

        result, weights = fedavg_aggregate(
            client_states, client_sizes, factors
        )
        if self.mode == "shadow":
            reference, _ = fedavg_aggregate(client_states, client_sizes)
            if any(not torch.equal(result[k], reference[k]) for k in result):
                raise AssertionError("Shadow mode changed FedAvg aggregation.")

        feature_ranks = saliency_deviation.abs().mean(dim=1)
        self.last_diagnostics = {
            "scores": scores.tolist(),
            "flagged": flags.tolist(),
            "raw_saliency": raw_saliency.tolist(),
            "raw_response": raw_response.tolist(),
            "saliency_z": normalized[:, 0].tolist(),
            "response_z": normalized[:, 1].tolist(),
            "threshold": self.k,
            "multipliers": factors.tolist(),
            "aggregation_weights": weights,
            "top_features": feature_ranks.topk(
                min(5, feature_ranks.shape[1]), dim=1
            ).indices.tolist(),
            "raw_thresholds": (med + self.k * scale).tolist(),
        }
        return result, (~flags).nonzero(as_tuple=True)[0].tolist()


def flatten(state, keys=None):
    keys = list(state) if keys is None else keys
    return torch.cat([state[k].detach().cpu().double().reshape(-1)
                      for k in keys if state[k].is_floating_point()])


def krum_scores(vectors, f):
    n = len(vectors)
    if n <= 2 * f + 2 or f < 0:
        raise ValueError("Krum screening requires n > 2f+2; f is a configured budget, not an attacker ID.")
    result = []
    for i in range(n):
        distances = torch.stack([(vectors[i] - vectors[j]).square().sum()
                                 for j in range(n) if i != j])
        result.append(distances.sort().values[:n-f-2].sum())
    return torch.stack(result)


def robust_upper_z(values):
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite distances/scores; aggregation cannot continue.")
    median = torch.quantile(values, 0.5)
    mad = torch.quantile((values - median).abs(), 0.5)
    # Relative numerical floor, not a fitted security threshold.
    floor = max(1e-12, float(median.abs()) * 1e-6)
    scale = max(float(1.4826 * mad), floor)
    return ((values - median) / scale).clamp_min(0), float(median), scale


class LegacyHistoryShield:
    """Experimental peer/history admission with bounded sample-share influence.

    Krum is an existing algorithm (Blanchard et al., 2017). The additions here
    are hypotheses, NOT a security proof or established novelty. A shortlist
    can contain attackers. Consistently passing attacks can contaminate history.
    Default adaptive selection imposes NO fixed shortlist on aggregation.
    This corrects retention/amplification mechanics, not detector blindness.
    No labels, sample features, validation data, trigger, or attacker IDs enter
    this class. Stable client IDs and individual model updates are assumed.
    """
    def __init__(self, final_keys, f=1, core_size=0, mode="filter",
                 spatial_scope="whole", history_policy="hard", k=3.0,
                 history_limit=1.5, rho=0.1, admission_patience=2,
                 recovery_patience=6, trust_cutoff=0.5,
                 risk_decay=0.8, risk_gain=1.0, clip_factor=1.0,
                 selection_rule="adaptive", weight_rule="capped", clip_rule="mad"):
        self.final_keys = tuple(final_keys)
        self.f, self.core_size, self.mode = int(f), int(core_size), mode
        self.scope, self.policy = spatial_scope, history_policy
        self.k, self.history_limit, self.rho = float(k), float(history_limit), float(rho)
        self.patience, self.recovery = int(admission_patience), int(recovery_patience)
        self.cutoff, self.decay, self.gain = float(trust_cutoff), float(risk_decay), float(risk_gain)
        self.clip_factor = float(clip_factor)
        self.selection_rule, self.weight_rule, self.clip_rule = selection_rule, weight_rule, clip_rule
        self.memory = {}
        self.last_diagnostics = {}
        values = (k, history_limit, rho, trust_cutoff, risk_decay, risk_gain, clip_factor)
        if not all(math.isfinite(v) for v in values) or risk_gain < 0 or clip_factor < 0:
            raise ValueError("History/penalty parameters must be finite with nonnegative gain/clipping.")
        if not self.final_keys or len(set(self.final_keys)) != len(self.final_keys) or f < 0 or core_size < 0:
            raise ValueError("Need unique final-layer keys and nonnegative f/core_size.")
        if mode not in ("shadow", "filter") or spatial_scope not in ("whole", "final"):
            raise ValueError("Invalid shield mode or spatial scope.")
        if selection_rule not in ("adaptive", "topk") or weight_rule not in ("capped", "renormalized"):
            raise ValueError("Unknown selection/weight rule.")
        if clip_rule not in ("mad", "median_core"):
            raise ValueError("Unknown clipping rule.")
        if core_size and selection_rule != "topk":
            raise ValueError("--core-size applies only with --selection-rule topk.")
        if history_policy not in ("none", "ungated", "proportional", "hard"):
            raise ValueError("Unknown history policy.")
        if not (0 < rho <= 1 and 0 < trust_cutoff <= 1 and 0 <= risk_decay < 1):
            raise ValueError("Invalid reference/risk parameters.")
        if k <= 0 or history_limit <= 0 or admission_patience < 1 or recovery_patience < admission_patience:
            raise ValueError("Invalid thresholds or probation lengths.")

    def aggregate(self, global_state, states, sizes, client_ids, evidence=None):
        validate(states, sizes)
        if len(client_ids) != len(states) or len(set(client_ids)) != len(states):
            raise ValueError("Participant IDs must be unique and match states.")
        validate([global_state] + list(states), [1] + list(sizes))
        if any(k not in global_state or not global_state[k].is_floating_point() for k in self.final_keys):
            raise ValueError("Final-layer keys must name floating tensors in the shared model.")
        n = len(states)
        deltas = [{k: s[k] - global_state[k] for k in s if s[k].is_floating_point()}
                  for s in states]
        whole = torch.stack([flatten(d) for d in deltas])
        final = torch.stack([flatten(d, self.final_keys) for d in deltas])
        if evidence is None:
            scores = krum_scores(whole if self.scope == "whole" else final, self.f)
            m = self.core_size or n - self.f - 2
            if not 1 <= m <= n - self.f - 2:
                raise ValueError("core_size must be between 1 and n-f-2 (0 chooses n-f-2).")
            order = torch.argsort(scores, stable=True)
            core, anchor = set(order[:m].tolist()), int(order[0])
            distances = torch.linalg.vector_norm(final - final[anchor], dim=1)
            whole_z, whole_med, whole_scale = robust_upper_z(scores)
            layer_z, layer_med, layer_scale = robust_upper_z(distances)
            peer_score = torch.maximum(whole_z, layer_z)
        else:
            if len(evidence) != n or self.selection_rule != "adaptive" or self.clip_rule != "mad":
                raise ValueError("Probe evidence requires adaptive participation and MAD norm clipping.")
            # No Krum selector or anchor is hidden inside the redesigned defence.
            peer_score = self.k * torch.tensor([d["Audit_Score"] for d in evidence], dtype=torch.float64)
            if not torch.isfinite(peer_score).all() or (peer_score < 0).any():
                raise ValueError("Invalid probe evidence.")
            scores = torch.zeros(n, dtype=torch.float64)
            order, core, anchor, m = torch.arange(n), set(range(n)), 0, n
            whole_med = whole_scale = layer_med = layer_scale = None
        norms = torch.linalg.vector_norm(whole, dim=1)
        if self.clip_rule == "median_core":
            radius = max(float(torch.quantile(norms[order[:m]], 0.5)), 1e-12)
        else:
            # Avoid clipping every naturally larger honest update at the median.
            # MAD is a heuristic here; a majority of honest clients is not proof
            # that their non-IID norm distribution is unimodal.
            _, norm_median, norm_scale = robust_upper_z(norms)
            radius = max(norm_median + self.k * norm_scale, 1e-12)
        clipping = (radius * self.clip_factor / norms.clamp_min(1e-12)).clamp_max(1)
        if self.clip_factor == 0:
            clipping = torch.ones_like(norms)
        details, factors = [], []
        for position, cid in enumerate(client_ids):
            u = final[position]
            norm = float(torch.linalg.vector_norm(u))
            informative = norm > 1e-10
            old = self.memory.get(int(cid), {"mu": None, "nu": 0., "risk": 0.,
                                            "streak": 0, "peer_streak": 0, "buffer": []})
            item = copy.deepcopy(old)
            ready = old["mu"] is not None and self.policy != "none"
            temporal = 0.0
            if ready and informative:
                angular = 0.0
                if float(old["mu"].norm()) > 1e-10:
                    cosine = float(torch.dot(u, old["mu"]) /
                                   (u.norm() * old["mu"].norm()).clamp_min(1e-30))
                    angular = 1 - max(-1., min(1., cosine))
                temporal = abs(math.log((norm + 1e-12) / (old["nu"] + 1e-12))) + angular
            peer_bad = float(peer_score[position]) > self.k
            temporal_bad = ready and informative and temporal > self.history_limit
            eligible = self.selection_rule == "adaptive" or position in core
            coverage = evidence is None or bool(evidence[position]["Audit_Complete"])
            peer_good = eligible and not peer_bad and informative and coverage
            item["peer_streak"] = old["peer_streak"] + 1 if peer_good else 0
            qualifying = peer_good and not temporal_bad
            item["streak"] = old["streak"] + 1 if qualifying else 0
            item["buffer"] = (old["buffer"] + [u.clone()])[-self.recovery:] if peer_good else []
            # Sustained fresh peer agreement permits risk decay against a stale
            # personal reference. Absence is NOT an observation or recovery.
            recovery_evidence = False
            recovery_vector, recovery_norm = None, None
            if peer_good and item["peer_streak"] >= self.recovery:
                candidates = torch.stack(item["buffer"])
                recovery_vector = torch.quantile(candidates, 0.5, dim=0)
                candidate_norms = candidates.norm(dim=1)
                recovery_norm = float(torch.quantile(candidate_norms, 0.5))
                if float(recovery_vector.norm()) > 1e-10 and recovery_norm > 1e-10:
                    angles = 1 - (candidates @ recovery_vector / (
                        candidate_norms * recovery_vector.norm()).clamp_min(1e-30)).clamp(-1, 1)
                    shifts = ((candidate_norms + 1e-12) / (recovery_norm + 1e-12)).log().abs() + angles
                    # Repeated peer-screened observations must also agree with
                    # each other. Consistent poisoning can still pass this gate.
                    recovery_evidence = bool((shifts <= self.history_limit).all())
            excess = max(0., float(peer_score[position]) / self.k - 1.)
            if temporal_bad and not recovery_evidence:
                excess = max(excess, temporal / self.history_limit - 1.)
            if excess > 0:
                previous_risk = old["risk"] if self.decay > 0 else 0.
                increment = min(excess, 10.)**2
                if evidence is not None:
                    increment = max(1., increment)
                # Preserve the old control unchanged; the new detector supplies
                # an explicit, smaller cap. A current alert still gets zero weight.
                cap = self.risk_cap if evidence is not None and hasattr(self, "risk_cap") else 20.
                item["risk"] = min(cap, previous_risk + self.gain * increment)
            elif qualifying or recovery_evidence:
                item["risk"] = self.decay * old["risk"]
            # Missing, uninformative or nonqualifying observations do not
            # automatically rehabilitate a client.
            trust = math.exp(-item["risk"])
            admitted, recovered = False, False
            if informative and self.policy != "none":
                if self.policy in ("ungated", "proportional"):
                    admitted = True
                else:
                    admitted = qualifying and trust >= self.cutoff and item["streak"] >= self.patience
                    recovered = (ready and temporal_bad and recovery_evidence and trust >= self.cutoff)
                if recovered:
                    # A bounded EMA step, not an immediate overwrite by the
                    # latest update or by an elapsed-time timeout.
                    item["mu"] = old["mu"] + self.rho * (recovery_vector - old["mu"])
                    item["nu"] = old["nu"] + self.rho * (recovery_norm - old["nu"])
                    item["streak"] = 0
                elif admitted:
                    rate = self.rho * trust if self.policy == "proportional" else self.rho
                    if old["mu"] is None and self.policy == "hard":
                        probation = torch.stack(item["buffer"][-self.patience:])
                        item["mu"] = torch.quantile(probation, 0.5, dim=0)
                        item["nu"] = float(torch.quantile(probation.norm(dim=1), 0.5))
                        if float(item["mu"].norm()) <= 1e-10:
                            item["mu"], item["nu"], admitted = None, 0., False
                    else:
                        item["mu"] = u.clone() if old["mu"] is None else old["mu"] + rate * (u - old["mu"])
                        item["nu"] = norm if old["mu"] is None else old["nu"] + rate * (norm - old["nu"])
            if self.policy == "none":
                item["mu"], item["nu"] = None, 0.
            self.memory[int(cid)] = item
            # Current confirmed alerts receive zero weight; no positive leakage floor.
            blocked = evidence is not None and (peer_bad or temporal_bad)
            factors.append(trust if eligible and not blocked else 0.)
            details.append({
                "Reference_Core": int(position in core), "Aggregation_Eligible": int(eligible),
                "Krum_Score": float(scores[position]),
                "Peer_Score": float(peer_score[position]), "History_Score": temporal,
                "Score": max(float(peer_score[position]) / self.k, temporal / self.history_limit),
                "Flagged": int(peer_bad or temporal_bad), "Informative": int(informative),
                "Trust": trust, "Risk": item["risk"], "History_Ready_Before": int(ready),
                "History_Admitted": int(admitted), "History_Recovered": int(recovered),
                "Reference_Changed": int(admitted or recovered),
                "Peer_Flagged": int(peer_bad), "History_Flagged": int(temporal_bad),
                "Recovery_Evidence": int(recovery_evidence),
                "Gate_Reason": ("peer_alert" if peer_bad else "history_alert" if temporal_bad
                    else "outside_topk" if not eligible else "uninformative" if not informative else "audit_not_run" if not coverage
                    else "low_trust" if trust < self.cutoff else "probation" if item["streak"] < self.patience
                    else "qualified"),
                "Qualifying_Streak": item["streak"], "Peer_Streak": item["peer_streak"],
                "Clip_Factor": float(clipping[position]), "Krum_Median": whole_med,
                "Krum_Scale": whole_scale, "Layer_Median": layer_med, "Layer_Scale": layer_scale,
            })
        if self.mode == "shadow":
            result, weights = fedavg_aggregate(states, sizes)
            clipping = torch.ones_like(norms)
        else:
            nominal = torch.tensor(sizes, dtype=torch.float64)
            nominal /= nominal.sum()
            weights = nominal * torch.tensor(factors, dtype=torch.float64)
            if self.weight_rule == "renormalized" and float(weights.sum()) > 0:
                weights /= weights.sum()
            effective = weights * clipping
            if not torch.isfinite(effective).all() or (effective < 0).any() or float(effective.sum()) > 1 + 1e-12:
                raise ValueError("Invalid effective aggregation coefficients.")
            if self.weight_rule == "capped" and bool((effective > nominal + 1e-12).any()):
                raise AssertionError("A client exceeded its original sample share.")
            # Retain discarded mass at the PREVIOUS global model. Do not
            # amplify surviving clients by renormalizing an accepted subset.
            # If all coefficients are zero, keep the previous global model.
            if bool((weights == nominal).all()) and bool((clipping == 1).all()):
                result, _ = fedavg_aggregate(states, sizes)
            else:
                result = {}
                for key, value in global_state.items():
                    if value.is_floating_point():
                        delta = sum(float(w) * (s[key].double() - value.double())
                                    for w, s in zip(effective, states))
                        result[key] = (value.double() + delta).to(value)
                    else:
                        if any(not torch.equal(s[key], value) for s in states):
                            raise ValueError(f"Unequal nonfloating global/client buffer {key}.")
                        result[key] = value.clone()
            weights = weights.tolist()
        residual = max(0., 1. - sum(w * float(c) for w, c in zip(weights, clipping)))
        for i, row in enumerate(details):
            row.update({"Aggregation_Weight": weights[i],
                        "Effective_Delta_Multiplier": weights[i] * float(clipping[i]),
                        "Applied_Clip_Factor": float(clipping[i]), "Shadow": int(self.mode == "shadow"),
                        "Server_Residual_Weight": residual,
                        "Weight_Amplification": weights[i] / (sizes[i] / sum(sizes)),
                        "Effective_Weight_Amplification": weights[i] * float(clipping[i]) / (sizes[i] / sum(sizes))})
        self.last_diagnostics = {"clients": details, "anchor_client": int(client_ids[anchor]),
                                 "radius": radius, "core_size": m}
        return result, self.last_diagnostics


class SparseProbeAudit:
    """Full-search calibration against clean-data-trained surrogate updates.

    The surrogates use discovery records, never the confirmation/validation
    labels. They are NOT certified honest-client distributions. Calibration is
    an empirical envelope, not a p-value, FPR guarantee, or proof of cleanliness.
    A previously poisoned global model can contaminate every surrogate's start.
    """
    def __init__(self, model, discovery_X, discovery_y, verification_X, verification_y,
                 classes, seed=2026, steps=16, restarts=2, fraction=.12,
                 bound=8., learning_rate=.2, min_gain=.10,
                 bank_size=24, enabled=True, control_steps=16,
                 control_learning_rate=.001, control_neighbors=3,
                 control_margin=.02, control_dominance=(.6, .9)):
        if steps < 1 or restarts < 1 or bank_size < 0 or classes < 2:
            raise ValueError("Invalid probe search sizes.")
        numbers = (fraction, bound, learning_rate, min_gain, control_learning_rate,
                   control_margin, *control_dominance)
        if not all(math.isfinite(v) for v in numbers) or not (
                0 < fraction <= 1 and bound > 0 and learning_rate > 0
                and 0 < min_gain <= 2 and control_learning_rate > 0
                and 0 < control_margin <= 2 and control_steps >= 1):
            raise ValueError("Invalid probe search bounds/thresholds.")
        if not control_dominance or len(set(control_dominance)) != len(control_dominance) or not all(
                1 / classes < d < 1 for d in control_dominance):
            raise ValueError("Control dominance values must be distinct and between 1/classes and 1.")
        if not 1 <= control_neighbors < 1 + classes * len(control_dominance):
            raise ValueError("Need positive control_neighbors smaller than the number of controls.")
        self.template = copy.deepcopy(model).cpu().eval().requires_grad_(False)
        self.DX, self.Dy = discovery_X.detach().cpu().float().clone(), discovery_y.detach().cpu().long().clone()
        self.VX, self.Vy = verification_X.detach().cpu().float().clone(), verification_y.detach().cpu().long().clone()
        if self.DX.ndim != 2 or self.VX.ndim != 2 or self.DX.shape[1] != self.VX.shape[1]:
            raise ValueError("Probe feature dimensions do not match.")
        if self.Dy.shape != (len(self.DX),) or self.Vy.shape != (len(self.VX),):
            raise ValueError("Trusted labels must be one-dimensional and match the records.")
        if any(bool(((y < 0) | (y >= classes)).any()) for y in (self.Dy, self.Vy)):
            raise ValueError("Unknown trusted label.")
        self.classes, self.seed, self.steps, self.restarts = classes, seed, steps, restarts
        self.budget = max(1, math.ceil(fraction * self.DX.shape[1]))
        self.bound, self.lr, self.min_gain = float(bound), float(learning_rate), float(min_gain)
        self.bank_size, self.enabled = bank_size, enabled
        self.control_steps, self.control_lr = int(control_steps), float(control_learning_rate)
        self.neighbors, self.margin = int(control_neighbors), float(control_margin)
        self.profiles = [{"name": "balanced", "probabilities": [1. / classes] * classes}]
        for d in control_dominance:
            for c in range(classes):
                probabilities = [(1. - d) / (classes - 1)] * classes
                probabilities[c] = float(d)
                self.profiles.append({"name": f"class_{c}_mass_{d:g}", "probabilities": probabilities})
        self.round, self.bank, self.last_probes = 0, [], []
        self.last_calibration = {}
        self.folds = torch.zeros(len(self.Vy), dtype=torch.long)
        for c in range(classes):
            discovery = (self.Dy == c).nonzero(as_tuple=True)[0]
            verification = (self.Vy == c).nonzero(as_tuple=True)[0]
            if len(discovery) < 2 or len(verification) < 4:
                raise ValueError("Each class needs >=2 discovery and >=4 confirmation records.")
            self.folds[verification[1::2]] = 1
        if not torch.isfinite(self.DX).all() or not torch.isfinite(self.VX).all():
            raise ValueError("Nonfinite trusted probes.")

    def _model(self, state):
        model = copy.deepcopy(self.template)
        model.load_state_dict(state)
        return model.eval().requires_grad_(False)

    def _controls(self, global_state):
        """Same global start; full-discovery weighted loss, AdamW, fixed steps.

        No client class counts, malicious IDs, or actual trigger enter this
        training. A profile changes the loss mass assigned to each true class.
        This models label skew only; device/domain skew is not simulated here.
        """
        result = []
        counts = torch.bincount(self.Dy, minlength=self.classes).float()
        for index, profile in enumerate(self.profiles):
            model = self._model(global_state).train().requires_grad_(True)
            probability = torch.tensor(profile["probabilities"], dtype=torch.float32)
            row_weight = probability[self.Dy] / counts[self.Dy]
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed + self.round * 100003 + index * 3571)
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.control_lr, weight_decay=1e-4)
                with torch.enable_grad():
                    for _ in range(self.control_steps):
                        optimizer.zero_grad(set_to_none=True)
                        losses = torch.nn.functional.cross_entropy(model(self.DX), self.Dy, reduction="none")
                        loss = (losses * row_weight).sum()
                        if not torch.isfinite(loss):
                            raise ValueError("Nonfinite clean-control loss.")
                        loss.backward()
                        optimizer.step()
            model.eval().requires_grad_(False)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite clean-control parameters.")
            result.append(model)
        return result

    @torch.no_grad()
    def _signature(self, model):
        probabilities = self._probabilities(model, self.DX)
        return torch.stack([probabilities[self.Dy == c].mean(0)
                            for c in range(self.classes)]).flatten()

    @staticmethod
    def transform(X, probe):
        return X * (1 - probe["mask"]) + probe["pattern"] * probe["mask"]

    def _search(self, candidate, base, target):
        X = self.DX[self.Dy != target]
        with torch.no_grad():
            clean_difference = (candidate(X).softmax(1)[:, target] - base(X).softmax(1)[:, target]).mean()
        best, best_gain = None, -math.inf
        for restart in range(self.restarts):
            # Same starting probes for each client, independent of client order.
            generator = torch.Generator().manual_seed(self.seed + self.round * 100003 + target * 1009 + restart * 7919)
            logits = torch.nn.Parameter(.05 * torch.randn(X.shape[1], generator=generator))
            pattern = torch.nn.Parameter((2 * torch.rand(X.shape[1], generator=generator) - 1) * self.bound)
            optimizer = torch.optim.Adam([logits, pattern], lr=self.lr)
            with torch.enable_grad():
                for step in range(self.steps):
                    optimizer.zero_grad(set_to_none=True)
                    soft = logits.sigmoid()
                    soft = soft * (self.budget / soft.sum().clamp_min(1e-12)).clamp_max(1.)
                    hard = torch.zeros_like(soft)
                    hard[torch.argsort(soft, descending=True, stable=True)[:self.budget]] = 1.
                    # First explore smoothly, then optimize the actual sparse mask
                    # with a straight-through gradient. This is a heuristic search.
                    mask = soft if step < self.steps // 2 else hard + soft - soft.detach()
                    changed = X * (1 - mask) + pattern.clamp(-self.bound, self.bound) * mask
                    cp, gp = candidate(changed).log_softmax(1), base(changed).log_softmax(1)
                    loss = -(cp[:, target] - gp[:, target]).mean()
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite probe objective.")
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        pattern.clamp_(-self.bound, self.bound)
                        logits.clamp_(-8., 8.)
                with torch.no_grad():
                    mask = torch.zeros_like(logits)
                    mask[torch.argsort(logits, descending=True, stable=True)[:self.budget]] = 1.
                    probe = {"target": int(target), "mask": mask.detach().clone(),
                             "pattern": pattern.detach().clone(), "created_round": self.round}
                    changed = self.transform(X, probe)
                    gain = float((candidate(changed).softmax(1)[:, target] - base(changed).softmax(1)[:, target]).mean() - clean_difference)
                if gain > best_gain:
                    best, best_gain = probe, gain
        best["discovery_gain"] = best_gain
        return best

    @staticmethod
    @torch.no_grad()
    def _probabilities(model, X):
        return torch.cat([model(X[i:i+4096]).softmax(1) for i in range(0, len(X), 4096)])

    @staticmethod
    def calibrate(gains, source_support, signatures, client_count,
                  neighbors, min_gain, margin):
        """Calibrate the MAXIMUM statistic, not each probe separately.

        Real clients and clean surrogates undergo the same frozen interventions.
        Each surrogate excludes itself when selecting its comparison controls.
        Neighbours are selected from clean discovery-output signatures only.
        """
        if gains.ndim != 3 or gains.shape[2] != 2 or gains.shape[1] == 0:
            raise ValueError("Expected model x probe x two-fold confirmation gains.")
        if source_support.shape != gains.shape or source_support.dtype != torch.bool:
            raise ValueError("Confirmation support must be a matching boolean tensor.")
        if signatures.ndim != 2 or signatures.shape[0] != gains.shape[0]:
            raise ValueError("Invalid clean-discovery signatures.")
        if not torch.isfinite(gains).all() or not torch.isfinite(signatures).all():
            raise ValueError("Nonfinite confirmation evidence.")
        control_count = len(gains) - client_count
        if (client_count < 1 or not 1 <= neighbors < control_count
            or not all(math.isfinite(v) and v > 0 for v in (min_gain, margin))):
            raise ValueError("Invalid control calibration sizes/thresholds.")
        references = signatures[client_count:]
        matched, distances, envelopes = [], [], []
        for index, signature in enumerate(signatures):
            distance = ((references - signature).double().square()).mean(1)
            if index >= client_count:
                distance[index - client_count] = math.inf  # Leave itself out.
            chosen = torch.argsort(distance, stable=True)[:neighbors]
            matched.append(chosen.tolist())
            distances.append(distance[chosen].tolist())
            # A conservative envelope across behaviourally nearby controls,
            # separately for each SAME probe/fold. Never compare different probes.
            envelopes.append(gains[client_count:][chosen].max(dim=0).values)
        envelope = torch.stack(envelopes)
        excess = (gains - envelope).clamp_min(0.)
        eligible = source_support.all(2) & (gains > min_gain).all(2)
        probe_statistic = torch.where(eligible, excess.min(2).values,
                                      torch.zeros_like(excess[:, :, 0]))
        maxima, indices = probe_statistic.max(1)
        null_maxima = maxima[client_count:]
        threshold = float(null_maxima.max()) + margin
        scores = maxima[:client_count] / threshold
        return {"scores": scores, "raw": maxima, "indices": indices,
                "threshold": threshold, "null_maxima": null_maxima,
                "matched_controls": matched, "signature_distances": distances,
                "envelope": envelope, "excess": excess}

    def evaluate(self, global_state, states, client_ids):
        self.round += 1
        if not self.enabled:
            self.last_probes, self.last_calibration = [], {"enabled": False}
            return [{"Audit_Score": 0., "Audit_Complete": 0, "Audit_Flagged": 0,
                     "Audit_Raw_Max": None, "Audit_Threshold": None,
                     "Control_Count": 0, "Matched_Controls": "",
                     "Probe_Count": 0, "Probe_Target": None, "Probe_Features": "",
                     "Probe_Values": "", "Probe_Gain_Fold0": None, "Probe_Gain_Fold1": None,
                     "Probe_Envelope_Fold0": None, "Probe_Envelope_Fold1": None,
                     "Probe_Excess_Fold0": None, "Probe_Excess_Fold1": None,
                     "Probe_Triggered_Target_Rate": None, "Probe_Budget": self.budget}
                    for _ in states]
        if len(states) != len(client_ids) or not states:
            raise ValueError("Each submission needs an ID.")
        base = self._model(global_state)
        candidates = [self._model(state) for state in states]
        controls = self._controls(global_state)
        models = candidates + controls
        signatures = torch.stack([self._signature(model) for model in models])
        fresh = []
        for index, model in enumerate(models):
            for target in range(self.classes):
                probe = self._search(model, base, target)
                probe["origin_client"] = int(client_ids[index]) if index < len(states) else None
                probe["origin_control"] = (self.profiles[index - len(states)]["name"]
                                           if index >= len(states) else None)
                fresh.append(probe)
        probes = list(self.bank) + fresh
        self.last_probes = probes
        # Controls and real models are evaluated on the same UNION of patterns.
        # Each control gets its own equally budgeted discovery search as well.
        transformed = torch.stack([self.transform(self.VX, p) for p in probes])
        P, B, F = transformed.shape
        targets = torch.tensor([p["target"] for p in probes], dtype=torch.long)
        gather_index = targets[:, None, None].expand(P, B, 1)
        flat = transformed.reshape(P * B, F)
        with torch.no_grad():
            base_trigger = self._probabilities(base, flat).reshape(P, B, self.classes).gather(2, gather_index).squeeze(2)
            base_clean = self._probabilities(base, self.VX)[:, targets].T
            gains, supports, rates = [], [], []
            required_sources = min(2, self.classes - 1)
            for model in models:
                changed = self._probabilities(model, flat).reshape(P, B, self.classes)
                clean = self._probabilities(model, self.VX)[:, targets].T
                response = (changed.gather(2, gather_index).squeeze(2) - base_trigger) - (clean - base_clean)
                per_fold, support_fold = [], []
                for fold in (0, 1):
                    class_means = torch.stack([
                        response[:, (self.Vy == c) & (self.folds == fold)].mean(1)
                        for c in range(self.classes)], dim=1)
                    keep = torch.arange(self.classes)[None, :] != targets[:, None]
                    per_fold.append((class_means * keep).sum(1) / (self.classes - 1))
                    support_fold.append(((class_means > self.min_gain / 2) & keep).sum(1) >= required_sources)
                gains.append(torch.stack(per_fold, dim=1))
                supports.append(torch.stack(support_fold, dim=1))
                eligible = self.Vy[None, :] != targets[:, None]
                rates.append(((changed.argmax(2) == targets[:, None]) & eligible).sum(1) / eligible.sum(1))
            gains, supports, rates = torch.stack(gains), torch.stack(supports), torch.stack(rates)
            calibrated = self.calibrate(gains, supports, signatures, len(states),
                                        self.neighbors, self.min_gain, self.margin)
        rows = []
        for i in range(len(states)):
            j = int(calibrated["indices"][i])
            probe, score = probes[j], float(calibrated["scores"][i])
            columns = probe["mask"].nonzero(as_tuple=True)[0].tolist()
            rows.append({"Audit_Score": score, "Audit_Complete": 1,
                "Audit_Flagged": int(score > 1.), "Audit_Raw_Max": float(calibrated["raw"][i]),
                "Audit_Threshold": calibrated["threshold"], "Control_Count": len(controls),
                "Matched_Controls": ";".join(self.profiles[c]["name"] for c in calibrated["matched_controls"][i]),
                "Probe_Count": len(probes), "Probe_Target": int(probe["target"]),
                "Probe_Features": ";".join(map(str, columns)),
                "Probe_Values": ";".join(f"{float(probe['pattern'][c]):.7g}" for c in columns),
                "Probe_Gain_Fold0": float(gains[i, j, 0]), "Probe_Gain_Fold1": float(gains[i, j, 1]),
                "Probe_Envelope_Fold0": float(calibrated["envelope"][i, j, 0]),
                "Probe_Envelope_Fold1": float(calibrated["envelope"][i, j, 1]),
                "Probe_Excess_Fold0": float(calibrated["excess"][i, j, 0]),
                "Probe_Excess_Fold1": float(calibrated["excess"][i, j, 1]),
                "Probe_Triggered_Target_Rate": 100 * float(rates[i, j]), "Probe_Budget": self.budget})
        self.last_calibration = {
            "enabled": True, "rule": "leave-one-control-out full-search maximum + absolute margin",
            "not_a_statistical_FPR_guarantee": True, "threshold": calibrated["threshold"],
            "margin": self.margin, "neighbors": self.neighbors,
            "control_steps": self.control_steps, "control_learning_rate": self.control_lr,
            "profiles": self.profiles, "null_maxima": calibrated["null_maxima"].tolist(),
            "control_worst_probe_indices": calibrated["indices"][len(states):].tolist(),
            "model_order": ([f"client_{cid}" for cid in client_ids] +
                            [p["name"] for p in self.profiles]),
            "matched_control_indices": calibrated["matched_controls"],
            "signature_distances": calibrated["signature_distances"],
            "class_balanced_gains_model_probe_fold": gains.tolist(),
            "source_support_model_probe_fold": supports.tolist()}
        # Only discovery evidence determines the bank; confirmation outcomes do
        # not choose future probes. Keep one strongest fresh probe per target.
        additions = []
        for target in range(self.classes):
            choices = [p for p in fresh if p["target"] == target]
            additions.append(max(choices, key=lambda p: p["discovery_gain"]))
        self.bank = (self.bank + additions)[-self.bank_size:] if self.bank_size else []
        return rows


class FedMediShield(LegacyHistoryShield):
    """Sparse-probe admission + historical references + capped aggregation.

    Uses trusted labeled TRAINING probes, stable IDs and individually visible
    updates. Does not use an oracle trigger, target, attacker list or validation.
    The whole model is inspected behaviorally; temporal memory uses the final
    layer. Probe-negative is limited search evidence, not proof of honesty.
    """
    def __init__(self, model, discovery_X, discovery_y, verification_X, verification_y,
                 classes, final_keys, mode="filter", history_policy="hard",
                 k=3., history_limit=1.5, rho=.1, admission_patience=2,
                 recovery_patience=6, trust_cutoff=.5, risk_decay=.8, risk_gain=1.,
                 clip_factor=1., audit_options=None, risk_cap=2.,
                 stop_flagged_fraction=.5, stop_min_mass=.25):
        super().__init__(final_keys, mode=mode, history_policy=history_policy, k=k,
            history_limit=history_limit, rho=rho, admission_patience=admission_patience,
            recovery_patience=recovery_patience, trust_cutoff=trust_cutoff,
            risk_decay=risk_decay, risk_gain=risk_gain, clip_factor=clip_factor)
        if not all(math.isfinite(v) for v in (risk_cap, stop_flagged_fraction, stop_min_mass)) or not (
                risk_cap >= 0 and 0 < stop_flagged_fraction <= 1 and 0 < stop_min_mass <= 1):
            raise ValueError("Invalid risk cap or filtering validity bounds.")
        self.risk_cap = float(risk_cap)
        self.stop_flagged_fraction, self.stop_min_mass = float(stop_flagged_fraction), float(stop_min_mass)
        self.audit = SparseProbeAudit(model, discovery_X, discovery_y,
            verification_X, verification_y, classes, **(audit_options or {}))

    def aggregate(self, global_state, states, sizes, client_ids):
        validate([global_state] + list(states), [1] + list(sizes))
        if len(client_ids) != len(states) or len(set(client_ids)) != len(states):
            raise ValueError("Participant IDs must be unique and match submissions.")
        before = copy.deepcopy(self.export_state())
        evidence = self.audit.evaluate(global_state, states, client_ids)
        # The off ablation bypasses admission evidence deliberately. Labelled
        # explicitly in settings; it is not claimed to use the protected gate.
        if not self.audit.enabled:
            for d in evidence:
                d["Audit_Complete"] = 1
        result, diagnostic = super().aggregate(global_state, states, sizes, client_ids, evidence)
        for row, audit in zip(diagnostic["clients"], evidence):
            row.update(audit)
            if row["Gate_Reason"] == "peer_alert":
                row["Gate_Reason"] = "probe_alert"
            if not self.audit.enabled:
                row["Audit_Complete"] = 0
            # These legacy geometric diagnostics do not apply to this method.
            for key in ("Krum_Score", "Krum_Median", "Krum_Scale", "Layer_Median", "Layer_Scale", "Reference_Core"):
                row[key] = None
        diagnostic.update(anchor_client=None, core_size=None,
                          audit_round=self.audit.round, probe_count=len(self.audit.last_probes))
        flagged = sum(row["Flagged"] for row in diagnostic["clients"]) / len(states)
        effective_mass = sum(row["Effective_Delta_Multiplier"] for row in diagnostic["clients"])
        reasons = []
        if flagged >= self.stop_flagged_fraction:
            reasons.append(f"flagged fraction {flagged:.3f} >= {self.stop_flagged_fraction:g}")
        if self.mode == "filter" and effective_mass < self.stop_min_mass:
            reasons.append(f"effective update mass {effective_mass:.6f} < {self.stop_min_mass:g}")
        diagnostic.update(flagged_fraction=flagged, effective_mass=effective_mass,
                          validity_reasons=reasons, filter_applied=self.mode == "filter" and not reasons)
        for row in diagnostic["clients"]:
            row["Detector_Validity_Warning"] = int(bool(reasons))
        if self.mode == "filter" and reasons:
            # Transaction rollback: neither suspicious references nor an invalid
            # probe bank survive a round that the caller never applies.
            self.import_state(before)
            raise DetectorCollapseError("Filtering stopped: " + "; ".join(reasons), diagnostic)
        return result, diagnostic

    def export_state(self):
        return {"memory": self.memory, "audit_round": self.audit.round, "probe_bank": self.audit.bank}

    def import_state(self, state):
        self.memory, self.audit.round, self.audit.bank = state["memory"], state["audit_round"], state["probe_bank"]

    def probe_manifest(self):
        return [{"target": p["target"], "features": p["mask"].nonzero(as_tuple=True)[0].tolist(),
                 "values": p["pattern"][p["mask"].bool()].tolist(),
                 "discovery_gain": p["discovery_gain"], "created_round": p["created_round"],
                 "origin_client": p["origin_client"],
                 "origin_control": p.get("origin_control")} for p in self.audit.last_probes]


def fltrust_aggregate(global_state, states, sizes, root_state):
    """FLTrust equations (2)-(5), Cao et al., NDSS 2021, on model deltas.

    Local optimizer/step budgets are harness choices, not paper replication.
    Zero root norm or no positive cosine holds the global model (explicit
    degenerate-case convention). Zero cosine is NOT a malicious-client label.
    """
    validate([global_state, root_state] + list(states), [1, 1] + list(sizes))
    deltas = [{k: s[k] - global_state[k] for k in s if s[k].is_floating_point()} for s in states]
    root = flatten({k: root_state[k] - global_state[k] for k in global_state if global_state[k].is_floating_point()})
    vectors = torch.stack([flatten(d) for d in deltas])
    norms, root_norm = vectors.norm(dim=1), float(root.norm())
    scores = (vectors @ root / (norms * root_norm).clamp_min(1e-30)).clamp(0., 1.)
    scores = torch.where(norms > 1e-12, scores, torch.zeros_like(scores))
    if root_norm <= 1e-12:
        scores.zero_()
    weights = scores / scores.sum() if float(scores.sum()) > 0 else torch.zeros_like(scores)
    normalization = torch.where(norms > 1e-12, root_norm / norms.clamp_min(1e-12), torch.zeros_like(norms))
    coefficients = weights * normalization
    output = {}
    for k, old in global_state.items():
        if old.is_floating_point():
            output[k] = (old.double() + sum(float(c) * d[k].double() for c, d in zip(coefficients, deltas))).to(old)
        else:
            if any(not torch.equal(s[k], old) for s in [root_state] + list(states)):
                raise ValueError(f"Unequal nonfloating buffer {k}.")
            output[k] = old.clone()
    rows = [{"Aggregation_Weight": float(weights[i]), "Applied_Clip_Factor": float(normalization[i]),
             "Root_Cosine_Trust": float(scores[i]), "Root_Update_Norm": root_norm,
             "Root_Norm_Normalization": float(normalization[i]), "Selected": int(weights[i] > 0),
             "FLTrust_Degenerate_Hold": int(float(weights.sum()) == 0)} for i in range(len(states))]
    return output, rows


class FLTrust:
    """Root-data local AdamW update, then the published FLTrust aggregation."""
    def __init__(self, model, X, y, steps=8, learning_rate=.001):
        self.template = copy.deepcopy(model).cpu()
        self.X, self.y = X.detach().cpu().float().clone(), y.detach().cpu().long().clone()
        if len(self.y) < 1 or steps < 1 or learning_rate <= 0:
            raise ValueError("FLTrust requires trusted root data and a positive optimization budget.")
        self.steps, self.lr = steps, learning_rate

    def aggregate(self, global_state, states, sizes, client_ids):
        root = copy.deepcopy(self.template)
        root.load_state_dict(global_state)
        root.train()
        optimizer = torch.optim.AdamW(root.parameters(), lr=self.lr, weight_decay=1e-4)
        # Full root batches, deterministic. Root data never includes evaluation.
        with torch.enable_grad():
            for _ in range(self.steps):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(root(self.X), self.y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite root training loss.")
                loss.backward()
                optimizer.step()
        return fltrust_aggregate(global_state, states, sizes, root.state_dict())
