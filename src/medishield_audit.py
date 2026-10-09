"""Audit saved Fed-MediShield contrast5 measurements; no training or inference.

Compare the previous raw score with ONE specified round-relative alternative.
Threshold settings are fixed by the supplied clean calibration contract. Nothing
is automatically tuned or promoted to a live defender. This is a diagnostic,
not a replacement for medishield_candidate.py.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

VERSION = "medishield_cohort_audit_1"
SUPPORTED_SOURCE = "4e8d50b8101a4149e6b0c44eba3fa611c972c72bc64922cff4c70c5e94db47ce"
PEER_FLOOR = 1.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    if not values or not 0 <= q <= 1:
        raise ValueError("Invalid quantile input.")
    return values[math.ceil((len(values) - 1) * q)]


def phase(row, config):
    return "initial" if row["Round"] <= config["initial_rounds"] else "later"


def with_peers(rows, clients):
    groups, result, seen = defaultdict(list), [], set()
    for row in rows:
        key = (row["Seed"], row["Condition"], row["Round"], row["Client"])
        if key in seen:
            raise ValueError("Duplicate score observation.")
        seen.add(key)
        if any(not math.isfinite(row[k]) or row[k] < 0 for k in ("Score", "Full_Update_Norm")):
            raise ValueError("Nonfinite or negative saved measurement.")
        groups[key[:3]].append(row)
    for key, group in sorted(groups.items()):
        if {r["Client"] for r in group} != set(range(clients)):
            raise ValueError("This audit requires the recorded full-participation cohort each round.")
        for row in sorted(group, key=lambda r: r["Client"]):
            peer = statistics.median(r["Score"] for r in group if r["Client"] != row["Client"])
            result.append({**row, "Peer_Median": peer})
    return result


def fit_profiles(rows, config, threshold_fit, relative, seeds, clients):
    if not rows or any(r["Condition"] != "clean" or r["Seed"] not in seeds for r in rows):
        raise ValueError("Fitting requires only the specified clean seeds.")
    profiles = {}
    for cid in range(clients):
        for ph in ("initial", "later"):
            group = [r for r in rows if r["Client"] == cid and phase(r, config) == ph]
            if any(sum(r["Seed"] == seed for r in group) < 3 for seed in seeds):
                raise ValueError(f"Insufficient clean seed coverage for client {cid}/{ph}.")
            anchor = max(PEER_FLOOR, statistics.median(r["Peer_Median"] for r in group))
            values = [operational_score(r, anchor, relative) for r in group]
            centre = statistics.median(values)
            mad = statistics.median(abs(v - centre) for v in values)
            threshold = max(threshold_fit["alert_floor"], quantile(values, threshold_fit["quantile"]) + threshold_fit["margin"])
            profiles[(cid, ph)] = {
                "anchor": anchor, "threshold": threshold,
                "width": max(config["penalty_width_floor"], config["penalty_relative_width"] * threshold, 1.4826 * mad),
                "clip_bound": max(config["norm_epsilon"], config["clip_multiplier"] * max(r["Full_Update_Norm"] for r in group)),
                "fit_seeds": sorted(seeds), "observations": len(group)}
    return profiles


def operational_score(row, anchor, relative):
    if relative:
        return row["Score"] * anchor / max(PEER_FLOOR, row["Peer_Median"])
    return row["Score"]


def review(rows, profiles, config, relative, truth):
    """Replay influence arithmetic along recorded observations, never model training.

    Evaluator truth is attached after scoring/weight decisions; no fitting uses it.
    Continuing arithmetic after a would-stop round is diagnostic only, reported
    as such. No counterfactual ASR or classifier accuracy is computed.
    """
    grouped, memory, decisions, stops = defaultdict(list), defaultdict(float), [], []
    for row in rows:
        grouped[row["Round"]].append(row)
    for rnd, group in sorted(grouped.items()):
        total = sum(r["Client_Size"] for r in group)
        current = []
        for row in sorted(group, key=lambda r: r["Client"]):
            cid = row["Client"]
            profile = profiles[(cid, phase(row, config))]
            score = operational_score(row, profile["anchor"], relative)
            flagged = score > profile["threshold"]
            severity = min(config["severity_cap"], max(0., score - profile["threshold"]) / profile["width"])
            clip = min(1., profile["clip_bound"] / row["Full_Update_Norm"]) if row["Full_Update_Norm"] > config["norm_epsilon"] else 1.
            if flagged:
                memory[cid] = min(config["risk_cap"], memory[cid] + config["memory_gain"] * severity ** 2)
            elif row["Informative"] and clip == 1.:
                memory[cid] *= 1. - config["risk_recovery"]
            if memory[cid] < config["norm_epsilon"]:
                memory[cid] = 0.
            trust = math.exp(-config["risk_gain"] * severity ** 2 - memory[cid])
            nominal = row["Client_Size"] / total
            key = (row["Seed"], row["Condition"], rnd, cid)
            if key not in truth:
                raise ValueError("Missing evaluator truth; it must not default to honest.")
            current.append({"Seed": row["Seed"], "Condition": row["Condition"], "Round": rnd, "Client": cid,
                "Phase": phase(row, config), "Raw_Score": row["Score"], "Peer_Median": row["Peer_Median"],
                "Calibration_Anchor": profile["anchor"], "Score": score, "Threshold": profile["threshold"],
                "Flagged": int(flagged), "Trust": trust, "Nominal_Weight": nominal,
                "Proposed_Coefficient": nominal * trust * clip, "Informative": row["Informative"],
                "Is_Poisoned_Update": int(truth[key])})
        if (sum(r["Flagged"] for r in current) / len(current) >= config["stop_flagged_fraction"]
                or sum(r["Proposed_Coefficient"] for r in current if r["Informative"]) < config["stop_min_mass"]):
            stops.append(rnd)
        decisions.extend(current)
    clean = [r for r in decisions if not r["Is_Poisoned_Update"]]
    poisoned = [r for r in decisions if r["Is_Poisoned_Update"]]
    def retention(group):
        return 100 * sum(r["Proposed_Coefficient"] for r in group) / sum(r["Nominal_Weight"] for r in group) if group else None
    delays = {}
    for cid in sorted({r["Client"] for r in poisoned}):
        participating = [r for r in poisoned if r["Client"] == cid]
        delays[str(cid)] = next((i + 1 for i, r in enumerate(participating) if r["Flagged"]), None)
    summary = {"Seed": rows[0]["Seed"], "Condition": rows[0]["Condition"],
        "False_Alerts": sum(r["Flagged"] for r in clean), "Unpoisoned_Observations": len(clean),
        "FPR_Percent": 100 * sum(r["Flagged"] for r in clean) / len(clean) if clean else None,
        "Proposed_Unpoisoned_Retention": retention(clean), "Detected_Poisoned": sum(r["Flagged"] for r in poisoned),
        "Poisoned_Observations": len(poisoned), "Detection_Percent": 100 * sum(r["Flagged"] for r in poisoned) / len(poisoned) if poisoned else None,
        "Proposed_Poisoned_Retention": retention(poisoned), "First_Detection_Poisoned_Participation": delays,
        "Would_Stop_Rounds": stops, "ASR": None, "Training_Executed": False,
        "Interpretation": "Fixed recorded trajectory; proposed influence only. No alternative training outcome or statistical guarantee."}
    return summary, decisions


def failures(summary, criteria, attack=False):
    reasons = []
    if summary["FPR_Percent"] is None or summary["FPR_Percent"] > criteria["Max_FPR"]:
        reasons.append("unpoisoned FPR exceeds limit")
    if summary["Proposed_Unpoisoned_Retention"] is None or summary["Proposed_Unpoisoned_Retention"] < criteria["Min_Retention"]:
        reasons.append("unpoisoned retention below minimum")
    if summary["Would_Stop_Rounds"]:
        reasons.append("one or more rounds would trigger a stop")
    if attack:
        if summary["Detection_Percent"] is None or summary["Detection_Percent"] < criteria["Min_Detection"]:
            reasons.append("poisoned detection below minimum")
        if summary["Proposed_Poisoned_Retention"] is None or summary["Proposed_Poisoned_Retention"] > criteria["Max_Poisoned_Retention"]:
            reasons.append("poisoned retention above maximum")
        if any(v is None or v > criteria["Latest_First_Alert"] for v in summary["First_Detection_Poisoned_Participation"].values()):
            reasons.append("first detection missing or too late")
    return reasons


def load_saved(path):
    """Read contrast5 measurements and provenance. Never import archive source."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP contains duplicate member names.")
        candidates = [n for n in names if n == "calibration.json" or n.endswith("/calibration.json")]
        if len(candidates) != 1:
            raise ValueError("Supply the contrast5 results.zip containing one calibration.json.")
        prefix = candidates[0][:-len("calibration.json")]
        def raw(name):
            info = archive.getinfo(prefix + name)
            if info.file_size > 32 * 1024 * 1024:
                raise ValueError("Unexpectedly large archive member: " + name)
            return archive.read(info)
        def obj(name):
            return json.loads(raw(name))
        def csvrows(name):
            return list(csv.DictReader(io.StringIO(raw(name).decode("utf-8-sig"))))
        calibration = obj("calibration.json")
        if (calibration.get("version") != "medishield_contrast_5"
                or calibration.get("source_sha256") != SUPPORTED_SOURCE
                or hashlib.sha256(raw("sources/medishield_candidate.py")).hexdigest() != SUPPORTED_SOURCE):
            raise ValueError("This audit supports the supplied contrast5 release; source identity differs.")
        if calibration["artifact_sha256"] != digest({k: v for k, v in calibration.items() if k != "artifact_sha256"}):
            raise ValueError("Calibration checksum mismatch.")
        if calibration["contract_sha256"] != digest(calibration["contract"]):
            raise ValueError("Calibration contract checksum mismatch.")
        cfg, training = calibration["shield_config"], calibration["contract"]["training"]
        if digest(cfg) != digest(calibration["contract"]["shield_config"]):
            raise ValueError("Calibration configuration mismatch.")
        clients, horizon = len(training["client_sizes"]), calibration["rounds"]
        if clients != calibration["participants"]:
            raise ValueError("This audit covers full participation only.")
        clean = csvrows("calibration_observations.csv")
        attack = csvrows("capture_review/seed_42/attacked/scores.csv")
        seeds = calibration["calibration_seeds"]
        if seeds != [42, 44, 45]:
            raise ValueError("This bounded audit expects calibration model seeds 42, 44, 45.")
        truth, information, evidence = {}, {}, {}
        for condition in ("clean", "attacked"):
            base = f"capture_review/seed_42/{condition}/"
            for row in csvrows(base + "replayed_decisions.csv"):
                key = (42, condition, int(row["Round"]), int(row["Client"]))
                if key in truth:
                    raise ValueError("Duplicate decision identity.")
                truth[key], information[key] = bool(int(row["Is_Poisoned_Update"])), bool(int(row["Informative"]))
                if condition == "clean" and truth[key]:
                    raise ValueError("Clean capture contains poisoned submissions.")
            for rnd in range(1, horizon + 1):
                saved = obj(base + f"round_{rnd:04d}.json")
                for i, cid in enumerate(saved["client_ids"]):
                    evidence[(42, condition, rnd, cid)] = saved["evidence"]["scores"][i]
        # Read classifier deltas only to reproduce the informative-update gate.
        # NumPy loads recorded arrays; no model is constructed or evaluated.
        import numpy as np
        for seed in (44, 45):
            folder = f"calibration_controls/seed_{seed}/"
            manifest = obj(folder + "config.json")
            if manifest.get("attack_applied") is not False or manifest.get("aggregation") != "sample_weighted_FedAvg":
                raise ValueError("Control is not explicitly clean FedAvg.")
            if digest(manifest["training_contract"]) != digest(training):
                raise ValueError("Control training protocol mismatch.")
            committed = obj(folder + "rounds.json")
            if [r["Round"] for r in committed] != list(range(1, horizon + 1)):
                raise ValueError("Incomplete clean control.")
            for metadata in committed:
                rnd = metadata["Round"]
                for name, checksum in metadata["artifacts"].items():
                    if hashlib.sha256(raw(folder + name)).hexdigest() != checksum:
                        raise ValueError("Control evidence checksum mismatch.")
                record = obj(folder + f"evidence/round_{rnd:04d}.json")
                with np.load(io.BytesIO(raw(folder + f"captures/round_{rnd:04d}.npz")), allow_pickle=False) as captured:
                    keys = captured["parameter_keys"].tolist()
                    w, b = keys.index("network.6.weight"), keys.index("network.6.bias")
                    ids, steps = captured["client_ids"].tolist(), captured["optimizer_steps"].tolist()
                    if ids != record["client_ids"] or steps != record["optimizer_steps"]:
                        raise ValueError("Control identities/steps disagree.")
                    for i, cid in enumerate(ids):
                        dw = (captured[f"client_{cid}_{w:03d}"].astype(float) - captured[f"global_{w:03d}"].astype(float)) / steps[i]
                        db = (captured[f"client_{cid}_{b:03d}"].astype(float) - captured[f"global_{b:03d}"].astype(float)) / steps[i]
                        if not np.isfinite(dw).all() or not np.isfinite(db).all():
                            raise ValueError("Nonfinite recorded classifier delta.")
                        key = (seed, "clean", rnd, cid)
                        information[key] = bool((np.sqrt((dw * dw).sum(axis=1) + db * db) > cfg["norm_epsilon"]).any())
                        truth[key] = False
                        evidence[key] = record["evidence"]["scores"][i]
        for rows in (clean, attack):
            for row in rows:
                for key in ("Seed", "Round", "Client", "Client_Size", "Optimizer_Steps"):
                    row[key] = int(row[key])
                for key in ("Score", "Full_Update_Norm"):
                    row[key] = float(row[key])
                key = (row["Seed"], row["Condition"], row["Round"], row["Client"])
                if key not in evidence or not math.isclose(row["Score"], evidence[key], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("CSV score does not match saved evidence.")
                cid = row["Client"]
                if (not 1 <= row["Round"] <= horizon or row["Client_Size"] != training["client_sizes"][cid]
                        or row["Optimizer_Steps"] != training["prescribed_steps"][cid]):
                    raise ValueError("Observation horizon/count/budget mismatch.")
                row["Informative"] = information[key] and row["Full_Update_Norm"] > cfg["norm_epsilon"]
        if len(clean) != len(seeds) * clients * horizon or len(attack) != clients * horizon:
            raise ValueError("Missing score observations.")
        gate = obj("replay_gate.json")
        criteria = {"Max_FPR": gate["Max_Replay_Unpoisoned_FPR_Percent"], "Min_Retention": gate["Min_Replay_Unpoisoned_Retention_Percent"],
            "Min_Detection": gate["Min_Replay_Detection_Percent"], "Max_Poisoned_Retention": gate["Max_Replay_Poisoned_Retention_Percent"],
            "Latest_First_Alert": gate["Latest_First_Detected_Poisoned_Participation"]}
        if criteria != {"Max_FPR": 5., "Min_Retention": 90., "Min_Detection": 70., "Max_Poisoned_Retention": 20., "Latest_First_Alert": 3}:
            raise ValueError("Screening limits differ from the agreed experiment; audit will not relax them.")
        old_attack = next(r for r in gate["Results"] if r["Condition"] == "attacked" and r["Seed"] == 42)
        return with_peers(clean, clients), with_peers(attack, clients), truth, calibration, criteria, old_attack


def audit(path):
    clean, attack, truth, calibration, criteria, old_attack = load_saved(path)
    config, fitting = calibration["shield_config"], calibration["threshold_fit"]
    seeds, clients = calibration["calibration_seeds"], len(calibration["profiles"])
    results, details = {}, {}
    for relative in (False, True):
        name = "cohort_relative" if relative else "raw_control"
        checks, rejected = [], []
        for seed in seeds:
            fit_rows = [r for r in clean if r["Seed"] != seed]
            held = [r for r in clean if r["Seed"] == seed]
            profiles = fit_profiles(fit_rows, config, fitting, relative, set(seeds) - {seed}, clients)
            summary, rows = review(held, profiles, config, relative, truth)
            summary.update(Check="held_out_clean_seed", Fit_Seeds=sorted(set(seeds) - {seed}))
            reasons = failures(summary, criteria)
            summary.update(Passed=not reasons, Failed_Criteria=reasons)
            checks.append(summary)
            details[f"{name}_held_seed{seed}"] = rows
            rejected.extend(f"held-out clean seed {seed}: {r}" for r in reasons)
        profiles = fit_profiles(clean, config, fitting, relative, set(seeds), clients)
        summary, rows = review(attack, profiles, config, relative, truth)
        summary.update(Check="development_attack_replay", Fit_Seeds=seeds)
        reasons = failures(summary, criteria, attack=True)
        summary.update(Passed=not reasons, Failed_Criteria=reasons)
        checks.append(summary)
        details[f"{name}_attack_seed42"] = rows
        rejected.extend("development attack: " + r for r in reasons)
        if not relative:
            if (summary["Detected_Poisoned"] != old_attack["TP"] or summary["False_Alerts"] != old_attack["FP"]
                    or not math.isclose(summary["Proposed_Unpoisoned_Retention"], old_attack["Proposed_Unpoisoned_Retention"], abs_tol=1e-7)
                    or not math.isclose(summary["Proposed_Poisoned_Retention"], old_attack["Proposed_Poisoned_Retention"], abs_tol=1e-7)):
                raise ValueError("Raw-score control does not reproduce the original result; comparison invalid.")
        results[name] = {"Passed_All_Checks": not rejected, "Failed_Criteria": rejected, "Checks": checks}
    return {"version": VERSION, "input_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "candidate_count": 1, "parameter_search_performed": False, "criteria": criteria, "results": results,
        "rule": "S_i * anchor_i_phase / max(1, median(S_j for j != i)); anchor fitted on clean fitting seeds only",
        "old_raw_control_reproduced": True, "Model_Training_Executed": False, "Model_Inference_Executed": False,
        "ASR": None, "Live_Deployment_Approved": False,
        "next_step": "Reject this candidate; do not start live training." if not results["cohort_relative"]["Passed_All_Checks"] else
                     "Development screen passed; requires a separately implemented and frozen live evaluation. No deployment automatically authorized.",
        "limitations": ["Development data inspected during design; leave-one-seed-out checks are not final independent evidence.",
            "Recorded trajectory weights do not establish a new model's accuracy or ASR.",
            "Normalizing aggregate score can mask coordinated attacks. Peer scores themselves depend on earlier cohort comparisons.",
            "Only three clean initialization seeds, fixed partition, one recorded attack seed and one trigger setting."]}, details


def write_outputs(folder, report, details):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "audit_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for name, rows in details.items():
        with (folder / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    lines = ["Fed-MediShield cached-score audit", "No training or model inference.", ""]
    for method, result in report["results"].items():
        lines.append(method + ": " + ("PASS development screen" if result["Passed_All_Checks"] else "FAIL"))
        for check in result["Checks"]:
            lines.append(f"  {check['Check']} seed={check['Seed']}: FPR={check['FPR_Percent']:.2f}% retained={check['Proposed_Unpoisoned_Retention']:.2f}%")
        lines.extend("  FAILED: " + reason for reason in result["Failed_Criteria"])
    lines.extend(["", report["next_step"]])
    text = "\n".join(lines)
    (folder / "audit_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print("Saved:", folder.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=VERSION)
    parser.add_argument("--results-zip", required=True, help="Completed contrast5 results.zip with clean controls and captured-score review.")
    parser.add_argument("--out", required=True, help="New directory for the arithmetic audit; existing directories are preserved.")
    args = parser.parse_args(argv)
    if Path(args.out).exists():
        parser.error("Output already exists. Choose a new --out; no training needs repeating.")
    report, details = audit(args.results_zip)
    write_outputs(Path(args.out), report, details)
    # A completed audit can reject a candidate. Rejection is recorded in JSON;
    # nonzero exit codes are reserved for actual input/code errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
