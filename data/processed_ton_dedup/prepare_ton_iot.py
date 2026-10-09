"""Prepare the ToN-IoT network train/test CSV (train_test_network.csv) for the federated IDS pipeline.

Output format is identical to data/processed_grouped (federated_data.pt, feature_names.json, scaler, manifest),
so train_defended.py and medishield_candidate.py run on it unchanged via --data.

Feature policy (fixed before any model was trained on this data):
  * dropped identifiers: src_ip, dst_ip, src_port, dst_port (attacker hosts 192.168.1.30-39 would leak the label)
  * dropped free text / content identifiers: dns_query, ssl_subject, ssl_issuer, http_uri, http_user_agent, weird_addl
  * dropped near-constant categorical columns: absent ('-') in >= 99.5% of rows
  * remaining categorical columns: one-hot (absence '-' is its own category)
  * numeric columns: log1p(max(x, 0)) for nonnegative counts, then StandardScaler fit on the training split only
With --dedup (used for the paper data) exact duplicates of (encoded features, label) are removed, keeping
the first; without it a single 15,344-row identical Scanning vector forms one group that lands wholly in
validation (Scanning recall 0%, see addendum A29). Remaining identical vectors (conflicting labels) are never
split across train/validation (grouped split, as in prepare_federated_source.py). The ToN-IoT file has no separate official test set; this script
holds out a grouped 20% validation split and never uses it for fitting.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

CLASS_ORDER = ["normal", "backdoor", "ddos", "dos", "injection", "mitm", "password", "ransomware", "scanning", "xss"]
DISPLAY = {"normal": "Normal", "backdoor": "Backdoor", "ddos": "DDoS", "dos": "DoS", "injection": "Injection",
           "mitm": "MITM", "password": "Password", "ransomware": "Ransomware", "scanning": "Scanning", "xss": "XSS"}
IDENTIFIERS = ["src_ip", "dst_ip", "src_port", "dst_port"]
FREE_TEXT = ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent", "weird_addl"]
LABELS = ["label", "type"]
ABSENT_MAX_SHARE = 0.995


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def fingerprints(X):
    X = np.array(X, copy=True)
    X[X == 0] = 0
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def conflict_count(hashes, y):
    counts = pd.DataFrame({"group": hashes, "label": y}).groupby("group").label.nunique()
    return int((counts > 1).sum())


def merge_groups(groups, hashes):
    _, observed = np.unique(hashes, return_inverse=True)
    pairs = np.unique(np.column_stack((observed, groups)), axis=0)
    parent = np.arange(int(groups.max()) + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    same = pairs[1:, 0] == pairs[:-1, 0]
    for a, b in zip(pairs[:-1, 1][same], pairs[1:, 1][same]):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(parent))])
    _, merged = np.unique(roots[groups], return_inverse=True)
    return merged


def encode(frame):
    """Raw frame -> (numeric matrix before scaling, feature names, policy record). Deterministic, label-free."""
    frame = frame.drop(columns=IDENTIFIERS + FREE_TEXT + LABELS)
    categorical, numeric, dropped = [], [], []
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric.append(column)
        elif (values.astype(str) == "-").mean() >= ABSENT_MAX_SHARE:
            dropped.append(column)
        else:
            categorical.append(column)
    parts, names = [], []
    for column in numeric:
        v = frame[column].to_numpy(dtype=np.float64)
        if (v >= 0).all():
            v = np.log1p(v)
        parts.append(v[:, None]); names.append(column)
    for column in categorical:
        values = frame[column].astype(str)
        for category in sorted(values.unique()):
            parts.append((values == category).to_numpy(dtype=np.float64)[:, None])
            names.append(f"{column}={category}")
    policy = {"identifiers_dropped": IDENTIFIERS, "free_text_dropped": FREE_TEXT,
              "near_constant_dropped": dropped, "absent_max_share": ABSENT_MAX_SHARE,
              "numeric_log1p_then_standardized": numeric, "one_hot": categorical}
    return np.hstack(parts), names, policy


def make_clients(y, n_classes, n, alpha, seed):
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(n)]
    for c in range(n_classes):
        indices = np.flatnonzero(y == c)
        rng.shuffle(indices)
        counts = rng.multinomial(len(indices), rng.dirichlet(np.full(n, alpha)))
        for i, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
            clients[i].extend(part.tolist())
    if any(not part for part in clients):
        raise ValueError("Dirichlet draw produced an empty client; partition was not silently repaired.")
    for part in clients:
        rng.shuffle(part)
    assert np.array_equal(np.sort(np.concatenate(clients)), np.arange(len(y)))
    return {i: indices for i, indices in enumerate(clients)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="data/raw_ton/train_test_network.csv")
    p.add_argument("--out", default="data/processed_ton")
    p.add_argument("--classes", nargs="+", default=CLASS_ORDER, help="Subset of ToN-IoT types; 'normal' is required and becomes class 0.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-passes", type=int, default=12)
    p.add_argument("--dedup", action="store_true", help="Drop exact duplicates of (encoded features, label), keeping the first.")
    args = p.parse_args()
    classes = ["normal"] + [c for c in CLASS_ORDER if c in args.classes and c != "normal"]
    if "normal" not in args.classes or set(args.classes) - set(CLASS_ORDER):
        p.error(f"--classes must include 'normal' and only contain {CLASS_ORDER}.")
    out, raw = Path(args.out), Path(args.raw)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Use a new --out path; existing data is preserved.")

    frame = pd.read_csv(raw, low_memory=False)
    frame = frame[frame["type"].isin(classes)].reset_index(drop=True)
    y = frame["type"].map({c: i for i, c in enumerate(classes)}).to_numpy(dtype=np.int64)
    if not np.array_equal(frame["label"].to_numpy(), (y != 0).astype(int)):
        raise ValueError("The binary 'label' column disagrees with 'type'.")
    X, names, policy = encode(frame)
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite encoded features.")
    dedup = None
    if args.dedup:
        keep = ~pd.DataFrame(np.column_stack([X, y])).duplicated(keep="first").to_numpy()
        dedup = {"rows_before": int(len(y)), "rows_after": int(keep.sum()),
                 "removed_per_class": {DISPLAY[c]: int(((~keep) & (y == i)).sum()) for i, c in enumerate(classes)}}
        X, y = X[keep], y[keep]
        print("Exact (features, label) duplicates removed:", dedup["removed_per_class"], flush=True)
    print(f"{len(y):,} rows, {X.shape[1]} features, classes {classes}", flush=True)

    raw_hash = fingerprints(X)
    _, groups = np.unique(raw_hash, return_inverse=True)
    history = []
    for attempt in range(1, args.max_passes + 1):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        train, val = next(splitter.split(np.zeros((len(y), 1)), y, groups))
        if len(np.unique(y[train])) != len(classes) or len(np.unique(y[val])) != len(classes):
            raise ValueError("A grouped split lacks a class; no row-wise fallback was used.")
        scaler = StandardScaler().fit(X[train])
        stored = scaler.transform(X).astype(np.float32)
        final_hash = fingerprints(stored)
        shared = np.intersect1d(final_hash[train], final_hash[val])
        history.append({"pass": attempt, "groups": int(len(np.unique(groups))), "train_rows": len(train),
                        "validation_rows": len(val), "shared_float32_fingerprints": len(shared)})
        print(f"Pass {attempt}: train={len(train):,} validation={len(val):,} shared fingerprints={len(shared):,}", flush=True)
        if len(shared) == 0:
            break
        groups = merge_groups(groups, final_hash)
    else:
        raise RuntimeError(f"No stable split after {args.max_passes} passes. Nothing saved. {history}")

    clients = make_clients(y[train], len(classes), args.num_clients, args.alpha, args.seed)
    k = len(classes)
    provenance = {"schema_version": "ton_iot_network_grouped_v1", "conflicting_label_policy": "preserve_within_split",
                  "scope": "ToN-IoT train_test_network.csv; grouped 80/20 split; no separate official test set used.",
                  "official_test_loaded": False, "scaler_fit_on_final_training_only": True,
                  "split_method": "First fixed seeded fold of StratifiedGroupKFold(5); monotone grouping of float32 collisions.",
                  "row_duplicates_retained": not args.dedup, "deduplication": dedup, "feature_policy": policy,
                  "raw_conflicting_groups": conflict_count(raw_hash, y),
                  "train_conflicting_float32_groups": conflict_count(final_hash[train], y[train]),
                  "validation_conflicting_float32_groups": conflict_count(final_hash[val], y[val])}
    manifest = {**provenance, "settings": vars(args), "source_file": {"file": str(raw), "sha256": sha256(raw), "rows_used": len(y)},
                "classes": [DISPLAY[c] for c in classes], "features": names, "passes": history,
                "train_rows": len(train), "validation_rows": len(val),
                "class_counts": {"train": np.bincount(y[train], minlength=k).tolist(),
                                 "validation": np.bincount(y[val], minlength=k).tolist()},
                "clients": {str(i): {"rows": len(idx), "class_counts": np.bincount(y[train][idx], minlength=k).tolist()}
                            for i, idx in clients.items()},
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "torch": str(torch.__version__)}}
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"train_X": torch.from_numpy(stored[train].copy()), "train_y_type": torch.from_numpy(y[train].copy()),
                "train_y_binary": torch.from_numpy((y[train] != 0).astype(np.int64)),
                "val_X": torch.from_numpy(stored[val].copy()), "val_y_type": torch.from_numpy(y[val].copy()),
                "val_y_binary": torch.from_numpy((y[val] != 0).astype(np.int64)),
                "client_indices": clients, "class_names": [DISPLAY[c] for c in classes], "preparation": provenance},
               out / "federated_data.pt")
    np.savez_compressed(out / "split_indices.npz", train=train, validation=val, groups=groups)
    joblib.dump(scaler, out / "scaler.pkl")
    (out / "feature_names.json").write_text(json.dumps({"features": names}, indent=2), encoding="utf-8")
    (out / "partition_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    shutil.copyfile(__file__, out / "prepare_ton_iot.py")
    print(f"Completed {out}: {X.shape[1]} features, train={len(train):,}, validation={len(val):,}.")
    print("Train class counts:", dict(zip(manifest["classes"], manifest["class_counts"]["train"])))
    print("Rows per client:", [v["rows"] for v in manifest["clients"].values()])


if __name__ == "__main__":
    main()
