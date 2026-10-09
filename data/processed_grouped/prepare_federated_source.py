"""Rebuild the six-file CICIoMT development subset from raw training CSVs.

Preserves duplicate records and conflicting labels, but keeps identical final
float32 feature vectors in one split. Never reads the official test CSVs.
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

FILES = {
    "Benign_train.pcap.csv": 0,
    "TCP_IP-DDoS-ICMP1_train.pcap.csv": 1,
    "TCP_IP-DoS-ICMP1_train.pcap.csv": 2,
    "Recon-OS_Scan_train.pcap.csv": 3,
    "MQTT-Malformed_Data_train.pcap.csv": 4,
    "ARP_Spoofing_train.pcap.csv": 5,
}
CLASSES = ["Benign", "DDoS", "DoS", "Recon", "MQTT", "Spoofing"]
DROP = {
    "flowid", "srcip", "dstip", "sourceip", "destinationip", "timestamp",
    "label", "attacktype", "attack", "attackcategory", "srcmac", "dstmac",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fingerprints(X):
    X = np.array(X, copy=True)
    X[X == 0] = 0  # Normalize signed zero before hashing.
    return pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()


def conflict_count(hashes, y):
    counts = pd.DataFrame({"group": hashes, "label": y}).groupby("group").label.nunique()
    return int((counts > 1).sum())


def merge_groups(groups, hashes):
    """Only merge: never separate groups identified on previous passes."""
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


def read_raw(root):
    arrays, labels, sources = [], [], []
    feature_names = None
    for filename, class_id in FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw training CSV: {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate column names: {filename}")
        columns = [c for c in frame if
                   ''.join(ch for ch in c.lower() if ch.isalnum()) not in DROP
                   and not c.lower().startswith('unnamed:')]
        if feature_names is None:
            feature_names = columns
            if len(feature_names) != 45:
                raise ValueError(f"Expected the audited 45-feature schema, found {len(columns)}: {columns}")
        if set(columns) != set(feature_names):
            raise ValueError(f"Feature schema differs in {filename}")
        values = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values).all(axis=1)
        sources.append({"file": filename, "sha256": sha256(path), "class_id": class_id,
                        "raw_rows": len(values), "nonfinite_rows_removed": int((~finite).sum()),
                        "retained_rows": int(finite.sum())})
        if not finite.any():
            raise ValueError(f"No finite numeric rows in {filename}")
        arrays.append(values[finite])
        labels.append(np.full(int(finite.sum()), class_id, dtype=np.int64))
        print(f"{filename}: {int(finite.sum()):,} retained; {int((~finite).sum()):,} nonfinite removed", flush=True)
    return np.concatenate(arrays), np.concatenate(labels), feature_names, sources


def make_clients(y, n, alpha, seed):
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(n)]
    for c in range(6):
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out", default="data/processed_grouped")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-clients", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-passes", type=int, default=12)
    args = p.parse_args()
    if args.num_clients < 3 or args.max_passes < 1 or not np.isfinite(args.alpha) or args.alpha <= 0:
        p.error("Need >=3 clients, positive alpha and >=1 pass.")
    if not 0 <= args.seed <= 2**32 - 1:
        p.error("Seed must be in [0, 2**32 - 1].")
    root, out = Path(args.raw_dir), Path(args.out)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}. Use a new --out path; existing data is preserved.")

    X, y, names, sources = read_raw(root)
    raw_hash = fingerprints(X)
    _, groups = np.unique(raw_hash, return_inverse=True)
    history = []
    # Splitting is provisional until final stored features pass the overlap check.
    # Every pass refits a NEW scaler using only that pass's training rows.
    for attempt in range(1, args.max_passes + 1):
        print(f"Grouping/scaling pass {attempt}/{args.max_passes}...", flush=True)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
        train, val = next(splitter.split(np.zeros((len(y), 1)), y, groups))
        if len(np.unique(y[train])) != 6 or len(np.unique(y[val])) != 6:
            raise ValueError("A grouped split lacks a class. Audit group sizes; no row-wise fallback was used.")
        scaler = StandardScaler().fit(X[train])
        stored = scaler.transform(X).astype(np.float32)
        if not np.isfinite(stored).all():
            raise ValueError("Scaling produced nonfinite features; inspect raw feature ranges.")
        final_hash = fingerprints(stored)
        shared = np.intersect1d(final_hash[train], final_hash[val])
        history.append({"pass": attempt, "groups": int(len(np.unique(groups))),
                        "train_rows": len(train), "validation_rows": len(val),
                        "shared_float32_fingerprints": len(shared)})
        print(f"Pass {attempt}: train={len(train):,}, validation={len(val):,}, shared final fingerprints={len(shared):,}", flush=True)
        if len(shared) == 0:
            break
        groups = merge_groups(groups, final_hash)
    else:
        raise RuntimeError(f"No stable split after {args.max_passes} passes. Nothing saved. Pass history: {history}")

    raw_overlap = np.intersect1d(raw_hash[train], raw_hash[val])
    if len(raw_overlap):
        raise AssertionError("Raw feature groups crossed splits.")
    clients = make_clients(y[train], args.num_clients, args.alpha, args.seed)
    provenance = {
        "schema_version": "ciciomt_grouped_v1",
        "conflicting_label_policy": "preserve_within_split",
        "scope": "Six selected raw training CSVs; development subset, not full CICIoMT2024.",
        "official_test_loaded": False,
        "scaler_fit_on_final_training_only": True,
        "privacy": "Pooled training preprocessing in a simulator; trusted calibration reserved by the runner.",
        "split_method": "First fixed seeded fold of StratifiedGroupKFold(5); monotone grouping of float32 collisions.",
        "row_duplicates_retained": True,
        "raw_conflicting_groups": conflict_count(raw_hash, y),
        "train_conflicting_float32_groups": conflict_count(final_hash[train], y[train]),
        "validation_conflicting_float32_groups": conflict_count(final_hash[val], y[val]),
        "raw_overlap": 0, "float32_overlap": 0,
    }
    manifest = {
        **provenance, "settings": vars(args), "source_files": sources,
        "classes": CLASSES, "features": names, "passes": history,
        "train_rows": len(train), "validation_rows": len(val),
        "actual_validation_fraction": len(val) / len(y),
        "class_counts": {"train": np.bincount(y[train], minlength=6).tolist(),
                         "validation": np.bincount(y[val], minlength=6).tolist()},
        "clients": {str(i): {"rows": len(idx), "class_counts": np.bincount(y[train][idx], minlength=6).tolist()}
                    for i, idx in clients.items()},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "torch": str(torch.__version__)},
    }
    out.mkdir(parents=True, exist_ok=False)
    torch.save({
        "train_X": torch.from_numpy(stored[train].copy()),
        "train_y_type": torch.from_numpy(y[train].copy()),
        "train_y_binary": torch.from_numpy((y[train] != 0).astype(np.int64)),
        "val_X": torch.from_numpy(stored[val].copy()),
        "val_y_type": torch.from_numpy(y[val].copy()),
        "val_y_binary": torch.from_numpy((y[val] != 0).astype(np.int64)),
        "client_indices": clients, "preparation": provenance,
    }, out / "federated_data.pt")
    np.savez_compressed(out / "split_indices.npz", train=train, validation=val, groups=groups)
    joblib.dump(scaler, out / "scaler.pkl")
    write_json(out / "feature_names.json", {"features": names})
    write_json(out / "partition_manifest.json", manifest)
    shutil.copyfile(__file__, out / "prepare_federated_source.py")
    print(f"Completed {out}: train={len(train):,}, validation={len(val):,}; final feature overlap=0.", flush=True)
    print(f"Conflicting-label training groups retained: {provenance['train_conflicting_float32_groups']}; no labels were reassigned.")


if __name__ == "__main__":
    main()
