"""Data-quality gate — run before training so a bad CSV fails fast.

    python training/check_quality.py datasets/isl_landmarks.csv 63
"""
import sys

import numpy as np
import pandas as pd

N_META = 3  # label, source, frame_id


def check(path: str, expected_dim: int) -> int:
    df = pd.read_csv(path, header=None)
    feature_cols = df.shape[1] - N_META
    failures: list[str] = []
    warnings: list[str] = []

    if feature_cols != expected_dim:
        failures.append(f"Wrong dim: got {feature_cols}, expected {expected_dim}")

    counts = df.iloc[:, 0].value_counts()
    ratio = counts.min() / counts.max()
    if ratio < 0.5:
        warnings.append(f"Imbalance: min/max class ratio = {ratio:.2f}")

    feats = df.iloc[:, N_META:].astype(float)
    if feats.isna().any().any():
        failures.append("NaN values present")
    if np.isinf(feats.to_numpy()).any():
        failures.append("Inf values present")

    sources = df.iloc[:, 1].nunique()
    if sources < 2:
        warnings.append(f"Only {sources} source(s); want >= 2")

    for w in warnings:
        print(f"WARN: {w}")
    if failures:
        for fail in failures:
            print(f"FAIL: {fail}")
        return 1
    print("Data quality: OK")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_quality.py <landmarks.csv> <expected_dim>")
        raise SystemExit(2)
    raise SystemExit(check(sys.argv[1], int(sys.argv[2])))
