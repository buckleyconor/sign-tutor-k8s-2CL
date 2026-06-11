"""Landmark-space augmentation (applied per-batch at training time).

Cheap regularisation on the 63-float vector: small noise, in-plane rotation, and
mirror flip. Cleaner than pixel augmentation — no risk of distorting the hand
into something unrealistic.
"""
import numpy as np


def augment_one_hand(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``vec``: (63,) normalised landmark vector. Returns an augmented (63,)."""
    pts = vec.reshape(21, 3).copy()
    # Gaussian noise — MediaPipe is itself noisy.
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    # In-plane rotation around the wrist (±10°).
    theta = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    # Mirror with 50% probability — one-handed signs are mostly handedness-agnostic.
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
    return pts.flatten().astype(np.float32)
