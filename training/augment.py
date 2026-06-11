"""Landmark-space augmentation (applied per-batch at training time).

Cheap regularisation on the 63-float vector: small noise and in-plane rotation.
Cleaner than pixel augmentation — no risk of distorting the hand into something
unrealistic.

Mirror flip is **opt-in and off by default**. spec_04 §8 originally recommended
it ("one-handed signs are mostly handedness-agnostic"), but an ISL ablation
showed it consistently *lowers* accuracy (~1-4 pts) — a mirrored handshape can
resemble a different letter. Enable it only for a dataset known to benefit.
"""
import numpy as np


def augment_one_hand(vec: np.ndarray, rng: np.random.Generator,
                     mirror: bool = False) -> np.ndarray:
    """``vec``: (63,) normalised landmark vector. Returns an augmented (63,).

    ``mirror``: if True, flip x with 50% probability (off by default).
    """
    pts = vec.reshape(21, 3).copy()
    # Gaussian noise — MediaPipe is itself noisy.
    pts += rng.normal(0, 0.01, pts.shape).astype(np.float32)
    # In-plane rotation around the wrist (±10°).
    theta = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pts = pts @ R.T
    if mirror and rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
    return pts.flatten().astype(np.float32)
