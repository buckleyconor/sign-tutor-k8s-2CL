"""Deterministic synthetic landmark fixtures for unit tests."""

import numpy as np


def synthetic_hand(seed: int = 0, translate=(0, 0, 0), scale: float = 1.0):
    """Return a deterministic (21, 3) landmark array with the wrist at origin."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1, 1, size=(21, 3)).astype(np.float32)
    base[0] = 0.0  # wrist
    return base * scale + np.array(translate, dtype=np.float32)
