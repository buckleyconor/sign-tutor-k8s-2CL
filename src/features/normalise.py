"""Landmark normalisation — translation/scale invariant feature vectors."""
import numpy as np


def normalise_one_hand(landmarks: np.ndarray) -> np.ndarray:
    """``landmarks``: (21, 3) array. Returns a (63,) normalised float32 vector.

    - Translate so the wrist (landmark 0) is at the origin.
    - Scale so the distance wrist(0) -> middle-finger MCP(9) is 1.0.
    - Flatten (21, 3) -> (63,).

    A degenerate input (wrist and MCP coincident) is handled by falling back to
    a unit scale, so the output is always finite.
    """
    assert landmarks.shape == (21, 3), f"expected (21, 3), got {landmarks.shape}"
    wrist = landmarks[0]
    centred = landmarks - wrist
    scale = np.linalg.norm(centred[9])
    if scale < 1e-6:
        scale = 1.0
    return (centred / scale).astype(np.float32).flatten()
