"""Feature layer — turn MediaPipe detections into model input vectors."""

import numpy as np

from src.registry import Language

from .normalise import normalise_one_hand

__all__ = ["build_feature_vector", "normalise_one_hand"]


def build_feature_vector(language: Language, detections) -> np.ndarray | None:
    """Build the language-appropriate input vector from MediaPipe detections.

    Returns ``None`` when there is no usable detection.
    """
    if not detections:
        return None
    # One-handed languages (ASL/ISL): take whichever hand was detected first.
    return normalise_one_hand(detections[0][1])
