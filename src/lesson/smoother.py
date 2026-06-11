"""Prediction smoothing — suppress frame-by-frame jitter.

Raw predictions jitter heavily. The displayed prediction is the modal class over
a rolling window; the displayed confidence is the average confidence for that
modal class. This eliminates flicker without noticeable lag.
"""
from collections import Counter, deque

import numpy as np


class PredictionSmoother:
    def __init__(self, window: int = 15):
        self._window = window
        self._preds: deque[int] = deque(maxlen=window)
        self._confs: deque[float] = deque(maxlen=window)

    def update(self, pred_idx: int, confidence: float):
        self._preds.append(pred_idx)
        self._confs.append(confidence)

    def smoothed(self) -> tuple[int, float] | None:
        """Return ``(modal_idx, avg_conf)`` once warmed up, else ``None``."""
        if len(self._preds) < self._window // 2:
            return None
        modal, _ = Counter(self._preds).most_common(1)[0]
        avg_conf = float(
            np.mean([c for p, c in zip(self._preds, self._confs) if p == modal])
        )
        return modal, avg_conf

    def reset(self):
        self._preds.clear()
        self._confs.clear()
