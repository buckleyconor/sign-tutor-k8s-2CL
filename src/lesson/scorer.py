"""Traffic-light scoring.

RED   : conf < amber_min OR predicted != target
AMBER : amber_min <= conf < green_min AND predicted == target
GREEN : conf >= green_min AND predicted == target, sustained for hold_seconds
"""

import time
from enum import Enum


class Light(Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


class TrafficLightScorer:
    def __init__(
        self,
        target_idx: int,
        hold_seconds: float = 1.0,
        amber_min: float = 0.50,
        green_min: float = 0.80,
        clock=time.monotonic,
    ):
        self._target = target_idx
        self._hold = hold_seconds
        self._amber_min = amber_min
        self._green_min = green_min
        self._clock = clock  # injectable for deterministic tests
        self._green_since: float | None = None

    def set_target(self, target_idx: int):
        self._target = target_idx
        self._green_since = None

    def evaluate(self, pred_idx: int, conf: float) -> tuple[Light, bool]:
        """Return ``(light, completed)``."""
        now = self._clock()
        if pred_idx != self._target or conf < self._amber_min:
            self._green_since = None
            return Light.RED, False
        if conf < self._green_min:
            self._green_since = None
            return Light.AMBER, False
        # Green territory
        if self._green_since is None:
            self._green_since = now
        completed = (now - self._green_since) >= self._hold
        return Light.GREEN, completed
