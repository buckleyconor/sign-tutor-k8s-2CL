"""MediaPipe Hands wrapper.

Outputs 21 landmarks per detected hand. We always run with ``max_hands=2`` even
though ASL/ISL are one-handed — the cost is negligible and it keeps the capture
layer language-agnostic.
"""

import numpy as np

try:
    import mediapipe as mp
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    # Only catch a genuinely-missing package. Other ImportErrors (e.g. a
    # transitive C-lib failure like "libGL.so.1: cannot open") must propagate
    # with their real message, not be masked as "mediapipe is required".
    raise ModuleNotFoundError(
        "mediapipe is required for hand tracking. Install it inside the "
        "tutor-app container (see Dockerfile)."
    ) from exc


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        # static_image_mode=True is used for dataset extraction (independent
        # images); False for the live webcam stream (uses tracking between
        # frames).
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray):
        """Return a list of ``(handedness, landmarks_array)`` tuples.

        ``landmarks_array`` is shape ``(21, 3)``.
        """
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        out = []
        for hand_idx, hand_lms in enumerate(result.multi_hand_landmarks):
            handedness = result.multi_handedness[hand_idx].classification[0].label
            arr = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            out.append((handedness, arr))
        return out
