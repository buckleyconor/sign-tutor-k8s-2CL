"""Lesson controller — wires capture -> features -> inference -> scoring.

This is the application brain behind the Gradio UI. It keeps per-session state
(active language, current target letter, smoother, scorer) and turns each webcam
frame into an annotated frame plus a traffic-light score.

The fuzzy bits (landmark overlay drawing, celebration animation) are left as
clearly marked TODOs — they do not affect the deterministic core that the test
suite exercises.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import yaml

from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.inference.triton_client import TritonClassifier
from src.lesson.scorer import Light, TrafficLightScorer
from src.lesson.smoother import PredictionSmoother
from src.registry import Language

_LIGHT_COLOURS = {
    Light.RED: "#E74C3C",
    Light.AMBER: "#FFBF00",  # aligned to the UI accent
    Light.GREEN: "#27AE60",
}


def render_light(light: Light = Light.RED) -> str:
    colour = _LIGHT_COLOURS[light]
    return (
        f'<div style="width:60px;height:60px;border-radius:50%;'
        f'background:{colour};margin:auto;box-shadow:0 0 12px {colour};"></div>'
    )


def _quality_colour(pct: float) -> str:
    # Visual mapping requested for the bar: red <=40, amber 41-75, green 75+.
    # (Independent of the scorer's completion thresholds, which still gate
    # letter advancement.)
    if pct <= 40:
        return "#E74C3C"
    if pct <= 75:
        return "#FFBF00"
    return "#27AE60"


def render_quality_bar(pct: float) -> str:
    """Render the 0-100% quality bar with a target marker at 90%.

    The fill width/colour are inline so each update reflects the latest value;
    smoothness comes from the controller's EMA (small, frequent steps) plus the
    CSS transition on ``.quality-fill`` (see ``ui/theme.py``).
    """
    pct = max(0.0, min(100.0, float(pct)))
    colour = _quality_colour(pct)
    return (
        '<div class="quality-wrap">'
        '<div class="quality-track">'
        f'<div class="quality-fill" style="width:{pct:.0f}%;background:{colour};">'
        "</div>"
        '<div class="quality-target" style="left:90%;"></div>'
        "</div>"
        '<div class="quality-meta">'
        f"<span>Quality: {pct:.0f}%</span>"
        '<span class="quality-target-label">target 90%</span>'
        "</div>"
        "</div>"
    )


def _load_thresholds(path: Path = Path("configs/thresholds.yaml")) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


class LessonController:
    def __init__(self, languages: dict[str, Language], triton_url: str | None = None):
        self._languages = languages
        self._triton_url = triton_url or os.environ.get("TRITON_URL", "triton:8000")
        self._cfg = _load_thresholds()
        self._tracker = HandTracker(max_hands=2)
        self._clients: dict[str, TritonClassifier] = {}
        self._references: dict[str, dict[str, np.ndarray]] = {}

        # MediaPipe's graph is not thread-safe and navigation mutates the same
        # smoother/scorer/index a streaming frame reads. One participant per pod,
        # but multiple tabs (and the frame vs. button race) still need this lock.
        # The webcam stream is also pinned to concurrency_limit=1 in ui/app.py.
        self._lock = threading.Lock()

        self._active: Language | None = None
        self._letter_idx: int = 0
        self._quality_ema: float = 0.0  # smoothed 0-1 quality for the bar
        self._ema_alpha: float = float(self._cfg.get("quality_ema_alpha", 0.35))
        self._smoother = PredictionSmoother(
            window=int(self._cfg.get("smoothing_window_frames", 15))
        )
        self._scorer: TrafficLightScorer | None = None
        if languages:
            self.set_language(next(iter(languages)))

    # ------------------------------------------------------------------ state
    def _client_for(self, lang: Language) -> TritonClassifier:
        if lang.code not in self._clients:
            self._clients[lang.code] = TritonClassifier(
                url=self._triton_url, model_name=lang.triton_model_name
            )
        return self._clients[lang.code]

    def _reference_image(self, lang: Language, letter: str) -> np.ndarray | None:
        cache = self._references.setdefault(lang.code, {})
        if letter not in cache:
            path = lang.references_dir / f"{letter}.png"
            if not path.exists():
                cache[letter] = None  # e.g. ISL Q.png is currently missing
            else:
                import cv2

                img = cv2.imread(str(path))
                cache[letter] = (
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
                )
        return cache[letter]

    def set_language(self, code: str):
        self._active = self._languages[code]
        self._letter_idx = 0
        self._quality_ema = 0.0
        self._smoother.reset()
        self._new_scorer()

    def _new_scorer(self):
        self._scorer = TrafficLightScorer(
            target_idx=self._letter_idx,
            hold_seconds=float(self._cfg.get("hold_seconds_for_complete", 1.0)),
            amber_min=float(self._cfg.get("amber_min_confidence", 0.50)),
            green_min=float(self._cfg.get("green_min_confidence", 0.80)),
        )

    @property
    def current_letter(self) -> str:
        return self._active.classes[self._letter_idx]

    def render_light(self, light: Light = Light.RED) -> str:
        return render_light(light)

    # ------------------------------------------------------------- navigation
    def next_letter(self):
        if self._active is None:
            return
        self._letter_idx = (self._letter_idx + 1) % len(self._active.classes)
        self._quality_ema = 0.0
        self._smoother.reset()
        self._new_scorer()

    def skip(self):
        self.next_letter()

    # ----------------------------------------------------------- UI view hooks
    # These return ``(reference_image, quality_html, status_md)`` so the webcam
    # stream is no longer the only thing that can paint the reference panel.
    def current_view(self) -> tuple[np.ndarray | None, str, str]:
        if self._active is None:
            return None, render_quality_bar(0.0), ""
        reference = self._reference_image(self._active, self.current_letter)
        status = (
            f"Sign **{self.current_letter}** — match the reference, then hold steady."
        )
        return reference, render_quality_bar(self._quality_ema), status

    def on_language_change(self, code: str):
        with self._lock:
            self.set_language(code)
            return self.current_view()

    def on_next(self):
        with self._lock:
            self.next_letter()
            return self.current_view()

    def on_skip(self):
        with self._lock:
            self.skip()
            return self.current_view()

    # --------------------------------------------------------------- per-frame
    def on_frame(self, frame: np.ndarray, lang_code: str):
        """Process one webcam frame.

        Returns ``(reference_image, quality_html, status_md)``. The webcam
        preview renders client-side, so we no longer echo the frame back to the
        input component (that churned the stream and bought nothing while the
        landmark overlay is still a TODO). The whole body is lock-guarded
        because MediaPipe's graph is not thread-safe.
        """
        with self._lock:
            if self._active is None or lang_code != self._active.code:
                self.set_language(lang_code)
            lang = self._active
            reference = self._reference_image(lang, self.current_letter)

            target_q = 0.0  # 0-1 confidence *for the target class* this frame
            status = f"Sign **{self.current_letter}** — show your hand to the camera."

            detections = self._tracker.process(frame) if frame is not None else []
            feat = build_feature_vector(lang, detections)
            if feat is not None:
                logits = self._client_for(lang).infer(feat)
                probs = _softmax(logits)
                pred_idx = int(np.argmax(probs))
                self._smoother.update(pred_idx, float(probs[pred_idx]))
                smoothed = self._smoother.smoothed()
                if smoothed is None:
                    status = "Hold steady…"
                else:
                    s_idx, s_conf = smoothed
                    _light, completed = self._scorer.evaluate(s_idx, s_conf)
                    target_q = s_conf if s_idx == self._letter_idx else 0.0
                    status = (
                        f"Sign **{self.current_letter}** — "
                        f"predicted **{lang.classes[s_idx]}** ({s_conf:.0%})"
                    )
                    if completed:
                        self.next_letter()  # resets _quality_ema to 0
                        reference = self._reference_image(lang, self.current_letter)
                        target_q = 0.0
                        status = (
                            f"✅ **{lang.classes[s_idx]}** complete! "
                            f"Next: {self.current_letter}"
                        )

            # Ease the bar toward this frame's quality so it grows/recedes
            # smoothly instead of snapping.
            self._quality_ema += self._ema_alpha * (
                target_q * 100.0 - self._quality_ema
            )
            return reference, render_quality_bar(self._quality_ema), status


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
