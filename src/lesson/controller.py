"""Lesson controller — wires capture -> features -> inference -> scoring.

This is the application brain behind the Gradio UI. It keeps per-session state
(active language, current target letter, smoother, scorer) and turns each webcam
frame into an annotated frame plus a traffic-light score.

The fuzzy bits (landmark overlay drawing, celebration animation) are left as
clearly marked TODOs — they do not affect the deterministic core that the test
suite exercises.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.inference.triton_client import TritonClassifier
from src.lesson.scorer import Light, TrafficLightScorer
from src.lesson.smoother import PredictionSmoother
from src.registry import Language

log = logging.getLogger("tutor.controller")

# Verbose per-frame diagnostics (frame shape, per-step timing, hand counts).
# On by default while we chase the streaming failure; set TUTOR_FRAME_DEBUG=0 to
# quieten it. Errors are always logged regardless of this flag.
_FRAME_DEBUG = os.environ.get("TUTOR_FRAME_DEBUG", "1") not in ("0", "false", "")

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


def render_target_letter(letter: str) -> str:
    """Heading for the reference panel: ``Target Letter: A``."""
    return (
        '<div style="font-size:16pt;font-weight:600;color:#111;">'
        f'Target Letter: <span style="color:#32CD32;">{letter}</span>'
        "</div>"
    )


def render_progress(completed: int, total: int) -> str:
    """Dotted progress readout: filled dots for completed letters, ``o`` for the
    rest, e.g. ``Progress: ●●●ooooo… (3/26)``."""
    total = max(0, int(total))
    completed = max(0, min(total, int(completed)))
    done = "●" * completed
    todo = "o" * (total - completed)
    return (
        '<div style="font-family:monospace;font-size:14pt;color:#111;">'
        f'Progress: <span style="color:#32CD32;">{done}</span>'
        f'<span style="color:#999;">{todo}</span>'
        f'  <span style="font-size:10pt;">({completed}/{total})</span>'
        "</div>"
    )


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

        # Serialises all access to MediaPipe + the smoother/scorer/index, which
        # streaming frames and navigation (Skip/Next) both touch. The webcam
        # stream runs with concurrency_limit=30 (ui/app.py), so on_frame can be
        # dispatched from several Gradio worker threads — this lock is what makes
        # that safe. If logs pin the failure on MediaPipe being called across
        # threads, the next step is to pin frame processing to one thread.
        self._lock = threading.Lock()

        self._active: Language | None = None
        self._letter_idx: int = 0
        self._quality_ema: float = 0.0  # smoothed 0-1 quality for the bar
        self._ema_alpha: float = float(self._cfg.get("quality_ema_alpha", 0.35))
        # The "Next letter" button stays locked (grey) until the quality bar
        # crosses this threshold, then latches unlocked (lime) until the user
        # navigates. 90% matches the target marker on the bar.
        self._unlock_threshold: float = float(self._cfg.get("unlock_quality_pct", 90.0))
        self._unlocked: bool = False
        self._completed: int = 0  # letters finished (Next clicked while unlocked)
        self._smoother = PredictionSmoother(
            window=int(self._cfg.get("smoothing_window_frames", 15))
        )
        self._scorer: TrafficLightScorer | None = None
        # Streaming diagnostics.
        self._frame_count: int = 0
        self._error_count: int = 0
        if languages:
            self.set_language(next(iter(languages)))
        log.info(
            "LessonController ready: triton_url=%s languages=%s frame_debug=%s",
            self._triton_url,
            list(languages),
            _FRAME_DEBUG,
        )

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
        self._unlocked = False
        self._completed = 0
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
        self._unlocked = False
        self._smoother.reset()
        self._new_scorer()

    def skip(self):
        self.next_letter()

    # ----------------------------------------------------------- UI view hooks
    # Navigation hooks paint the reference panel and *re-lock* the Next button.
    # They return ``(reference_image, quality_html, status_md, next_btn_update)``.
    # The webcam stream (``on_frame``) deliberately does NOT repaint the
    # reference — pushing a fresh image 5x/second kept the "Sign this letter"
    # panel perpetually reloading (blank) and saturated the session's event
    # channel, which in turn made Skip take tens of seconds to land.
    def current_view(self) -> tuple:
        """``(target_letter, reference, progress, quality, status, next_btn)``."""
        locked = gr.update(interactive=False)
        if self._active is None:
            return (
                render_target_letter("—"),
                None,
                render_progress(0, 0),
                render_quality_bar(0.0),
                "",
                locked,
            )
        total = len(self._active.classes)
        reference = self._reference_image(self._active, self.current_letter)
        status = (
            f"Sign **{self.current_letter}** — match the reference, then hold steady."
        )
        return (
            render_target_letter(self.current_letter),
            reference,
            render_progress(self._completed, total),
            render_quality_bar(self._quality_ema),
            status,
            locked,
        )

    def initial_view(self) -> tuple:
        """Fired on every page load/refresh. Resets the lesson to the first
        letter (A) so a refresh always restarts from the beginning, then returns
        ``current_view`` with the app's opening status prompt."""
        with self._lock:
            if self._active is not None:
                self.set_language(self._active.code)  # back to letter 0, progress 0
            tl, ref, prog, quality, _status, locked = self.current_view()
        return tl, ref, prog, quality, "<-- Click the record button to begin!", locked

    def on_language_change(self, code: str):
        with self._lock:
            self.set_language(code)
            return self.current_view()

    def on_next(self):
        with self._lock:
            # Only a *completed* letter (Next clicked while unlocked) counts
            # toward progress; Skip deliberately does not.
            if self._unlocked:
                self._completed += 1
            self.next_letter()
            return self.current_view()

    def on_skip(self):
        with self._lock:
            self.skip()
            return self.current_view()

    # --------------------------------------------------------------- per-frame
    def on_frame(self, frame: np.ndarray, lang_code: str):
        """Process one webcam frame.

        Returns ``(quality_html, status_md, next_btn_update)`` — deliberately
        lightweight. The reference image is owned by the navigation hooks, not
        the stream, so the "Sign this letter" panel stays put and the per-frame
        payload is tiny (this is what keeps Skip/Next responsive). The webcam
        preview renders client-side, so we don't echo the frame back either.

        Advancing letters is now manual: when the quality bar crosses the
        unlock threshold the Next button latches active (lime); the user clicks
        it to move on. The whole body is lock-guarded because MediaPipe's graph
        is not thread-safe.
        """
        with self._lock:
            self._frame_count += 1
            n = self._frame_count
            # Log the first few frames in full, then 1-in-50, so the pod logs
            # show the stream is alive without flooding.
            verbose = _FRAME_DEBUG and (n <= 5 or n % 50 == 0)
            if verbose:
                shape = None if frame is None else getattr(frame, "shape", "n/a")
                dtype = None if frame is None else getattr(frame, "dtype", "n/a")
                log.info(
                    "on_frame #%d thread=%s lang=%s frame_shape=%s dtype=%s",
                    n,
                    threading.current_thread().name,
                    lang_code,
                    shape,
                    dtype,
                )

            if self._active is None or lang_code != self._active.code:
                self.set_language(lang_code)
            lang = self._active

            target_q = 0.0  # 0-1 confidence *for the target class* this frame
            status = f"Sign **{self.current_letter}** — show your hand to the camera."

            # Each pipeline stage is isolated so a failure (a) names the culprit
            # in the logs and (b) does NOT raise out of the handler — a raised
            # exception makes Gradio mark the whole stream "Error" and stop
            # delivering output (the silent-stream failure seen in diagnostic_3).
            try:
                t0 = time.perf_counter()
                detections = self._tracker.process(frame) if frame is not None else []
                t_track = time.perf_counter() - t0

                feat = build_feature_vector(lang, detections)
                if verbose:
                    log.info(
                        "  frame #%d hands=%d feat=%s track=%.1fms",
                        n,
                        len(detections),
                        None if feat is None else f"{feat.shape}",
                        t_track * 1000,
                    )

                if feat is not None:
                    t1 = time.perf_counter()
                    logits = self._client_for(lang).infer(feat)
                    t_infer = time.perf_counter() - t1
                    probs = _softmax(logits)
                    pred_idx = int(np.argmax(probs))
                    self._smoother.update(pred_idx, float(probs[pred_idx]))
                    smoothed = self._smoother.smoothed()
                    if smoothed is None:
                        status = "Hold steady…"
                    else:
                        s_idx, s_conf = smoothed
                        target_q = s_conf if s_idx == self._letter_idx else 0.0
                        status = (
                            f"Sign **{self.current_letter}** — "
                            f"predicted **{lang.classes[s_idx]}** ({s_conf:.0%})"
                        )
                    if verbose:
                        log.info(
                            "  frame #%d infer=%.1fms pred=%s conf=%.2f",
                            n,
                            t_infer * 1000,
                            lang.classes[pred_idx],
                            float(probs[pred_idx]),
                        )
            except Exception:
                self._error_count += 1
                # Full traceback to the pod logs; rate-limited so a persistent
                # failure doesn't spam, but the first occurrences are captured.
                if self._error_count <= 20 or self._error_count % 100 == 0:
                    log.exception(
                        "on_frame #%d FAILED (error #%d) lang=%s — keeping stream alive",
                        n,
                        self._error_count,
                        lang_code,
                    )
                # Surface a short hint in the UI without killing the stream.
                status = "⚠️ Detection error — see server logs (tutor-app)."
                return render_quality_bar(self._quality_ema), status, gr.update()

            # Ease the bar toward this frame's quality so it grows/recedes
            # smoothly instead of snapping.
            self._quality_ema += self._ema_alpha * (
                target_q * 100.0 - self._quality_ema
            )

            # Latch the Next button active once (lime); navigation re-locks it.
            # Emitting an update only on the transition avoids re-rendering the
            # button on every frame.
            btn_update = gr.update()
            if not self._unlocked and self._quality_ema >= self._unlock_threshold:
                self._unlocked = True
                status = f"✅ Great — click **Next letter** to continue ({self.current_letter} ✓)"
                btn_update = gr.update(interactive=True)

            return render_quality_bar(self._quality_ema), status, btn_update


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()
