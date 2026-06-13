"""Gradio app — split-screen tutor (top) + embedded terminal (bottom).

Run with ``python -m src.ui.app``. The webcam is browser-side (HTTPS via the
NGINX Ingress); the terminal executes server-side inside this pod.
"""

import base64
import logging
import os
from pathlib import Path

import gradio as gr

from src.lesson.controller import (
    LessonController,
    render_progress,
    render_quality_bar,
    render_target_letter,
)
from src.registry import load_registry
from src.terminal.executor import run_command
from src.ui.theme import CSS, THEME

LANGS = load_registry()

_LOGO_DIR = Path("Logos")


def _logo_data_uri(filename: str) -> str:
    """Inline a PNG logo as a base64 data URI (no runtime file-serving needed).
    Returns '' if the file is missing so the title still renders off-cluster."""
    try:
        data = base64.b64encode((_LOGO_DIR / filename).read_bytes()).decode()
        return f"data:image/png;base64,{data}"
    except OSError:
        return ""


def _title_html() -> str:
    """Title bar: heading on the left, Dell + NVIDIA logos pinned top-right."""
    dell = _logo_data_uri("DellTech_Logo_Prm_Wht_rgb.png")
    nvidia = _logo_data_uri("nvidia-logo-horiz-1cwht-16x9.png")
    logos = "".join(
        f'<img src="{uri}" alt="{alt}" style="height:34px;width:auto;"/>'
        for uri, alt in ((dell, "Dell Technologies"), (nvidia, "NVIDIA"))
        if uri
    )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'width:100%;gap:16px;">'
        '<span style="font-size:16pt;font-weight:600;color:#fff;">'
        "Sign Language Tutor, Train your first vision model"
        "</span>"
        '<span style="display:flex;align-items:center;gap:24px;flex:0 0 auto;">'
        f"{logos}</span>"
        "</div>"
    )


# Scroll the embedded terminal to the newest line after each command runs.
_SCROLL_TERMINAL_JS = (
    "() => { setTimeout(() => {"
    " const e = document.querySelector('#lab-terminal .cm-scroller');"
    " if (e) { e.scrollTop = e.scrollHeight; } }, 60); }"
)


def build_app() -> gr.Blocks:
    controller = LessonController(LANGS)
    default_code = next(iter(LANGS), "asl")

    with gr.Blocks(title="Sign Language Tutor", theme=THEME, css=CSS) as demo:
        gr.HTML(_title_html(), elem_id="app-title")
        with gr.Row(elem_id="lang-lesson"):
            lang = gr.Dropdown(
                choices=[(lng.name, lng.code) for lng in LANGS.values()],
                value=default_code,
                label="Language",
            )
            gr.Dropdown(  # registered in the layout on creation; no handler yet
                choices=[("Alphabet", "alphabet")],
                value="alphabet",
                label="Lesson",
            )

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="Live feed",
                    elem_id="live-feed",
                )
            with gr.Column(scale=1):
                target_letter = gr.HTML(value=render_target_letter("—"))
                target = gr.Image(label="Reference", interactive=False)
                progress = gr.HTML(value=render_progress(0, 0))
                quality = gr.HTML(value=render_quality_bar(0.0))
                status = gr.Markdown(
                    value="<-- Click the record button to begin!",
                    elem_id="feedback-status",
                )
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    # Locked (grey) by default; the controller latches it active
                    # (lime) once the quality bar crosses 90%. Colours come from
                    # the disabled/enabled CSS in ui/theme.py.
                    next_btn = gr.Button(
                        "Next letter",
                        elem_id="next-letter-btn",
                        interactive=False,
                    )

        # Embedded terminal — bottom 25%, full width (Module 2 lab commands).
        with gr.Row():
            with gr.Column(elem_id="lab-terminal"):
                term_out = gr.Code(
                    label="Terminal",
                    language="shell",
                    interactive=False,
                    lines=20,
                    elem_id="terminal-output",
                )
                with gr.Row():
                    term_in = gr.Textbox(
                        label="Enter your Commands here",
                        placeholder="python training/train_classifier.py ...",
                        elem_id="command-input",
                        lines=2,
                        max_lines=12,  # accept multi-line / pasted commands
                        scale=8,
                    )
                    exec_btn = gr.Button(
                        "Execute Command", elem_id="execute-btn", scale=1
                    )

        # Navigation hooks own the reference panel + Next-button lock state.
        nav_views = [target_letter, target, progress, quality, status, next_btn]
        # The stream stays lightweight: quality bar, status, and the one-shot
        # button-unlock update — no reference image. Pushing the image here is
        # what made the panel reload-blank and starved Skip in the HAR.
        stream_views = [quality, status, next_btn]
        # Continuous capture. ``concurrency_limit`` MUST be >1: with =1 a single
        # streaming session holds the only worker slot for the whole time_limit
        # window, so the webcam sends one frame then stalls for ~30s (the spinner
        # in debug_2.har). MediaPipe thread-safety is handled by the controller's
        # lock instead. ``time_limit`` is large so the stream doesn't reset/rejoin
        # mid-lesson. The frame is not echoed back to `cam` (browser renders it).
        cam.stream(
            controller.on_frame,
            inputs=[cam, lang],
            outputs=stream_views,
            stream_every=0.15,
            concurrency_limit=30,
            concurrency_id="frame",
            time_limit=3600,
            show_progress="hidden",
        )
        # Paint the reference letter on load and on every navigation — no longer
        # dependent on a webcam frame arriving first. ``initial_view`` differs
        # only in the opening status prompt ("Click the record button…").
        demo.load(controller.initial_view, outputs=nav_views)
        lang.change(controller.on_language_change, inputs=lang, outputs=nav_views)
        skip_btn.click(controller.on_skip, outputs=nav_views)
        next_btn.click(controller.on_next, outputs=nav_views)
        # Both Enter and the "Execute Command" button run the typed command.
        run_io = dict(inputs=[term_in, term_out], outputs=[term_out, term_in])
        term_in.submit(run_command, **run_io).then(
            None, None, None, js=_SCROLL_TERMINAL_JS
        )
        exec_btn.click(run_command, **run_io).then(
            None, None, None, js=_SCROLL_TERMINAL_JS
        )
    return demo


if __name__ == "__main__":
    # Line-buffered, timestamped logs to stdout so `kubectl logs deploy/tutor-app`
    # shows the per-frame diagnostics and any on_frame traceback in real time.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    # show_error surfaces server-side exceptions to the browser too (lab/debug
    # convenience). Disable for a hardened production deploy.
    show_error = os.environ.get("GRADIO_SHOW_ERROR", "1") not in ("0", "false", "")
    build_app().launch(
        server_name="0.0.0.0", server_port=7860, show_error=show_error
    )
