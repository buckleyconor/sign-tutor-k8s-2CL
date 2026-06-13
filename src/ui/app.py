"""Gradio app — split-screen tutor (top) + embedded terminal (bottom).

Run with ``python -m src.ui.app``. The webcam is browser-side (HTTPS via the
NGINX Ingress); the terminal executes server-side inside this pod.
"""

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

TITLE_TEXT = "Sign Language Tutor app, Powered by Dell Technologies and NVIDIA"
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
        gr.HTML(TITLE_TEXT, elem_id="app-title")
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
                    label="Terminal", language="shell", interactive=False, lines=20
                )
                term_in = gr.Textbox(
                    label="Enter your Commands here",
                    placeholder="python training/train_classifier.py ...",
                    elem_id="command-input",
                    lines=2,
                    max_lines=12,  # accept multi-line / pasted commands
                )

        # Navigation hooks own the reference panel + Next-button lock state.
        nav_views = [target_letter, target, progress, quality, status, next_btn]
        # The stream stays lightweight: quality bar, status, and the one-shot
        # button-unlock update — no reference image. Pushing the image here is
        # what made the panel reload-blank and starved Skip in the HAR.
        stream_views = [quality, status, next_btn]
        # Throttle to a rate the MediaPipe+Triton pipeline can sustain and pin to
        # one in-flight frame, so the queue can't back up into the freeze/spinner
        # seen in the HAR. The frame is no longer echoed to `cam`.
        cam.stream(
            controller.on_frame,
            inputs=[cam, lang],
            outputs=stream_views,
            stream_every=0.2,
            concurrency_limit=1,
            concurrency_id="frame",
            show_progress="hidden",
        )
        # Paint the reference letter on load and on every navigation — no longer
        # dependent on a webcam frame arriving first. ``initial_view`` differs
        # only in the opening status prompt ("Click the record button…").
        demo.load(controller.initial_view, outputs=nav_views)
        lang.change(controller.on_language_change, inputs=lang, outputs=nav_views)
        skip_btn.click(controller.on_skip, outputs=nav_views)
        next_btn.click(controller.on_next, outputs=nav_views)
        term_in.submit(
            run_command, inputs=[term_in, term_out], outputs=[term_out, term_in]
        ).then(None, None, None, js=_SCROLL_TERMINAL_JS)
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
