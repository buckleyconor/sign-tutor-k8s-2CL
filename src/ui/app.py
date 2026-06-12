"""Gradio app — split-screen tutor (top) + embedded terminal (bottom).

Run with ``python -m src.ui.app``. The webcam is browser-side (HTTPS via the
NGINX Ingress); the terminal executes server-side inside this pod.
"""

import gradio as gr

from src.lesson.controller import LessonController, render_quality_bar
from src.registry import load_registry
from src.terminal.executor import run_command
from src.ui.theme import CSS, THEME

LANGS = load_registry()


def build_app() -> gr.Blocks:
    controller = LessonController(LANGS)
    default_code = next(iter(LANGS), "asl")

    with gr.Blocks(title="Sign Language Tutor", theme=THEME, css=CSS) as demo:
        gr.Markdown(
            "This lab teaches **fingerspelling** — one component of sign language."
        )
        with gr.Row():
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
                cam = gr.Image(sources=["webcam"], streaming=True, label="Live feed")
            with gr.Column(scale=1):
                target = gr.Image(label="Sign this letter", interactive=False)
                quality = gr.HTML(value=render_quality_bar(0.0))
                status = gr.Markdown()
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next letter")

        # Embedded terminal — bottom 25%, full width (Module 2 lab commands).
        with gr.Row():
            with gr.Column(elem_id="lab-terminal"):
                term_out = gr.Code(
                    label="Terminal", language="shell", interactive=False, lines=10
                )
                term_in = gr.Textbox(
                    label="Command",
                    placeholder="python training/train_classifier.py ...",
                )

        views = [target, quality, status]
        # Throttle to a rate the MediaPipe+Triton pipeline can sustain and pin to
        # one in-flight frame, so the queue can't back up into the freeze/spinner
        # seen in the HAR. The frame is no longer echoed to `cam`.
        cam.stream(
            controller.on_frame,
            inputs=[cam, lang],
            outputs=views,
            stream_every=0.2,
            concurrency_limit=1,
            concurrency_id="frame",
            show_progress="hidden",
        )
        # Paint the reference letter on load and on every navigation — no longer
        # dependent on a webcam frame arriving first.
        demo.load(controller.current_view, outputs=views)
        lang.change(controller.on_language_change, inputs=lang, outputs=views)
        skip_btn.click(controller.on_skip, outputs=views)
        next_btn.click(controller.on_next, outputs=views)
        term_in.submit(
            run_command, inputs=[term_in, term_out], outputs=[term_out, term_in]
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
