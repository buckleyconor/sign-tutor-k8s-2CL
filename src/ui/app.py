"""Gradio app — split-screen tutor (top) + embedded terminal (bottom).

Run with ``python -m src.ui.app``. The webcam is browser-side (HTTPS via the
NGINX Ingress); the terminal executes server-side inside this pod.
"""
import gradio as gr

from src.lesson.controller import LessonController
from src.lesson.scorer import Light
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
            lesson = gr.Dropdown(
                choices=[("Alphabet", "alphabet")],
                value="alphabet",
                label="Lesson",
            )

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(sources=["webcam"], streaming=True, label="Live feed")
            with gr.Column(scale=1):
                target = gr.Image(label="Sign this letter", interactive=False)
                light = gr.HTML(value=controller.render_light(Light.RED))
                status = gr.Markdown()
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    next_btn = gr.Button("Next letter")

        # Embedded terminal — bottom 25%, full width (Module 2 lab commands).
        with gr.Row():
            with gr.Column(elem_id="lab-terminal"):
                term_out = gr.Code(label="Terminal", language="shell",
                                   interactive=False, lines=10)
                term_in = gr.Textbox(
                    label="Command",
                    placeholder="python training/train_classifier.py ...",
                )

        cam.stream(
            controller.on_frame,
            inputs=[cam, lang],
            outputs=[cam, target, light, status],
        )
        lang.change(lambda code: controller.set_language(code), inputs=lang)
        skip_btn.click(lambda: controller.skip())
        next_btn.click(lambda: controller.next_letter())
        term_in.submit(
            run_command, inputs=[term_in, term_out], outputs=[term_out, term_in]
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
