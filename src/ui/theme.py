"""UI palette — single source of truth.

Black background, lime-green element borders, white/grey text, amber for
highlights only. Retune here without touching layout code.
"""
import gradio as gr

BG = "#000000"      # black background
BORDER = "#32CD32"  # lime green element borders
TEXT = "#E0E0E0"    # white/grey body text
ACCENT = "#FFBF00"  # amber — highlights only

CSS = f"""
.gradio-container, .gradio-container * {{
    background-color: {BG};
    color: {TEXT};
}}
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-box, .gradio-container .gr-panel {{
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
}}
.gradio-container .highlight, .gradio-container .amber {{ color: {ACCENT}; }}
#lab-terminal textarea {{
    background: {BG}; color: {TEXT};
    font-family: monospace; border: 1px solid {BORDER} !important;
}}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
).set(body_background_fill=BG, body_text_color=TEXT)
