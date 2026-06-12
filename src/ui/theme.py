"""UI palette — single source of truth.

Light-grey background, lime-green element borders, black text, amber for
highlights only. Retune here without touching layout code.
"""

import gradio as gr

BG = "#EDEDED"  # light grey background
PANEL = "#F7F7F7"  # slightly lighter panel/input fill
BORDER = "#32CD32"  # lime green element borders
TEXT = "#111111"  # black body text
ACCENT = "#FFBF00"  # amber — highlights only
TERMINAL_HEIGHT = "220px"  # fixed terminal viewport; scrolls past this

CSS = f"""
.gradio-container {{
    background-color: {BG};
    color: {TEXT};
}}
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-box, .gradio-container .gr-panel {{
    background-color: {PANEL};
    color: {TEXT};
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
}}
.gradio-container label, .gradio-container .label-wrap span {{ color: {TEXT}; }}
.gradio-container .highlight, .gradio-container .amber {{ color: {ACCENT}; }}

/* Embedded terminal: fixed-height viewport that scrolls instead of growing. */
#lab-terminal .cm-editor {{
    max-height: {TERMINAL_HEIGHT};
    background: {PANEL};
}}
#lab-terminal .cm-scroller {{
    max-height: {TERMINAL_HEIGHT};
    overflow: auto;
}}
#lab-terminal textarea {{
    max-height: {TERMINAL_HEIGHT};
    overflow: auto;
    background: {PANEL}; color: {TEXT};
    font-family: monospace; border: 1px solid {BORDER} !important;
}}

/* Quality bar (replaces the traffic light). Smoothness = controller EMA +
   this CSS transition on the fill. */
.quality-wrap {{ width: 100%; margin: 4px 0 2px; }}
.quality-track {{
    position: relative;
    width: 100%;
    height: 22px;
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 11px;
    overflow: hidden;
}}
.quality-fill {{
    height: 100%;
    border-radius: 11px 0 0 11px;
    transition: width 180ms linear, background-color 180ms linear;
}}
.quality-target {{
    position: absolute;
    top: -2px;
    width: 2px;
    height: 26px;
    background: {TEXT};
}}
.quality-meta {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8em;
    color: {TEXT};
    margin-top: 2px;
}}
.quality-target-label {{ color: {ACCENT}; }}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
).set(body_background_fill=BG, body_text_color=TEXT)
