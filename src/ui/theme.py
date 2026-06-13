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
LIME = "#32CD32"  # active "Next letter" button
LOCKED = "#D3D3D3"  # light grey — locked "Next letter" button
BLACK = "#000000"  # black panels (title, terminal, feedback, etc.)
WHITE = "#FFFFFF"  # text on black panels
TERMINAL_HEIGHT = "420px"  # fixed terminal viewport (~20 lines); scrolls past

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

/* Title frame: black panel, white text, 16pt. */
#app-title {{
    background: {BLACK};
    color: {WHITE};
    font-size: 16pt;
    font-weight: 600;
    padding: 12px 16px;
    border-radius: 6px;
    border: 1px solid {BORDER};
}}

/* Language / Lesson row: black frame, white text. */
#lang-lesson, #lang-lesson .block, #lang-lesson .form,
#lang-lesson .wrap, #lang-lesson .gr-box {{
    background: {BLACK} !important;
}}
#lang-lesson, #lang-lesson * {{ color: {WHITE} !important; }}
#lang-lesson input, #lang-lesson .secondary-wrap,
#lang-lesson .wrap-inner {{ background: {BLACK} !important; color: {WHITE} !important; }}

/* Live feed: black background + white text before the webcam starts. */
#live-feed, #live-feed .block {{ background: {BLACK} !important; }}
#live-feed, #live-feed * {{ color: {WHITE} !important; }}

/* Embedded terminal: black, fixed-height viewport that scrolls (auto-scrolled
   to the newest line by JS in ui/app.py) instead of growing. */
#lab-terminal .cm-editor, #lab-terminal .cm-scroller,
#lab-terminal .cm-content, #lab-terminal .cm-gutters {{
    background: {BLACK} !important;
    color: {WHITE};
}}
#lab-terminal .cm-editor {{ max-height: {TERMINAL_HEIGHT}; }}
#lab-terminal .cm-scroller {{
    max-height: {TERMINAL_HEIGHT};
    overflow: auto;
}}
/* Command input: black box, white monospace text. */
#command-input textarea, #lab-terminal #command-input textarea {{
    max-height: {TERMINAL_HEIGHT};
    overflow: auto;
    background: {BLACK} !important; color: {WHITE} !important;
    font-family: monospace; border: 1px solid {BORDER} !important;
}}

/* Quality bar (replaces the traffic light). Smoothness = controller EMA +
   this CSS transition on the fill. */
.quality-wrap {{ width: 100%; margin: 4px 0 2px; }}
.quality-track {{
    position: relative;
    width: 100%;
    height: 22px;
    background: {BLACK};
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
    background: {WHITE};
}}
.quality-meta {{
    display: flex;
    justify-content: space-between;
    font-size: 0.8em;
    color: {TEXT};
    margin-top: 2px;
}}
.quality-target-label {{ color: {ACCENT}; }}

/* "Next letter" button: locked = light grey, active = lime, always black text.
   The controller toggles `interactive`, which renders Gradio's `disabled`
   attribute — so the state is driven entirely off `[disabled]` here. */
#next-letter-btn button {{
    background: {LOCKED} !important;
    color: {TEXT} !important;
    border: 1px solid {LOCKED} !important;
    cursor: not-allowed;
    opacity: 1 !important;
}}
#next-letter-btn button:not([disabled]) {{
    background: {LIME} !important;
    color: {TEXT} !important;
    border: 1px solid {LIME} !important;
    cursor: pointer;
}}

/* Feedback status box: black panel, white 14pt text. */
#feedback-status {{
    background: {BLACK} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px;
    padding: 8px 12px;
    min-height: 2.4em;
}}
#feedback-status, #feedback-status * {{
    color: {WHITE} !important;
    font-size: 14pt !important;
}}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
).set(body_background_fill=BG, body_text_color=TEXT)
