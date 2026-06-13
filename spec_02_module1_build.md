# Module 1 — Detailed Build Spec

**Multi-Language Fingerspelling Alphabet**
**Spec 2 of 5**

---

## 1. Scope of this spec

This document is the implementation plan for Module 1 — fingerspelled alphabet recognition for ASL and ISL. It assumes the architecture in Spec 1. Each section below maps to a single, scoped deliverable. Every section ends with an exit criterion — something concrete you can demonstrate before moving on.

---

## 2. Repository Layout

```
sign-tutor-k8s/
├── README.md
├── Dockerfile                  # NVIDIA PyTorch 25.03-py3 base — tutor-app image
├── k8s/
│   ├── chart/                  # Helm chart — one release per participant namespace
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   └── templates/          # Deployments, Services, PVCs, Ingress, Certificate
│   ├── cluster-setup/          # One-time admin guides and manifests
│   └── scripts/
│       ├── deploy-participant.sh
│       └── reset-participant.sh
├── configs/
│   └── thresholds.yaml         # completion cutoffs, smoothing window, bar EMA
├── languages/
│   ├── asl/
│   │   ├── config.yaml
│   │   ├── model.onnx          # pre-built; bundled into onnx-models ConfigMap at deploy time
│   │   └── references/         # A.png ... Z.png
│   ├── isl/
│   │   ├── config.yaml
│   │   └── references/         # A.png ... Z.png — NOTE: Q.png currently missing (add, or UI must handle gracefully); model.onnx produced during lab
├── src/
│   ├── capture/                # webcam + MediaPipe wrapper
│   ├── features/               # landmark normalisation, packing
│   ├── inference/              # Triton client wrapper
│   ├── lesson/                 # lesson controller, scoring
│   ├── ui/                     # Gradio app (split-screen + embedded terminal)
│   ├── terminal/               # embedded-terminal command executor
│   │   ├── executor.py         # subprocess runner: allowlist, timeout, truncation
│   │   └── config.py           # allowlist, blocked patterns, limits
│   └── registry.py             # language registry loader
├── training/
│   ├── extract_landmarks.py
│   ├── train_classifier.py
│   ├── export_onnx.py
│   ├── augment.py
│   └── check_quality.py
├── triton_repo/                # Triton model repository config (config.pbtxt only)
│   ├── asl_classifier/
│   │   └── config.pbtxt        # model.plan built by init container at pod start
│   └── isl_classifier/
│       ├── config.pbtxt        # copied to shared PVC by participant; model.plan built during lab
│       └── config.dynamic.pbtxt # optional dynamic-batching variant (lab §10.2)
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

> **Exit criterion:** repo with this layout committed and pushed; `k8s/chart/` renders valid K8s manifests via `helm template`.

---

---

## 3. Environment Setup

### 3.1 Kubernetes deployment

The system runs as a Helm chart. It is developed and tested on a single-node **microk8s** cluster on the dev host, then deployed unchanged to **Canonical Charmed Kubernetes** for production labs — both have RTX PRO 6000 Blackwell (sm_120) GPUs, so the same TensorRT engines run in both. Each lab participant gets a dedicated namespace with two Deployments:

- **triton** — `nvcr.io/nvidia/tritonserver:26.04-py3`. An init container runs `trtexec` at startup to compile the bundled ASL ONNX model into a TensorRT engine for the cluster GPU (RTX PRO 6000, sm_120). The main container serves on ports 8000 (HTTP) and 8001 (gRPC) with poll-mode hot-reload. It mounts the shared `triton-models` PVC as its model repository.
- **tutor-app** — built from the Dockerfile below. Runs MediaPipe, Gradio, the lesson controller, and the embedded-terminal executor. Connects to Triton via the in-namespace `triton` Service. Exposed via NGINX Ingress with TLS at `https://p1.lab.internal`. It mounts the shared `triton-models` PVC **read-write**, so engines built here (via the base image's bundled `trtexec`) land directly in Triton's repository — no cross-pod copy.

> **Shared model repository:** `triton-models` is one PVC mounted read-write in tutor-app and read-only in triton. Its StorageClass is set in `values.yaml` — `microk8s-hostpath` on the dev host, Ceph RBD on Charmed Kubernetes. This is what makes the lab a sequence of local commands inside the tutor-app pod rather than cross-pod `kubectl cp`/`exec`.

Deploy a participant namespace:

```bash
bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
```

Reset a participant (wipe and redeploy):

```bash
bash k8s/scripts/reset-participant.sh p1 p1.lab.internal
```

### 3.2 tutor-app Dockerfile

```dockerfile
# TRT must match the Triton image (26.04 = TRT 10.16) so engines built in this
# pod load in Triton. pytorch:25.03 (TRT 10.9) builds but won't deserialize there.
FROM nvcr.io/nvidia/pytorch:26.04-py3
RUN pip install --no-cache-dir \
    mediapipe \
    opencv-python-headless \
    gradio \
    tritonclient[http,grpc] \
    pyyaml \
    pytest pytest-cov
WORKDIR /app
COPY . /app
CMD ["python", "-m", "src.ui.app"]
```

> **Note:** `opencv-python-headless` avoids GUI/X11 dependencies inside the container; Gradio handles all UI rendering in the browser. The webcam is accessed via the Gradio streaming component, not `/dev/video0` — browser-side webcam requires HTTPS, which the NGINX Ingress provides.

> **No extra tooling needed for the terminal:** the embedded terminal runs the lab's commands locally in this pod. `python` and the training scripts are already present; `trtexec` ships in the `nvcr.io/nvidia/pytorch` base image (no separate TensorRT install); `curl` covers Triton inspection. The image does **not** need `kubectl` or a cluster ServiceAccount for the lab flow — read-only `kubectl` in the terminal is optional and, if enabled, requires a get/describe/logs-only Role.

> **Exit criterion:** `helm upgrade --install` deploys both pods to a healthy state; Triton's `/v2/health/ready` returns 200 (check via `kubectl exec -n lab-p1 deploy/tutor-app -- curl -s http://triton:8000/v2/health/ready`); Gradio loads on `https://p1.lab.internal`.

---

## 4. Language Registry

This is the keystone of the multi-language design. Every other component reads from the registry.

### 4.1 Per-language config schema

```yaml
# languages/asl/config.yaml
name: "American Sign Language"
code: "asl"
input_hands: 1            # one-handed sign language
classes: ["A","B","C","D","E","F","G","H","I","J","K","L","M",
          "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
triton_model_name: "asl_classifier"
references_dir: "references"
notes:
  dynamic_letters: ["J", "Z"]   # captured as static snapshots
  attribution: "Reference images: <source>"
```

### 4.2 Registry loader

```python
# src/registry.py
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Language:
    name: str
    code: str
    input_hands: int
    classes: list[str]
    triton_model_name: str
    references_dir: Path
    notes: dict

def load_registry(root: Path = Path("languages")) -> dict[str, Language]:
    registry = {}
    for cfg_path in root.glob("*/config.yaml"):
        with open(cfg_path) as f:
            data = yaml.safe_load(f)
        lang_dir = cfg_path.parent
        registry[data["code"]] = Language(
            name=data["name"],
            code=data["code"],
            input_hands=data["input_hands"],
            classes=data["classes"],
            triton_model_name=data["triton_model_name"],
            references_dir=lang_dir / data["references_dir"],
            notes=data.get("notes", {}),
        )
    return registry
```

> **Exit criterion:** `load_registry()` returns two Language objects (asl, isl) with the expected fields. Unit-tested.

---

## 5. Capture & Hand Tracking

### 5.1 MediaPipe wrapper

```python
# src/capture/hands.py
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_hands: int = 2,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb_frame: np.ndarray):
        """Returns list of (handedness, landmarks_array) tuples.
           landmarks_array is shape (21, 3)."""
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
```

### 5.2 Why max_hands = 2 always

Even though ASL/ISL only need one hand, we always run with `max_hands=2`. The cost is negligible and it lets the same capture layer serve all three languages without switching modes mid-session.

---

## 6. Feature Layer

### 6.1 Why normalise

Raw MediaPipe coordinates depend on where the hand is in the frame and how big it appears. To make the classifier robust to position and distance, we normalise so every input vector represents the same hand-shape in the same canonical space.

### 6.2 Single-hand normalisation (ASL/ISL)

- Translate so the wrist (landmark 0) is at the origin.
- Scale so the distance from wrist (0) to middle-finger MCP (9) is 1.0.
- Flatten (21, 3) → (63,) float32 vector.

```python
# src/features/normalise.py
import numpy as np

def normalise_one_hand(landmarks: np.ndarray) -> np.ndarray:
    """landmarks: (21, 3) array. Returns (63,) normalised vector."""
    assert landmarks.shape == (21, 3)
    wrist = landmarks[0]
    centred = landmarks - wrist
    scale = np.linalg.norm(centred[9])
    if scale < 1e-6:
        scale = 1.0
    return (centred / scale).astype(np.float32).flatten()
```

### 6.3 Feature dispatcher

```python
# src/features/__init__.py
import numpy as np
from src.registry import Language
from .normalise import normalise_one_hand

def build_feature_vector(language: Language, detections) -> np.ndarray | None:
    if not detections:
        return None
    # One-handed languages (ASL/ISL): take whichever hand was detected
    return normalise_one_hand(detections[0][1])
```

> **Exit criterion:** feature builder returns the right shape per language; unit tests cover translation/scale invariance.

---

## 7. Training Pipeline

### 7.1 Dataset extraction

Source images go through MediaPipe to produce CSV rows of `(label, 63 floats)`. Done once, then reused across many training runs.

```bash
# training/extract_landmarks.py (skeleton) — ASL set is bundled in-repo
python extract_landmarks.py \
    --src datasets/asl_dataset2/asl_alphabet_train/asl_alphabet_train \
    --dst languages/asl/landmarks.csv \
    --hands 1
```

### 7.2 Classifier (one-handed)

```python
# training/model_one_hand.py
import torch.nn as nn

class OneHandClassifier(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    def forward(self, x):
        return self.net(x)
```

### 7.3 Training script outline

- Load CSV, split 80/10/10 train/val/test (stratified by class).
- Train for ~50 epochs, batch size 256, Adam lr=1e-3, cross-entropy loss.
- Augmentation: small Gaussian noise on coordinates (sigma=0.01) and ±5° rotations in the XY plane.
- Save best checkpoint by validation accuracy.
- Report per-class precision/recall and confusion matrix on test set.

### 7.5 ONNX export

```python
# training/export_onnx.py (excerpt)
dummy = torch.randn(1, 63)
torch.onnx.export(
    model, dummy, "model.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=17,
)
```

### 7.6 TensorRT engine build

```bash
# training/build_engine.sh
trtexec --onnx=model.onnx \
        --saveEngine=model.plan \
        --fp16 \
        --minShapes=input:1x63 \
        --optShapes=input:32x63 \
        --maxShapes=input:64x63
```

> TensorRT engines are tied to GPU architecture. In this cluster, the target is sm_120 (RTX PRO 6000 Blackwell). Engines built on an H100 (sm_90) won't run here, and vice versa. The ONNX is the portable artefact; the engine is hardware-specific.
>
> **Container version matters:** TRT 10.9 (`tritonserver:25.03`) has no kernel implementations for Blackwell and fails with `Could not find any implementation for node`. Use `tritonserver:26.04-py3` (TRT 10.16.01) or later. In K8S the init container handles the engine build; verify the image tag in `k8s/chart/values.yaml`. Rollback: switch `config.pbtxt` to `platform: "onnxruntime_onnx"` with the `.onnx` file and skip the engine build step.

> **Exit criterion:** per-language `model.plan` file exists; `trtexec` validation passes; standalone Python harness can run inference on it.

---

## 8. Triton Deployment

### 8.1 Model repository layout

```
triton_repo/
├── asl_classifier/
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
├── isl_classifier/
│   ├── config.pbtxt
│   └── 1/model.plan

```

### 8.2 config.pbtxt (one-handed)

```
name: "asl_classifier"
platform: "tensorrt_plan"
max_batch_size: 64
input  [ { name: "input",  data_type: TYPE_FP32, dims: [63] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [26] } ]
instance_group [ { count: 1, kind: KIND_GPU } ]
```

### 8.3 Triton client wrapper

```python
# src/inference/triton_client.py
import numpy as np
import tritonclient.http as httpclient

class TritonClassifier:
    def __init__(self, url: str = "triton:8000", model_name: str = "asl_classifier"):
        self._client = httpclient.InferenceServerClient(url=url)
        self._model_name = model_name

    def infer(self, x: np.ndarray) -> np.ndarray:
        """x: (D,) or (B, D). Returns (num_classes,) softmax-able logits."""
        if x.ndim == 1:
            x = x[None, :]
        inp = httpclient.InferInput("input", x.shape, "FP32")
        inp.set_data_from_numpy(x.astype(np.float32))
        resp = self._client.infer(self._model_name, [inp])
        return resp.as_numpy("output")[0]
```

> **Threading caveat.** `tritonclient.http` runs on `geventhttpclient`, whose
> greenlet hub is bound to the thread that constructed the client. Because the
> webcam stream dispatches `on_frame` across many Gradio worker threads, the
> controller must hold **one `TritonClassifier` per thread** (`threading.local`,
> §10.1) — a single shared instance raises `greenlet.error: cannot switch to a
> different thread` and silently kills the stream.

> **Exit criterion:** from a Python REPL inside `tutor-app`, calling `TritonClassifier` on a known-good landmark vector returns the correct letter with high confidence.

---

## 9. Scoring & Smoothing

### 9.1 Smoothing

```python
# src/lesson/smoother.py
from collections import deque, Counter
import numpy as np

class PredictionSmoother:
    def __init__(self, window: int = 15):
        self._window = window
        self._preds = deque(maxlen=window)
        self._confs = deque(maxlen=window)

    def update(self, pred_idx: int, confidence: float):
        self._preds.append(pred_idx)
        self._confs.append(confidence)

    def smoothed(self) -> tuple[int, float] | None:
        if len(self._preds) < self._window // 2:
            return None
        modal, count = Counter(self._preds).most_common(1)[0]
        avg_conf = float(np.mean(
            [c for p, c in zip(self._preds, self._confs) if p == modal]
        ))
        return modal, avg_conf
```

### 9.2 Traffic-light scorer

```python
# src/lesson/scorer.py
from enum import Enum
import time

class Light(Enum):
    RED = "red"; AMBER = "amber"; GREEN = "green"

class TrafficLightScorer:
    def __init__(self, target_idx: int, hold_seconds: float = 1.0,
                 amber_min: float = 0.50, green_min: float = 0.80):
        self._target = target_idx
        self._hold = hold_seconds
        self._amber_min = amber_min
        self._green_min = green_min
        self._green_since: float | None = None

    def evaluate(self, pred_idx: int, conf: float) -> tuple[Light, bool]:
        """Returns (light, completed)."""
        now = time.monotonic()
        if pred_idx != self._target or conf < self._amber_min:
            self._green_since = None
            return Light.RED, False
        if conf < self._green_min:
            self._green_since = None
            return Light.AMBER, False
        # Green territory
        if self._green_since is None:
            self._green_since = now
        completed = (now - self._green_since) >= self._hold
        return Light.GREEN, completed
```

> **Exit criterion:** with a synthetic stream of `(pred, conf)` values, scorer transitions through RED → AMBER → GREEN → completed at the right thresholds. Unit-tested.

---

## 10. Gradio UI

### 10.1 Layout

Gradio's `gr.Blocks` API gives precise control over layout. Use a top Row for the language/lesson selector, a Row containing two Columns for the split-screen, and a bottom Row (~25% height, full width) for the embedded terminal.

**Theme.** The UI is themed light-grey-background / lime-green borders / black text, with amber reserved for highlights. This is applied with a light `gr.Themes` base plus a small CSS override; the palette lives in one place so it can be retuned without touching layout code. The same stylesheet also fixes the terminal to a scrolling viewport (Bug 3) and styles the quality bar (§10.2).

> A later cosmetic pass layers **black panels** on top of this base (via
> `elem_id` hooks, all in `theme.py`): the title bar (white 16pt heading +
> Dell/NVIDIA logos top-right), the language/lesson row, the terminal output,
> the command input, and the feedback status box. The terminal also gets a lime
> **"Execute Command"** button beside the input. The snippet below shows the
> base palette; see `src/ui/theme.py` for the full current rule set.

```python
# src/ui/theme.py
import gradio as gr

# Palette — single source of truth
BG     = "#EDEDED"   # light grey background
PANEL  = "#F7F7F7"   # slightly lighter panel/input fill
BORDER = "#32CD32"   # lime green element borders
TEXT   = "#111111"   # black body text
ACCENT = "#FFBF00"   # amber — highlights only
TERMINAL_HEIGHT = "220px"

CSS = f"""
.gradio-container {{ background-color: {BG}; color: {TEXT}; }}
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-box, .gradio-container .gr-panel {{
    background-color: {PANEL}; color: {TEXT};
    border: 1px solid {BORDER} !important; border-radius: 6px;
}}
.gradio-container .highlight, .gradio-container .amber {{ color: {ACCENT}; }}

/* Terminal: fixed-height viewport that scrolls instead of growing. */
#lab-terminal .cm-editor {{ max-height: {TERMINAL_HEIGHT}; background: {PANEL}; }}
#lab-terminal .cm-scroller {{ max-height: {TERMINAL_HEIGHT}; overflow: auto; }}

/* Quality bar — fill width/colour are inline (see §10.2); the CSS transition
   plus the controller's EMA give the smooth grow/recede. */
.quality-track {{ position: relative; width: 100%; height: 22px;
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 11px;
    overflow: hidden; }}
.quality-fill {{ height: 100%; transition: width 180ms linear,
    background-color 180ms linear; }}
.quality-target {{ position: absolute; top: -2px; width: 2px; height: 26px;
    background: {TEXT}; }}
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.gray,
).set(body_background_fill=BG, body_text_color=TEXT)
```

```python
# src/ui/app.py (skeleton)
import gradio as gr
from src.registry import load_registry
from src.lesson.controller import (LessonController, render_quality_bar,
                                   render_progress, render_target_letter)
from src.ui.theme import THEME, CSS
from src.terminal.executor import run_command   # see §12

LANGS = load_registry()

def build_app():
    controller = LessonController(LANGS)
    with gr.Blocks(title="Sign Language Tutor", theme=THEME, css=CSS) as demo:
        with gr.Row():
            lang = gr.Dropdown(
                choices=[(l.name, l.code) for l in LANGS.values()],
                value="asl", label="Language")
            lesson = gr.Dropdown(
                choices=[("Alphabet", "alphabet")],
                value="alphabet", label="Lesson")
        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(sources=["webcam"], streaming=True,
                               label="Live feed")
            with gr.Column(scale=1):
                target_letter = gr.HTML(render_target_letter("—"))  # "Target Letter: A"
                target = gr.Image(label="Reference", interactive=False)
                progress = gr.HTML(render_progress(0, 0))            # "Progress: ●●●ooo"
                quality = gr.HTML(value=render_quality_bar(0.0))
                status = gr.Markdown("<-- Click the record button to begin!",
                                     elem_id="feedback-status")
                with gr.Row():
                    skip_btn = gr.Button("Skip")
                    # Locked (grey) until the quality bar hits 90%, then it
                    # latches active (lime); the user clicks it to advance.
                    next_btn = gr.Button("Next letter", elem_id="next-letter-btn",
                                         interactive=False)
        # Embedded terminal — bottom 25%, full width (Module 2 lab commands)
        with gr.Row():
            with gr.Column(elem_id="lab-terminal"):
                term_out = gr.Code(label="Terminal", language="shell",
                                   interactive=False, lines=20,
                                   elem_id="terminal-output")
                with gr.Row():
                    term_in = gr.Textbox(label="Enter your Commands here",
                                         elem_id="command-input", lines=2,
                                         max_lines=12, scale=8)  # multi-line paste
                    exec_btn = gr.Button("Execute Command",
                                         elem_id="execute-btn", scale=1)
                # Enter and the button both run the command; a trailing .then(js)
                # auto-scrolls the terminal to the newest line (see ui/app.py).
                for trigger in (term_in.submit, exec_btn.click):
                    trigger(run_command, inputs=[term_in, term_out],
                            outputs=[term_out, term_in])

        # Navigation hooks own the reference panel + Next-button lock state.
        nav_views = [target_letter, target, progress, quality, status, next_btn]
        # The stream stays lightweight — quality bar, status, one-shot button
        # unlock — and does NOT repaint the reference image (pushing a fresh
        # image 5x/s kept the panel reloading-blank and starved Skip). The frame
        # is not echoed back to `cam` either (the preview renders client-side).
        stream_views = [quality, status, next_btn]
        cam.stream(controller.on_frame, inputs=[cam, lang], outputs=stream_views,
                   stream_every=0.15, concurrency_limit=30, concurrency_id="frame",
                   time_limit=3600, show_progress="hidden")
        # initial_view resets to letter A (refresh restarts the lesson) and shows
        # the opening prompt; the others repaint the active letter on navigation.
        demo.load(controller.initial_view, outputs=nav_views)
        lang.change(controller.on_language_change, inputs=lang, outputs=nav_views)
        skip_btn.click(controller.on_skip, outputs=nav_views)
        next_btn.click(controller.on_next, outputs=nav_views)
    return demo

if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, show_error=True)
```

> **Why the concurrency, lock, and per-thread client matter.** The per-frame
> pipeline (MediaPipe + a blocking Triton HTTP call) is heavy, and three things
> are load-bearing:
> - **`concurrency_limit=30` (NOT 1).** A streaming event holds its worker slot
>   for the whole `time_limit` window; with `concurrency_limit=1` the webcam
>   sends one frame then stalls ~30 s (perpetual spinner). It must be > 1 so
>   frames flow continuously.
> - **A `threading.Lock`** around `on_frame`/navigation in the controller, since
>   on_frame now runs on many worker threads and MediaPipe + the smoother/scorer
>   are shared mutable state.
> - **A per-thread Triton client.** `tritonclient[http]` is built on
>   `geventhttpclient`, whose greenlet hub is bound to the creating thread; a
>   shared client called from another worker raises `greenlet.error: cannot
>   switch to a different thread`, which Gradio reports as a silent stream
>   "Error". The controller caches the client in `threading.local` (§9.1).
> - **No reference image in the stream outputs**, and `cam` is not echoed back.

### 10.2 Quality-bar rendering

The feedback widget is a 0–100% bar with a target marker at 90%. The fill's width and colour are inline so each update reflects the latest value; the colour uses **semantic** red/amber/green that carries meaning (wrong/close/match) and is deliberately exempt from the chrome palette. The amber is aligned to the UI accent (`#FFBF00`). Smoothness comes from two places: the controller eases the value with an EMA (small, frequent steps at ~5 fps), and the CSS `transition` on `.quality-fill` animates each step.

```python
def _quality_colour(pct: float) -> str:        # red <=40, amber 41-75, green 75+
    if pct <= 40: return "#E74C3C"
    if pct <= 75: return "#FFBF00"
    return "#27AE60"

def render_quality_bar(pct: float) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    colour = _quality_colour(pct)
    return (
        '<div class="quality-wrap"><div class="quality-track">'
        f'<div class="quality-fill" style="width:{pct:.0f}%;background:{colour};"></div>'
        '<div class="quality-target" style="left:90%;"></div></div>'
        f'<div class="quality-meta"><span>Quality: {pct:.0f}%</span>'
        '<span class="quality-target-label">target 90%</span></div></div>')
```

The continuous value is derived in the controller from the smoothed confidence for the *target* class (0 when the predicted class isn't the target), then EMA-eased:

```python
target_q = s_conf if s_idx == self._letter_idx else 0.0
self._quality_ema += self._ema_alpha * (target_q * 100.0 - self._quality_ema)
```

**Advancement is manual.** Once the eased quality bar crosses the 90% target,
the controller latches the **Next letter** button active (lime); the user clicks
it to move on, and navigation re-locks it for the next letter. There is no
automatic advance. `TrafficLightScorer` (§9.2) is retained — its unit tests
still pass — but it no longer drives the live lesson flow; the 90% bar crossing
is the single completion gate.

### 10.3 Performance considerations

- Resize incoming webcam frames to 480p before MediaPipe — full HD is wasteful.
- Drop frames if the inference call is still in flight; never queue.
- Reuse the `HandTracker` across frames; reuse the `TritonClassifier` **per
  thread** (`threading.local`) — see §9.1 / §10.1 for the gevent reason.
- Pre-load reference images into memory at startup (small enough to fit easily).

> **Exit criterion:** end-to-end demo with ASL — load page, select ASL alphabet, sign a letter, see GREEN, advance. 25+ FPS sustained.

---

## 11. Adding ISL (one-handed extension)

Adding ISL is the moment of truth for the framework. If we built the registry correctly, this is a data-only task.

- Use the ISL-HS dataset (DCU) or capture ~200 images per letter using webcam.
- Run `extract_landmarks.py` to produce `languages/isl/landmarks.csv`.
- Train a `OneHandClassifier` with the same script.
- Export to ONNX via `kubectl exec -n lab-p1 deploy/tutor-app -- python training/export_onnx.py ...`.
- Copy the ONNX to the triton pod and run `trtexec` via `kubectl exec` to produce `/models/isl_classifier/1/model.plan`.
- Write `config.pbtxt` to `/models/isl_classifier/config.pbtxt` in the triton pod.
- Triton poll mode picks up the new model within 5 seconds — no pod restart needed.
- Languages PVC already contains `languages/isl/config.yaml` and reference images (pre-populated at deploy time).
- Restart the tutor-app to reload the language registry: `kubectl rollout restart deploy/tutor-app -n lab-p1`.
- ISL appears in the language dropdown automatically — no UI code change.

> **Exit criterion:** switching the language dropdown to ISL works; sign 'A' in ISL; system gives correct GREEN.

---

## 12. Embedded Terminal Executor

The terminal at the bottom of the UI lets a participant run the Module-2 lab commands without leaving the browser. It is a **server-side** executor: each submitted line runs as a subprocess inside the `tutor-app` pod, so every command operates on the volumes mounted there (including the shared `triton-models` PVC). The security model is specified in Spec 1 §6.5; this section is the implementation contract.

### 12.1 Executor config

```python
# src/terminal/config.py
COMMAND_TIMEOUT_SECONDS = 300   # 5 min — long enough to run training in the foreground
MAX_OUTPUT_LINES = 500
WORKDIR = "/app"

# First token of the command must be in this set
ALLOWLIST = {
    "kubectl", "curl", "ls", "cat", "echo", "python", "python3",
    "trtexec", "nvidia-smi", "mkdir", "cp", "head", "tail", "grep", "wc",
    "pwd", "date", "env", "whoami", "uptime", "df", "top", "ps", "free", "dmesg",
}
# mkdir/cp are needed for the Module-2 deploy step (create the model-version
# dir and place config.pbtxt on the shared triton-models PVC). They can only
# touch the pod's own mounts; BLOCKED_PATTERNS still rejects rm -rf, /etc, etc.

# Reject if any of these substrings appear anywhere in the command
BLOCKED_PATTERNS = (
    "delete", "rm -rf", "shred", "dd ", "mkfs",
    "> /etc", ">/etc", "/proc/sys", "docker",
)

# kubectl is inspection-only: the verb (2nd token) must be in this set
KUBECTL_READONLY_VERBS = {"get", "describe", "logs", "version", "api-resources"}
```

### 12.2 Executor

```python
# src/terminal/executor.py
import os, shlex, subprocess, threading
from src.terminal import config

def _reject(cmd: str) -> str | None:
    """Return an error string if the command is not permitted, else None."""
    lowered = cmd.lower()
    for pat in config.BLOCKED_PATTERNS:
        if pat in lowered:
            return f"Blocked: command contains disallowed pattern '{pat.strip()}'."
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return "Could not parse command (unbalanced quotes?)."
    if not tokens:
        return None
    prog = os.path.basename(tokens[0])
    if prog not in config.ALLOWLIST:
        allowed = ", ".join(sorted(config.ALLOWLIST))
        return f"Command '{prog}' is not permitted. Allowed commands: {allowed}"
    if prog == "kubectl" and (len(tokens) < 2
                              or tokens[1] not in config.KUBECTL_READONLY_VERBS):
        return ("kubectl is inspection-only here. Allowed verbs: "
                + ", ".join(sorted(config.KUBECTL_READONLY_VERBS)))
    return None

def run_command(cmd: str, history: str = ""):
    """Generator — streams the terminal buffer as the command runs, so the
       participant sees per-epoch training output live. Yields
       (terminal_text, cleared_input). NAMESPACE is injected so kubectl/scripts
       default to the participant's namespace."""
    cmd = (cmd or "").strip()
    prior = history.rstrip()
    if not cmd:
        yield prior, ""
        return

    def render(body_lines):
        return (prior + ("\n" if prior else "") + "\n".join(body_lines)).strip()

    err = _reject(cmd)
    if err is not None:
        yield render([f"$ {cmd}", err]), ""
        return

    env = {**os.environ, "NAMESPACE": os.environ.get("NAMESPACE", "lab-p1")}
    lines = [f"$ {cmd}"]
    proc = subprocess.Popen(
        cmd, shell=True, cwd=config.WORKDIR, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    # Watchdog kills the process at the deadline even if it produces no output.
    timer = threading.Timer(config.COMMAND_TIMEOUT_SECONDS, proc.kill)
    timer.start()
    truncated = False
    try:
        for line in proc.stdout:                      # streams line-by-line
            lines.append(line.rstrip("\n"))
            if len(lines) > config.MAX_OUTPUT_LINES:
                lines = lines[: config.MAX_OUTPUT_LINES]
                lines.append("[Output truncated — re-run with a narrower filter]")
                proc.kill()
                truncated = True
                break
            yield render(lines), ""                   # live update to the UI
        proc.wait()
    finally:
        timer.cancel()

    if not truncated and proc.returncode and proc.returncode < 0:
        lines.append(f"[Killed after {config.COMMAND_TIMEOUT_SECONDS}s timeout]")
    yield render(lines), ""
```

The UI wires this generator straight to the input box (note the prior buffer is fed back in so history accumulates):

```python
term_in.submit(run_command, inputs=[term_in, term_out], outputs=[term_out, term_in])
```

> **Note on `shell=True`:** the allowlist/blocked-pattern checks are a guardrail against accidental damage in a lab, not a hardened sandbox — `python` is allowlisted, so the terminal is as capable as a Python REPL. The real containment is the pod boundary: non-root user, no privileged escalation, namespace-scoped RBAC, and a read-only `kubectl`. Treat the allowlist as UX (clear errors, fewer foot-guns), not as the security perimeter.

> **Streaming & timeout:** the executor is a **generator** — it `yield`s the growing buffer line-by-line, so a 2–3 minute training run shows per-epoch progress live in the terminal rather than dumping at the end. The 5-minute (`COMMAND_TIMEOUT_SECONDS = 300`) watchdog is generous enough to run extraction and training in the **foreground**, which keeps the lab interactive; it still cuts off a genuinely stuck command. The one thing to avoid is a deliberately endless stream (`nvidia-smi --loop`, `tail -f`) — that will simply run until the 5-minute cap.

> **Exit criterion:** allowlisted commands run and stream output back; a disallowed command (e.g. `kubectl delete ...`) returns the helpful rejection message; a command exceeding the timeout is killed; output beyond 500 lines is truncated with the notice. Unit-tested (see Spec 3 §2.7).

---

## 13. Configuration Summary

```yaml
# configs/thresholds.yaml
amber_min_confidence: 0.50
green_min_confidence: 0.80
hold_seconds_for_complete: 1.0
smoothing_window_frames: 15

# Per-language overrides allowed:

```
