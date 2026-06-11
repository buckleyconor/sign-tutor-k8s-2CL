# Multi-Language Sign Language Tutor — Master Architecture

**Spec 1 of 5**
**Canonical Charmed Kubernetes Lab — Dell Customer Solution Center**

| | |
|---|---|
| Author | Conor (Dell CSC) |
| Version | 0.2 |
| Date | June 2026 |

---

## 1. Executive Summary

This document specifies the architecture for a multi-language sign language tutor lab running on Canonical Charmed Kubernetes with NVIDIA RTX PRO 6000 Blackwell GPUs. The system uses computer vision and a hand-landmark classifier to recognise finger-spelled letters in real time, compare them against a reference, and provide visual feedback to a learner via a split-screen interface.

The platform supports two sign languages from day one: American Sign Language (ASL) and Irish Sign Language (ISL). Both use one-handed alphabets with identical input shapes (63 normalised landmarks → 26 classes). The architecture is deliberately language-agnostic — adding additional sign languages requires only a new dataset and a per-language model, not a code rewrite.

The full ML stack is NVIDIA-centric: training in PyTorch on a RTX PRO 6000 GPU, optimisation via TensorRT, and serving via Triton Inference Server. This makes the lab a credible enterprise reference implementation as well as a learning project.

### 1.1 Key Decisions

| Decision | Rationale |
|---|---|
| Hand tracking via MediaPipe Hands | CPU-only, fast, well-documented, runs natively on aarch64. Outputs 21 landmarks per hand, suitable for downstream classification. |
| Per-language classifier model | ASL and ISL alphabets differ at the letter level. Per-language models keep each one small, fast, and accurate. Both share the same input shape (63 features) since they are both one-handed. |
| PyTorch → ONNX → TensorRT pipeline | Standard NVIDIA inference path. Demonstrates real enterprise tooling rather than ad-hoc scripts. |
| Triton Inference Server | Provides HTTP/gRPC endpoint, model versioning, dynamic batching, and clean separation between UI and ML. Identical interface scales from a single GPU node to a fleet of H100s. |
| Gradio for UI | Python-native, fast to iterate, supports webcam input out of the box, easy to demo. Streamlit is an alternative but Gradio's webcam component is more mature. |
| Local-only first release | No cloud dependencies. Runs entirely on-cluster. Simplifies demo logistics and keeps customer data on-prem. |
| Embedded terminal in Gradio UI | Server-side command executor (Python subprocess) with per-command execution, command allowlist, and namespace isolation. Enables participants to run `kubectl`, `curl`, `trtexec`, and other lab commands directly in the browser without switching to a separate terminal window.


---

## 2. Problem Statement & Scope

### 2.1 Problem

Sign language is a primary mode of communication for the Deaf and Hard-of-Hearing community. Recognition tools — even basic finger-spelling tutors — make learning more accessible and provide a foundation for accessibility products in healthcare, education, customer service, and video conferencing. There is no shortage of academic work in this area, but production-grade reference implementations on modern NVIDIA hardware are far rarer.

### 2.2 In Scope (Module 1 + Module 2)

- Real-time hand tracking via webcam input.
- Recognition of static finger-spelled alphabet letters in ASL and ISL.
- Split-screen UI: live feed + reference image + traffic-light score.
- Lesson progression: A→Z guided practice with feedback.
- Per-language model training and TensorRT optimisation.
- Triton-based inference serving.
- Module 2: small set of static or near-static common words per language.

### 2.3 Out of Scope (this version)

- Continuous signing / sentence-level translation.
- Dynamic letters (J, Z in ASL/ISL) — handled as static snapshots only.
- Facial expression analysis (a real component of full sign languages).
- Speech-to-sign avatar generation.
- Multi-user / cloud deployment.
- Accessibility certification — this is a learning tool, not a production assistive device.

### 2.4 Success Criteria

- Per-letter top-1 accuracy ≥ 90% on held-out test set, per language.
- End-to-end latency (frame in → classification out) < 100 ms.
- Stable 25+ FPS in the UI with no dropped frames during a 5-minute session.
- Adding a new sign language requires zero code changes — dataset and config only.

---

## 3. Sign Language Domain Notes

Sign languages are not universal. ASL and ISL are mutually unintelligible languages with distinct alphabets, even though they share the same one-handed fingerspelling approach.

| Language | Hands | Notes | Implication for our system |
|---|---|---|---|
| **ASL** | One | American Sign Language. Best-documented, most datasets available. J and Z involve motion. | Single-hand model. 21 landmarks input. Well-trodden ground. |
| **ISL** | One | Irish Sign Language. Officially recognised in Ireland (ISL Act 2017). Closer to ASL than to other sign languages but with letter-level differences. Limited public datasets. | Same input shape as ASL classifier. Custom dataset capture likely required. Strong local relevance for Cork-based CSC. |

---

## 4. System Architecture

### 4.1 High-level component diagram

```
+----------------------------------------------------------+
|            K8S namespace (lab-p1)                        |
|                                                          |
|   +---------+    +-----------------+    +-----------+    |
|   | Webcam  |--->| MediaPipe Hands |--->| Feature   |    |
|   | (browser|    | (CPU, tutor-app |    | builder   |    |
|   |  HTTPS) |    |  pod)           |    +-----+-----+    |
|   +---------+    +-----------------+          |          |
|                                               v          |
|   +-----------------+    +-------------+    +-------+    |
|   | Gradio UI       |<---| Score /     |<---|Triton |    |
|   | (NGINX Ingress) |    | Smoothing   |    |+TRT   |    |
|   | +-------------+ |    +-------------+    +-------+    |
|   | | Embedded     | |                          |
|   | | Terminal     | |                          |
|   | | (server-side | |                          |
|   | |  subprocess) | |                          |
|   | +-------------+ |                          |
|   +-----------------+                          |
|                                                          |
+----------------------------------------------------------+
```

### 4.2 Layered view

The system is organised into five layers:

- **Capture layer** — webcam frames via OpenCV at 30 FPS.
- **Perception layer** — MediaPipe Hands extracts 21 landmarks per detected hand.
- **Feature layer** — landmarks normalised into model input tensor (63 floats for one-handed).
- **Inference layer** — Triton serves the per-language TensorRT engine; gRPC/HTTP call returns class probabilities.
- **Application layer** — Gradio UI, lesson controller, scoring, traffic-light feedback.

### 4.3 Data flow per frame

1. OpenCV captures frame from webcam (BGR, 640×480 default).
2. Frame is converted to RGB and passed to MediaPipe Hands.
3. MediaPipe returns 0, 1 or 2 hand landmark sets.
4. Feature builder normalises landmarks (translation/scale invariant) and assembles the language-appropriate input vector.
5. If a complete input is available, it is sent to Triton with the model name matching the active language (e.g. `asl_classifier`).
6. Triton returns 26 class logits.
7. Score smoother applies a rolling average over the last N frames to suppress jitter.
8. Lesson controller compares smoothed prediction to target letter and emits a traffic-light score.
9. Gradio renders annotated webcam frame + reference image + score + feedback text.

### 4.4 Multi-language design

Languages are first-class configuration, not hard-coded. The system loads a registry of supported languages at startup, each with its own model, label set, reference imagery, and required input shape.

```
languages/
  asl/
    config.yaml          # input_hands: 1, classes: [A..Z]
    model.onnx
    model.engine         # TensorRT engine
    references/          # 26 reference images (PNG)
      A.png
      B.png ...
  isl/
    config.yaml          # input_hands: 1, classes: [A..Z]
    model.onnx
    model.engine
    references/
```

Adding a new sign language (e.g. Auslan, French Sign Language) requires only: a labelled dataset, a `config.yaml`, a trained model, and a reference image set. No pipeline or UI code changes.

### 4.5 ASL vs ISL

ASL and ISL share the input shape (one hand, 63 features). Their classifiers are structurally identical — only weights and label sets differ. This means the feature pipeline, normalisation, and model architecture are the same for both languages. The only per-language customisation is the model weights and label set.

The shared input shape is a key design advantage: the UI, lesson controller, and scoring code work identically for both languages with zero conditional logic.

---

## 5. NVIDIA Stack & Hardware Compatibility

### 5.1 Verified components

The production lab deployment runs on Canonical Charmed Kubernetes with **NVIDIA RTX PRO 6000 Blackwell** GPU nodes (sm_120, x86_64).
The development host I am running on has microk8s single node cluster where we can deploy and test before moving to the production Canonical Charmed Kubernetes cluster.

| Component | Status | Notes |
|---|---|---|
| MediaPipe Hands | ✅ OK | CPU-based. Pip install on Ubuntu 22.04/24.04 (x86_64) works. |
| PyTorch (NVIDIA container) | ✅ OK | Use `nvcr.io/nvidia/pytorch:25.xx-py3` — Blackwell-optimised. Do not use generic pip wheels. |
| TensorRT | ✅ OK | `trtexec` ships in both `tritonserver:26.04-py3` **and** the `nvcr.io/nvidia/pytorch:25.xx-py3` tutor-app base, so the lab builds engines locally in the tutor-app pod. Target: sm_120 (RTX PRO 6000 Blackwell). **Requires a TRT 10.16+ image** — TRT 10.9 (25.03) fails at engine build time on Blackwell. The pre-bundled ASL engine is built at pod startup by the Triton init container; the ISL engine is built by the participant during the lab. |
| Triton Inference Server | ✅ OK | Use `26.04-py3` (TRT 10.16.01) — confirmed working on RTX PRO 6000. Poll-mode hot-reload enabled (5 s interval). |
| ONNX Runtime GPU | ✅ OK | Standard PyPI x86_64 wheel. Only needed for inference outside Triton. |
| TAO Toolkit | Optional | Useful for image-based Module 2 experiments. For Module 1 (landmarks-only), direct PyTorch is simpler. |
| OpenCV / Gradio | ✅ OK | Pure Python, no platform issues. |

> **Engine portability note:** TensorRT engines are tied to GPU architecture. Both of our environments — the microk8s dev host and the Charmed Kubernetes production cluster — use **RTX PRO 6000 Blackwell (sm_120)**, so an engine built in dev runs unchanged in production. Portability only becomes a concern for *foreign* architectures: a `model.plan` built on sm_120 will **not** run on an H100 (sm_90) or a consumer 4090 (sm_89). The ONNX file is the portable artefact; rebuild the engine on each non-Blackwell target.

### 5.2 Kubernetes deployment strategy

The system deploys to Canonical Charmed Kubernetes as a **Helm chart**, with one release per lab participant in an isolated namespace (`lab-p1`, `lab-p2`, …).

Each namespace contains:

- **triton** Deployment — `nvcr.io/nvidia/tritonserver:26.04-py3`. An init container runs `trtexec` at pod startup to compile ONNX → TensorRT engine for the cluster GPU (sm_120). The main container serves models with poll-mode hot-reload (5 s interval) on ports 8000 (HTTP) and 8001 (gRPC).
- **tutor-app** Deployment — built from `nvcr.io/nvidia/pytorch:25.xx-py3`. Runs MediaPipe, the Gradio UI, and the lesson controller. Communicates with Triton over the in-namespace `triton` Service. Exposed via NGINX Ingress with TLS (`https://p1.lab.internal`).
- **PersistentVolumeClaims** — `triton-models`, `languages`, `datasets`, `checkpoints`, and a read-only PVC for the shared ISL frames dataset. The `triton-models` PVC is the **shared model repository**: it is mounted **read-write in tutor-app** and read-only in the triton pod, so a TensorRT engine built in tutor-app appears directly in Triton's repository with no cross-pod copy. The StorageClass is parameterized in `values.yaml` (Ceph RBD on Charmed Kubernetes; the default `microk8s-hostpath` on the single-node dev host).

Training, ONNX export, and the `trtexec` engine build all run **inside the tutor-app pod** — which is also where the embedded terminal executes — writing the finished engine and `config.pbtxt` straight onto the shared `triton-models` PVC. Triton picks them up automatically via poll mode. No `kubectl cp` or cross-pod `exec` is required.

See `k8s/chart/` for the Helm templates and `k8s/scripts/deploy-participant.sh` for the full deployment procedure.

---

## 6. UI / UX Specification

### 6.1 Layout

The UI is a single-page split-screen layout. Left half: live annotated webcam feed. Right half: reference imagery and feedback panel. Top bar: language and lesson selector. Bottom of the UI has an embedded terminal where the user can run their training commands

```
+--------------------------------------------------------------+
| [Language: ASL ▾]  [Lesson: Alphabet ▾]   [Letter: D]  ●●●○○ |
+----------------------------+---------------------------------+
|                            |                                 |
|                            |   Sign the letter:              |
|     LIVE WEBCAM            |                                 |
|     (with landmarks        |          [reference image of D] |
|      overlaid)             |                                 |
|                            |   Quality:  [GREEN]             |
|                            |   Confidence: 96%               |
|                            |                                 |
|                            |   [ Skip ]   [ Next letter ]    |
---------------------------------------------------------------+
| TERMINAL: kubectl get pods --show-labels |                    |
+----------------------------+---------------------------------+
```

The terminal frame occupies the bottom 25% of the UI vertically and spans the full width. It uses Gradio's `gr.Code` component with a monospace font for command input and output display. Commands execute server-side via a Python subprocess handler (`src/terminal/executor.py`) with a configurable timeout (default 300 s / 5 min); the handler streams output back line-by-line so long-running lab commands (extraction, training) show progress live rather than dumping at the end. Output is limited to a maximum of 500 lines to prevent browser memory exhaustion. The terminal supports a scrollable output area; previous commands are retained in the output buffer for reference. The entire UI is black background with element borders in lime green and text in white/grey, with amber reserved for highlights.

### 6.2 Traffic-light scoring

| Light | Trigger | UX behaviour |
|---|---|---|
| 🔴 **RED** | `conf < 0.50`, OR `predicted != target` | Show "try again" hint. Highlight key landmarks that differ from reference (stretch goal). |
| 🟠 **AMBER** | `0.50 ≤ conf < 0.80` AND `predicted == target` | Show "close" hint. Encourage user to hold the sign more clearly. |
| 🟢 **GREEN** | `conf ≥ 0.80` AND `predicted == target` sustained for ≥ 1.0 s | Mark letter as completed; auto-advance after a brief celebration animation. |

Thresholds are configurable via a single YAML file so demos can be tuned to room lighting / camera quality without code changes.

### 6.3 Smoothing

Raw frame-by-frame predictions jitter heavily. The UI uses a 15-frame (~0.5 s) rolling window: the displayed prediction is the modal class across the window, and the displayed confidence is the average for that class. This eliminates flicker without adding noticeable lag.

### 6.4 Lesson flow

1. User selects a language (ASL / ISL).
2. User selects a lesson (Module 1: Alphabet; Module 2: Words).
3. System presents target sign with reference image.
4. User signs; traffic-light updates in real time.
5. On sustained GREEN, system advances to next target.
6. Session ends after the full lesson; summary screen shows per-letter best score.

### 6.5 Terminal Security Model

The embedded terminal is a server-side command executor running inside the `tutor-app` pod. Because it executes **inside** that pod, the entire Module-2 lab flow is a sequence of **local** commands — `python training/...` for extraction/training/export, the base image's bundled `trtexec` for the engine build, and `curl http://triton:8000/...` for inspecting Triton — all writing to volumes mounted in the same pod (including the shared `triton-models` PVC). No `kubectl cp`, no cross-pod `exec`, and no `rollout restart` are needed, which is what lets the terminal stay locked to a read-only, inspection-only `kubectl` posture. It is subject to the following security constraints:

| Constraint | Detail |
|---|---|
| **Namespace isolation** | The `NAMESPACE` environment variable (set at pod deployment, e.g. `lab-p1`) is injected into every command. `kubectl` commands default to the participant's namespace. Participants cannot access other namespaces. |
| **Command allowlist** | Only commands in the allowlist (see `src/terminal/config.py`) may execute. Disallowed commands produce a helpful error: `"Command '{cmd}' is not permitted. Allowed commands: {allowed_list}"`. |
| **Timeout** | All commands are killed by a watchdog after `command_timeout_seconds` (default: 300 s / 5 min). This is generous enough to run the lab's longest steps (landmark extraction ~1–2 min, training ~2–3 min) in the **foreground**, keeping the lab interactive — the executor streams output line-by-line so per-epoch progress appears live. A genuinely stuck command is still cut off at the cap; deliberately endless streams (`nvidia-smi --loop`, `tail -f`) will run until it. |
| **Output limit** | Maximum 500 lines of output per command. Excess output is truncated with a notice: `[Output truncated — see full output via kubectl logs]`. |
| **No container escape** | The executor runs as the same non-root user as the `tutor-app` container. No privileged containers, no access to `docker socket`, no access to other namespaces' secrets or resources. |
| **Read-only kubectl** | Only `get`, `describe`, `logs`, and `exec --stdin --tty` (for inspecting the current pod) are permitted for `kubectl`. `delete`, `apply`, `cp`, `exec` into other pods, and other destructive operations are blocked. |
| **Working directory** | Commands execute in `/app` (the pod's working directory). Participants can `cd` but only within the pod's filesystem. |

**Allowlisted commands:** `kubectl`, `curl`, `ls`, `cat`, `echo`, `python`, `python3`, `trtexec`, `nvidia-smi`, `mkdir`, `cp`, `head`, `tail`, `grep`, `wc`, `pwd`, `date`, `env`, `whoami`, `uptime`, `df`, `top`, `ps`, `free`, `dmesg`. (`mkdir` and `cp` are needed for the Module-2 deploy step — creating the Triton model-version directory and placing `config.pbtxt` on the shared `triton-models` PVC. They operate only within the pod's own mounts; the blocked-pattern check still rejects `rm -rf`, writes to `/etc`, and the like.)

**Blocked command patterns:** any command containing `delete`, `rm -rf`, `shred`, `dd`, `mkfs`, or attempts to write to `/etc`, `/proc/sys`, or `docker`. These are detected via a simple prefix check in the executor.

---

## 7. Risks & Mitigation

| Risk | Severity | Mitigation |
|---|---|---|
| Limited public ISL alphabet datasets | High | Self-capture with webcam, augmented with rotations/lighting; document data provenance carefully. Engage with Irish Deaf Society for guidance and review. |

| MediaPipe missed-detections under poor lighting | Medium | Document recommended demo lighting; UI to display "no hand detected" clearly; consider a fallback CV-CUDA pose model later. |
| aarch64 wheel availability for niche dependencies | Low | Stay inside the NVIDIA PyTorch container; avoid exotic dependencies. Already-known workaround for `onnxruntime-gpu`. |
| Cultural sensitivity / misrepresenting Deaf community | Medium | Frame the lab clearly as a learning aid, not an assistive product. Use reference imagery from authoritative sources. Acknowledge this is fingerspelling, not full sign language. |
| Demo audience confuses fingerspelling with full signing | Low | Add explicit on-screen note: "This lab teaches fingerspelling — one component of sign language." |

---

## 8. Deliverables & Timeline

*Indicative effort, single developer, part-time:*

| Week | Milestone | Deliverable |
|---|---|---|
| 1 | Environment & dataset | K8S cluster running; ASL Kaggle dataset processed to landmark CSV. |
| 2 | ASL classifier | Trained PyTorch ASL model, ONNX export, TensorRT engine, Triton serving, smoke test. |
| 3 | UI + scoring | Gradio split-screen, traffic-light scoring, smoothing, ASL alphabet lesson playable end-to-end. |
| 4 | ISL | ISL dataset captured, classifier trained, plugged in via language registry. Adding a second language proves the framework. |
| 4 | ISL (one-handed) | ISL dataset captured, classifier trained, plugged in via language registry. Adding a second language proves the framework. |
| 6 | Module 2 + polish | Static-word lesson set per language. Demo-ready polish, latency tuning, documentation. |

---

## 9. Companion Specifications

- **Spec 2 — Module 1 Detailed Build Spec**: end-to-end implementation steps for the ASL alphabet classifier, ISL extension, and integration with Triton + Gradio.
- **Spec 3 — Test & Validation Plan**: unit, integration, and acceptance tests; data quality checks; performance benchmarks.
- **Spec 4 — Data Acquisition & Preparation**: ASL Kaggle dataset, DCU ISL-HS dataset, self-capture protocol, augmentation strategy.
- **Spec 5 — Lab Guide**: hands-on participant guide for adding ISL to a running Kubernetes deployment (3–4 hours).
