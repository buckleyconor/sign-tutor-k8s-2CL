# Multi-Language Sign Language Tutor

Real-time fingerspelling recognition (ASL + ISL) on the NVIDIA stack — MediaPipe
→ PyTorch → ONNX → TensorRT → Triton — served through a Gradio split-screen UI
with an embedded terminal, on Kubernetes.

> This lab teaches **fingerspelling** — one component of sign language.

## Environments

Developed on a single-node **microk8s** cluster, deployed to **Canonical Charmed
Kubernetes**. Both have RTX PRO 6000 Blackwell (sm_120) GPUs, so TensorRT engines
built in dev run unchanged in production.

## Layout

```
src/            application (capture, features, inference, lesson, ui, terminal, registry)
training/       extraction, augmentation, training, ONNX export, engine build
triton_repo/    Triton config.pbtxt per model
languages/      per-language config + GUI reference images (asl/, isl/)  [provided]
datasets/       training images (ASL Kaggle, ISL frames)                 [provided, gitignored]
configs/        thresholds.yaml (traffic-light cutoffs, smoothing)
k8s/            Helm chart + per-participant deploy scripts
tests/          unit (no GPU/Triton), integration (needs Triton)
```

## Quick start (dev)

```bash
pip install -r requirements-dev.txt
make unit                 # deterministic unit tests — no GPU/Triton/webcam
make helm-template        # render manifests

# Deploy a participant namespace (needs a cluster + images)
bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
```

## The lab (Module 2)

Participants extend the running ASL app to ISL entirely from the **embedded
terminal** in the UI — extract landmarks, train, export ONNX, build the TensorRT
engine with the bundled `trtexec` onto the shared model PVC, drop in
`config.pbtxt`, and Triton hot-loads it via poll mode. See
`spec_05_lab_guide.md`.

## Specs

`spec_01` architecture · `spec_02` Module 1 build · `spec_03` test plan ·
`spec_04` data acquisition · `spec_05` lab guide.

## Known gaps

- `languages/isl/references/Q.png` is missing (25/26 letters); the GUI shows a
  blank reference for Q until it's added.
- Held-out ASL validation currently uses the small bundled 28-image test set;
  the Rasband varied set is recommended for an honest ≥90% gate.
