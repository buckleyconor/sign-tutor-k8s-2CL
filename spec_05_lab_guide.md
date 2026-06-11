# Lab Guide — Add a New Sign Language

**Sign Language Tutor — Hands-on NVIDIA Stack Lab**
**Spec 5 of 5**
**Target duration: 3–4 hours**

---

## 0. Before you start

### 0.1 What this lab is

You will take a working sign-language recognition application that supports American Sign Language (ASL) and extend it to also support Irish Sign Language (ISL). Along the way you will touch every layer of NVIDIA's vision-AI stack: GPU-accelerated training in **PyTorch** on a **Blackwell** GPU, model optimisation with **TensorRT**, and production serving via **Triton Inference Server**.

The end deliverable is your trained ISL model running live in the same application. When you sign an ISL letter at the webcam, your model — running on the Triton server you just deployed to — recognises it and the traffic-light score turns green.

### 0.2 What this lab is *not*

This is not an app-development exercise. The application is already built; you are extending it. The technical learning happens in the NVIDIA tooling between data preparation and live inference. If you find yourself editing UI code, you have gone off-route — come back to the lab guide.

### 0.3 What you will learn

By the end you will have first-hand experience of:

- Extracting training features from raw images using MediaPipe on GPU.
- Understanding how landmark-space augmentation improves model robustness.
- Training a small classifier in the NVIDIA PyTorch container on Blackwell hardware.
- Exporting to ONNX and building a TensorRT engine for the cluster GPU (`sm_120` architecture).
- Deploying a new model to a running Triton server without restarting it.
- Understanding how the same pipeline scales from a single workstation GPU to a data-centre H100.

### 0.4 Prerequisites

You should be comfortable with:

- A Linux command line (`cd`, `cat`, `ls`, reading a YAML file).
- Copy-pasting commands into the embedded terminal (the facilitator will show you any syntax you haven't seen before — every command you need is given in full).
- Reading Python (you will not need to write much).

You do **not** need prior experience with sign language, computer vision, or model training.

### 0.5 What is already running before you sit down

The facilitator has prepared:

- A Kubernetes namespace `lab-p1` (your participant ID may differ — ask the facilitator).
- Two pods running in that namespace:
  - `triton-…`: hosts the ASL TensorRT model on port 8000 (HTTP) and 8001 (gRPC).
  - `tutor-app-…`: the Gradio UI, reachable at `https://p1.lab.internal`.
- A working ASL classifier already serving from Triton.
- A pre-collected dataset of ISL hand images at `datasets/isl_frames/` inside the tutor-app pod — 26 letter subdirectories, roughly 28–37 frames each (882 images total).
- The ISL language configuration and reference images are already in the pod; your job is to produce and deploy the trained model.

> **Facilitator note:** the ISL reference set in `languages/isl/references/` is currently missing `Q.png` (25 of 26 letters present). The model still trains and predicts Q, but the GUI has no reference image to show for it — add `Q.png` before the session, or expect the Q reference panel to be blank.

### 0.6 How you run commands: the embedded terminal

Every command in this lab is run in the **embedded terminal at the bottom of the application UI** (black panel, lime-green border). That terminal executes commands *server-side, inside the `tutor-app` pod* — the same pod that holds the training scripts, the datasets, and (mounted read-write) Triton's shared model repository. That is why you never `ssh` anywhere or copy files between pods: you are already inside the one pod that can reach everything.

A few things to know about it:

- Your participant namespace is injected automatically as `$NAMESPACE` — you don't need to set it.
- `kubectl` is **inspection-only** here (`get`, `describe`, `logs`). You won't need it for the core lab; you inspect Triton over HTTP with `curl` instead.
- The terminal **streams output live** and allows up to **5 minutes** per command — long enough to run landmark extraction and training right here in the foreground and watch them progress, line by line, as they go.

> 💡 Avoid deliberately endless commands (`nvidia-smi --loop`, `tail -f`) — in a single terminal they'll just run until the 5-minute cap. Everything the lab asks you to run finishes comfortably inside it.

---

## 1. Orientation (15 minutes)

### 1.1 Verify the baseline works

Open a browser on the lab workstation and navigate to:

```
https://p1.lab.internal
```

You should see the sign-language tutor UI with ASL selected — black background, lime-green borders, white/grey text, with an embedded terminal across the bottom. Sign a letter (the reference image shows you how). The quality bar should turn green when you match.

> ✅ **Checkpoint:** ASL recognition works.

### 1.2 Inspect what is running

In the **embedded terminal**, ask Triton directly what it is serving (the `tutor-app` pod reaches Triton over the in-cluster `triton` Service):

```bash
curl -s http://triton:8000/v2/models/asl_classifier | python3 -m json.tool
```

You should see the ASL classifier with platform `tensorrt_plan`. There is no ISL classifier yet — you are about to create one.

```bash
# Confirm ISL is not yet serving (expect an error / not-found response)
curl -s http://triton:8000/v2/models/isl_classifier
```

You can also confirm Triton is healthy:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://triton:8000/v2/health/ready
```

### 1.3 Understand the extension point

Everything you need is already in this pod's working directory (`/app`). Look at the languages directory:

```bash
ls languages/
```

You will see two folders: `asl/` and `isl/`. The ISL folder already contains the language configuration and reference images for the UI — these were prepared in advance. Look at what is there:

```bash
ls languages/isl/
cat languages/isl/config.yaml
ls languages/isl/references/
```

The `config.yaml` declares the language name, letter classes, and — critically — the `triton_model_name: "isl_classifier"`. The application will look for a Triton model with that name. Your job is to train that model and deploy it.

> ✅ **Checkpoint:** You understand that the application already knows about ISL. Your job is to produce the trained model artefact and deploy it to Triton.

---

## 2. Extract the training data (20 minutes)

### 2.1 Understand the dataset

The raw ISL data is a collection of hand images organised by letter, already in this pod:

```bash
ls datasets/isl_frames/
ls datasets/isl_frames/A/
```

Each image shows a hand forming a letter of the ISL fingerspelling alphabet. Rather than training directly on images (which would require a larger model and much more data), this pipeline extracts 21 hand-landmark coordinates from each image using MediaPipe. The resulting 63-dimensional feature vector (21 landmarks × 3 coordinates, normalised relative to the wrist) is compact, pose-invariant, and fast to train on.

### 2.2 Extract landmarks

Extraction takes 1–2 minutes. The terminal streams MediaPipe's progress live, so just run it in the foreground and watch:

```bash
python training/extract_landmarks.py \
    --src datasets/isl_frames \
    --dst datasets/isl_landmarks.csv \
    --source-tag isl_frames \
    --hands 1
```

MediaPipe processes each image and writes one CSV row per successfully detected hand; frames where no hand is detected are silently skipped — this is normal. When it finishes, check the output:

```bash
wc -l datasets/isl_landmarks.csv
```

Expect around **870 rows** out of the 882 frames — MediaPipe detects a hand in ~99% of these clean, background-removed images, skipping only a handful. Anything above ~600 is fine to proceed.

### 2.3 Verify data quality

```bash
python training/check_quality.py datasets/isl_landmarks.csv 63
```

This checks that the feature dimension is correct (63) and that the class distribution is not severely imbalanced. You may see **two warnings** — a class-imbalance one (the dynamic letters X and Z have fewer clean static frames, so the min/max ratio dips just under the threshold) and a single-source one. Both are expected for this one pre-collected dataset and are acceptable here; the check still reports `Data quality: OK`.

> ✅ **Checkpoint:** You have a `datasets/isl_landmarks.csv` file with ISL landmark data.

---

## 3. Augmentation (15 minutes)

### 3.1 Why augment

Your dataset, though real, is limited in diversity: 6 signers, controlled lighting, consistent angle. A model trained on it alone will work well in lab conditions but may struggle with different hand sizes, wrist tilts, or lighting variations. Augmentation generates additional training variants without requiring more real data.

### 3.2 How augmentation works in this pipeline

Look at the augmentation module:

```bash
cat training/augment.py
```

Unlike image-space augmentation (which would use a library like NVIDIA DALI to apply transforms to pixels), this pipeline augments in *landmark space* — directly perturbing the 63-dimensional feature vectors at training time. Two operations are applied to each training sample by default (a third, mirror flip, is opt-in — see below):

| Operation | Effect | Rationale |
|---|---|---|
| Gaussian noise | `± ~1%` jitter on each coordinate | MediaPipe itself is noisy; teaching the model to tolerate noise |
| In-plane rotation `± 10°` | Rotates landmarks around the wrist | Handles wrist tilt variation |

> A mirror-flip augmentation is available (`--mirror`) but **off by default**: an
> ISL ablation showed it consistently lowers accuracy by ~1–4 points, because a
> mirrored handshape can resemble a different letter. Leave it off.

### 3.3 Augmentation is automatic

You do not need to run a separate augmentation step. The training script applies these transforms to every training batch automatically — each epoch the model sees a slightly different version of every sample. This is why training for 50 epochs on ~700 samples produces a model that generalises beyond those 700 examples.

> ✅ **Checkpoint:** You understand what augmentation does and why the training script applies it automatically.

---

## 4. Train the classifier (30 minutes)

### 4.1 What you are training

A small Multi-Layer Perceptron (MLP) with three dense layers. Input: 63 floats. Output: 26 logits (one per letter A–Z). About 12,000 trainable parameters in total. On the cluster's Blackwell GPU, training takes 2–3 minutes.

This is deliberately tiny. For a well-bounded classification task with good features (which landmarks are), small models with clean data outperform large models with noisy data every time.

### 4.2 Run training

Training takes 2–3 minutes on the Blackwell GPU. Run it in the foreground — the terminal streams per-epoch progress (training loss, validation loss, validation accuracy) live as the epochs tick by:

```bash
python training/train_classifier.py \
    --dataset datasets/isl_landmarks.csv \
    --epochs 50 \
    --checkpoint-dir checkpoints/isl \
    --csv-file checkpoints/isl/train_log.csv
```

Validation accuracy should rise quickly and plateau in the **90–96%** range.

> 💡 You'll see a line like `Small dataset: batch_size 256 -> 44 (~16
> batches/epoch)`. That's expected and intentional: the ISL set (~700 rows) is
> far smaller than the ASL set the default batch size targets, so the trainer
> shrinks the batch to give enough gradient updates per epoch. Without it the
> model badly underfits (~60% accuracy). No action needed.

> 💡 **Want to see the GPU at work?** Run `nvidia-smi` *before* you start training to note the baseline, then again right after it finishes. The model is tiny so utilisation comes in short bursts; don't use `nvidia-smi --loop` in this single terminal — a continuous command would occupy it until the 5-minute cap.

### 4.3 Inspect the result

When training finishes (the stream stops at the final epoch), review the log (CSV: epoch, train loss, val loss, val accuracy):

```bash
cat checkpoints/isl/train_log.csv
```

Look at the final few epochs. You are looking for:

- **Validation accuracy ≥ 85%.** Below this, your model may struggle in the live demo.
- **Validation loss still decreasing (not diverging).** If it starts climbing while training loss falls, the model is overfitting — reduce `--epochs` or the dataset is too small.

If you fall short of 85%, see Section 9 (troubleshooting).

> ✅ **Checkpoint:** `checkpoints/isl/best.pt` exists and the training log shows validation accuracy ≥ 85%.

---

## 5. Export and optimise (20 minutes)

### 5.1 ONNX export

PyTorch is great for training, but for production inference NVIDIA has a faster path. The first step is exporting to ONNX, an open intermediate format that separates the model architecture from the training framework.

```bash
python training/export_onnx.py \
    --checkpoint checkpoints/isl/best.pt \
    --output languages/isl/model.onnx
```

Verify the file was produced:

```bash
ls -lh languages/isl/model.onnx
```

### 5.2 Build the TensorRT engine

This is the NVIDIA-specific optimisation step. TensorRT compiles the ONNX graph into a binary engine tailored to the target GPU's architecture. In this cluster, that target is `sm_120` (RTX PRO 6000 Blackwell).

`trtexec` ships in this pod's PyTorch base image — and crucially that base (`pytorch:26.04`) carries the **same TensorRT version as Triton** (10.16), so the engine you build here will load in Triton. Triton's model repository is mounted **read-write** here at `/models` (the shared `triton-models` PVC), so you build the engine **straight into Triton's repository** — no copying between pods. First create the model-version directory, then build:

```bash
mkdir -p /models/isl_classifier/1
```

Now build the engine (this runs in well under a minute for a model this small; the build log streams in the terminal and ends with `&&&& PASSED` on success):

```bash
trtexec \
    --onnx=languages/isl/model.onnx \
    --saveEngine=/models/isl_classifier/1/model.plan \
    --minShapes=input:1x63 \
    --optShapes=input:32x63 \
    --maxShapes=input:64x63 \
    --useCudaGraph
```

A few flags worth understanding:

- `--minShapes / --optShapes / --maxShapes` — tell TensorRT the range of batch sizes to expect. The engine is optimised for the `optShapes` size (32 samples) and handles the full 1–64 range. For the streaming UI, single-sample (batch=1) inference is the hot path.
- `--useCudaGraph` — captures kernel launch graphs for lower latency on repeated calls.

> **Why no `--fp16`?** FP16 (half-precision) is beneficial for large models where memory bandwidth is the bottleneck. For a 12,000-parameter MLP like this one, FP16 quantisation shifts logits enough to change the top-1 predicted class on a large fraction of inputs — effectively breaking the classifier. Use FP32 here; the latency difference at this model size is negligible (both are sub-millisecond).

> **A note worth remembering:** TensorRT engines are tied to GPU architecture. The `model.plan` you just built will run on this cluster's RTX PRO 6000 (sm_120). It will **not** run on an H100 (sm_90) or a 4090 (sm_89). To deploy to different hardware, you rebuild from the same ONNX on that hardware. The ONNX is the portable artefact; the engine is hardware-specific.

Check the engine was produced:

```bash
ls -lh /models/isl_classifier/1/model.plan
```

> ✅ **Checkpoint:** `/models/isl_classifier/1/model.plan` exists on the shared model repository — which Triton is already watching.

---

## 6. Deploy to Triton (15 minutes)

### 6.1 Triton's model repository

Triton serves models from a folder structure on a persistent volume. New models are deployed by adding folders; updates happen by adding numbered version subdirectories. There is no separate "deploy step" — the directory layout *is* the deployment.

Look at the current state of the shared repository (mounted here at `/models`):

```bash
ls -la /models/
```

You will see `asl_classifier/` (the pre-existing model) and your new `isl_classifier/1/model.plan`. Now add the configuration file.

### 6.2 Write the config

A ready-made `config.pbtxt` for ISL is bundled in the repo at `triton_repo/isl_classifier/config.pbtxt`. Look at it first — this is what tells Triton how to serve your engine:

```bash
cat triton_repo/isl_classifier/config.pbtxt
```

```
name: "isl_classifier"
platform: "tensorrt_plan"
max_batch_size: 64
input  [ { name: "input",  data_type: TYPE_FP32, dims: [63] } ]
output [ { name: "output", data_type: TYPE_FP32, dims: [26] } ]
instance_group [ { count: 1, kind: KIND_GPU } ]
```

It declares: a TensorRT model called `isl_classifier`, taking 63-float inputs (one hand's landmarks) and producing 26-float outputs (one logit per letter), served on the GPU. Copy it into place on the shared repository:

```bash
cp triton_repo/isl_classifier/config.pbtxt /models/isl_classifier/config.pbtxt
```

> **Note on `max_batch_size`:** Setting this to 64 lets Triton manage batching — it adds the batch dimension automatically and will queue up to 64 concurrent requests. For the streaming UI (which sends one frame at a time), Triton dispatches each request immediately; the batch headroom is there for when you scale to concurrent users.

### 6.3 Triton picks it up automatically

The Triton pod runs with `--model-control-mode=poll`, which means it watches the repository for changes and loads new models automatically (poll interval: 5 seconds).

Wait a few seconds, then confirm:

```bash
curl -s http://triton:8000/v2/models/isl_classifier | python3 -m json.tool
```

You should see `isl_classifier` with platform `tensorrt_plan`. The ISL model is now live.

> ✅ **Checkpoint:** Triton is serving your ISL model.

---

## 7. Activate in the application (5 minutes)

### 7.1 The language is already registered

The application loaded `languages/*/config.yaml` at startup, and the ISL config was pre-prepared — so **ISL is already in the registry and the language dropdown**. It simply had no model to talk to until now. Verify the config:

```bash
cat languages/isl/config.yaml
```

Check that `triton_model_name: "isl_classifier"` matches the Triton model name you just deployed. The reference images for the UI are also already in place:

```bash
ls languages/isl/references/
```

### 7.2 No restart needed

Because the ISL language was registered at startup and Triton hot-loaded your engine via poll mode, **there is nothing to restart**. The moment Triton reports `isl_classifier` as ready (which you confirmed in §6.3), selecting ISL in the UI will route to your new model. This is the whole point of poll-mode serving and a pre-registered language: deploying a model is a data operation, not a redeploy.

> ✅ **Checkpoint:** ISL is registered and its model is being served — no application restart required.

---

## 8. Test live (15 minutes)

### 8.1 Open the UI

Refresh `https://p1.lab.internal` in your browser.

The language dropdown lists both American Sign Language and Irish Sign Language (ISL was registered at startup). Select **ISL** — it now resolves to the model you just deployed, where moments ago it had none.

### 8.2 Sign a few letters

Use the reference image as guidance. Hold each sign clearly and steadily for 1–2 seconds. The quality bar should rise as the model's confidence grows — the bar only turns green when the model consistently predicts the target letter above the **80%** confidence threshold (the green cutoff in `configs/thresholds.yaml`).

> 💡 **Tip:** The smoother accumulates predictions over 15 frames (about 3 seconds at the streaming rate). Hold the sign steady rather than moving — the system is looking for consistency, not speed.

> ✅ **Checkpoint:** Your trained ISL model is recognising your live signing through the Triton-hosted TensorRT engine.

That sentence is worth re-reading. *Your* model, *your* engine, on the same infrastructure that could scale up to a data centre, recognising *your* hand in real time. You have just deployed a working production inference pipeline on Blackwell.

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| A command returns `[Killed after 300s timeout]` | The command genuinely stalled, or you used a continuous flag | Re-run without `--loop`/`-f`; extraction and training both finish well inside the 5-minute cap |
| `extract_landmarks.py` produces very few rows | Many images lack a detectable hand | Normal for some frames — proceed if you have ≥ 600 rows |
| Quality check fails on imbalance | One letter class has very few detections | Expected with ~35 frames/letter; proceed to training anyway |
| Training accuracy plateaus below 80% | Insufficient data diversity | Reduce `--val-split` to 0.1 and retrain; the pre-collected dataset is small by design |
| `export_onnx.py` fails | Checkpoint path wrong | Verify `checkpoints/isl/best.pt` exists first: `ls checkpoints/isl/` |
| `trtexec: command not found` | Engine build run somewhere without TensorRT | Run it in the embedded terminal (the tutor-app pod's PyTorch image bundles `trtexec`); if needed use the full path `/usr/src/tensorrt/bin/trtexec` |
| Triton logs `Version tag does not match` / model `UNAVAILABLE` after deploy | Engine built by a TRT version different from Triton's | The tutor-app and Triton images must share a TRT version (both 10.16 here). The engine builds fine but won't deserialize if they differ — flag to the facilitator |
| Triton reports model as `UNAVAILABLE` | `config.pbtxt` missing/typo, or `model.plan` not in `1/` | Re-check `ls -la /models/isl_classifier/1/` and that `config.pbtxt` is at `/models/isl_classifier/config.pbtxt`; ask the facilitator to check `triton` pod logs |
| Curl shows ISL not ready after 10s+ | Poll hasn't picked it up, or build incomplete | Confirm the `trtexec` output ended with `&&&& PASSED`; wait one more poll interval (5s) |
| UI shows ISL but always "no hand detected" | Webcam permission denied | Check browser permissions; allow camera for `p1.lab.internal` |
| Quality bar stays red | Model accuracy too low or sign not held long enough | Hold sign steady for 2+ seconds; check training val accuracy |
| Predictions look random for every sign | Wrong model deployed | Verify `triton_model_name` in `config.yaml` matches the folder name under `/models/` |

---

## 10. Going further (optional, 30+ minutes)

If you finish early, try one or more of:

### 10.1 Profile end-to-end latency

Instrument the lesson controller and measure: frame capture → MediaPipe → normalise → Triton → score. Identify the slowest stage. Hint: MediaPipe (CPU) is almost always the bottleneck — your Triton call is sub-millisecond.

### 10.2 Dynamic batching for concurrent users

The current config serves requests one at a time. Add dynamic batching to Triton for high-concurrency scenarios.

A dynamic-batching variant is bundled alongside the base config. Look at the extra block it adds, then copy it over the live config on the shared repository:

```bash
cat triton_repo/isl_classifier/config.dynamic.pbtxt
```

```
# ...same as the base config, plus:
dynamic_batching {
  preferred_batch_size: [ 8, 16 ]
  max_queue_delay_microseconds: 5000
}
```

```bash
cp triton_repo/isl_classifier/config.dynamic.pbtxt /models/isl_classifier/config.pbtxt
```

Triton's poll mode reloads the model with the new config within ~5 seconds — confirm with `curl -s http://triton:8000/v2/models/isl_classifier/config | python3 -m json.tool`.

The `max_queue_delay_microseconds: 5000` (5 ms) cap means Triton won't hold a request longer than 5 ms waiting for a batch to fill — so single-user streaming latency is unaffected, while concurrent users get batched together automatically. This is the capability that makes Triton relevant at data-centre scale.

### 10.3 Compare INT8 to FP32

Rebuild the engine with `--int8` and a calibration dataset. Measure the accuracy drop versus the latency gain. This is the kind of calibration enterprise teams do routinely for production inference.

### 10.4 Per-class confusion analysis

Look at the training log's per-epoch validation accuracy. Which letters does your model confuse most often? In ISL fingerspelling, certain pairs (M/N, R/U) share similar static shapes. Now you have hard evidence of where to focus your next data capture session.

---

## 11. What you take away

You have completed an end-to-end NVIDIA inference pipeline:

| Stage | Tool |
|---|---|
| Feature extraction | MediaPipe (CPU, in PyTorch container) |
| Data augmentation | Landmark-space augmentation (training loop) |
| Training | PyTorch in `nvcr.io/nvidia/pytorch` container on Blackwell |
| Format conversion | ONNX (`torch.onnx.export`) |
| Optimisation | TensorRT `trtexec` (sm_120, FP32) |
| Serving | Triton Inference Server (`tensorrt_plan`, poll-mode hot-reload) |
| Orchestration | Kubernetes (Helm chart, per-participant namespace isolation) |
| Validation | Accuracy gate + live webcam test |

The application happens to be a sign-language tutor. The pipeline you ran is identical to the one a real enterprise team would use for production defect detection, medical imaging triage, retail shelf analytics, or any other vision-classification problem. Only the dataset and the label set change.
