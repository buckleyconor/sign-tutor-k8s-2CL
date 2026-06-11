# Test & Validation Plan

**Sign Language Tutor — Module 1**
**Spec 3 of 5**

---

## 1. Testing Philosophy

This is a vision-and-ML project, which means many components are inherently noisy. Tests are organised so that the deterministic parts (config loading, normalisation, scoring logic) are tested rigorously and quickly, while the inherently fuzzy parts (model accuracy, end-to-end UI behaviour) are tested with appropriate tolerances and acceptance thresholds.

**Three layers of tests:**

- **Unit tests** — pure-Python, no GPU, no webcam, no Triton. Fast (< 5s total). Run on every commit.
- **Integration tests** — require Triton to be running with at least one model loaded. Slower (~30s).
- **Acceptance tests** — manual or semi-manual demos with real webcam input. Run before each release.

---

## 2. Unit Tests

Framework: `pytest` with `pytest-cov`. Target coverage: 80%+ on `src/` excluding the UI module.

### 2.1 Registry

| Test | Assertion |
|---|---|
| `test_loads_two_languages` | Given fixture languages dir, registry contains keys `'asl'`, `'isl'`. |
| `test_asl_isl_one_handed` | `registry['asl'].input_hands == 1`; same for isl. |
| `test_classes_have_26_letters` | Both languages have `len(classes) == 26` with A..Z. |
| `test_missing_config_raises` | Loading from an empty directory yields an empty registry without crashing. |
| `test_malformed_yaml_fails_clearly` | A `config.yaml` with missing required fields raises a clear `KeyError` or `ValueError`, not a cryptic stack trace. |

### 2.2 Feature normalisation

| Test | Assertion |
|---|---|
| `test_output_shape_one_hand` | `normalise_one_hand((21,3))` returns shape `(63,)` and dtype float32. |
| `test_translation_invariance` | Translating all 21 landmarks by a constant vector produces an identical normalised output (within 1e-6). |
| `test_scale_invariance` | Multiplying all landmarks by 2.0 produces an identical normalised output. |
| `test_wrist_at_origin` | After normalisation, the first three values of the output (the wrist) are all 0.0. |
| `test_zero_scale_handled` | If wrist and middle MCP coincide (degenerate input), function returns finite values, not NaN/Inf. |


### 2.3 Smoother

| Test | Assertion |
|---|---|
| `test_returns_none_until_warmup` | `smoothed()` returns `None` until at least `window/2` samples received. |
| `test_modal_class_wins` | After feeding 10 'A' and 5 'B', `smoothed()` returns idx for 'A'. |
| `test_confidence_average_class_only` | Returned confidence averages confs only for the modal class. |
| `test_window_eviction` | After `window+1` samples, oldest sample is no longer influencing output. |

### 2.4 Traffic-light scorer

| Test | Assertion |
|---|---|
| `test_wrong_class_is_red` | `evaluate(wrong_idx, conf=0.99)` → RED. |
| `test_correct_low_conf_is_red` | `evaluate(target, conf=0.4)` → RED. |
| `test_correct_mid_conf_is_amber` | `evaluate(target, conf=0.65)` → AMBER. |
| `test_correct_high_conf_is_green` | `evaluate(target, conf=0.9)` → GREEN, `completed=False` (initially). |
| `test_completion_after_hold` | Sustained GREEN for `hold_seconds` yields `completed=True`. Use a fake monotonic clock. |
| `test_amber_resets_green_timer` | Brief drop to AMBER then back to GREEN restarts the hold timer. |
| `test_no_completion_at_exact_threshold_minus_epsilon` | Boundary: `hold_seconds - 1ms` does not yet complete; `hold_seconds` does. |

### 2.5 Feature dispatcher

- `test_one_handed_language_dispatches_to_normalise_one_hand`
- `test_no_detections_returns_none_for_one_handed`

### 2.6 Test fixtures

Generate synthetic fixtures rather than relying on real webcam captures for unit tests. A small helper produces deterministic 21×3 landmark arrays representing canonical hand poses.

```python
# tests/fixtures/synthetic.py
import numpy as np

def synthetic_hand(seed: int = 0, translate=(0,0,0), scale=1.0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1, 1, size=(21, 3)).astype(np.float32)
    base[0] = 0.0  # wrist
    return base * scale + np.array(translate, dtype=np.float32)
```

### 2.7 Terminal executor

Pure-Python, no real subprocess where avoidable (monkeypatch `subprocess.run`); the rejection logic needs no subprocess at all. These guard the embedded terminal's safety contract (Spec 1 §6.5, Spec 2 §12).

| Test | Assertion |
|---|---|
| `test_allowlisted_command_runs` | An allowlisted command (`echo hi`) executes and its output is returned. |
| `test_disallowed_command_rejected` | A non-allowlisted program (`scp ...`) returns the `"is not permitted"` message and never spawns a subprocess. |
| `test_blocked_pattern_rejected` | Commands containing `delete`, `rm -rf`, `dd `, or `docker` are rejected by pattern even if the program is allowlisted (e.g. `kubectl delete pod x`). |
| `test_kubectl_readonly_verbs_only` | `kubectl get pods` is allowed; `kubectl apply -f x` / `kubectl cp ...` / `kubectl exec ...` are rejected with the read-only message. |
| `test_timeout_kills_command` | A command exceeding `COMMAND_TIMEOUT_SECONDS` (monkeypatched short) returns the `[Killed after …]` notice rather than hanging. |
| `test_output_truncated_at_500_lines` | Output longer than `MAX_OUTPUT_LINES` is truncated and the truncation notice is appended. |
| `test_namespace_injected` | The `NAMESPACE` env var is present in the subprocess environment, defaulting when unset. |
| `test_empty_command_is_noop` | Submitting a blank line returns the existing buffer and does not spawn a subprocess. |
| `test_unbalanced_quotes_handled` | A command that fails `shlex.split` returns a clear parse error, not a stack trace. |

---

## 3. Integration Tests

These require a running Triton container with at least the `asl_classifier` model loaded. Marked with `@pytest.mark.integration` so they can be skipped in fast CI.

| Test | Assertion |
|---|---|
| `test_triton_health` | `/v2/health/ready` returns 200 within a 10s timeout. |
| `test_model_loaded` | Each language's expected Triton model is reported as READY. |
| `test_inference_returns_correct_shape` | Calling `TritonClassifier.infer` with a `(63,)` vector returns `(26,)`. |
| `test_inference_known_letter` | Feed a held-out test landmark for 'A' through Triton; argmax is 'A' with confidence > 0.7. |
| `test_e2e_test_set_accuracy` | Across the held-out test set for each language, top-1 accuracy ≥ 90%. |
| `test_inference_latency_p95` | p95 latency for 100 sequential single-sample calls < 30 ms. |
| `test_input_validation` | Wrong input shape produces a clear error rather than a silent failure. |

### 3.1 Triton + MediaPipe pipeline test

One integration test runs static reference images of known signs through the full MediaPipe → normalise → Triton pipeline and verifies the predicted letter.

```python
# tests/integration/test_pipeline.py
import cv2, pytest
from src.capture.hands import HandTracker
from src.features import build_feature_vector
from src.inference.triton_client import TritonClassifier
from src.registry import load_registry

@pytest.mark.integration
@pytest.mark.parametrize("letter", ["A", "B", "C", "L", "Y"])
def test_known_image_predicts_correctly(letter):
    langs = load_registry()
    asl = langs["asl"]
    tracker = HandTracker()
    classifier = TritonClassifier(model_name=asl.triton_model_name)

    img = cv2.imread(f"tests/fixtures/asl_reference/{letter}.png")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = tracker.process(rgb)
    feat = build_feature_vector(asl, detections)
    assert feat is not None, "MediaPipe failed to detect hand in reference image"

    logits = classifier.infer(feat)
    pred = asl.classes[logits.argmax()]
    assert pred == letter
```

---

## 4. Model Validation

### 4.1 Per-language acceptance

Before a language is considered demo-ready, its model must satisfy:

- Top-1 accuracy ≥ 90% on the held-out test set.
- No single class with recall < 75% (i.e. no letter is systematically missed).
- Confusion matrix saved to `docs/` for review.
- Test set captures lighting and angle variation, not just clean lab conditions.

### 4.2 Regression suite

After any retrain, run the saved test set through the new model. Compare per-class metrics to the previous best — a drop of more than 3 percentage points on any class blocks promotion of the new model.

### 4.3 Slice-level analysis

- Performance with skin-tone variation in source images (use a diverse evaluation set).
- Performance with lighting variation (bright / dim / backlit).
- Performance with hand size variation (proxy: distance from camera).

> These slices are about confidence in the demo, not about claiming the model is production-ready. Be honest in any internal report about where it weakens.

---

## 5. Performance Benchmarks

| Metric | Target | Measurement |
|---|---|---|
| MediaPipe latency per frame | < 35 ms | Time `HandTracker.process` over 1000 frames; report mean and p95. |
| Triton inference latency | < 10 ms | `trtexec --loadEngine=model.plan` and Triton client microbenchmark. |
| End-to-end frame → light | < 100 ms | Instrument the lesson controller; record timestamps at each stage. |
| UI sustained FPS | ≥ 25 | Run a 5-minute session, count frames rendered. |
| GPU memory (Triton) | < 1 GB | `nvidia-smi` during a session with all three models loaded. |

---

## 6. Acceptance / Demo Tests

Manual checklist; tick before any demo. Two people if possible — one signing, one observing.

- [ ] Cold start: `bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal` brings UI to ready state in under 5 minutes (dominated by TRT engine build in the Triton init container).
- [ ] `https://p1.lab.internal` loads with a valid TLS certificate (no browser warning).
- [ ] Language switch ASL → ISL works without restart; reference images update.
- [ ] RED / AMBER / GREEN transitions feel responsive (no perceptible lag) when sign is held steady.
- [ ] Sustained GREEN of ~1 s reliably advances to next letter.
- [ ] Skip and Next buttons work; lesson can be exited cleanly.
- [ ] No hand visible: UI shows "no hand detected" rather than a stale prediction.
- [ ] Two different signers (different hand sizes, skin tones if possible) both achieve ≥ 80% letters in a single pass through the alphabet.
- [ ] No console errors during a 10-minute session (`kubectl logs -n lab-p1 deploy/tutor-app`).
- [ ] Network disconnected from internet: lab still runs (all images cached on nodes, everything is local).
- [ ] UI renders with the intended palette: black background, lime-green element borders, white/grey text, amber used only for highlights and the AMBER traffic light.
- [ ] Embedded terminal runs the full Module-2 command flow (extract → train → export → `trtexec` → `config.pbtxt`) locally in the tutor-app pod; the new ISL model is served by Triton afterwards.
- [ ] Embedded terminal rejects a disallowed command (e.g. `kubectl delete pod`) with the helpful allowlist message, and truncates very long output at 500 lines.

---

## 7. Data Quality Checks

Before training, the landmark CSV produced by `extract_landmarks.py` is sanity-checked:

- Per-class sample count balanced within 2x ratio (e.g. no class with 5x more samples than another).
- No NaN or Inf values.
- All vectors of expected length (63 or 126).
- No duplicate rows.
- For self-captured data, frame-source diversity check: at least 3 different recording sessions per class.

> These are simple pandas checks — make them part of the training script's startup, so a bad CSV fails fast rather than producing a bad model.

---

## 8. Continuous Integration

Suggested layout for a lightweight CI run:

```yaml
stages:
  - lint (ruff + black --check)
  - unit (pytest -m "not integration", < 30s)
  - integration (only on main branch; deploys a test namespace via deploy-participant.sh, runs integration suite, tears down)
  - acceptance (manual gate)
```

> Acceptance is intentionally manual — webcam-driven tests are flaky in headless CI and not worth automating for a lab project.

---

## 9. Summary

The test plan above keeps the deterministic core of the system honest while accepting that the ML and UX layers need pragmatic, tolerance-based validation. Unit tests should run in seconds and gate every commit; integration and acceptance tests gate releases.

**If only one set of tests is implemented in the first cut, prioritise unit tests on the registry, normalisation, smoother, and scorer — those are the components most likely to subtly misbehave and least visible in a demo until the moment they fail.**
