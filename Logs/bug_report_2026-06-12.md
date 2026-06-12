# Sign-Language Tutor — Bug & Enhancement Report

**Date:** 2026-06-12
**Author:** Diagnostic pass from `Logs/conor.har` + `Logs/outputofresources.txt` + source review
**Build under test in the lab:** `conorbuckley/sl-tutor:1.0.3` (ArgoCD-managed, namespace `hol-1356-01-254696`, Dell lab registry) — **note this is a different image/registry than the local microk8s `sign-tutor/tutor-app:0.6`.** Code fixes here apply to both; the lab will only pick them up after a new `sl-tutor` tag is pushed and ArgoCD syncs.

---

## Evidence summary

### From `outputofresources.txt` (lab cluster `kubectl get all -oyaml`)
- All three pods (`tutor-app`, `triton`, `static-responder`) **Running, `restartCount: 0`, no OOMKilled, no eviction, no `FailedScheduling`**. → The freezing is **not** k8s resource starvation. 
- `tutor-app` has healthy resources: `requests cpu=2/mem=8Gi/gpu=1`, `limits cpu=4/mem=16Gi/gpu=1`. Env is correct: `TRITON_URL=triton:8000`, `GRADIO_PORT=7860`, `GRADIO_HOST=0.0.0.0`. Init container correctly waits for Triton readiness.
- Liveness/readiness probes are `httpGet / :7860` — Gradio root returns 200 fast, so probes aren't restarting the pod (consistent with restartCount 0).

### From `conor.har` (browser session)
- Gradio is driven over the SSE queue: `POST /gradio_api/queue/join` then `GET /gradio_api/queue/data`.
- Captured events are **only** `fn_index:1` (language dropdown change) and `fn_index:4` (terminal `run_command`). **There is not a single webcam-stream (`on_frame`) event in the whole capture.** The webcam stream never established a working frame loop — matching "spinning ball / waiting / never becomes active."
- The terminal round-trips the **entire accumulated output** as the input value on every command (visible in the growing `data[1]` payloads) — confirms the unbounded-growth cosmetic issue.

---

## Bug 1 — Webcam preview freezes / lags / spins, recognises a sign only for a split second

**Severity: High (core feature).** Multiple compounding root causes, all in `src/ui/app.py` + `src/lesson/controller.py`:

1. **MediaPipe Hands is shared and not thread-safe.** `LessonController.__init__` builds one `HandTracker` (`self._tracker`) and `on_frame` calls `self._tracker.process(frame)` on it. Gradio's queue can dispatch `on_frame` from multiple worker threads; concurrent `.process()` calls on a single MediaPipe graph stall/deadlock the native graph. **This is the prime suspect for the hard freeze / spinner.**

2. **One global controller shared across all sessions/users.** `build_app()` creates a single `LessonController`; every browser session mutates the same `_letter_idx`, `_smoother`, `_scorer`, `_tracker`. In a multi-participant lab this corrupts state and serialises everyone through one MediaPipe graph.

3. **Unthrottled, heavy synchronous pipeline.** Each frame runs MediaPipe (CPU) **plus** a blocking Triton HTTP round-trip — tens to hundreds of ms — while `cam.stream(...)` has **no `stream_every` and no `concurrency_limit`**. The browser pushes ~30 fps; the handler can't keep up; the queue backs up and latency grows without bound → laggy, then stalls. The "split-second recognition" is the rare frame that gets through before backpressure wins.

4. **Output feeds back into the input webcam component.** `outputs=[cam, target, light, status]` returns the frame to the *same* `cam` component every tick. No overlay is drawn yet (it's a `TODO`), so this is pure churn and can re-initialise the stream → "active for a split second."

**Proposed resolution**
- Make capture **per-session and serialised**: create the `HandTracker` (and smoother/scorer/state) inside `gr.State` per session, or guard `.process()` with a `threading.Lock`. Simplest robust fix: one tracker guarded by a lock **and** `concurrency_limit=1` on the stream so frames never overlap.
- Add `stream_every=0.2` (≈5 fps) and `concurrency_limit=1`, `concurrency_id="frame"` to `cam.stream(...)` to bound the rate to what the pipeline can sustain.
- Stop returning the frame to `cam`; drop `cam` from `outputs` (re-add only when landmark overlay is actually drawn). Outputs become `[target, quality, status]`.
- Move per-user state out of the module-global controller (use `gr.State` or session-scoped construction).

---

## Bug 2 — "Sign this letter" reference image is blank

**Severity: High.** The PNGs are present (confirmed: 26 files baked into the image under `/app/languages/asl/references/`), so this is **not** a missing-file problem. Two code causes:

1. **The target image is only ever produced as an output of `on_frame`.** Because the webcam stream is broken (Bug 1), `on_frame` effectively never runs, so the image never appears.
2. **Navigation handlers don't update the UI.** `lang.change`, `skip_btn.click`, `next_btn.click` have **no `outputs`** and there is **no `demo.load`** to seed the first letter. So even with streaming fixed, changing language or pressing Next/Skip changes internal state but never repaints the reference image, light, or status.

**Proposed resolution**
- Add a `current_view()` helper on the controller returning `(reference_image, quality_html, status_md)` for the active letter, independent of the webcam.
- Wire it to `demo.load(...)` (seed on page open) and to `lang.change` / `next` / `skip` (repaint on navigation), outputting to `[target, quality, status]`.
- This decouples the reference image from the webcam entirely — it shows immediately and updates on every navigation.

---

## Bug 3 — Terminal output grows unbounded instead of scrolling in a fixed frame

**Severity: Low (cosmetic, functionally fine).** `gr.Code(lines=10)` sets an initial height but the component auto-grows to its content; there's no max-height/scroll. (`MAX_OUTPUT_LINES` in `executor.py` caps the buffer but the rendered box still expands within that cap.)

**Proposed resolution**
- CSS: give `#lab-terminal` (and its inner `.cm-editor`/`textarea`) a fixed `max-height` with `overflow:auto`, and keep the newest line in view. Pure CSS in `theme.py`; no logic change.

---

## Bug 4 — Black theme unappealing → light-grey / lime-green / black

**Severity: Cosmetic (requested).** Current `theme.py` is black bg + grey text + lime borders.

**Proposed resolution** — repalette in `src/ui/theme.py`:
- Background: light grey (`#EDEDED` / `#F2F2F2`)
- Element borders: lime green (`#32CD32`)
- Text: black (`#111111`)
- Keep amber as a sparing highlight. Switch the Gradio `Base` theme to a light body fill so component internals match.

---

## Bug 5 — Replace traffic light with a smooth 0–100% quality bar

**Severity: Enhancement (requested).** Today the score is a discrete red/amber/green HTML circle (`render_light`) driven by `TrafficLightScorer` returning a `Light` enum. Requested: a progress bar 0–100%, a marker line at 90%, smooth real-time grow/recede, colour **red ≤40% / amber 41–75% / green 75%+**.

**Proposed resolution**
- Derive a **continuous quality %** from the smoothed confidence for the *target* class (0 when the predicted class ≠ target, else `conf*100`). Apply an EMA so it eases up/down instead of jumping.
- Render an HTML/CSS bar: a track, a fill whose width = quality% with a CSS `transition: width/background 150ms` for smoothness, colour chosen by the thresholds, and an absolutely-positioned tick at 90% labelled "target."
- Keep the existing hold-to-complete logic (sustained ≥ green threshold for `hold_seconds`) for advancing letters; the bar reaching the 90% line is the visual cue.
- `TrafficLightScorer` keeps working for completion; add a `quality()` method (or return the continuous value alongside the light) so tests stay green.

---

## Cross-cutting note (not a reported bug, but relevant)
The single module-global `LessonController` (Bug 1.2) is a correctness risk for a **multi-participant lab** even after the freeze is fixed: all sessions share one letter index and scorer. The fix for Bug 1 (per-session state via `gr.State`) resolves this too and should be done together.

---

## Suggested implementation order
1. **Bug 1 + Bug 2 together** (they share the `app.py` wiring and the controller view/state refactor) — restores the core experience.
2. **Bug 5** (quality bar) — touches scorer + controller render + theme CSS.
3. **Bug 4** (palette) and **Bug 3** (terminal scroll) — both pure `theme.py` CSS.
4. Rebuild → bump tag → for the **lab**, push the new `sl-tutor` tag to the Dell registry and let ArgoCD sync (the local microk8s `:0.x` deploy does not affect the lab namespace).
