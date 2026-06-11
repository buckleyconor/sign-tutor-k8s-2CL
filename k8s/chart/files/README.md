# Chart files — release-time artefacts

Files Helm bundles into the release at `helm install`/`template` time via
`.Files.Get`.

## `asl/model.onnx` (required for a working ASL deploy)

The pre-built ASL classifier ONNX. It is **not** committed (gitignored: `*.onnx`)
because it's a generated artefact. Produce and place it before deploying:

```bash
# 1. Produce the ONNX (inside the tutor-app container — MediaPipe + torch)
bash training/build_asl_model.sh

# 2. Copy it into the chart so Helm can deliver it
cp languages/asl/model.onnx k8s/chart/files/asl/model.onnx

# 3. Deploy
bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
```

At install time the chart embeds this ONNX into the `asl-bootstrap` ConfigMap
(binaryData), and the Triton init container compiles it to a TensorRT engine for
the cluster GPU (sm_120). If the file is absent, the init container exits with a
clear error rather than serving a broken model.
