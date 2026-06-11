# Chart files — release-time artefacts

Files Helm bundles into the release at `helm install`/`template` time via
`.Files.Get`.

## `asl/model.onnx` (committed — we ship ASL pre-trained)

The pre-built ASL classifier ONNX **is committed** here so the chart deploys a
working ASL model out of the box — no training required to stand up the demo. At
install time the chart embeds it into the `asl-bootstrap` ConfigMap (binaryData),
and the Triton init container compiles it to a TensorRT engine for the cluster
GPU (sm_120). If the file is somehow absent, the init container exits with a
clear error rather than serving a broken model.

To regenerate / refresh it from the dataset:

```bash
bash training/build_asl_model.sh   # extract -> train -> export -> stage here
```

That script writes `languages/asl/model.onnx` and copies it to this directory.
The ISL model is **not** shipped — participants produce it during the lab.
