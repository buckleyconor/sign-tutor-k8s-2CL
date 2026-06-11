# Cluster setup (one-time, admin)

One-time admin steps before deploying participant namespaces.

## Environments

| | Dev | Production |
|---|---|---|
| Cluster | single-node **microk8s** | **Canonical Charmed Kubernetes** |
| GPU | RTX PRO 6000 Blackwell (sm_120) | RTX PRO 6000 Blackwell (sm_120) |
| Default StorageClass | `microk8s-hostpath` | Ceph RBD / CephFS |

Because both are sm_120, TensorRT engines built in dev run unchanged in prod.

## Prerequisites

- NVIDIA GPU Operator (or device plugin) so pods can request `nvidia.com/gpu`.
- NGINX Ingress controller.
- TLS: cert-manager issuing `tutor-tls`, or a pre-provisioned secret per host.
- StorageClass set in `k8s/chart/values.yaml` (`storageClass:`).

### microk8s quick enable

```bash
microk8s enable gpu hostpath-storage ingress
```

## ASL model delivery

The chart is self-contained per release via the **`asl-bootstrap` ConfigMap**
(`templates/asl-bootstrap-configmap.yaml`): it carries the ASL `config.pbtxt`
(inline) and the pre-built ASL `model.onnx` (binary, from
`k8s/chart/files/asl/model.onnx`). The Triton init container mounts it, compiles
the ONNX to a TensorRT engine for sm_120 on first start, and drops the config
into the shared `triton-models` PVC.

Produce and place the ONNX before deploying (see `k8s/chart/files/README.md`):

```bash
bash training/build_asl_model.sh                       # in the tutor-app container
cp languages/asl/model.onnx k8s/chart/files/asl/model.onnx
```

The ISL model is produced by participants during the lab and written directly to
the shared `triton-models` PVC — no ConfigMap needed.

## Per-participant deploy

```bash
bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
```
