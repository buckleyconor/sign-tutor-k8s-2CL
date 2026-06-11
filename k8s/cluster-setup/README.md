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

The bundled ASL `model.onnx` is delivered to the Triton init container via an
`onnx-models` ConfigMap (or an init-time copy from the `languages` PVC), and the
Triton repo `config.pbtxt` files via a `triton-repo` ConfigMap. The init
container compiles the engine for sm_120 on first start.

## Per-participant deploy

```bash
bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
```
