#!/usr/bin/env bash
# Deploy (or upgrade) one participant's namespace.
#   bash k8s/scripts/deploy-participant.sh p1 p1.lab.internal
set -euo pipefail

PARTICIPANT="${1:?usage: deploy-participant.sh <participant> <host>}"
HOST="${2:?usage: deploy-participant.sh <participant> <host>}"
NAMESPACE="lab-${PARTICIPANT}"
CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../chart" && pwd)"

kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install "tutor-${PARTICIPANT}" "${CHART_DIR}" \
  --namespace "${NAMESPACE}" \
  --set participant="${PARTICIPANT}" \
  --set host="${HOST}"

echo "Waiting for rollouts..."
kubectl rollout status deploy/triton -n "${NAMESPACE}" --timeout=5m
kubectl rollout status deploy/tutor-app -n "${NAMESPACE}" --timeout=5m
echo "Deployed. UI: https://${HOST}"
