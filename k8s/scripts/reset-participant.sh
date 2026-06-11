#!/usr/bin/env bash
# Wipe and redeploy one participant's namespace (fresh start between sessions).
#   bash k8s/scripts/reset-participant.sh p1 p1.lab.internal
set -euo pipefail

PARTICIPANT="${1:?usage: reset-participant.sh <participant> <host>}"
HOST="${2:?usage: reset-participant.sh <participant> <host>}"
NAMESPACE="lab-${PARTICIPANT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

helm uninstall "tutor-${PARTICIPANT}" --namespace "${NAMESPACE}" || true
kubectl delete namespace "${NAMESPACE}" --ignore-not-found --wait=true

bash "${SCRIPT_DIR}/deploy-participant.sh" "${PARTICIPANT}" "${HOST}"
