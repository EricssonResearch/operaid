#!/usr/bin/env bash
# OperAID — Reset deployment using helm upgrade (fast recovery)
# Resets the deployment to a clean state without deleting the namespace.
# Reads deployment-specific configuration from a profile JSON file.
#
# Usage:
#   ./reset-deployment.sh [--profile open5gs] [--namespace <ns>] [--charts-dir <path>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config for defaults
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-open5gs}"
NAMESPACE="${NAMESPACE:-}"
CHARTS_DIR="${OPENVERSO_CHARTS_DIR:-/home/ariel/git/openverso-charts}"
HELM_TIMEOUT="${HELM_TIMEOUT:-120}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)     DEPLOYMENT_PROFILE="$2"; shift 2 ;;
        --namespace)   NAMESPACE="$2"; shift 2 ;;
        --charts-dir)  CHARTS_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--profile open5gs] [--namespace <ns>] [--charts-dir <path>]"
            exit 0 ;;
        *) shift ;;
    esac
done

# Resolve profile
PROFILE_PATH="${SCRIPT_DIR}/deployments/${DEPLOYMENT_PROFILE}.json"
if [[ ! -f "${PROFILE_PATH}" ]]; then
    PROFILE_PATH="${DEPLOYMENT_PROFILE}"
fi
if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo "ERROR: Deployment profile not found: ${DEPLOYMENT_PROFILE}" >&2
    exit 1
fi

# Extract profile values
PROFILE_NAME=$(python3 -c "import json; print(json.load(open('${PROFILE_PATH}'))['name'])")
if [[ -z "${NAMESPACE}" ]]; then
    NAMESPACE=$(python3 -c "import json; print(json.load(open('${PROFILE_PATH}')).get('namespace', 'default'))")
fi
RELEASE_NAME=$(python3 -c "
import json
p = json.load(open('${PROFILE_PATH}'))
print(p.get('deployment', {}).get('release_name', '${PROFILE_NAME}'))
")

# Get component list for targeted reset
COMPONENTS=$(python3 -c "
import json
p = json.load(open('${PROFILE_PATH}'))
prefix = '${PROFILE_NAME}-'
comps = p.get('components', {})
for k, v in comps.items():
    print(v.get('deployment', prefix + k))
")

echo "[reset] Resetting ${PROFILE_NAME} deployment via helm upgrade..."

# Step 1: Delete any injected faults (NetworkPolicies, etc.)
echo "[reset] Removing injected faults..."
kubectl delete networkpolicy --all -n "${NAMESPACE}" 2>/dev/null || true

# Step 2: Reset any scaled-down deployments
echo "[reset] Resetting deployment replicas..."
for dep in ${COMPONENTS}; do
    kubectl scale deployment "${dep}" -n "${NAMESPACE}" --replicas=1 2>/dev/null || true
done

# Step 3: Remove bad patches that may conflict with helm
echo "[reset] Removing patched deployments..."
for dep in ${COMPONENTS}; do
    kubectl delete deployment "${dep}" -n "${NAMESPACE}" --ignore-not-found=true 2>/dev/null || true
done

# Step 3b: Wait for pods to fully terminate (handles stuck CreateContainerConfigError pods)
echo "[reset] Waiting for pods to terminate..."
SECONDS=0
MAX_WAIT=90
while [[ ${SECONDS} -lt ${MAX_WAIT} ]]; do
    REMAINING=$(kubectl get pods -n "${NAMESPACE}" --no-headers 2>/dev/null \
        | grep -cE "^($(echo ${COMPONENTS} | tr ' ' '|' | sed 's/$/-/'))" || echo "0")
    if [[ "${REMAINING}" -eq 0 ]]; then
        echo "[reset] All pods terminated after ${SECONDS}s"
        break
    fi
    sleep 2
done
if [[ ${SECONDS} -ge ${MAX_WAIT} ]]; then
    echo "[reset] WARNING: Pods still present after ${MAX_WAIT}s, force-deleting..."
    kubectl delete pods -n "${NAMESPACE}" --all --force --grace-period=0 2>/dev/null || true
    sleep 5
fi

# Step 4: Helm upgrade recreates deleted deployments with clean state (no --wait)
echo "[reset] Running helm upgrade to reset state..."
helm upgrade "${RELEASE_NAME}" "${CHARTS_DIR}/charts/${PROFILE_NAME}" \
    -n "${NAMESPACE}" \
    -f "${SCRIPT_DIR}/${PROFILE_NAME}-values.yaml" \
    --timeout "${HELM_TIMEOUT}s" 2>&1 || {
    echo "[reset] ERROR: Helm upgrade failed" >&2
    exit 1
}

# Upgrade UERANSIM if present
helm upgrade ueransim "${CHARTS_DIR}/charts/ueransim" \
    -n "${NAMESPACE}" \
    --timeout "${HELM_TIMEOUT}s" 2>&1 || true

# Step 5: Quick health check
echo "[reset] Verifying health..."
HEALTH_CMD="python3 ${SCRIPT_DIR}/health/health_check.py --profile ${PROFILE_PATH} -n ${NAMESPACE}"
if ${HEALTH_CMD} --wait -t 60; then
    echo "[reset] ✓ Reset successful"
    exit 0
else
    echo "[reset] ✗ Reset failed - health check did not pass"
    exit 1
fi
