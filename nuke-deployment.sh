#!/usr/bin/env bash
# OperAID — Nuke and redeploy
# Deletes the namespace entirely and redeploys from scratch.
# Reads deployment-specific configuration from a profile JSON file.
#
# Usage:
#   ./nuke-deployment.sh [--profile open5gs] [--namespace <ns>] [--charts-dir <path>]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config for defaults
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-open5gs}"
NAMESPACE="${NAMESPACE:-}"
CHARTS_DIR="${OPENVERSO_CHARTS_DIR:-/home/ariel/git/openverso-charts}"
HELM_TIMEOUT="${HELM_TIMEOUT:-180}"

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
HEALTH_CMD="python3 ${SCRIPT_DIR}/health/health_check.py --profile ${PROFILE_PATH} -n ${NAMESPACE}"

echo "================================================================"
echo "Nuking ${PROFILE_NAME} deployment in namespace '${NAMESPACE}'..."
echo "================================================================"

# Step 1: Uninstall Helm releases
echo "[1/5] Uninstalling Helm releases..."
helm uninstall "${RELEASE_NAME}" -n "${NAMESPACE}" 2>/dev/null || true
helm uninstall ueransim -n "${NAMESPACE}" 2>/dev/null || true
sleep 3

# Step 2: Delete namespace (removes all remaining resources)
echo "[2/5] Deleting namespace '${NAMESPACE}'..."
kubectl delete namespace "${NAMESPACE}" --wait=true --timeout=60s 2>/dev/null || true
sleep 3

# Step 3: Recreate namespace and deploy MongoDB
echo "[3/5] Deploying MongoDB..."
kubectl create namespace "${NAMESPACE}" 2>/dev/null || true
MONGODB_MANIFEST="${SCRIPT_DIR}/mongodb-standalone.yaml"
if [[ -f "${MONGODB_MANIFEST}" ]]; then
    kubectl apply -f "${MONGODB_MANIFEST}" || {
        echo "ERROR: MongoDB deployment failed" >&2
        exit 1
    }
    kubectl wait --for=condition=available deployment/open5gs-mongodb -n "${NAMESPACE}" --timeout=90s 2>/dev/null || true
else
    echo "  No MongoDB manifest found, skipping."
fi

# Step 4: Deploy via Helm (no --wait to avoid timeout on startup ordering)
echo "[4/5] Deploying ${PROFILE_NAME}..."
helm install "${RELEASE_NAME}" "${CHARTS_DIR}/charts/${PROFILE_NAME}" \
    -n "${NAMESPACE}" \
    -f "${SCRIPT_DIR}/${PROFILE_NAME}-values.yaml" \
    --timeout "${HELM_TIMEOUT}s" 2>&1 || {
    echo "ERROR: Helm install failed" >&2
    exit 1
}

# Install UERANSIM
helm install ueransim "${CHARTS_DIR}/charts/ueransim" \
    -n "${NAMESPACE}" \
    --timeout "${HELM_TIMEOUT}s" 2>&1 || {
    echo "WARNING: UERANSIM Helm install failed, continuing without it" >&2
}

# Step 5: Health check with generous timeout (pods need time for startup ordering)
echo "[5/5] Waiting for deployment to become healthy..."
${HEALTH_CMD} --wait -t 180

echo "================================================================"
echo "Done. ${PROFILE_NAME} redeployed in namespace '${NAMESPACE}'."
echo "================================================================"
