#!/usr/bin/env bash
# OperAID — Single experiment runner
# Runs one fault scenario: inject → diagnose → remediate → verify → cleanup
#
# Usage:
#   ./run_experiment.sh --api-key <KEY> --model <MODEL> --scenario <1|2|3> [options]
#
# The API key can also be provided via OPENROUTER_API_KEY environment variable.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config defaults
# shellcheck source=config.env
source "${SCRIPT_DIR}/config.env"

# ---------------------------------------------------------------------------
# Defaults (can be overridden by CLI args or config.env)
# ---------------------------------------------------------------------------
API_KEY="${OPENROUTER_API_KEY:-}"
MODEL="${LLM_MODEL:-z-ai/glm-5}"
NAMESPACE="${NAMESPACE:-default}"
MAX_TURNS="${LLM_MAX_TURNS:-3}"
TEMPERATURE="${LLM_TEMPERATURE:-0.0}"
MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"
BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}"
USE_TOOLS="true"
SCENARIO=""
OUTPUT_DIR=""
HEALTH_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-120}"
REMEDIATION_TIMEOUT="${REMEDIATION_STEP_TIMEOUT:-60}"
REQUEST_TIMEOUT="${LLM_REQUEST_TIMEOUT:-240}"
MAX_RETRIES="${TRY_QUERY_AGAIN:-10}"
DEPLOYMENT_PROFILE="${DEPLOYMENT_PROFILE:-open5gs}"
FIRST_RUN="false"

# Resolve profile path
PROFILE_PATH="${SCRIPT_DIR}/deployments/${DEPLOYMENT_PROFILE}.json"
if [[ ! -f "${PROFILE_PATH}" ]]; then
    PROFILE_PATH="${DEPLOYMENT_PROFILE}"
fi

# Load profile and extract namespace if not overridden
if [[ -f "${PROFILE_PATH}" ]] && [[ "${NAMESPACE}" == "default" || "${NAMESPACE}" == "open5gs" ]]; then
    PROFILE_NS=$(python3 -c "
import json, sys
with open('${PROFILE_PATH}') as f:
    p = json.load(f)
print(p.get('namespace', 'default'))
" 2>/dev/null || echo "default")
    # Only override if namespace wasn't explicitly set via CLI
    if [[ "${CLI_NAMESPACE_SET:-false}" == "false" ]]; then
        NAMESPACE="${PROFILE_NS}"
    fi
fi

# Scenario descriptions — loaded from profile via Python helper
load_scenario_description() {
    local scenario_num="$1"
    python3 -c "
import json, sys
with open('${PROFILE_PATH}') as f:
    profile = json.load(f)
ns = '${NAMESPACE}'
scenarios = profile.get('scenario_targets', {})
sc = scenarios.get('${scenario_num}', {})
desc = sc.get('description', 'A fault has been injected. Diagnose the root cause.')
print(desc.format(namespace=ns))
"
}

# Fault injection — loaded from profile via Python helper
inject_fault() {
    local scenario_num="$1"
    local ns="$2"
    local output_log="$3"

    python3 -c "
import json, subprocess, sys

with open('${PROFILE_PATH}') as f:
    profile = json.load(f)

fi_config = profile.get('fault_injection', {}).get('${scenario_num}', {})
method = fi_config.get('method', '')
ns = '${ns}'

if method == 'kubectl_apply':
    script_dir = '${SCRIPT_DIR}'
    f = fi_config.get('file', '')
    if not f.startswith('/'):
        f = script_dir + '/' + f
    result = subprocess.run(['kubectl', 'apply', '-f', f], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f'ERROR: kubectl apply failed (exit {result.returncode})', file=sys.stderr)
        sys.exit(result.returncode)
elif method == 'kubectl_patch':
    target = fi_config.get('target_deployment', '')
    patch = fi_config.get('patch', '')
    result = subprocess.run(
        ['kubectl', 'patch', 'deployment', target, '-n', ns, '--type=json', '-p', patch],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f'ERROR: kubectl patch failed (exit {result.returncode})', file=sys.stderr)
        sys.exit(result.returncode)
    # Wait for rollout
    rollout = subprocess.run(['kubectl', 'rollout', 'status', f'deployment/{target}', '-n', ns, '--timeout=60s'],
                   capture_output=True, text=True)
    if rollout.returncode != 0:
        print(f'ERROR: rollout status check failed (exit {rollout.returncode})', file=sys.stderr)
        sys.exit(rollout.returncode)
    import time; time.sleep(5)
elif method == 'kubectl_scale':
    target = fi_config.get('target_deployment', '')
    replicas = fi_config.get('replicas', 0)
    result = subprocess.run(
        ['kubectl', 'scale', 'deployment', target, '-n', ns, f'--replicas={replicas}'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f'ERROR: kubectl scale failed (exit {result.returncode})', file=sys.stderr)
        sys.exit(result.returncode)
    import time; time.sleep(5)
else:
    print(f'ERROR: Unknown injection method: {method}', file=sys.stderr)
    sys.exit(1)
" 2>&1 | tee "${output_log}"
}

# Fallback scripts: reset (fast) first, then nuke (full redeploy) if reset fails
RESET_SCRIPT="${SCRIPT_DIR}/reset-deployment.sh"
NUKE_SCRIPT="${SCRIPT_DIR}/nuke-deployment.sh"

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)       API_KEY="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --scenario)      SCENARIO="$2"; shift 2 ;;
        --namespace)     NAMESPACE="$2"; CLI_NAMESPACE_SET="true"; shift 2 ;;
        --max-turns)     MAX_TURNS="$2"; shift 2 ;;
        --temperature)   TEMPERATURE="$2"; shift 2 ;;
        --no-tools)      USE_TOOLS="false"; shift ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --base-url)      BASE_URL="$2"; shift 2 ;;
        --max-retries)   MAX_RETRIES="$2"; shift 2 ;;
        --request-timeout) REQUEST_TIMEOUT="$2"; shift 2 ;;
        --profile)       DEPLOYMENT_PROFILE="$2"; shift 2 ;;
        --first-run)     FIRST_RUN="true"; shift ;;
        -h|--help)
            echo "Usage: $0 --api-key <KEY> --model <MODEL> --scenario <1|2|3> [options]"
            echo ""
            echo "Options:"
            echo "  --api-key KEY         OpenRouter API key (or set OPENROUTER_API_KEY)"
            echo "  --model MODEL         LLM model identifier"
            echo "  --scenario N          Scenario number (1, 2, or 3)"
            echo "  --namespace NS        Kubernetes namespace (default: from profile)"
            echo "  --max-turns N         Max diagnosis turns (default: 3)"
            echo "  --temperature T       LLM temperature (default: 0.0)"
            echo "  --no-tools            Disable diagnostic tool access"
            echo "  --output-dir DIR      Output directory (auto-generated if not set)"
            echo "  --base-url URL        API base URL"
            echo "  --max-retries N       Max API retries (default: 10)"
            echo "  --request-timeout S   API request timeout in seconds (default: 240)"
            echo "  --profile NAME        Deployment profile name or path (default: open5gs)"
            echo "  --first-run           Signal first run in suite (forces nuke for clean baseline)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "${API_KEY}" ]]; then
    echo "ERROR: --api-key is required (or set OPENROUTER_API_KEY)" >&2; exit 1
fi
if [[ -z "${SCENARIO}" ]]; then
    echo "ERROR: --scenario is required (1, 2, or 3)" >&2; exit 1
fi
if [[ ! "${SCENARIO}" =~ ^[123]$ ]]; then
    echo "ERROR: --scenario must be 1, 2, or 3" >&2; exit 1
fi
if [[ ! -f "${PROFILE_PATH}" ]]; then
    echo "ERROR: Deployment profile not found: ${PROFILE_PATH}" >&2; exit 1
fi

# Build tool condition label
TOOL_CONDITION="all-tools"
[[ "${USE_TOOLS}" == "false" ]] && TOOL_CONDITION="no-tools"

# Auto-generate output dir
if [[ -z "${OUTPUT_DIR}" ]]; then
    TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
    MODEL_SHORT="$(echo "${MODEL}" | sed 's|.*/||; s|[^a-zA-Z0-9._-]|_|g')"
    OUTPUT_DIR="${SCRIPT_DIR}/suite_results/${MODEL_SHORT}-${TOOL_CONDITION}/${TIMESTAMP}/scenario_${SCENARIO}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "OperAID Experiment"
echo "================================================================"
echo "  Model:      ${MODEL}"
echo "  Scenario:   ${SCENARIO} (${TOOL_CONDITION})"
echo "  Namespace:  ${NAMESPACE}"
echo "  Profile:    ${DEPLOYMENT_PROFILE}"
echo "  Max turns:  ${MAX_TURNS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Pre-flight health check
# ---------------------------------------------------------------------------
# - First run (--first-run flag): always nuke to establish clean baseline.
# - Subsequent runs: single health check. Safe because the previous run's
#   LLM diagnosis phase (20-90s) already provides a natural stabilization
#   window for any CrashLoopBackOff to surface.
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Pre-flight health check..."

if [[ "${FIRST_RUN}" == "true" ]]; then
    echo "  First run in suite. Performing full nuke for clean baseline..."
    if ! "${NUKE_SCRIPT}" --profile "${DEPLOYMENT_PROFILE}" --namespace "${NAMESPACE}" 2>&1; then
        echo "ERROR: Nuke + redeploy failed" >&2
        RESULT_JSON=$(python3 -c "
import json
result = {
    'scenario': ${SCENARIO},
    'model': '${MODEL}',
    'tool_condition': '${TOOL_CONDITION}',
    'success': False,
    'llm_success': False,
    'source': 'nuke_failed',
    'root_cause': 'Nuke + redeploy failed',
    'root_cause_verified': False,
    'turns_used': 0,
    'duration_s': 0,
    'timestamp': '',
    'api_errors': 0,
    'failure_reason': 'nuke_failed',
}
print(json.dumps(result, indent=2))
")
        echo "${RESULT_JSON}" > "${OUTPUT_DIR}/result.json"
        exit 1
    fi
    echo "  ✓ Clean baseline established."
elif python3 "${SCRIPT_DIR}/health/health_check.py" --profile "${PROFILE_PATH}" -n "${NAMESPACE}"; then
    echo "  ✓ Cluster healthy. Skipping nuke."
else
    if "${RESET_SCRIPT}" --profile "${DEPLOYMENT_PROFILE}" --namespace "${NAMESPACE}" 2>&1; then
        echo "  Reset succeeded."
    else
        echo "  Reset failed. Attempting nuke + redeploy..." >&2
        if ! "${NUKE_SCRIPT}" --profile "${DEPLOYMENT_PROFILE}" --namespace "${NAMESPACE}" 2>&1; then
            echo "ERROR: Nuke + redeploy failed" >&2
            RESULT_JSON=$(python3 -c "
import json
result = {
    'scenario': ${SCENARIO},
    'model': '${MODEL}',
    'tool_condition': '${TOOL_CONDITION}',
    'success': False,
    'llm_success': False,
    'source': 'nuke_failed',
    'root_cause': 'Nuke + redeploy failed',
    'root_cause_verified': False,
    'turns_used': 0,
    'duration_s': 0,
    'timestamp': '',
    'api_errors': 0,
    'failure_reason': 'nuke_failed',
}
print(json.dumps(result, indent=2))
")
            echo "${RESULT_JSON}" > "${OUTPUT_DIR}/result.json"
            exit 1
        fi
        echo "  ✓ Nuke + redeploy succeeded."
    fi

    if ! python3 "${SCRIPT_DIR}/health/health_check.py" --profile "${PROFILE_PATH}" -n "${NAMESPACE}" --wait -t "${HEALTH_TIMEOUT}"; then
        echo "ERROR: Cluster still not healthy after recovery attempt" >&2
        RESULT_JSON=$(python3 -c "
import json
result = {
    'scenario': ${SCENARIO},
    'model': '${MODEL}',
    'tool_condition': '${TOOL_CONDITION}',
    'success': False,
    'llm_success': False,
    'source': 'health_check_failed',
    'root_cause': 'Cluster not healthy before fault injection',
    'root_cause_verified': False,
    'turns_used': 0,
    'duration_s': 0,
    'timestamp': '',
    'api_errors': 0,
    'failure_reason': 'pre_flight_health_check',
}
print(json.dumps(result, indent=2))
")
        echo "${RESULT_JSON}" > "${OUTPUT_DIR}/result.json"
        exit 1
    fi
    echo "  ✓ Cluster recovered. Proceeding with experiment."
fi

# ---------------------------------------------------------------------------
# Step 2: Inject fault
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Injecting fault (scenario ${SCENARIO})..."

inject_fault "${SCENARIO}" "${NAMESPACE}" "${OUTPUT_DIR}/inject.log"
echo "  Fault injected."

# Verify fault injection for scenarios 2 and 3
if [[ "${SCENARIO}" == "2" ]]; then
    ENVFROM_CHECK=$(python3 -c "
import json, subprocess
result = subprocess.run(
    ['kubectl', 'get', 'deployment', 'open5gs-smf', '-n', '${NAMESPACE}', '-o', 'json'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print('error')
    exit(0)
d = json.loads(result.stdout)
containers = d.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
for c in containers:
    for ef in c.get('envFrom', []):
        cm = ef.get('configMapRef', {}).get('name', '')
        if 'smf-extra-config-missing' in cm:
            print('present')
            exit(0)
print('absent')
" 2>/dev/null)
    if [[ "${ENVFROM_CHECK}" != "present" ]]; then
        echo "  ERROR: Fault injection failed — SMF does not reference missing ConfigMap 'smf-extra-config-missing'" >&2
        exit 1
    else
        echo "  ✓ Fault verified: SMF references missing ConfigMap"
    fi
elif [[ "${SCENARIO}" == "3" ]]; then
    UPF_REPLICAS=$(python3 -c "
import json, subprocess
result = subprocess.run(
    ['kubectl', 'get', 'deployment', 'open5gs-upf', '-n', '${NAMESPACE}', '-o', 'json'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print('0')
    exit(0)
d = json.loads(result.stdout)
print(d.get('spec', {}).get('replicas', 0))
" 2>/dev/null)
    if [[ "${UPF_REPLICAS}" != "0" ]]; then
        echo "  ERROR: Fault injection failed — UPF replicas is ${UPF_REPLICAS}, expected 0" >&2
        exit 1
    else
        echo "  ✓ Fault verified: UPF scaled to 0 replicas"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3: LLM Diagnosis
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Running LLM diagnosis..."
DIAGNOSIS_FILE="${OUTPUT_DIR}/diagnosis.json"

SCENARIO_DESC=$(load_scenario_description "${SCENARIO}")

TOOL_FLAG=""
[[ "${USE_TOOLS}" == "false" ]] && TOOL_FLAG="--no-tools"

python3 "${SCRIPT_DIR}/engine/diagnose.py" \
    --api-key "${API_KEY}" \
    --model "${MODEL}" \
    --namespace "${NAMESPACE}" \
    --profile "${PROFILE_PATH}" \
    --max-turns "${MAX_TURNS}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --base-url "${BASE_URL}" \
    --max-retries "${MAX_RETRIES}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --scenario-description "${SCENARIO_DESC}" \
    --output "${DIAGNOSIS_FILE}" \
    ${TOOL_FLAG}

# Parse diagnosis result
HAS_REMEDIATION=$(python3 -c "
import json, sys
with open('${DIAGNOSIS_FILE}') as f:
    d = json.load(f)
print('true' if d.get('has_actionable_remediation') else 'false')
")

# ---------------------------------------------------------------------------
# Step 4: Execute remediation
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Executing remediation..."
LLM_SUCCESS="false"
SOURCE="llm"

if [[ "${HAS_REMEDIATION}" == "true" ]]; then
    # Execute LLM-proposed remediation
    python3 -c "
import json, sys
sys.path.insert(0, '${SCRIPT_DIR}')
from engine.diagnose import execute_remediation
with open('${DIAGNOSIS_FILE}') as f:
    result = json.load(f)
rem = execute_remediation(result, '${NAMESPACE}')
with open('${OUTPUT_DIR}/remediation.json', 'w') as f:
    json.dump(rem, f, indent=2)
print('Remediation executed:', 'success' if rem['all_success'] else 'partial')
"

    # Verify health after LLM remediation
    echo "  Waiting for cluster to stabilize..."
    if python3 "${SCRIPT_DIR}/health/health_check.py" --profile "${PROFILE_PATH}" -n "${NAMESPACE}" --wait -t "${HEALTH_TIMEOUT}"; then
        # Scenario-specific fault verification
        SCENARIO_FAULT_CLEARED="true"
        if [[ "${SCENARIO}" == "1" ]]; then
            if kubectl get networkpolicy block-amf-to-smf -n "${NAMESPACE}" &>/dev/null; then
                echo "  ✗ NetworkPolicy 'block-amf-to-smf' still present — fault NOT remediated."
                SCENARIO_FAULT_CLEARED="false"
            else
                echo "  ✓ NetworkPolicy 'block-amf-to-smf' removed."
            fi
        elif [[ "${SCENARIO}" == "2" ]]; then
            ENVFROM_CHECK=$(python3 -c "
import json, subprocess
result = subprocess.run(
    ['kubectl', 'get', 'deployment', 'open5gs-smf', '-n', '${NAMESPACE}', '-o', 'json'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print('error')
    exit(0)
d = json.loads(result.stdout)
containers = d.get('spec', {}).get('template', {}).get('spec', {}).get('containers', [])
for c in containers:
    for ef in c.get('envFrom', []):
        cm = ef.get('configMapRef', {}).get('name', '')
        if 'smf-extra-config-missing' in cm:
            print('still_present')
            exit(0)
print('cleared')
" 2>/dev/null)
            if [[ "${ENVFROM_CHECK}" == "still_present" ]]; then
                echo "  ✗ SMF still references non-existent ConfigMap 'smf-extra-config-missing' — fault NOT remediated."
                SCENARIO_FAULT_CLEARED="false"
            else
                echo "  ✓ SMF ConfigMap reference cleared."
            fi
        elif [[ "${SCENARIO}" == "3" ]]; then
            UPF_REPLICAS=$(python3 -c "
import json, subprocess
result = subprocess.run(
    ['kubectl', 'get', 'deployment', 'open5gs-upf', '-n', '${NAMESPACE}', '-o', 'json'],
    capture_output=True, text=True
)
if result.returncode != 0:
    print('0')
    exit(0)
d = json.loads(result.stdout)
print(d.get('spec', {}).get('replicas', 0))
" 2>/dev/null)
            if [[ "${UPF_REPLICAS}" == "0" ]]; then
                echo "  ✗ UPF still scaled to 0 replicas — fault NOT remediated."
                SCENARIO_FAULT_CLEARED="false"
            else
                echo "  ✓ UPF replicas restored to ${UPF_REPLICAS}."
            fi
        fi

        if [[ "${SCENARIO_FAULT_CLEARED}" == "true" ]]; then
            LLM_SUCCESS="true"
            echo "  ✓ LLM remediation successful!"
        else
            echo "  ✗ LLM remediation failed (fault still present). Applying groundtruth fallback..."
            SOURCE="groundtruth"
        fi
    else
        echo "  ✗ LLM remediation failed. Applying groundtruth fallback..."
        SOURCE="groundtruth"
    fi
else
    echo "  No actionable remediation from LLM. Applying groundtruth fallback..."
    SOURCE="groundtruth"
fi

# Fallback: try reset first (fast), then nuke if reset fails
FINAL_SUCCESS="false"
if [[ "${LLM_SUCCESS}" == "false" ]]; then
    echo "  Attempting fast reset via helm upgrade..."
    if "${RESET_SCRIPT}" --profile "${DEPLOYMENT_PROFILE}" --namespace "${NAMESPACE}" 2>&1 | tee "${OUTPUT_DIR}/reset.log"; then
        echo "  ✓ Fast reset successful"
        FINAL_SUCCESS="true"
    else
        echo "  ✗ Fast reset failed. Falling back to full nuke + redeploy..."
        if "${NUKE_SCRIPT}" --profile "${DEPLOYMENT_PROFILE}" --namespace "${NAMESPACE}" 2>&1 | tee "${OUTPUT_DIR}/nuke_redeploy.log"; then
            echo "  ✓ Nuke + redeploy successful"
            FINAL_SUCCESS="true"
        else
            echo "  ✗ WARNING: Nuke + redeploy failed! Manual intervention may be needed."
        fi
    fi
else
    FINAL_SUCCESS="true"
fi

# ---------------------------------------------------------------------------
# Step 5: Record result
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Recording result..."

# Extract diagnosis metadata
RESULT_JSON=$(python3 -c "
import json
with open('${DIAGNOSIS_FILE}') as f:
    d = json.load(f)
meta = d.get('session_meta', {})
diag = d.get('diagnosis', {})
root_cause_verified = '${LLM_SUCCESS}' == 'true'
# success tracks final cluster health, llm_success tracks if LLM fixed it
result = {
    'scenario': ${SCENARIO},
    'model': '${MODEL}',
    'tool_condition': '${TOOL_CONDITION}',
    'success': '${FINAL_SUCCESS}' == 'true',
    'llm_success': '${LLM_SUCCESS}' == 'true',
    'source': '${SOURCE}',
    'root_cause': diag.get('root_cause', 'unknown'),
    'root_cause_verified': root_cause_verified,
    'has_actionable_remediation': d.get('has_actionable_remediation', False),
    'turns_used': meta.get('turns_used', 0),
    'duration_s': meta.get('duration_s', 0),
    'timestamp': meta.get('timestamp', ''),
    'api_errors': len(d.get('api_errors', [])),
    'total_input_tokens': meta.get('total_input_tokens', 0),
    'total_output_tokens': meta.get('total_output_tokens', 0),
    'total_tokens': meta.get('total_tokens', 0),
}
print(json.dumps(result, indent=2))
")

echo "${RESULT_JSON}" > "${OUTPUT_DIR}/result.json"

# Print summary
echo ""
echo "================================================================"
echo "Result Summary"
echo "================================================================"
echo "${RESULT_JSON}"
echo "================================================================"
