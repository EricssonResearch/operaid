# OperAID

**OperAID** is an open-source testbed for evaluating LLM agents as autonomous operators of 5G Core networks deployed on Kubernetes. It provides a closed-loop pipeline: **Fault Injection → Agentic Diagnosis → Remediation → Execution-Based Verification**.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/EricssonResearch/operaid)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENROUTER_API_KEY="sk-..."

# Run a single experiment
./run_experiment.sh --api-key "$OPENROUTER_API_KEY" --model z-ai/glm-5 --scenario 1

# Run without tools (no-tools condition)
./run_experiment.sh --api-key "$OPENROUTER_API_KEY" --model z-ai/glm-5 --scenario 1 --no-tools

# Run a full suite (YAML config)
python3 run_suite.py --suite suites/tool_impact.yaml --api-key "$OPENROUTER_API_KEY"

# Generate paper figures from results
python3 visualize_suite.py --stats paper/figures/suite_statistics.json --csv summary.csv -o paper/figures
```

## Prerequisites

- Kubernetes cluster (KinD recommended)
- [openverso-charts](https://github.com/Gradiant/openverso-charts) — Helm charts for Open5GS + UERANSIM deployment
- Python 3.10+
- `kubectl` and `helm` configured and pointing to your cluster
- OpenRouter API key (or any OpenAI-compatible API)

Set the charts path in `config.env` or via environment variable:
```bash
export OPENVERSO_CHARTS_DIR=/path/to/openverso-charts
```

## Project Structure

```
operaid/
├── config.env                 # Main configuration (LLM, K8s, timeouts)
├── run_experiment.sh          # Single experiment runner
├── run_suite.py               # Suite runner with YAML config
├── regenerate_stats.py        # Regenerate suite_statistics.json + summary.csv from per-run data
├── nuke-deployment.sh         # Full cleanup / redeploy
├── reset-deployment.sh        # Fast reset via helm upgrade
├── visualize.py               # Single-run visualizer
├── visualize_suite.py         # Suite visualizer (generates paper figures)
├── scenario_definitions.json  # Scenario definitions + expected remediation
├── deployments/
│   └── open5gs.json           # Open5GS deployment profile (faults, health, tools)
├── engine/
│   ├── diagnose.py            # Multi-turn LLM diagnosis engine
│   └── profile.py             # Deployment profile loader
├── tools/
│   ├── kubectl_tools.py       # Built-in kubectl diagnostic tools + custom tool registry
│   └── __init__.py
├── health/
│   └── health_check.py        # Generic deployment health check
├── scenarios/
│   ├── scenario_1_netpol.yaml # S1: NetworkPolicy fault
│   ├── scenario_2_configmap.yaml # S2: ConfigMap fault
│   └── scenario_3_upf_scale.yaml # S3: UPF scaling fault
├── suites/
│   ├── tool_impact.yaml       # 5 models × 2 conditions × 3 scenarios × 30 runs
│   ├── single_model_quick.yaml
│   ├── validation_test.yaml
│   ├── temperature_sweep.yaml
│   └── scenario_deep_dive.yaml
└── suite_results/             # Experiment run outputs (git-ignored)
```

## Fault Scenarios

| ID | Type | Description |
|----|------|-------------|
| S1 | Network | NetworkPolicy blocks AMF→SMF SBI (port 7777) |
| S2 | Configuration | SMF references non-existent ConfigMap → CrashLoopBackOff |
| S3 | Scaling | UPF scaled to 0 replicas → no user plane |

## Configuration

### config.env

Main configuration file sourced by `run_experiment.sh`:
- `LLM_PROVIDER` / `LLM_MODEL` — LLM provider and model
- `LLM_MAX_TURNS` — max diagnosis turns (default: 3)
- `LLM_MAX_TOKENS` — max output tokens (default: 4096)
- `NAMESPACE` — Kubernetes namespace (default: from deployment profile)
- `DEPLOYMENT_PROFILE` — deployment profile name or path (default: open5gs)
- Various timeouts for health checks, remediation, and API calls

### Deployment Profiles

Deployment profiles (`deployments/*.json`) define everything specific to a target deployment:

- **Components** — NF names and deployment prefixes
- **Context prompt** — deployment description injected into the LLM system prompt
- **Fault injection** — method and parameters per scenario
- **Health check** — expected deployments and readiness criteria
- **Custom tools** — additional diagnostic tools beyond the built-in set

Example (`deployments/open5gs.json`):

```json
{
  "name": "open5gs",
  "namespace": "open5gs",
  "context_prompt": "Open5GS is a 5G Core network running on Kubernetes...",
  "components": { "amf": { "deployment": "open5gs-amf" }, ... },
  "fault_injection": { "1": { "method": "kubectl_apply", "file": "scenarios/scenario_1_netpol.yaml" }, ... },
  "health_check": { "check_type": "deployments", "expected_deployments": [...] },
  "custom_tools": []
}
```

### Suite YAML

Suite YAML files (in `suites/`) define experiment matrices:

```yaml
common:
  profile: open5gs
  scenarios: [1, 2, 3]
  runs_per_scenario: 30
  max_turns: 3
  temperature: 0.0
  max_tokens: 4096
  custom_tools: []

experiments:
  - name: "glm-5-all-tools"
    model: "z-ai/glm-5"
    use_tools: true
  - name: "glm-5-no-tools"
    model: "z-ai/glm-5"
    use_tools: false
```

### Custom Tools

Custom diagnostic tools can be added via the deployment profile or suite YAML. Each tool defines an OpenAI function-calling schema and an executor:

```json
{
  "custom_tools": [
    {
      "schema": {
        "type": "function",
        "function": {
          "name": "get_configmaps",
          "description": "List all ConfigMaps in the namespace.",
          "parameters": {"type": "object", "properties": {}, "required": []}
        }
      },
      "executor": {"type": "kubectl", "command": "get configmaps"}
    },
    {
      "schema": {
        "type": "function",
        "function": {
          "name": "check_endpoint",
          "description": "Check an HTTP endpoint.",
          "parameters": {
            "type": "object",
            "properties": {"port": {"type": "integer"}},
            "required": ["port"]
          }
        }
      },
      "executor": {"type": "shell", "command": "curl -sf http://localhost:{port}/health || echo 'unhealthy'"}
    }
  ]
}
```

Two executor types are supported:
- **`kubectl`** — runs a kubectl command template. `{arg}` placeholders are substituted from tool arguments. Blocked commands (edit, exec, etc.) are rejected.
- **`shell`** — runs an arbitrary shell command. `{namespace}` and tool arguments are available as placeholders.

Suite-level `custom_tools` override the profile's list. The default is `[]` (built-in tools only).

### Built-in Diagnostic Tools

| Tool | Description |
|------|-------------|
| `get_pods` | List all pods with status and restart counts |
| `get_events` | List Kubernetes events for the namespace |
| `describe_pod(name)` | Detailed pod information including events |
| `get_pod_logs(name)` | Container stdout/stderr logs |
| `get_deployment(name)` | Deployment spec, status, and conditions |
| `get_networkpolicies` | List NetworkPolicies and their rules |
| `run_kubectl(cmd)` | Execute arbitrary read-only kubectl commands |

### Experiment Runner Flow

Each experiment run follows this pipeline:

1. **Pre-flight health check** — first run in a suite performs a full nuke for a clean baseline; subsequent runs skip the nuke if the cluster is healthy (the previous run's LLM diagnosis phase provides a natural stabilization window)
2. **Fault injection** — deterministic fault applied via profile-defined method
3. **Agentic diagnosis** — multi-turn LLM reasoning with tool access (configurable)
4. **Remediation** — proposed `kubectl` commands executed against the cluster
5. **Execution-based verification** — health checks confirm all NFs are ready
6. **Fallback recovery** — if LLM remediation fails, fast reset (helm upgrade) then full nuke as last resort

## Visualization

All figures use the **seaborn rocket** palette. To regenerate paper figures:

```bash
# Generate figures from suite statistics
python3 visualize_suite.py \
    --stats suite_results/tool-impact-analysis/20260401-144241/suite_statistics.json \
    --suite-dir suite_results/tool-impact-analysis/20260401-144241 \
    --pricing paper/pricing.csv \
    -o paper/figures
```

### Regenerating Statistics

If you have per-run data but need to regenerate the aggregated files:

```bash
python3 regenerate_stats.py suite_results/tool-impact-analysis/20260401-144241
```

This recreates:
- `suite_statistics.json` — model/scenario aggregated metrics
- `summary.csv` — per-run results table
- `api_error_summary.csv` — API error breakdown

## Key Results (900 experiments, April 2026)

| Metric | Value |
|--------|-------|
| **Overall LLM success rate** | 36.0% |
| **Average with tools** | 70.7% |
| **Average without tools** | 7.1% |
| **Best performing model** | Qwen3.5-35b-a3b (93.3% with tools) |
| **Best small model** | Qwen3.5-35b-a3b — 3B active params, 93.3% success |

**Scenario breakdown:**
- **S1 (NetworkPolicy)**: 16.0% success — most challenging scenario
- **S2 (ConfigMap)**: 42.0% success — 0% without tools for 4/5 models
- **S3 (UPF Scale)**: 49.3% success — highest overall success rate

**Key findings:**
- **Tool access** raises average success from 7.1% to 70.7% (+63.6pp)
- **Small models** (3B active params) achieve 93.3% with tools
- **S2 (ConfigMap)**: 0% without tools for most models — validates the Validity Gap
- **Failure modes**: 68% "no_remediation" (API limits), 31% "wrong_diagnosis" when tools unavailable

## Citation

```bibtex
@inproceedings{operaid2026,
  title={OperAID: Benchmarking LLM Agents for Autonomous Kubernetes Fault Remediation},
  author={de Castro, Ariel G. and Vandikas, Konstantinos and Ferlin-Reiter, Simone and Chiesa, Marco and Rothenberg, Christian E.},
  booktitle={IEEE NetSoft Trust 6G-Net Workshop},
  year={2026}
}
```
