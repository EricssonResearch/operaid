"""
Kubectl diagnostic tools for LLM function calling.

Each tool has:
- A schema (OpenAI function-calling format) for the LLM
- An execute() dispatcher that runs the actual kubectl command

Supports deployment profiles for name resolution across different deployments.
Users can register additional tools via register_tool().
"""

import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional

# Configurable timeout (seconds) - matches revelens pattern
KUBECTL_TIMEOUT = int(os.environ.get("KUBECTL_COMMAND_TIMEOUT", "10"))

# Dangerous/interactive commands that can hang the terminal
_BLOCKED_SUBCOMMANDS = frozenset(
    {
        "edit",
        "exec",
        "port-forward",
        "watch",
        "proxy",
        "attach",
        "run",
        "cp",
        "debug",
        "wait",
        "krew",
    }
)

# ---------------------------------------------------------------------------
# Tool registry — builtin kubectl tools + user-extensible via register_tool()
# ---------------------------------------------------------------------------

ToolExecutor = Callable[[str, Dict[str, Any], str, Optional[Dict[str, Any]]], str]


def _builtin_execute(
    tool_name: str,
    arguments: Dict[str, Any],
    namespace: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Default executor for all builtin kubectl tools."""
    prof = profile or _active_profile

    if tool_name == "get_pods":
        return _run(["kubectl", "get", "pods", "-n", namespace, "-o", "wide"])

    elif tool_name == "describe_pod":
        pod = _resolve_name(arguments.get("pod_name", ""), namespace, "pods", prof)
        return _run(["kubectl", "describe", "pod", pod, "-n", namespace])

    elif tool_name == "get_pod_logs":
        pod = _resolve_name(arguments.get("pod_name", ""), namespace, "pods", prof)
        cmd = ["kubectl", "logs", pod, "-n", namespace, "--tail=80"]
        if arguments.get("previous", False):
            cmd.append("--previous")
        return _run(cmd)

    elif tool_name == "get_events":
        return _run(
            [
                "kubectl",
                "get",
                "events",
                "-n",
                namespace,
                "--sort-by=.lastTimestamp",
            ]
        )

    elif tool_name == "get_networkpolicies":
        return _run(
            [
                "kubectl",
                "get",
                "networkpolicies",
                "-n",
                namespace,
                "-o",
                "yaml",
            ]
        )

    elif tool_name == "get_deployment":
        name = _resolve_name(
            arguments.get("deployment_name", ""), namespace, "deployments", prof
        )
        return _run(
            [
                "kubectl",
                "get",
                "deployment",
                name,
                "-n",
                namespace,
                "-o",
                "yaml",
            ]
        )

    elif tool_name == "run_kubectl":
        raw_cmd = arguments.get("command", "").strip()
        parts = raw_cmd.split()
        if not parts:
            return "ERROR: empty command"
        subcommand = parts[0].lower()
        if subcommand in _BLOCKED_SUBCOMMANDS:
            return f"ERROR: '{subcommand}' is blocked (can hang terminal). Blocked commands: {sorted(_BLOCKED_SUBCOMMANDS)}"
        if subcommand not in _READONLY_SUBCOMMANDS:
            return f"ERROR: '{subcommand}' is not a read-only command. Allowed: {sorted(_READONLY_SUBCOMMANDS)}"
        cmd = ["kubectl"] + parts + ["-n", namespace]
        return _run(cmd)

    else:
        return f"ERROR: Unknown tool '{tool_name}'"


_BUILTIN_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_pods",
            "description": "List all pods in the namespace with their status, restarts, and age.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_pod",
            "description": "Get detailed information about a specific pod including events, conditions, and container statuses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod_name": {
                        "type": "string",
                        "description": "Name of the pod to describe.",
                    }
                },
                "required": ["pod_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pod_logs",
            "description": "Get the last 80 lines of logs from a pod. Use previous=true to get logs from the previous (crashed) container.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pod_name": {
                        "type": "string",
                        "description": "Name of the pod.",
                    },
                    "previous": {
                        "type": "boolean",
                        "description": "If true, get logs from the previous (crashed) container instance.",
                        "default": False,
                    },
                },
                "required": ["pod_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": "Get recent Kubernetes events in the namespace, sorted by timestamp. Useful for spotting scheduling failures, crashes, and policy violations.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_networkpolicies",
            "description": "List all NetworkPolicies in the namespace with their pod selectors and rules.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_deployment",
            "description": "Get the full YAML spec of a deployment, including replicas, containers, volumes, and env references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "deployment_name": {
                        "type": "string",
                        "description": "Name of the deployment.",
                    }
                },
                "required": ["deployment_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_kubectl",
            "description": "Run an arbitrary read-only kubectl command. Use this for commands not covered by the other tools (e.g., get services, get configmaps). Write operations are NOT allowed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The kubectl command to run (without 'kubectl' prefix). Example: 'get svc -o wide'",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


_extra_tools: Dict[str, Dict[str, Any]] = {}


def clear_extra_tools() -> None:
    _extra_tools.clear()


def register_tool(
    schema: Dict[str, Any],
    executor: ToolExecutor,
) -> None:
    """Register a custom diagnostic tool.

    Args:
        schema: OpenAI function-calling format tool schema with a "function" key.
        executor: Callable(tool_name, arguments, namespace, profile) -> str
    """
    name = schema["function"]["name"]
    if name in _extra_tools:
        raise ValueError(f"Tool '{name}' is already registered")
    _extra_tools[name] = {"schema": schema, "executor": executor}


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return all tool schemas (builtin + registered extras)."""
    return _BUILTIN_SCHEMAS + [t["schema"] for t in _extra_tools.values()]


def get_tool_names() -> List[str]:
    """Return all registered tool names."""
    builtin_names = [s["function"]["name"] for s in _BUILTIN_SCHEMAS]
    return builtin_names + list(_extra_tools.keys())


def load_profile_tools(profile: Optional[Dict[str, Any]]) -> None:
    """Load custom tools from a deployment profile, replacing any previously loaded profile tools.

    Each entry in profile["custom_tools"] must have:
      - "schema": OpenAI function-calling tool schema (with "function" key)
      - "executor": dict with execution config:
          {"type": "kubectl", "command": "get configmap {name} -o yaml"}
          {"type": "shell", "command": "curl -s http://localhost:8080/health"}
        Template variables {arg_name} are substituted from tool arguments.
        {namespace} is always available.
    """
    clear_extra_tools()

    if not profile:
        return

    custom_tools = profile.get("custom_tools")
    if not custom_tools:
        return

    builtin_names = {s["function"]["name"] for s in _BUILTIN_SCHEMAS}

    for tool_def in custom_tools:
        schema = tool_def["schema"]
        exec_config = tool_def["executor"]
        name = schema["function"]["name"]

        if name in builtin_names:
            print(f"  WARNING: Custom tool '{name}' conflicts with builtin, skipping")
            continue

        if exec_config["type"] == "kubectl":
            cmd_template = exec_config["command"]

            def _make_kubectl_executor(template: str) -> ToolExecutor:
                def executor(
                    tool_name: str,
                    arguments: Dict[str, Any],
                    namespace: str,
                    profile: Optional[Dict[str, Any]] = None,
                ) -> str:
                    params = {**arguments, "namespace": namespace}
                    cmd_str = template.format(**params)
                    parts = cmd_str.split()
                    subcommand = parts[0].lower() if parts else ""
                    if subcommand in _BLOCKED_SUBCOMMANDS:
                        return f"ERROR: '{subcommand}' is blocked"
                    return _run(["kubectl"] + parts + ["-n", namespace])

                return executor

            register_tool(schema, _make_kubectl_executor(cmd_template))

        elif exec_config["type"] == "shell":
            cmd_template = exec_config["command"]

            def _make_shell_executor(template: str) -> ToolExecutor:
                def executor(
                    tool_name: str,
                    arguments: Dict[str, Any],
                    namespace: str,
                    profile: Optional[Dict[str, Any]] = None,
                ) -> str:
                    params = {**arguments, "namespace": namespace}
                    cmd_str = template.format(**params)
                    return _run(["sh", "-c", cmd_str])

                return executor

            register_tool(schema, _make_shell_executor(cmd_template))

        else:
            print(
                f"  WARNING: Unknown executor type '{exec_config['type']}' for tool '{name}', skipping"
            )


TOOL_SCHEMAS = get_tool_schemas()


def _run(cmd: List[str]) -> str:
    """Run a command and return stdout+stderr, truncated to 8000 chars."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=KUBECTL_TIMEOUT,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        output = output.strip()
        if len(output) > 8000:
            output = output[:8000] + "\n... (truncated)"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: Command timed out after {KUBECTL_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


# Read-only kubectl subcommands allowed for run_kubectl
_READONLY_SUBCOMMANDS = frozenset(
    {
        "get",
        "describe",
        "logs",
        "top",
        "explain",
        "api-resources",
        "api-versions",
        "cluster-info",
        "events",
        "auth",
    }
)


def _resolve_name(
    short_name: str,
    namespace: str,
    resource: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Resolve a short name (e.g. 'smf') to a full name (e.g. 'open5gs-smf').

    If a profile is provided, uses profile prefix rules first.
    Otherwise falls back to cluster-based resolution.
    """
    # Profile-based resolution
    if profile:
        from engine.profile import resolve_name as profile_resolve

        return profile_resolve(short_name, profile, resource, namespace)

    # Cluster-based resolution (fallback)
    result = subprocess.run(
        [
            "kubectl",
            "get",
            resource,
            "-n",
            namespace,
            "--no-headers",
            "-o",
            "custom-columns=NAME:.metadata.name",
        ],
        capture_output=True,
        text=True,
        timeout=KUBECTL_TIMEOUT,
    )
    names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
    if short_name in names:
        return short_name
    # Try common prefix patterns
    for name in names:
        if name.endswith(f"-{short_name}"):
            return name
    return short_name  # Fall back to original if no match found


# Global profile reference (set via set_profile)
_active_profile: Optional[Dict[str, Any]] = None


def set_profile(profile: Optional[Dict[str, Any]]) -> None:
    """Set the active deployment profile for name resolution."""
    global _active_profile
    _active_profile = profile


def get_profile() -> Optional[Dict[str, Any]]:
    """Get the active deployment profile."""
    return _active_profile


def execute(
    tool_name: str,
    arguments: Dict[str, Any],
    namespace: str,
    profile: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatch a tool call and return the string result.

    Checks registered extra tools first, then falls back to builtin kubectl tools.
    """
    if tool_name in _extra_tools:
        return _extra_tools[tool_name]["executor"](
            tool_name, arguments, namespace, profile
        )
    return _builtin_execute(tool_name, arguments, namespace, profile)
