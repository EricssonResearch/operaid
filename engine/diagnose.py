"""
OperAID Diagnosis Engine — multi-turn LLM agent for Kubernetes fault remediation.

Supports OpenRouter (and other OpenAI-compatible APIs) with function calling.
The engine runs a multi-turn loop where the LLM can request diagnostic tools,
then produces a final diagnosis with remediation steps that are executed and verified.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# Add project root to path so we can import tools
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.kubectl_tools import (
    get_tool_schemas,
    execute as execute_tool,
    get_tool_names,
    load_profile_tools,
)
from engine.profile import load_profile, get_context_prompt, get_namespace, resolve_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KUBECTL_TIMEOUT = 30
REMEDIATION_STEP_TIMEOUT = int(os.environ.get("REMEDIATION_STEP_TIMEOUT", "60"))

# Interactive/dangerous commands that must never be executed
_BLOCKED_COMMANDS = frozenset(
    {
        "edit",
        "exec",
        "port-forward",
        "watch",
        "proxy",
        "attach",
    }
)

# Dangerous shell patterns that could harm the system (from revelens safety checks)
_DANGEROUS_PATTERNS = [
    r"rm\s+-rf",  # rm -rf
    r"mkfs",  # disk formatting
    r"dd\s+if=",  # disk operations
    r":\(\)\s*\{",  # fork bomb start
    r"chmod\s+-R\s+777",  # dangerous permissions
    r">\s*/dev/sd",  # direct disk write
    r"curl.*\|\s*bash",  # curl pipe to bash
    r"wget.*\|\s*bash",  # wget pipe to bash
]


def _is_dangerous(cmd: str) -> bool:
    """Check if command contains dangerous shell patterns (from revelens safety checks)."""
    cmd_lower = cmd.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if re.search(pattern, cmd_lower):
            return True
    return False


_DIAGNOSTIC_SUBCOMMANDS = frozenset(
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

# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------
# Only {deployment_context} and {context} are format placeholders.
# All other braces (JSON examples) are escaped as {{ }}.

_SYSTEM_PROMPT_TOOLS_TEMPLATE = """You are a Kubernetes expert diagnosing a fault in a {deployment_context}.

{context}

## Available Tools
To investigate, request tool calls using the provided function calling interface.
{tool_list}

## Final Answer
When you have enough information, respond with JSON:
{{"root_cause": "one-line description",
 "reasoning": "investigation explanation",
 "remediation_steps": ["kubectl cmd 1", "kubectl cmd 2"]}}

Rules:
- Always investigate first before proposing a fix.
- Remediation steps must be concrete kubectl commands with no placeholders.
- Do NOT include diagnostic commands (get, describe, logs) as remediation steps.
- Do NOT use interactive commands: kubectl edit, kubectl exec (without --command), watch, port-forward.
- For ConfigMap/Deployment fixes use kubectl patch, kubectl delete + kubectl apply, or kubectl set.
- Fix the root cause, not the symptoms."""

_SYSTEM_PROMPT_NO_TOOLS_TEMPLATE = """You are a Kubernetes expert diagnosing a fault in a {deployment_context}.

{context}

Provide your diagnosis directly.

## Final Answer
Respond with JSON:
{{"root_cause": "one-line description",
 "reasoning": "investigation explanation",
 "remediation_steps": ["kubectl cmd 1", "kubectl cmd 2"]}}

Rules:
- Remediation steps must be concrete kubectl commands with no placeholders.
- Do NOT include diagnostic commands (get, describe, logs) as remediation steps.
- Do NOT use interactive commands: kubectl edit, kubectl exec (without --command), watch, port-forward.
- For ConfigMap/Deployment fixes use kubectl patch, kubectl delete + kubectl apply, or kubectl set.
- Fix the root cause, not the symptoms."""

# Backward-compatible defaults for when no profile is provided
_DEFAULT_DEPLOYMENT_CONTEXT = "Kubernetes deployment"
_DEFAULT_CONTEXT = "This is a cloud-native application running on Kubernetes."


def _format_tool_list() -> str:
    schemas = get_tool_schemas()
    lines = []
    for s in schemas:
        func = s.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        short_desc = desc.split(".")[0] if desc else ""
        lines.append(f"- {name}: {short_desc}" if short_desc else f"- {name}")
    return "Available tools:\n" + "\n".join(lines)


def _build_system_prompt(use_tools: bool, profile: Optional[Dict] = None) -> str:
    if profile:
        deployment_context = profile.get("name", _DEFAULT_DEPLOYMENT_CONTEXT)
        context = profile.get("context_prompt", _DEFAULT_CONTEXT)
    else:
        deployment_context = _DEFAULT_DEPLOYMENT_CONTEXT
        context = _DEFAULT_CONTEXT

    template = (
        _SYSTEM_PROMPT_TOOLS_TEMPLATE if use_tools else _SYSTEM_PROMPT_NO_TOOLS_TEMPLATE
    )
    tool_list = _format_tool_list() if use_tools else ""
    return template.format(
        deployment_context=deployment_context, context=context, tool_list=tool_list
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_diagnostic(cmd: str) -> bool:
    """Return True if the kubectl command is read-only / diagnostic."""
    parts = cmd.strip().split()
    # Strip leading 'kubectl' if present
    if parts and parts[0] == "kubectl":
        parts = parts[1:]
    if not parts:
        return True
    return parts[0].lower() in _DIAGNOSTIC_SUBCOMMANDS


def _is_blocked(cmd: str) -> bool:
    """Return True if the command contains a blocked interactive subcommand."""
    parts = cmd.strip().split()
    if parts and parts[0] == "kubectl":
        parts = parts[1:]
    if not parts:
        return False
    return parts[0].lower() in _BLOCKED_COMMANDS


def _extract_json(text: str) -> Optional[Dict]:
    """Try to extract JSON from LLM text output."""
    # Try the whole text first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find JSON block in markdown
    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r'\{[^{}]*"root_cause"[^{}]*"remediation_steps"[^{}]*\}',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except (json.JSONDecodeError, TypeError):
                continue
    # Try to find any JSON object
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i : j + 1])
                        except (json.JSONDecodeError, TypeError):
                            break
    return None


def _run_remediation_step(cmd: str, namespace: str) -> Tuple[bool, str]:
    """Execute a single remediation kubectl command. Returns (success, output)."""
    if _is_dangerous(cmd):
        return False, f"BLOCKED: dangerous pattern detected: {cmd}"
    if _is_blocked(cmd):
        return False, f"BLOCKED: interactive command not allowed: {cmd}"
    if _is_diagnostic(cmd):
        return True, f"SKIPPED: diagnostic command (not remediation): {cmd}"

    # Ensure namespace is included
    full_cmd = cmd.strip()
    if not full_cmd.startswith("kubectl"):
        full_cmd = f"kubectl {full_cmd}"
    if (
        f"-n {namespace}" not in full_cmd
        and f"--namespace {namespace}" not in full_cmd
        and f"-n={namespace}" not in full_cmd
    ):
        full_cmd += f" -n {namespace}"

    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=REMEDIATION_STEP_TIMEOUT,
        )
        output = result.stdout.strip()
        if result.stderr:
            output += "\n" + result.stderr.strip()
        return result.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {REMEDIATION_STEP_TIMEOUT}s"
    except Exception as e:
        return False, f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Diagnosis Engine
# ---------------------------------------------------------------------------


class DiagnosisEngine:
    """Multi-turn LLM diagnosis engine with tool calling."""

    def __init__(
        self,
        api_key: str,
        model: str,
        namespace: str = "default",
        max_turns: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://openrouter.ai/api/v1",
        use_tools: bool = True,
        max_retries: int = 10,
        request_timeout: int = 240,
        profile: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.namespace = namespace
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_tools = use_tools
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.profile = profile
        load_profile_tools(profile)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=2,
        )

        self.conversation: List[Dict[str, Any]] = []
        self.api_errors: List[Dict[str, Any]] = []
        self.turn_timings: List[Dict[str, Any]] = []

    def _get_system_prompt(self) -> str:
        return _build_system_prompt(self.use_tools, self.profile)

    def _query_llm(self, user_message: str) -> Dict[str, Any]:
        """Query the LLM with retry logic. Returns the response message dict."""
        self.conversation.append({"role": "user", "content": user_message})

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self._get_system_prompt()},
                        *self.conversation,
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.use_tools:
                    kwargs["tools"] = get_tool_schemas()
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                # Capture token usage
                if hasattr(response, "usage") and response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens or 0
                    self.total_output_tokens += response.usage.completion_tokens or 0

                # Build response dict
                result = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

                self.conversation.append(result)
                return result

            except Exception as e:
                error_info = {
                    "attempt": attempt,
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self.api_errors.append(error_info)
                print(
                    f"  [LLM] Attempt {attempt}/{self.max_retries} failed: {e}",
                    flush=True,
                )
                if attempt < self.max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError(
                        f"LLM query failed after {self.max_retries} attempts: {e}"
                    )

    def _process_tool_calls(self, tool_calls: List[Dict]) -> str:
        """Execute tool calls and return combined results."""
        results = []
        for tc in tool_calls:
            func = tc["function"]
            name = func["name"]
            try:
                args = json.loads(func["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = {}

            print(f"    [Tool] {name}({args})", flush=True)
            output = execute_tool(name, args, self.namespace, self.profile)
            results.append(f"[{name}] {output}")

            # Add tool result to conversation for function calling
            self.conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": output,
                }
            )

        return "\n\n".join(results)

    def diagnose(self, scenario_description: str) -> Dict[str, Any]:
        """Run the multi-turn diagnosis loop. Returns the full session result."""
        self.conversation = []
        self.api_errors = []
        self.turn_timings = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        session_start = time.time()
        diagnosis_result = None

        for turn in range(1, self.max_turns + 1):
            turn_start = time.time()
            print(f"\n  [Turn {turn}/{self.max_turns}]", flush=True)

            user_msg = (
                scenario_description
                if turn == 1
                else "Based on the information gathered so far, please provide your "
                "final diagnosis and remediation steps as JSON."
            )

            response = self._query_llm(user_msg)

            if "tool_calls" in response and response["tool_calls"]:
                tool_output = self._process_tool_calls(response["tool_calls"])
                turn_duration = time.time() - turn_start
                self.turn_timings.append(
                    {
                        "turn": turn,
                        "duration_s": round(turn_duration, 1),
                        "action": "tool_calls",
                        "tools_called": [
                            tc["function"]["name"] for tc in response["tool_calls"]
                        ],
                        "input_tokens": self.total_input_tokens,
                        "output_tokens": self.total_output_tokens,
                    }
                )
                print(f"    Tools called, continuing...", flush=True)
                continue

            content = response.get("content", "")
            parsed = _extract_json(content)

            turn_duration = time.time() - turn_start
            self.turn_timings.append(
                {
                    "turn": turn,
                    "duration_s": round(turn_duration, 1),
                    "action": "final_answer",
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }
            )

            if parsed and "remediation_steps" in parsed:
                diagnosis_result = parsed
                diagnosis_result["turns_used"] = turn
                break
            elif turn == self.max_turns:
                diagnosis_result = parsed or {
                    "root_cause": "unknown",
                    "reasoning": content[:500] if content else "No response",
                    "remediation_steps": [],
                }
                diagnosis_result["turns_used"] = turn

        if diagnosis_result is None:
            print(
                f"\n  [Final] Requesting diagnosis after tool exploration...",
                flush=True,
            )
            response = self._query_llm(
                "You have used all your investigation turns. Based on the information "
                "gathered, provide your final diagnosis and remediation steps as JSON now."
            )
            content = response.get("content", "")
            parsed = _extract_json(content)
            if parsed and "remediation_steps" in parsed:
                diagnosis_result = parsed
            else:
                diagnosis_result = parsed or {
                    "root_cause": "unknown",
                    "reasoning": content[:500]
                    if content
                    else "No response after final prompt",
                    "remediation_steps": [],
                }
            diagnosis_result["turns_used"] = self.max_turns

        session_duration = time.time() - session_start

        if diagnosis_result is None:
            diagnosis_result = {
                "root_cause": "unknown",
                "reasoning": "Engine did not produce a result",
                "remediation_steps": [],
                "turns_used": self.max_turns,
            }

        # Check if remediation steps contain only diagnostics
        actionable_steps = [
            s
            for s in diagnosis_result.get("remediation_steps", [])
            if not _is_diagnostic(s) and not _is_blocked(s) and not _is_dangerous(s)
        ]

        return {
            "diagnosis": diagnosis_result,
            "session_meta": {
                "model": self.model,
                "namespace": self.namespace,
                "use_tools": self.use_tools,
                "max_turns": self.max_turns,
                "temperature": self.temperature,
                "duration_s": round(session_duration, 1),
                "turns_used": diagnosis_result.get("turns_used", self.max_turns),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
            },
            "turn_timings": self.turn_timings,
            "api_errors": self.api_errors,
            "has_actionable_remediation": len(actionable_steps) > 0,
            "conversation_history": self.conversation,
        }


# ---------------------------------------------------------------------------
# Remediation Execution
# ---------------------------------------------------------------------------


def execute_remediation(diagnosis_result: Dict, namespace: str) -> Dict[str, Any]:
    """Execute the remediation steps from a diagnosis result."""
    steps = diagnosis_result.get("diagnosis", {}).get("remediation_steps", [])
    results = []
    all_success = True

    for i, step in enumerate(steps, 1):
        if _is_dangerous(step):
            results.append({"step": i, "command": step, "status": "blocked_dangerous"})
            all_success = False
            continue
        if _is_diagnostic(step):
            results.append({"step": i, "command": step, "status": "skipped_diagnostic"})
            continue
        if _is_blocked(step):
            results.append({"step": i, "command": step, "status": "blocked"})
            all_success = False
            continue

        print(f"    [Remediation {i}] {step}", flush=True)
        success, output = _run_remediation_step(step, namespace)
        results.append(
            {
                "step": i,
                "command": step,
                "success": success,
                "output": output[:2000],
            }
        )
        if not success:
            all_success = False

    return {"steps": results, "all_success": all_success}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="OperAID Diagnosis Engine")
    parser.add_argument(
        "--api-key", required=True, help="API key (or set OPENROUTER_API_KEY env)"
    )
    parser.add_argument(
        "--model", required=True, help="Model identifier (e.g. z-ai/glm-5)"
    )
    parser.add_argument(
        "--namespace", default=None, help="Kubernetes namespace (overrides profile)"
    )
    parser.add_argument(
        "--profile", default=None, help="Deployment profile (name or path)"
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--no-tools", action="store_true", help="Disable tool access")
    parser.add_argument(
        "--scenario-description", required=True, help="Fault scenario description"
    )
    parser.add_argument(
        "--output", "-o", default="diagnosis.json", help="Output JSON file"
    )
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--request-timeout", type=int, default=240)

    args = parser.parse_args()

    profile = None
    if args.profile:
        profile = load_profile(args.profile)

    namespace = args.namespace
    if namespace is None:
        namespace = profile.get("namespace", "default") if profile else "default"

    engine = DiagnosisEngine(
        api_key=args.api_key,
        model=args.model,
        namespace=namespace,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        use_tools=not args.no_tools,
        max_retries=args.max_retries,
        request_timeout=args.request_timeout,
        profile=profile,
    )

    print(
        f"[OperAID] Diagnosing with {args.model} ({'tools' if not args.no_tools else 'no-tools'})",
        flush=True,
    )
    result = engine.diagnose(args.scenario_description)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[OperAID] Result saved to {output_path}", flush=True)
    print(
        f"  Root cause: {result['diagnosis'].get('root_cause', 'unknown')}", flush=True
    )
    print(
        f"  Turns: {result['session_meta']['turns_used']}, Duration: {result['session_meta']['duration_s']}s",
        flush=True,
    )
    print(
        f"  Actionable remediation: {result['has_actionable_remediation']}", flush=True
    )


if __name__ == "__main__":
    main()
