"""
Deployment profile loader for OperAID.

Provides a unified way to load deployment profiles and extract
deployment-specific configuration for all OperAID components.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _find_profile(name_or_path: str, profiles_dir: Optional[str] = None) -> Path:
    """Resolve a profile name or path to a Path object.

    Accepts:
      - A full path: "/home/ariel/git/operaid/deployments/open5gs.json"
      - A relative path: "deployments/open5gs.json"
      - A bare name: "open5gs" (resolved to deployments/open5gs.json)
    """
    p = Path(name_or_path)

    # If it's an existing file, use it directly
    if p.exists():
        return p

    # If profiles_dir is given, try there
    if profiles_dir:
        candidate = Path(profiles_dir) / p
        if not candidate.suffix:
            candidate = candidate.with_suffix(".json")
        if candidate.exists():
            return candidate

    # Default: look relative to project root (engine/profile.py -> project root is parent)
    project_root = Path(__file__).resolve().parent.parent
    candidate = project_root / "deployments" / p
    if not candidate.suffix:
        candidate = candidate.with_suffix(".json")
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Deployment profile not found: {name_or_path}")


def load_profile(
    name_or_path: str, profiles_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Load a deployment profile and return its contents as a dict."""
    path = _find_profile(name_or_path, profiles_dir)
    with open(path) as f:
        return json.load(f)


def get_namespace(profile: Dict[str, Any]) -> str:
    """Get the Kubernetes namespace from a profile."""
    return profile.get("namespace", "default")


def get_context_prompt(profile: Dict[str, Any]) -> str:
    """Get the deployment context string for LLM prompts."""
    return profile.get("context_prompt", "")


def get_expected_deployments(profile: Dict[str, Any]) -> list:
    """Get the list of expected deployment names from a profile."""
    hc = profile.get("health_check", {})
    if hc.get("check_type") == "deployments":
        return hc.get("expected_deployments", [])
    components = profile.get("components", {})
    return [comp["deployment"] for comp in components.values() if "deployment" in comp]


def get_name_prefix(profile: Dict[str, Any], component: str) -> str:
    """Get the full name prefix for a component (e.g. 'open5gs-' for 'smf')."""
    prefixes = profile.get("name_prefixes", {})
    prefix = prefixes.get(component, prefixes.get("default", ""))
    return prefix


def get_scenario_description(
    profile: Dict[str, Any], scenario_id: str, namespace: Optional[str] = None
) -> str:
    """Get the scenario description for a given scenario ID."""
    ns = namespace or get_namespace(profile)
    scenarios = profile.get("scenario_targets", {})
    if scenario_id not in scenarios:
        return ""
    desc = scenarios[scenario_id].get("description", "")
    return desc.format(namespace=ns)


def get_fault_injection(profile: Dict[str, Any], scenario_id: str) -> Dict[str, Any]:
    """Get fault injection configuration for a scenario."""
    fi = profile.get("fault_injection", {})
    return fi.get(scenario_id, {})


def resolve_name(
    short_name: str,
    profile: Dict[str, Any],
    resource: str = "deployments",
    namespace: str = "",
) -> str:
    """Resolve a short name to a full deployment name using profile prefix rules.

    Falls back to subprocess-based resolution if the short name doesn't match directly.
    """
    import subprocess

    # Try exact match first
    expected = get_expected_deployments(profile)
    if short_name in expected:
        return short_name

    # Try prefix matching
    prefixes = profile.get("name_prefixes", {})
    prefix = prefixes.get(short_name, prefixes.get("default", ""))
    candidate = f"{prefix}{short_name}"
    if candidate in expected:
        return candidate

    # Fallback: query the cluster
    ns = namespace or get_namespace(profile)
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                resource,
                "-n",
                ns,
                "--no-headers",
                "-o",
                "custom-columns=NAME:.metadata.name",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        names = [n.strip() for n in result.stdout.splitlines() if n.strip()]
        if short_name in names:
            return short_name
        for name in names:
            if name.endswith(f"-{short_name}"):
                return name
    except Exception:
        pass

    return short_name
