"""
Generic health check for Kubernetes deployments.

Reads expected deployments from a deployment profile JSON file.
Returns exit code 0 if healthy, 1 otherwise.

Usage:
    python3 health_check.py --profile deployments/open5gs.json -n open5gs
    python3 health_check.py --profile deployments/free5gc.json -n free5gc
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

KUBECTL_TIMEOUT = 15


def load_profile(profile_path: str) -> dict:
    """Load a deployment profile JSON file."""
    path = Path(profile_path)
    if not path.exists():
        # Try relative to script parent directory
        alt = Path(__file__).resolve().parent.parent / profile_path
        if alt.exists():
            path = alt
        else:
            print(f"ERROR: Profile not found: {profile_path}", file=sys.stderr)
            sys.exit(1)
    with open(path) as f:
        return json.load(f)


def get_expected_deployments(profile: dict) -> list[str]:
    """Extract the list of expected deployment names from a profile."""
    hc = profile.get("health_check", {})
    if hc.get("check_type") == "deployments":
        return hc.get("expected_deployments", [])
    # Fallback: extract from components
    components = profile.get("components", {})
    deployments = []
    for key, comp in components.items():
        if "deployment" in comp:
            deployments.append(comp["deployment"])
    return deployments


def check_deployments_ready(
    namespace: str, expected: list[str]
) -> tuple[bool, list[str]]:
    """Check if all expected deployments are available and ready.

    Returns (all_ready, list_of_issues).
    """
    issues = []
    try:
        result = subprocess.run(
            [
                "kubectl",
                "get",
                "deployments",
                "-n",
                namespace,
                "-o",
                "jsonpath={range .items[*]}{.metadata.name} {.status.readyReplicas}/{.status.replicas}\n{end}",
            ],
            capture_output=True,
            text=True,
            timeout=KUBECTL_TIMEOUT,
        )
        if result.returncode != 0:
            return False, [f"kubectl failed: {result.stderr.strip()}"]

        deploy_status = {}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                status = parts[1]
                deploy_status[name] = status

        for dep in expected:
            if dep not in deploy_status:
                issues.append(f"{dep}: NOT FOUND")
            else:
                status = deploy_status[dep]
                ready, total = status.split("/")
                if ready == "<none>" or ready == "" or ready == "0":
                    issues.append(f"{dep}: NOT READY ({status})")
                elif ready != total:
                    issues.append(f"{dep}: PARTIAL ({status})")

    except subprocess.TimeoutExpired:
        return False, ["kubectl timed out"]
    except Exception as e:
        return False, [f"Error: {e}"]

    return len(issues) == 0, issues


def wait_for_healthy(
    namespace: str, expected: list[str], timeout: int = 120, interval: int = 5
) -> bool:
    """Wait up to `timeout` seconds for all deployments to become ready."""
    start = time.time()
    while time.time() - start < timeout:
        healthy, issues = check_deployments_ready(namespace, expected)
        if healthy:
            elapsed = time.time() - start
            print(f"✓ All {len(expected)} deployments healthy after {elapsed:.0f}s")
            return True
        remaining = timeout - (time.time() - start)
        if remaining > interval:
            time.sleep(interval)
    _, issues = check_deployments_ready(namespace, expected)
    print(f"✗ Health check failed after {timeout}s:")
    for issue in issues:
        print(f"  - {issue}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Kubernetes Deployment Health Check")
    parser.add_argument(
        "--profile", "-p", required=True, help="Path to deployment profile JSON file"
    )
    parser.add_argument(
        "--namespace",
        "-n",
        default=None,
        help="Kubernetes namespace (overrides profile)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Max seconds to wait for healthy state",
    )
    parser.add_argument(
        "--wait",
        "-w",
        action="store_true",
        help="Wait for healthy state (with retries)",
    )
    args = parser.parse_args()

    profile = load_profile(args.profile)
    namespace = args.namespace or profile.get("namespace", "default")
    expected = get_expected_deployments(profile)

    if not expected:
        print("ERROR: No deployments found in profile", file=sys.stderr)
        sys.exit(1)

    if args.wait:
        healthy = wait_for_healthy(namespace, expected, args.timeout)
    else:
        healthy, issues = check_deployments_ready(namespace, expected)
        if healthy:
            print(f"✓ All {len(expected)} deployments healthy")
        else:
            print("✗ Health check failed:")
            for issue in issues:
                print(f"  - {issue}")

    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
