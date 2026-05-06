#!/usr/bin/env python3
"""
OperAID Suite Runner — run multiple experiments with YAML configuration.

Suite YAML files can override config.env defaults. The runner executes
run_experiment.sh for each (model, scenario, run) combination and aggregates
results into summary.csv and suite_statistics.json.

Usage:
    python3 run_suite.py --suite suites/tool_impact.yaml --api-key <KEY>
    python3 run_suite.py --model z-ai/glm-5 --scenarios 1,2,3 --runs 30 --api-key <KEY>
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))
from engine.profile import (
    load_profile,
    get_scenario_description as profile_scenario_desc,
)

# Scenario descriptions — fallback when no profile is provided
SCENARIO_DESCRIPTIONS = {
    1: "NetworkPolicy blocks inter-service communication (services register, sessions fail)",
    2: "Deployment references non-existent ConfigMap (CrashLoopBackOff)",
    3: "Deployment scaled to 0 replicas (service down)",
}


def load_suite_config(suite_path: str) -> Dict[str, Any]:
    """Load a YAML suite configuration file."""
    with open(suite_path) as f:
        return yaml.safe_load(f)


def build_experiment_list(
    config: Dict[str, Any], cli_args: argparse.Namespace
) -> List[Dict]:
    """Build a flat list of experiments from suite config or CLI args."""
    experiments = []

    if "experiments" in config:
        # YAML suite with explicit experiment list
        common = config.get("common", {})
        for exp in config["experiments"]:
            merged = {**common, **exp}
            model = merged.get("model", cli_args.model)
            scenarios = merged.get("scenarios", [1, 2, 3])
            if isinstance(scenarios, str):
                scenarios = [int(s) for s in scenarios.split(",")]
            runs = merged.get("runs_per_scenario", cli_args.runs)
            use_tools = merged.get("use_tools", True)
            name = merged.get(
                "name", f"{model}-{'all-tools' if use_tools else 'no-tools'}"
            )

            for scenario in scenarios:
                for run in range(1, runs + 1):
                    experiments.append(
                        {
                            "name": name,
                            "model": model,
                            "scenario": scenario,
                            "run": run,
                            "runs_per_scenario": runs,
                            "use_tools": use_tools,
                            "max_turns": merged.get("max_turns", cli_args.max_turns),
                            "temperature": merged.get(
                                "temperature", cli_args.temperature
                            ),
                            "max_tokens": merged.get("max_tokens", 4096),
                            "base_url": merged.get("base_url", cli_args.base_url),
                            "profile": merged.get("profile", cli_args.profile),
                            "custom_tools": merged.get("custom_tools", []),
                        }
                    )
    else:
        # CLI-driven: single model, all scenarios
        models = (
            [cli_args.model] if cli_args.model else config.get("models", ["z-ai/glm-5"])
        )
        if isinstance(models, str):
            models = [models]
        scenarios = [int(s) for s in cli_args.scenarios.split(",")]
        runs = cli_args.runs

        for model in models:
            for use_tools in (
                [True, False] if cli_args.both_conditions else [not cli_args.no_tools]
            ):
                tool_label = "all-tools" if use_tools else "no-tools"
                model_short = model.split("/")[-1] if "/" in model else model
                name = f"{model_short}-{tool_label}"
                for scenario in scenarios:
                    for run in range(1, runs + 1):
                        experiments.append(
                            {
                                "name": name,
                                "model": model,
                                "scenario": scenario,
                                "run": run,
                                "runs_per_scenario": runs,
                                "use_tools": use_tools,
                                "max_turns": cli_args.max_turns,
                                "temperature": cli_args.temperature,
                                "max_tokens": 4096,
                                "base_url": cli_args.base_url,
                                "profile": cli_args.profile,
                                "custom_tools": [],
                            }
                        )

    return experiments


def run_single_experiment(
    exp: Dict,
    api_key: str,
    output_base: str,
    namespace: str,
    verbose: bool = False,
    resume: bool = False,
) -> Optional[Dict]:
    """Run a single experiment via run_experiment.sh and return the result.

    If resume=True and result.json exists, load and return existing result.
    """
    output_dir = os.path.join(
        output_base, exp["name"], f"scenario_{exp['scenario']}", f"run_{exp['run']}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Check for existing result if resuming
    result_file = os.path.join(output_dir, "result.json")
    if resume and os.path.exists(result_file):
        try:
            with open(result_file) as f:
                data = json.load(f)
            # Validate required fields to detect partial writes
            required = ["scenario", "model", "llm_success", "source"]
            if all(k in data for k in required):
                if verbose:
                    success = data.get("success", False)
                    status = "SUCCESS (cached)" if success else "FAILED (cached)"
                    print(f"  Result: {status} (loaded from existing result)")
                return data
            else:
                print(f"  WARNING: result.json is incomplete, re-running...")
        except Exception as e:
            print(f"  WARNING: Failed to load existing result: {e}, re-running...")

    cmd = [
        str(SCRIPT_DIR / "run_experiment.sh"),
        "--api-key",
        api_key,
        "--model",
        exp["model"],
        "--scenario",
        str(exp["scenario"]),
        "--namespace",
        namespace,
        "--max-turns",
        str(exp["max_turns"]),
        "--temperature",
        str(exp["temperature"]),
        "--output-dir",
        output_dir,
        "--base-url",
        exp["base_url"],
    ]
    if not exp["use_tools"]:
        cmd.append("--no-tools")
    if exp.get("profile"):
        cmd.extend(["--profile", exp["profile"]])
    if exp.get("first_run"):
        cmd.append("--first-run")

    try:
        if verbose:
            # Stream output in real-time
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                # Truncate long lines (e.g., LLM responses)
                line_stripped = line.rstrip()
                if len(line_stripped) > 200:
                    print(f"  {line_stripped[:200]}...")
                else:
                    print(f"  {line_stripped}")
            process.wait()
            result_code = process.returncode
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            result_code = result.returncode

        # Read the result.json produced by run_experiment.sh
        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
                # Show result summary in verbose mode
                if verbose:
                    success = data.get("success", False)
                    root_cause = data.get("diagnosis", {}).get("root_cause", "unknown")
                    status = "✓ SUCCESS" if success else "✗ FAILED"
                    print(f"  Result: {status}")
                    print(
                        f"  Root cause: {root_cause[:80]}{'...' if len(root_cause) > 80 else ''}"
                    )
                return data
        else:
            print(
                f"  WARNING: No result.json for {exp['name']} S{exp['scenario']} R{exp['run']}"
            )
            if result_code != 0:
                print(f"  Exit code: {result_code}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {exp['name']} S{exp['scenario']} R{exp['run']}")
        return None
    except Exception as e:
        print(f"  ERROR: {exp['name']} S{exp['scenario']} R{exp['run']}: {e}")
        return None


def write_summary_csv(results: List[Dict], output_path: str):
    """Write results to summary.csv."""
    if not results:
        return
    fieldnames = [
        "scenario",
        "run",
        "model",
        "source",
        "success",
        "root_cause_verified",
        "turns_used",
        "diagnosis_duration_s",
        "timestamp",
        "failure_reason",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "scenario": r.get("scenario", ""),
                    "run": r.get("run", ""),
                    "model": r.get("model", ""),
                    "source": r.get("source", ""),
                    "success": str(r.get("success", False)).lower(),
                    "root_cause_verified": str(
                        r.get("root_cause_verified", False)
                    ).lower(),
                    "turns_used": r.get("turns_used", 0),
                    "diagnosis_duration_s": r.get("duration_s", 0),
                    "timestamp": r.get("timestamp", ""),
                    "failure_reason": r.get("failure_reason", "unknown"),
                }
            )


def write_api_error_csv(stats: Dict, output_path: str):
    """Write API error summary CSV."""
    fieldnames = [
        "model",
        "total_runs",
        "runs_with_errors",
        "total_api_errors",
        "fatal_api_errors",
        "retried_successfully",
        "error_rate",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, data in sorted(stats["models"].items()):
            # Only include tools runs for consistency
            if "no-tools" in name:
                continue
            short_name = name.replace("-all-tools", "").replace("-no-tools", "")
            total_runs = data["total_runs"]
            runs_with_errors = data.get("runs_with_api_errors", 0)
            total_errors = data.get("api_errors", 0)
            writer.writerow(
                {
                    "model": short_name,
                    "total_runs": total_runs,
                    "runs_with_errors": runs_with_errors,
                    "total_api_errors": total_errors,
                    "fatal_api_errors": 0,  # Not tracked separately yet
                    "retried_successfully": runs_with_errors,  # Assume all retried
                    "error_rate": runs_with_errors / total_runs * 100
                    if total_runs > 0
                    else 0,
                }
            )


def compute_suite_statistics(results: List[Dict], experiments: List[Dict]) -> Dict:
    """Compute suite-level statistics from results."""
    stats: Dict[str, Any] = {
        "suite_overview": {},
        "models": {},
        "scenarios": {},
        "cross_analysis": {},
    }

    # Group by model name (experiment name includes tool condition)
    model_results: Dict[str, List[Dict]] = {}
    for exp, res in zip(experiments, results):
        if res is None:
            continue
        name = exp["name"]
        if name not in model_results:
            model_results[name] = []
        model_results[name].append(res)

    total_runs = sum(len(v) for v in model_results.values())
    total_llm_success = sum(
        1 for v in model_results.values() for r in v if r.get("llm_success")
    )
    total_cluster_success = sum(
        1 for v in model_results.values() for r in v if r.get("success")
    )

    stats["suite_overview"] = {
        "total_models": len(model_results),
        "total_runs": total_runs,
        "total_scenarios": 3,
        "runs_per_scenario_per_model": 30,
        "total_llm_successes": total_llm_success,
        "total_cluster_successes": total_cluster_success,
        "overall_success_rate_pct": (total_llm_success / total_runs * 100)
        if total_runs > 0
        else 0,
    }

    import numpy as np

    # Per-model stats
    model_ranking = []
    for name, res_list in sorted(model_results.items()):
        n = len(res_list)
        llm_successes = [r for r in res_list if r.get("llm_success")]
        llm_failures = [r for r in res_list if not r.get("llm_success")]
        llm_success_rate = len(llm_successes) / n * 100 if n > 0 else 0

        # Per-scenario breakdown
        scenario_stats = {}
        for s in [1, 2, 3]:
            s_runs = [r for r in res_list if r.get("scenario") == s]
            s_llm_ok = [r for r in s_runs if r.get("llm_success")]
            s_rate = len(s_llm_ok) / len(s_runs) * 100 if s_runs else 0
            avg_turns = (
                (sum(r.get("turns_used", 0) for r in s_llm_ok) / len(s_llm_ok))
                if s_llm_ok
                else None
            )
            avg_dur = (
                (sum(r.get("duration_s", 0) for r in s_llm_ok) / len(s_llm_ok))
                if s_llm_ok
                else None
            )
            scenario_stats[f"scenario_{s}"] = {
                "runs": len(s_runs),
                "llm_successes": len(s_llm_ok),
                "success_rate": s_rate,
                "avg_turns": avg_turns,
                "avg_duration_s": avg_dur,
            }

        # Duration/turns stats over LLM-successful runs
        dur_vals = [
            r.get("duration_s", 0) for r in llm_successes if r.get("duration_s")
        ]
        turn_vals = [
            r.get("turns_used", 0) for r in llm_successes if r.get("turns_used")
        ]
        # Token stats from diagnosis files
        token_vals = [
            r.get("total_tokens", 0) for r in res_list if r.get("total_tokens")
        ]
        input_token_vals = [
            r.get("total_input_tokens", 0)
            for r in res_list
            if r.get("total_input_tokens")
        ]
        output_token_vals = [
            r.get("total_output_tokens", 0)
            for r in res_list
            if r.get("total_output_tokens")
        ]
        api_error_vals = [r.get("api_errors", 0) for r in res_list]
        total_api_errors = sum(api_error_vals)
        runs_with_errors = sum(1 for e in api_error_vals if e > 0)

        # Failure mode breakdown (mutually exclusive)
        health_check_failed = sum(
            1
            for r in res_list
            if r.get("failure_reason", "") == "pre_flight_health_check"
        )
        api_failure = sum(
            1
            for r in res_list
            if not r.get("llm_success")
            and r.get("api_errors", 0) > 0
            and r.get("failure_reason", "") != "pre_flight_health_check"
        )
        no_remediation = sum(
            1
            for r in res_list
            if not r.get("llm_success")
            and not r.get("has_actionable_remediation", False)
            and r.get("failure_reason", "") != "pre_flight_health_check"
            and r.get("api_errors", 0) == 0
        )
        wrong_diagnosis = sum(
            1
            for r in res_list
            if not r.get("llm_success")
            and r.get("has_actionable_remediation", False)
            and r.get("failure_reason", "") != "pre_flight_health_check"
            and r.get("api_errors", 0) == 0
        )

        dur_stats = (
            {
                "mean": float(np.mean(dur_vals)) if dur_vals else None,
                "std": float(np.std(dur_vals, ddof=1)) if len(dur_vals) > 1 else None,
                "min": float(min(dur_vals)) if dur_vals else None,
                "max": float(max(dur_vals)) if dur_vals else None,
                "median": float(np.median(dur_vals)) if dur_vals else None,
            }
            if dur_vals
            else {}
        )
        turn_stats = (
            {
                "mean": float(np.mean(turn_vals)) if turn_vals else None,
                "std": float(np.std(turn_vals, ddof=1)) if len(turn_vals) > 1 else None,
                "min": int(min(turn_vals)) if turn_vals else None,
                "max": int(max(turn_vals)) if turn_vals else None,
                "median": float(np.median(turn_vals)) if turn_vals else None,
            }
            if turn_vals
            else {}
        )

        # Token stats
        token_stats = {
            "total_tokens": sum(token_vals) if token_vals else 0,
            "avg_tokens_per_run": float(np.mean(token_vals)) if token_vals else 0,
            "total_input_tokens": sum(input_token_vals) if input_token_vals else 0,
            "total_output_tokens": sum(output_token_vals) if output_token_vals else 0,
            "avg_input_tokens": float(np.mean(input_token_vals))
            if input_token_vals
            else 0,
            "avg_output_tokens": float(np.mean(output_token_vals))
            if output_token_vals
            else 0,
        }

        stats["models"][name] = {
            "total_runs": n,
            "llm_successful_runs": len(llm_successes),
            "groundtruth_fallback_runs": len(llm_failures),
            "failed_runs": len(llm_failures),
            "success_rate_pct": llm_success_rate,
            "scenarios": scenario_stats,
            "duration": dur_stats,
            "turns": turn_stats,
            "tokens": token_stats,
            "api_errors": total_api_errors,
            "runs_with_api_errors": runs_with_errors,
            "failure_modes": {
                "no_remediation": no_remediation,
                "wrong_diagnosis": wrong_diagnosis,
                "health_check_failed": health_check_failed,
                "api_failure": api_failure,
            },
        }
        model_ranking.append({"model": name, "success_rate_pct": llm_success_rate})

    # Scenario overview
    for s in [1, 2, 3]:
        s_all = [
            r
            for res_list in model_results.values()
            for r in res_list
            if r.get("scenario") == s
        ]
        s_llm_ok = [r for r in s_all if r.get("llm_success")]
        best_model = ""
        best_rate = -1
        for name, res_list in model_results.items():
            s_runs = [r for r in res_list if r.get("scenario") == s]
            s_succ = [r for r in s_runs if r.get("llm_success")]
            rate = len(s_succ) / len(s_runs) * 100 if s_runs else 0
            if rate > best_rate:
                best_rate = rate
                best_model = name
        stats["scenarios"][f"scenario_{s}"] = {
            "total_runs": len(s_all),
            "total_llm_successes": len(s_llm_ok),
            "success_rate_pct": len(s_llm_ok) / len(s_all) * 100 if s_all else 0,
            "best_model": best_model,
        }

    model_ranking.sort(key=lambda x: x["success_rate_pct"], reverse=True)
    stats["cross_analysis"] = {
        "best_model": model_ranking[0]["model"] if model_ranking else "",
        "best_model_success_rate": model_ranking[0]["success_rate_pct"]
        if model_ranking
        else 0,
        "worst_model": model_ranking[-1]["model"] if model_ranking else "",
        "worst_model_success_rate": model_ranking[-1]["success_rate_pct"]
        if model_ranking
        else 0,
        "model_ranking": model_ranking,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="OperAID Suite Runner")
    parser.add_argument("--suite", help="YAML suite configuration file")
    parser.add_argument("--api-key", help="API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--model", help="Model identifier")
    parser.add_argument(
        "--scenarios", default="1,2,3", help="Comma-separated scenario IDs"
    )
    parser.add_argument("--runs", type=int, default=30, help="Runs per scenario")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-tools", action="store_true")
    parser.add_argument(
        "--both-conditions",
        action="store_true",
        help="Run both all-tools and no-tools conditions",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Kubernetes namespace (from profile if not set)",
    )
    parser.add_argument(
        "--profile", default="open5gs", help="Deployment profile name or path"
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument(
        "--output-dir", help="Output directory (auto-generated if not set)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results (skip runs with result.json)",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: --api-key required or set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Load deployment profile
    try:
        profile = load_profile(args.profile)
        if args.namespace is None:
            args.namespace = profile.get("namespace", "default")
    except FileNotFoundError:
        print(f"WARNING: Profile not found: {args.profile}, using defaults")
        profile = None
        if args.namespace is None:
            args.namespace = "default"

    # Load suite config or use defaults
    config: Dict[str, Any] = {}
    if args.suite:
        config = load_suite_config(args.suite)
        print(f"Loaded suite config: {args.suite}")

    experiments = build_experiment_list(config, args)
    # Inject profile into each experiment
    if profile:
        for exp in experiments:
            exp["profile"] = args.profile
            # Merge suite-level custom_tools into profile dict
            suite_custom_tools = exp.get("custom_tools")
            if suite_custom_tools is not None:
                profile["custom_tools"] = suite_custom_tools
    total = len(experiments)
    print(f"Total experiments: {total}")

    # Output dir: suite_results/<suite_name>/<timestamp>/
    suite_name = config.get("name", Path(args.suite).stem) if args.suite else "manual"
    suite_dir = SCRIPT_DIR / "suite_results" / suite_name

    if args.output_dir:
        output_base = args.output_dir
    elif args.resume:
        candidates = sorted(suite_dir.iterdir()) if suite_dir.exists() else []
        if not candidates:
            print(
                f"ERROR: --resume but no existing runs found in {suite_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        output_base = str(candidates[-1])
        print(f"  --resume: auto-detected latest run: {output_base}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_base = str(suite_dir / timestamp)

    # If resuming into an existing output dir, validate suite match
    if args.resume and os.path.exists(output_base):
        saved_suite = os.path.join(output_base, "suite_config.yaml")
        if os.path.exists(saved_suite) and args.suite:
            import filecmp

            if not filecmp.cmp(args.suite, saved_suite, shallow=False):
                print(
                    f"ERROR: Output directory {output_base} belongs to a different suite.",
                    file=sys.stderr,
                )
                print(f"  Current suite: {args.suite}", file=sys.stderr)
                print(f"  Saved suite:   {saved_suite}", file=sys.stderr)
                print(
                    f"  Use a different --output-dir or remove --resume to start fresh.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  Resuming existing suite run from: {output_base}")

    os.makedirs(output_base, exist_ok=True)

    # Set up logging to file
    log_file = os.path.join(output_base, "runner.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Also redirect stdout to both console and file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files

        def write(self, text):
            for f in self.files:
                f.write(text)
                f.flush()

        def flush(self):
            for f in self.files:
                f.flush()

    sys.stdout = TeeOutput(sys.stdout, open(log_file, "a"))
    sys.stderr = TeeOutput(sys.stderr, open(log_file, "a"))

    start_time = datetime.now(timezone.utc)
    print(f"Logging to: {log_file}")
    print(f"Suite started at: {start_time.isoformat()}")
    print(f"{'=' * 60}")

    # Copy suite config into output dir for reproducibility
    if args.suite:
        import shutil

        shutil.copy2(args.suite, os.path.join(output_base, "suite_config.yaml"))

    # Run experiments
    results = []
    skipped = 0
    for i, exp in enumerate(experiments, 1):
        exp["first_run"] = i == 1
        model = exp.get("model", "unknown")
        scenario = exp.get("scenario", "?")
        run = exp.get("run", "?")
        use_tools = "tools" if exp.get("use_tools", True) else "no-tools"

        # Calculate total runs for this model+scenario combination
        runs_per_scenario = (
            exp.get("runs_per_scenario", 1) if "runs_per_scenario" in exp else 1
        )

        # Get scenario description from profile or fallback
        if profile:
            scenario_desc = profile_scenario_desc(
                profile, str(scenario), args.namespace
            )
            if not scenario_desc:
                scenario_desc = SCENARIO_DESCRIPTIONS.get(scenario, "Unknown scenario")
        else:
            scenario_desc = SCENARIO_DESCRIPTIONS.get(scenario, "Unknown scenario")

        # Check if already completed (for display purposes)
        output_dir = os.path.join(
            output_base, exp["name"], f"scenario_{exp['scenario']}", f"run_{exp['run']}"
        )
        result_file = os.path.join(output_dir, "result.json")
        already_done = args.resume and os.path.exists(result_file)

        status_prefix = "[CACHED]" if already_done else f"[{i}/{total}]"
        print(f"\n{'=' * 60}")
        print(f"{status_prefix} {exp['name']}")
        print(f"  Model: {model}")
        print(
            f"  Scenario: {scenario} | Run: {run}/{runs_per_scenario} | Mode: {use_tools}"
        )
        print(f"  Fault: {scenario_desc}")
        if already_done:
            print(f"  Status: Using cached result")
        print(f"{'=' * 60}")

        result = run_single_experiment(
            exp, api_key, output_base, args.namespace, verbose=True, resume=args.resume
        )
        if result:
            result["run"] = exp["run"]
        elif already_done:
            skipped += 1
        results.append(result)

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    # Report summary
    print(f"\n{'=' * 60}")
    print(f"Run Summary: {len(valid_results)}/{len(results)} runs completed")
    if skipped > 0:
        print(f"  ({skipped} runs loaded from cache)")
    print(f"{'=' * 60}")

    # Write summary CSV
    csv_path = os.path.join(output_base, "summary.csv")
    write_summary_csv(valid_results, csv_path)
    print(f"\nSummary CSV: {csv_path}")

    # Compute and write statistics
    stats = compute_suite_statistics(valid_results, experiments)
    stats_path = os.path.join(output_base, "suite_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Suite statistics: {stats_path}")

    # Write summary CSV tables
    write_api_error_csv(stats, os.path.join(output_base, "api_error_summary.csv"))
    print(f"API error summary: {output_base}/api_error_summary.csv")

    # Generate plots
    plots_dir = os.path.join(output_base, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nGenerating plots in {plots_dir}...")
    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_suite.py"),
                "--stats",
                stats_path,
                "--suite-dir",
                output_base,
                "-o",
                plots_dir,
            ],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARNING: Plot generation failed: {e}")

    # Print overview
    overview = stats["suite_overview"]
    print(f"\n{'=' * 60}")
    print(
        f"Suite Complete: {overview['total_llm_successes']}/{overview['total_runs']} "
        f"({overview['overall_success_rate_pct']:.1f}%)"
    )
    print(f"Output: {output_base}")

    # Print per-experiment results summary
    print(f"\n{'=' * 60}")
    print("Results Summary:")
    print(f"{'=' * 60}")
    for exp, res in zip(experiments, results):
        if res is None:
            status = "✗ ERROR (no result)"
            source = "N/A"
            llm_status = "N/A"
        else:
            success = res.get("success", False)
            llm_success = res.get("llm_success", False)
            source = res.get("source", "unknown")
            status = "✓ HEALTHY" if success else "✗ UNHEALTHY"
            llm_status = "LLM ✓" if llm_success else "LLM ✗"
        scenario = exp.get("scenario", "?")
        run = exp.get("run", "?")
        model_short = exp.get("model", "?").split("/")[-1]
        print(
            f"  S{scenario} R{run} [{model_short[:15]:<15}]: {status} | {llm_status} | source: {source}"
        )

    print(f"{'=' * 60}")

    end_time = datetime.now(timezone.utc)
    duration = end_time - start_time
    print(f"Suite ended at: {end_time.isoformat()}")
    print(f"Total duration: {duration}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
