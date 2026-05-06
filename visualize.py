#!/usr/bin/env python3
"""
OperAID — Single experiment run visualizer.

Reads a diagnosis.json + result.json from an experiment run directory
and generates summary plots.

Usage:
    python3 visualize.py --run-dir suite_results/<model>/scenario_1/run_1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# OperAID visual identity: seaborn rocket palette
sns.set_theme(style="whitegrid", palette="rocket")


def load_run_data(run_dir: Path) -> dict:
    """Load diagnosis.json and result.json from a run directory."""
    data = {}
    diag_path = run_dir / "diagnosis.json"
    result_path = run_dir / "result.json"

    if diag_path.exists():
        with open(diag_path) as f:
            data["diagnosis"] = json.load(f)
    if result_path.exists():
        with open(result_path) as f:
            data["result"] = json.load(f)
    return data


def plot_turn_timings(data: dict, output_dir: Path):
    """Plot per-turn timing breakdown."""
    diag = data.get("diagnosis", {})
    timings = diag.get("turn_timings", [])
    if not timings:
        return

    turns = [t["turn"] for t in timings]
    durations = [t["duration_s"] for t in timings]
    actions = [t.get("action", "unknown") for t in timings]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("rocket", n_colors=len(turns))
    bars = ax.bar(turns, durations, color=colors)

    for bar, action in zip(bars, actions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                action, ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Per-Turn Timing")
    ax.set_xticks(turns)
    fig.tight_layout()
    fig.savefig(output_dir / "turn_timings.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tool_usage(data: dict, output_dir: Path):
    """Plot which tools were used in each turn."""
    diag = data.get("diagnosis", {})
    timings = diag.get("turn_timings", [])
    tool_turns = [t for t in timings if t.get("tools_called")]
    if not tool_turns:
        return

    # Count tool usage
    tool_counts = {}
    for t in tool_turns:
        for tool in t.get("tools_called", []):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    tools = sorted(tool_counts.keys(), key=lambda x: tool_counts[x], reverse=True)
    counts = [tool_counts[t] for t in tools]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("rocket", n_colors=len(tools))
    ax.barh(tools, counts, color=colors)
    ax.set_xlabel("Times Called")
    ax.set_title("Tool Usage")
    fig.tight_layout()
    fig.savefig(output_dir / "tool_usage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_summary(data: dict, output_dir: Path):
    """Generate a text summary of the run."""
    result = data.get("result", {})
    diag = data.get("diagnosis", {})
    meta = diag.get("session_meta", {})

    lines = [
        "=" * 60,
        "OperAID Run Summary",
        "=" * 60,
        f"Model:     {meta.get('model', 'unknown')}",
        f"Scenario:  {result.get('scenario', '?')}",
        f"Tools:     {'enabled' if meta.get('use_tools') else 'disabled'}",
        f"Success:   {result.get('success', False)}",
        f"Source:    {result.get('source', '?')}",
        f"Turns:     {meta.get('turns_used', '?')}",
        f"Duration:  {meta.get('duration_s', '?')}s",
        f"Root cause: {diag.get('diagnosis', {}).get('root_cause', 'unknown')}",
        "",
        "Remediation steps:",
    ]
    for step in diag.get("diagnosis", {}).get("remediation_steps", []):
        lines.append(f"  - {step}")

    if diag.get("api_errors"):
        lines.append(f"\nAPI Errors: {len(diag['api_errors'])}")

    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="OperAID Single Run Visualizer")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    data = load_run_data(run_dir)
    if not data:
        print(f"ERROR: No data found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generate_summary(data, run_dir)
    plot_turn_timings(data, plots_dir)
    plot_tool_usage(data, plots_dir)

    print(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    main()
