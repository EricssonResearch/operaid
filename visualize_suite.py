#!/usr/bin/env python3
"""
OperAID — Suite Figure Generator.

Generates figures from suite statistics and experiment results.

Usage:
    python3 visualize_suite.py --stats suite_statistics.json --csv summary.csv -o output_dir
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Visual Identity
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    }
)

FIG_DPI = 300

# Model display names
MODEL_DISPLAY = {
    "glm-5": "GLM-5",
    "gpt-oss-120b": "GPT-OSS-120b",
    "kimi-k2.5": "Kimi-K2.5",
    "qwen3.5-35b-a3b": "Qwen3.5-35b-a3b",
    "qwen3.5-397b-a17b": "Qwen3.5-397b-a17b",
}

MODEL_COST_PER_1M: Dict[str, float] = {}


def load_cost_benefit_csv(csv_path: Path) -> List[Dict]:
    """Load precomputed cost-benefit data from cost_benefit_summary.csv."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "name": row["Model"],
                    "rate": float(row["Success%"]),
                    "avg_cost": float(row["Avg Cost ($)"]),
                    "cost_per_success": float(row["Cost/Success ($)"]),
                    "duration": float(row["Avg Duration (s)"]),
                    "tokens": int(row["Avg Tokens"]),
                }
            )
    print(f"  Loaded cost data for {len(rows)} models from {csv_path}")
    return rows


# Label positioning configuration for duration_vs_success plot
# Format: "model_name (condition)": (x_offset, y_offset, zorder)
LABEL_POSITIONS: Dict[str, Tuple[int, int, int]] = {
    # Tools variants
    "GLM-5 (tools)": (10, 15, 10),
    "Kimi-k2.5 (tools)": (10, 15, 10),
    "Qwen3.5-397b-a17b (tools)": (15, 0, 10),
    "GPT-OSS-120b (tools)": (10, -20, 10),
    "Qwen3.5-35b-a3b (tools)": (5, -20, 5),
    # No-tools variants
    "GLM-5 (no tools)": (10, -15, 5),
    "Kimi-k2.5 (no tools)": (10, 5, 5),
    "Qwen3.5-397b-a17b (no tools)": (10, 12, 5),
    "GPT-OSS-120b (no tools)": (10, -20, 5),
    "Qwen3.5-35b-a3b (no tools)": (10, -24, 5),
}

ALL_TOOLS = [
    "describe_pod",
    "get_deployment",
    "get_events",
    "get_networkpolicies",
    "get_pod_logs",
    "get_pods",
    "run_kubectl",
]

SCENARIO_LABELS = {
    1: "S1: NetworkPolicy",
    2: "S2: ConfigMap",
    3: "S3: UPF Scale",
}


def _short_name(model_key: str) -> str:
    for suffix in ["-all-tools", "-no-tools"]:
        if model_key.endswith(suffix):
            return model_key[: -len(suffix)]
    return model_key


def _is_tools(model_key: str) -> bool:
    return model_key.endswith("-all-tools")


def _binomial_ci(p: float, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0
    return z * math.sqrt(p * (1 - p) / n)


def load_tool_data(suite_dir: str) -> pd.DataFrame:
    """Load per-run tool usage data from diagnosis.json files."""
    rows = []
    suite_path = Path(suite_dir)
    for rf in sorted(glob.glob(str(suite_path / "*/*/run_*/result.json"))):
        try:
            r = json.load(open(rf))
        except:
            continue
        dn = os.path.dirname(rf)
        parts = dn.replace(str(suite_path) + "/", "").split("/")
        if len(parts) < 3:
            continue
        name = parts[0]
        scenario = int(parts[1].split("_")[1])
        run = int(parts[2].split("_")[1])

        dp = os.path.join(dn, "diagnosis.json")
        tool_counts: Counter = Counter()
        if os.path.exists(dp):
            try:
                d = json.load(open(dp))
                for tt in d.get("turn_timings", []):
                    for t in tt.get("tools_called", []):
                        tool_counts[t] += 1
            except:
                pass

        short = _short_name(name)
        rows.append(
            {
                "model_key": name,
                "model_short": short,
                "scenario": scenario,
                "run": run,
                "tool_condition": "all-tools"
                if name.endswith("-all-tools")
                else "no-tools",
                "llm_success": r.get("llm_success", r.get("success", False)),
                "duration_s": r.get("duration_s", 0),
                "turns_used": r.get("turns_used", 0),
                "tool_counts": dict(tool_counts),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure 1: model_comparison_accuracy_ci.png
# Original: Individual bars per model+condition, sorted, with avg lines
# ---------------------------------------------------------------------------
def plot_model_comparison_accuracy_ci(stats: Dict, output_dir: Path):
    """Generate model comparison accuracy chart with confidence intervals."""
    models_data = stats["models"]

    # Sort models by name for consistent ordering
    model_keys = sorted(models_data.keys())

    fig, ax = plt.subplots(figsize=(12, 5))

    # Color scheme - gradient from dark purple to pink/coral
    # Tools models: darker purples/grays
    # No-tools models: lighter corals/pinks
    model_colors = {
        "glm-5-all-tools": "#4A4458",  # Dark gray-purple
        "glm-5-no-tools": "#4A4458",  # Dark gray-purple
        "gpt-oss-120b-all-tools": "#8B5A8E",  # Medium purple
        "gpt-oss-120b-no-tools": "#8B5A8E",  # Medium purple
        "kimi-k2.5-all-tools": "#D75A7F",  # Pink-red
        "kimi-k2.5-no-tools": "#E8967D",  # Coral
        "qwen3.5-35b-a3b-all-tools": "#F4A460",  # Sandy orange
        "qwen3.5-35b-a3b-no-tools": "#F4C4A0",  # Light peach
        "qwen3.5-397b-a17b-all-tools": "#5D5D7A",  # Dark blue-gray
        "qwen3.5-397b-a17b-no-tools": "#9B7B9E",  # Light purple
    }

    x_pos = 0
    x_positions = []
    x_labels = []
    rates = []
    cis = []
    colors = []

    for key in model_keys:
        data = models_data[key]
        rate = data["success_rate_pct"]
        n = data["total_runs"]
        ci = _binomial_ci(rate / 100, n) * 100

        x_positions.append(x_pos)
        x_labels.append(key)
        rates.append(rate)
        cis.append(ci)
        colors.append(model_colors.get(key, "#5D3A6E"))
        x_pos += 1

    bars = ax.bar(
        x_positions, rates, color=colors, edgecolor="black", linewidth=0.5, zorder=3
    )
    ax.errorbar(
        x_positions,
        rates,
        yerr=cis,
        fmt="none",
        ecolor="black",
        capsize=3,
        capthick=1,
        zorder=4,
    )

    # Add value labels
    for i, (pos, rate) in enumerate(zip(x_positions, rates)):
        ax.text(
            pos,
            rate + cis[i] + 2,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Average lines - teal for tools, coral for no-tools
    tools_rates = [r for k, r in zip(model_keys, rates) if _is_tools(k)]
    no_tools_rates = [r for k, r in zip(model_keys, rates) if not _is_tools(k)]

    avg_tools = np.mean(tools_rates) if tools_rates else 0
    avg_no_tools = np.mean(no_tools_rates) if no_tools_rates else 0

    ax.axhline(
        avg_tools,
        color="#5DADE2",
        linestyle="--",
        linewidth=1.8,
        label=f"Avg Tools: {avg_tools:.1f}%",
    )
    ax.axhline(
        avg_no_tools,
        color="#E07B7B",
        linestyle="--",
        linewidth=1.8,
        label=f"Avg No-Tools: {avg_no_tools:.1f}%",
    )

    ax.set_ylabel("Success Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison: Overall Success Rate")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=20, ha="right", fontsize=13)
    ax.set_ylim(0, 110)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(
        output_dir / "model_comparison_accuracy_ci.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: tool_success_correlation.png
# Original: Sorted by lift, horizontal bars with n= labels
# ---------------------------------------------------------------------------
def plot_tool_success_correlation(tool_df: pd.DataFrame, output_dir: Path):
    """Generate tool success correlation chart."""
    if tool_df.empty:
        print("    (skipped: no tool data)")
        return

    tools_df = tool_df[tool_df["tool_condition"] == "all-tools"]
    tools = ALL_TOOLS
    used_rate = []
    not_used_rate = []
    lift = []
    usage_count = []

    for t in tools:
        used_mask = tools_df["tool_counts"].apply(lambda d: d.get(t, 0) > 0)
        used_runs = tools_df[used_mask]
        not_used_runs = tools_df[~used_mask]

        u_rate = used_runs["llm_success"].mean() * 100 if len(used_runs) > 0 else 0
        n_rate = (
            not_used_runs["llm_success"].mean() * 100 if len(not_used_runs) > 0 else 0
        )

        used_rate.append(u_rate)
        not_used_rate.append(n_rate)
        lift.append(u_rate - n_rate)
        usage_count.append(len(used_runs))

    total_tools_runs = len(tools_df)

    # Sort by lift descending (get_pods with +80.6pp at top)
    sorted_idx = sorted(range(len(lift)), key=lambda i: lift[i], reverse=True)
    tools = [tools[i] for i in sorted_idx]
    used_rate = [used_rate[i] for i in sorted_idx]
    not_used_rate = [not_used_rate[i] for i in sorted_idx]
    lift = [lift[i] for i in sorted_idx]
    usage_count = [usage_count[i] for i in sorted_idx]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )

    y = np.arange(len(tools))
    height = 0.35

    # Color scheme
    used_color = "#5D3A6E"  # Dark purple
    not_used_color = "#E8A0A0"  # Light pink/coral

    # Left panel: Used vs Not Used - bars side by side
    ax1.barh(
        y - height / 2, used_rate, height, color=used_color, label="Tool Used", zorder=3
    )
    ax1.barh(
        y + height / 2,
        not_used_rate,
        height,
        color=not_used_color,
        label="Tool Not Used",
        zorder=3,
    )

    # Add n= labels at end of bars
    for i, (u, n_count) in enumerate(zip(used_rate, usage_count)):
        ax1.text(u + 1, i - height / 2, f"n={n_count}", va="center", fontsize=12)
        ax1.text(
            not_used_rate[i] + 1,
            i + height / 2,
            f"n={total_tools_runs - n_count}",
            va="center",
            fontsize=12,
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels(tools)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.set_xlabel("Success Rate (%)")
    ax1.set_title("Success Rate: Tool Used vs Not Used")
    ax1.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2
    )  # Legend at bottom
    ax1.set_xlim(0, 110)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()  # Put highest lift (get_pods) at top

    # Right panel: Lift - horizontal bars with pp labels on right, bold
    lift_colors = [used_color if l > 0 else not_used_color for l in lift]
    ax2.barh(y, lift, color=lift_colors, zorder=3)

    # Add lift labels - always on right side, bold
    for i, l in enumerate(lift):
        x_pos = max(l + 2, 5) if l > 0 else l - 2
        ha = "left" if l >= 0 else "right"
        ax2.text(
            x_pos, i, f"{l:+.1f}pp", va="center", ha=ha, fontsize=9, fontweight="bold"
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels([])  # No labels on right panel
    ax2.set_xlabel("Success Rate Lift (pp)")
    ax2.set_title("Tool Impact: Success Rate Lift When Used")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.grid(axis="x", alpha=0.3)
    ax2.invert_yaxis()  # Match left panel ordering

    fig.tight_layout()
    fig.savefig(
        output_dir / "tool_success_correlation.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: tool_usage_by_scenario.png
# Original: HEATMAP format (not bar chart)
# ---------------------------------------------------------------------------
def plot_tool_usage_by_scenario(tool_df: pd.DataFrame, output_dir: Path):
    """Generate tool usage by scenario heatmap."""
    tools = [
        "describe_pod",
        "get_deployment",
        "get_events",
        "get_networkpolicies",
        "get_pod_logs",
        "get_pods",
        "run_kubectl",
    ]
    scenarios = ["S1: NetworkPolicy", "S2: ConfigMap", "S3: UPF Scale"]

    if tool_df.empty:
        print("    (skipped: no tool data)")
        return

    tools_df = tool_df[
        (tool_df["tool_condition"] == "all-tools") & (tool_df["llm_success"])
    ]

    matrix = []
    for t in tools:
        row = []
        for s in [1, 2, 3]:
            s_runs = tools_df[tools_df["scenario"] == s]
            if len(s_runs) == 0:
                row.append(0)
                continue
            total = sum(r["tool_counts"].get(t, 0) for _, r in s_runs.iterrows())
            row.append(total / len(s_runs))
        matrix.append(row)
    matrix = np.array(matrix)

    df = pd.DataFrame(matrix, index=tools, columns=scenarios)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Use YlOrRd colormap
    sns.heatmap(
        df,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=max(1.2, matrix.max()),
        ax=ax,
        cbar_kws={"label": "Avg Calls per Successful Run"},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Tool")
    ax.set_title(
        "Tool Usage by Scenario (Successful Runs)", fontsize=14, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "tool_usage_by_scenario.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: failure_mode_comparison.png
# Original: Red color scheme pie charts
# ---------------------------------------------------------------------------
def _aggregate_failure_modes(models_data: Dict, condition_filter) -> Dict:
    """Aggregate failure_modes across models matching condition_filter."""
    totals = Counter()
    total_failed = 0
    for key, data in models_data.items():
        if condition_filter(key):
            fm = data.get("failure_modes", {})
            total_failed += data.get("failed_runs", 0)
            for mode, count in fm.items():
                totals[mode] += count
    return totals, total_failed


def _failure_pie_data(failure_counts: Counter, total_failed: int):
    """Build sorted values/labels/colors for a failure mode pie chart."""
    mode_order = [
        "wrong_diagnosis",
        "no_remediation",
        "health_check_failed",
        "api_failure",
    ]
    mode_display = {
        "wrong_diagnosis": "Wrong Diagnosis",
        "no_remediation": "No Remediation",
        "health_check_failed": "Health Check Failed",
        "api_failure": "API/Network Error",
    }
    colors_palette = ["#C0392B", "#E74C3C", "#F39C12", "#F5B7B1"]

    values, labels, colors = [], [], []
    for mode in mode_order:
        count = failure_counts.get(mode, 0)
        if count == 0:
            continue
        pct = count / total_failed * 100 if total_failed > 0 else 0
        values.append(count)
        labels.append(f"{mode_display[mode]} ({pct:.1f}%)")
        colors.append(colors_palette[len(values) - 1])

    return values, labels, colors


def plot_failure_mode_comparison(stats: Dict, output_dir: Path):
    """Generate failure mode pie charts."""
    models_data = stats["models"]

    tools_fm, tools_total_failed = _aggregate_failure_modes(models_data, _is_tools)
    no_tools_fm, no_tools_total_failed = _aggregate_failure_modes(
        models_data, lambda k: not _is_tools(k)
    )

    if tools_total_failed == 0 and no_tools_total_failed == 0:
        print("    (skipped: no failures)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Tools condition pie ---
    tools_values, tools_labels, tools_colors = _failure_pie_data(
        tools_fm, tools_total_failed
    )
    wedges_t, texts_t, autotexts_t = axes[0].pie(
        tools_values,
        labels=None,
        colors=tools_colors,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=135,
        pctdistance=0.55,
        textprops={"fontsize": 12, "color": "white", "fontweight": "bold"},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    axes[0].set_title(
        f"All-Tools Condition\n({tools_total_failed} failures)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend(
        wedges_t,
        tools_labels,
        loc="center left",
        bbox_to_anchor=(-0.4, 0.5),
        fontsize=10,
        frameon=False,
    )

    # --- No-tools condition pie ---
    nt_values, nt_labels, nt_colors = _failure_pie_data(
        no_tools_fm, no_tools_total_failed
    )
    hide_small = len(nt_values) > 2
    wedges_nt, texts_nt, autotexts_nt = axes[1].pie(
        nt_values,
        labels=None,
        colors=nt_colors,
        autopct=lambda pct: f"{pct:.1f}%" if pct > 1 else "",
        startangle=90,
        pctdistance=0.55,
        textprops={"fontsize": 12, "color": "white", "fontweight": "bold"},
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    axes[1].set_title(
        f"No-Tools Condition\n({no_tools_total_failed} failures)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(
        wedges_nt,
        nt_labels,
        loc="center right",
        bbox_to_anchor=(1.4, 0.5),
        fontsize=10,
        frameon=False,
    )

    # Annotate tiny slices
    for i, (val, total) in enumerate(
        zip(nt_values, [no_tools_total_failed] * len(nt_values))
    ):
        pct = val / total * 100 if total > 0 else 0
        if 0 < pct < 2:
            angle = sum(nt_values[:i]) / total * 360 if total > 0 else 0
            axes[1].annotate(
                f"{pct:.1f}%",
                xy=(0.003, 1),
                xytext=(1.15, 0.95),
                fontsize=11,
                fontweight="bold",
                color="#C0392B",
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.5),
            )

    fig.suptitle(
        "Failure Mode Distribution by Condition", fontsize=14, fontweight="bold", y=0.98
    )
    plt.subplots_adjust(top=0.88, wspace=0.4)
    fig.savefig(
        output_dir / "failure_mode_comparison.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: duration_vs_success.png
# ---------------------------------------------------------------------------
def plot_duration_vs_success(stats: Dict, output_dir: Path):
    """Generate duration vs success scatter plot."""
    models_data = stats["models"]

    fig, ax = plt.subplots(figsize=(12, 8))

    model_colors = {
        "kimi-k2.5":        "#E07B7B",  # coral/red
        "qwen3.5-397b-a17b":"#F4A460",  # sandy brown
        "qwen3.5-35b-a3b":  "#5D3A6E",  # dark purple
        "gpt-oss-120b":     "#5D3A6E",  # dark purple
        "glm-5":            "#5D3A6E",  # dark purple
    }

    # Collect all points
    points = []
    for key, data in sorted(models_data.items()):
        short = _short_name(key)
        is_tools = _is_tools(key)

        rate    = data["success_rate_pct"]
        dur_mean = data.get("duration", {}).get("mean", 0) or 0
        dur_std  = data.get("duration", {}).get("std",  0) or 0
        n        = data.get("total_runs", 1) or 1
        dur_ci   = 1.96 * dur_std / math.sqrt(n) if n > 0 else 0

        label_suffix = " (tools)" if is_tools else " (no tools)"
        display = MODEL_DISPLAY.get(short, short) + label_suffix
        color   = model_colors.get(short, "#5D3A6E") if is_tools else "#E07B7B"

        points.append({
            "dur_mean": dur_mean,
            "rate":     rate,
            "dur_ci":   dur_ci,
            "color":    color,
            "display":  display,
            "is_tools": is_tools,
            "short":    short,
        })

    # Jitter coincident no-tools points that share the same (dur, rate) coordinate
    # so dots and labels don't completely overlap.
    JITTER_Y = 2.5  # percentage points of vertical separation
    seen_coords: Dict[Tuple[float, float], int] = {}
    for p in points:
        if p["is_tools"]:
            continue
        key_coord = (round(p["dur_mean"], 1), round(p["rate"], 1))
        count = seen_coords.get(key_coord, 0)
        if count > 0:
            p["rate"] = p["rate"] + count * JITTER_Y
        seen_coords[key_coord] = count + 1

    # Plot
    for p in points:
        ax.errorbar(
            p["dur_mean"], p["rate"],
            xerr=p["dur_ci"],
            fmt="o",
            markersize=10,
            color=p["color"],
            ecolor=p["color"],
            capsize=3,
            capthick=1,
            alpha=0.9,
            zorder=5,
        )

        disp = p["display"]
        if disp in LABEL_POSITIONS:
            xytext = (LABEL_POSITIONS[disp][0], LABEL_POSITIONS[disp][1])
            zorder = LABEL_POSITIONS[disp][2]
        elif p["rate"] > 80:
            xytext, zorder = (10,   8), 5
        elif p["rate"] > 50:
            xytext, zorder = (10, -15), 5
        else:
            xytext, zorder = (10, -10), 5

        ax.annotate(
            disp,
            (p["dur_mean"], p["rate"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#E8E8E8",
                edgecolor="gray",
                alpha=0.9,
            ),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
            zorder=zorder,
        )

    ax.set_xlabel("Average Diagnosis Duration (seconds)", fontsize=11)
    ax.set_ylabel("LLM Success Rate (%)", fontsize=11)
    ax.set_title(
        "Success Rate vs. Diagnosis Duration\n(Upper-left = ideal: fast + high success)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylim(-8, 105)
    ax.set_xlim(left=-10, right=140)
    ax.grid(True, alpha=0.3, linestyle="-")

    fig.tight_layout()
    fig.savefig(
        output_dir / "duration_vs_success.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: cost_benefit_analysis.png
# Original: Green/blue colors, bubble chart + bar chart
# ---------------------------------------------------------------------------
def plot_cost_benefit_analysis(
    stats: Dict, output_dir: Path, model_costs: Optional[Dict[str, float]] = None
):
    """Generate cost-benefit analysis chart."""
    csv_path = output_dir / "cost_benefit_summary.csv"
    precomputed = load_cost_benefit_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if precomputed:
        names = [r["name"] for r in precomputed]
        rates = [r["rate"] for r in precomputed]
        costs_per_success = [r["cost_per_success"] * 1000 for r in precomputed]
        speeds = [r["duration"] for r in precomputed]
    else:
        models_data = stats["models"]
        tools_models = {k: v for k, v in models_data.items() if _is_tools(k)}
        costs_dict = model_costs if model_costs else MODEL_COST_PER_1M

        names, rates, costs_per_success, speeds = [], [], [], []

        for key, data in sorted(tools_models.items()):
            short = _short_name(key)
            display = MODEL_DISPLAY.get(short, short)
            rate = data["success_rate_pct"]
            dur = data.get("duration", {}).get("mean", 30) or 30

            token_data = data.get("tokens", {})
            if token_data and token_data.get("avg_tokens_per_run"):
                avg_tokens = int(token_data["avg_tokens_per_run"])
            else:
                avg_turns = data.get("turns", {}).get("mean", 2) or 2
                avg_tokens = int(avg_turns * 6000)

            cost_per_1m = costs_dict.get(
                display, costs_dict.get(short, MODEL_COST_PER_1M.get(short, 0.30))
            )
            cost_per_run = (avg_tokens / 1_000_000) * cost_per_1m
            cps = cost_per_run / (rate / 100) if rate > 0 else float("inf")
            cps_millicents = cps * 1000

            names.append(display)
            rates.append(rate)
            costs_per_success.append(cps_millicents)
            speeds.append(dur)

    # Left panel: Bubble chart (green bubbles)
    bubble_sizes = [max(100, 400 - d * 5) for d in speeds]  # Bigger = faster

    ax1.scatter(
        costs_per_success,
        rates,
        s=bubble_sizes,
        c="#4CAF50",
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        zorder=5,
    )

    for i, (cps, rate, name) in enumerate(zip(costs_per_success, rates, names)):
        ax1.annotate(
            name, (cps, rate), textcoords="offset points", xytext=(5, 5), fontsize=9
        )

    # Add threshold lines
    ax1.axhline(
        85, color="#4CAF50", linestyle="--", alpha=0.5, label="85% success threshold"
    )
    ax1.axvline(
        5, color="#FFA500", linestyle="--", alpha=0.5, label="$0.005 cost threshold"
    )

    ax1.set_xlabel("Cost per Success (millicents)")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Cost-Effectiveness Analysis\n(bubble size = speed)")
    ax1.set_ylim(40, 100)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: Horizontal bar chart (green for rate, blue for cost)
    y = np.arange(len(names))
    height = 0.35

    # Sort by success rate descending
    sorted_idx = sorted(range(len(rates)), key=lambda i: rates[i], reverse=True)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_rates = [rates[i] for i in sorted_idx]
    sorted_cps = [costs_per_success[i] for i in sorted_idx]

    ax2.barh(
        y - height / 2, sorted_rates, height, color="#4CAF50", label="Success Rate (%)"
    )

    # Normalize cost for display on same scale
    max_cps = max(sorted_cps)
    normalized_cps = [c / max_cps * 100 for c in sorted_cps]
    ax2.barh(
        y + height / 2,
        normalized_cps,
        height,
        color="#2196F3",
        label="Cost per Success (scaled)",
    )

    # Add value labels - pp on right and bold
    for i, (rate, cps) in enumerate(zip(sorted_rates, sorted_cps)):
        ax2.text(
            rate + 2,
            i - height / 2,
            f"{rate:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        ax2.text(
            normalized_cps[i] + 2,
            i + height / 2,
            f"{cps:.2f}mc",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(sorted_names)
    ax2.set_xlabel("Value")
    ax2.set_title("Success Rate vs Cost per Success")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "cost_benefit_analysis.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------
def write_cost_benefit_csv(
    stats: Dict, output_dir: Path, model_costs: Optional[Dict[str, float]] = None
):
    """Write cost-benefit summary CSV (skip if already exists)."""
    csv_path = output_dir / "cost_benefit_summary.csv"
    if csv_path.exists():
        print(f"  (cost_benefit_summary.csv already exists, skipping)")
        return

    # Use provided costs or fall back to defaults
    costs_dict = model_costs if model_costs else MODEL_COST_PER_1M

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Model",
                "Success%",
                "Avg Cost ($)",
                "Cost/Success ($)",
                "Avg Duration (s)",
                "Avg Tokens",
            ]
        )

        models_data = stats["models"]
        for key, data in sorted(models_data.items()):
            if not _is_tools(key):
                continue
            short = _short_name(key)
            display = MODEL_DISPLAY.get(short, short)
            rate = data["success_rate_pct"]

            # Use real token data from stats
            token_data = data.get("tokens", {})
            if token_data and token_data.get("avg_tokens_per_run"):
                avg_tokens = int(token_data["avg_tokens_per_run"])
            else:
                avg_turns = data.get("turns", {}).get("mean", 2) or 2
                avg_tokens = int(avg_turns * 6000)

            # Get cost from CSV-derived dict or use default
            cost_per_1m = costs_dict.get(
                display, costs_dict.get(short, MODEL_COST_PER_1M.get(short, 0.30))
            )
            avg_cost = (avg_tokens / 1_000_000) * cost_per_1m
            cost_per_success = avg_cost / (rate / 100) if rate > 0 else 0
            avg_dur = data.get("duration", {}).get("mean", 0) or 0

            writer.writerow(
                [display, rate, avg_cost, cost_per_success, avg_dur, avg_tokens]
            )


def write_api_error_csv(stats: Dict, output_dir: Path):
    """Write API error summary CSV."""
    csv_path = output_dir / "api_error_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "total_runs",
                "runs_with_errors",
                "total_api_errors",
                "error_rate",
            ]
        )

        models_data = stats["models"]
        for key, data in sorted(models_data.items()):
            if not _is_tools(key):
                continue
            short = _short_name(key)
            total_runs = data["total_runs"]
            runs_with_errors = data.get("runs_with_api_errors", 0)
            total_errors = data.get("api_errors", 0)
            error_rate = runs_with_errors / total_runs * 100 if total_runs > 0 else 0

            writer.writerow(
                [short, total_runs, runs_with_errors, total_errors, f"{error_rate:.1f}"]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OperAID Paper Figure Generator")
    parser.add_argument("--stats", required=True, help="Path to suite_statistics.json")
    parser.add_argument("--suite-dir", help="Suite results directory for tool data")
    parser.add_argument("--pricing", help="Path to pricing CSV (model, cost_per_1m)")
    parser.add_argument("--output-dir", "-o", default="plots", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stats) as f:
        stats = json.load(f)

    tool_df = pd.DataFrame()
    if args.suite_dir:
        tool_df = load_tool_data(args.suite_dir)

    model_costs = {}
    if args.pricing:
        model_costs = load_pricing_csv(Path(args.pricing))

    if not model_costs:
        print(
            "  Warning: No pricing CSV provided. Cost-dependent figures will be skipped."
        )
        print(
            "  Provide --pricing <path> pointing to a CSV with columns: model, cost_per_1m"
        )

    print(f"Generating figures in {output_dir}...")

    # Generate figures
    figures = [
        (
            "model_comparison_accuracy_ci.png",
            lambda: plot_model_comparison_accuracy_ci(stats, output_dir),
        ),
        (
            "tool_success_correlation.png",
            lambda: plot_tool_success_correlation(tool_df, output_dir),
        ),
        (
            "tool_usage_by_scenario.png",
            lambda: plot_tool_usage_by_scenario(tool_df, output_dir),
        ),
        (
            "failure_mode_comparison.png",
            lambda: plot_failure_mode_comparison(stats, output_dir),
        ),
        (
            "duration_vs_success.png",
            lambda: plot_duration_vs_success(stats, output_dir),
        ),
        (
            "cost_benefit_analysis.png",
            lambda: plot_cost_benefit_analysis(stats, output_dir, model_costs),
        ),
    ]

    for name, func in figures:
        try:
            func()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback

            traceback.print_exc()

    # Write CSV files (pass model_costs to ensure consistency)
    write_cost_benefit_csv(stats, output_dir, model_costs)
    print(f"  ✓ cost_benefit_summary.csv")

    write_api_error_csv(stats, output_dir)
    print(f"  ✓ api_error_summary.csv")

    print(f"\nDone! {len(figures)} figures generated.")


if __name__ == "__main__":
    main()
