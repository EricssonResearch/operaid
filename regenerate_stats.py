#!/usr/bin/env python3
"""
Regenerate aggregated suite files from per-run result.json data.

Produces:
  - suite_statistics.json
  - summary.csv
  - api_error_summary.csv

Usage:
    python3 regenerate_stats.py <suite_dir>
"""

import argparse
import json
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_suite import compute_suite_statistics, write_summary_csv, write_api_error_csv

parser = argparse.ArgumentParser(description="Regenerate suite aggregated files")
parser.add_argument("suite_dir", help="Path to suite results directory")
args = parser.parse_args()

suite_dir = os.path.abspath(args.suite_dir)
if not os.path.isdir(suite_dir):
    print(f"ERROR: {suite_dir} is not a directory", file=sys.stderr)
    sys.exit(1)

experiments = []
results = []

for rf in sorted(glob.glob(os.path.join(suite_dir, "*/*/run_*/result.json"))):
    r = json.load(open(rf))
    parts = os.path.dirname(rf).replace(suite_dir + "/", "").split("/")
    name = parts[0]
    scenario = int(parts[1].split("_")[1])
    run = int(parts[2].split("_")[1])
    experiments.append(
        {
            "name": name,
            "model": r.get("model", ""),
            "scenario": scenario,
            "run": run,
            "runs_per_scenario": 30,
            "use_tools": "no-tools" not in name,
        }
    )
    r["run"] = run
    r["failure_reason"] = r.get("failure_reason", "unknown")
    results.append(r)

print(f"Found {len(results)} runs in {suite_dir}")

stats = compute_suite_statistics(results, experiments)

stats_path = os.path.join(suite_dir, "suite_statistics.json")
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2, default=str)
print(f"Written: {stats_path}")

csv_path = os.path.join(suite_dir, "summary.csv")
write_summary_csv(results, csv_path)
print(f"Written: {csv_path}")

error_csv_path = os.path.join(suite_dir, "api_error_summary.csv")
write_api_error_csv(stats, error_csv_path)
print(f"Written: {error_csv_path}")

print(f"\nPer-model breakdown:")
for k, v in stats["models"].items():
    fm = v["failure_modes"]
    print(
        f"  {k}: llm_ok={v['llm_successful_runs']}/{v['total_runs']}, "
        f"failures={v['failed_runs']}, modes={fm}"
    )
