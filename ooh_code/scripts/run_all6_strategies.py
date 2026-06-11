#!/usr/bin/env python3
"""
Run all 6 strategies (Only-home, Only-meeting-points, No-pricing, Static-pricing, DSPO, DRPO)
with the same 30 seeds and parameters as the existing DSPO/DRPO comparison experiment.
Collect net_profit and total_costs for Table 1 update.

Usage:
    python scripts/run_all6_strategies.py
    python scripts/run_all6_strategies.py --dry_run  # print commands without running
    python scripts/run_all6_strategies.py --analyze  # only analyze existing results
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ─── Match the existing 30-seed DSPO/DRPO experiment parameters ───
BASE_CONFIG = {
    "instance": "RC",
    "data_seed": 0,
    "data_seed_test": 1,
    "fuel_cost": 0.6,
    "home_failure": 0.1,
    "home_util": 1.4,
    "incentive_sens": -0.25,
    "k": 10,
    "learning_rate": 0.001,
    "outside_option_util": -1.0,
    "revenue": 50.0,
    "spo_loss_weight": 0.7,
    "spo_rampup_episodes": 10,
    "spo_warmup_episodes": 5,
    "max_episodes": 200,
    "save_count": 1,
    "batch_size": 256,
    "gpu": 0,
}

SEEDS = [40, 67, 97, 52, 29, 20, 17, 88, 63, 79, 60, 62, 7, 48, 56, 15, 66, 53,
         90, 70, 24, 74, 80, 28, 2, 95, 92, 26, 39, 82]

# ─── Strategy configurations ───
# Only-home: set meeting-point price to +100 → utility = base - 0.25*100 = -25 << home(1.4)
# Only-meeting-points: set home price to +100 → home utility = 1.4-25 = -23.6 << meeting(~0)
# No-pricing: all prices = 0
# Static-pricing: home surcharge=+2, meeting discount=-5 (matches Table 1 avg=5/2)
STRATEGIES = {
    "Only-home":            {"algo_name": "Baseline", "pricing": True,  "price_home": 0,   "price_pp": 100},
    "Only-meeting-points":  {"algo_name": "Baseline", "pricing": True,  "price_home": 100, "price_pp": 0},
    "No-pricing":           {"algo_name": "Baseline", "pricing": True,  "price_home": 0,   "price_pp": 0},
    "Static-pricing":       {"algo_name": "Baseline", "pricing": True,  "price_home": 2,   "price_pp": -5},
}
# DSPO and DRPO already have results; skip running them unless you want fresh runs.
SKIP_EXISTING = True   # set to False to re-run DSPO/DRPO too

# Metric extraction regexes (same as rc_completeness_experiments.py)
PATTERNS = {
    "net_profit":       re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_costs":      re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "quit_rate":        re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?)%"),
    "home_delivery":    re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?)"),
    "travel_costs":     re.compile(r"travel costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "service_costs":    re.compile(r"service costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "failure_costs":    re.compile(r"failure costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "charge_revenue":   re.compile(r"Charge revenue:\s*([+-]?\d+(?:\.\d+)?)"),
    "discount_costs":   re.compile(r"Discount costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "avg_charge":       re.compile(r"Avg\. Charge:\s*([+-]?\d+(?:\.\d+)?)"),
    "avg_discount":     re.compile(r"Avg\. Discount:\s*([+-]?\d+(?:\.\d+)?)"),
    "served_demand":    re.compile(r"Accepted customers:\s*(\d+)"),
    "total_demand":     re.compile(r"Total customers:\s*(\d+)"),
}


def build_command(strategy_name, config, seed, out_dir):
    """Build the run.py command for a given strategy and seed."""
    cfg = dict(BASE_CONFIG)
    cfg.update(config)

    label = strategy_name.replace("-", "_").replace(" ", "_")
    experiment_id = f"ALL6_{label}_{seed}"

    cmd = [
        sys.executable, "run.py",
        "--algo_name", cfg["algo_name"],
        "--instance", cfg["instance"],
        "--seed", str(seed),
        "--data_seed", str(cfg["data_seed"]),
        "--data_seed_test", str(cfg["data_seed_test"]),
        "--max_episodes", str(cfg["max_episodes"]),
        "--save_count", str(cfg["save_count"]),
        "--log_output", "file",
        "--debug", "False",
        "--gpu", str(cfg["gpu"]),
        "--batch_size", str(cfg["batch_size"]),
        "--fuel_cost", str(cfg["fuel_cost"]),
        "--home_failure", str(cfg["home_failure"]),
        "--home_util", str(cfg["home_util"]),
        "--incentive_sens", str(cfg["incentive_sens"]),
        "--k", str(cfg["k"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--outside_option_util", str(cfg["outside_option_util"]),
        "--revenue", str(cfg["revenue"]),
        "--spo_loss_weight", str(cfg["spo_loss_weight"]),
        "--spo_rampup_episodes", str(cfg["spo_rampup_episodes"]),
        "--spo_warmup_episodes", str(cfg["spo_warmup_episodes"]),
        "--experiment", experiment_id,
        "--folder_suffix", "_all6",
    ]

    if "pricing" in cfg:
        cmd += ["--pricing", str(cfg["pricing"])]
    if "price_home" in cfg:
        cmd += ["--price_home", str(cfg["price_home"])]
    if "price_pp" in cfg:
        cmd += ["--price_pp", str(cfg["price_pp"])]

    return cmd, experiment_id


def extract_metrics_from_log(log_path):
    """Extract metrics from a logfile."""
    metrics = {}
    try:
        text = Path(log_path).read_text(errors="replace")
        for key, pat in PATTERNS.items():
            m = pat.search(text)
            if m:
                metrics[key] = float(m.group(1))
    except Exception as e:
        print(f"  Warning: could not read {log_path}: {e}")
    return metrics


def find_log(experiment_id, seed, root="Experiments/Parcelpoint_py/pricing"):
    """Find log file for a given experiment run."""
    candidates = list(Path(root).rglob(f"*{experiment_id}*/{seed}/Logs/logfile.log"))
    if not candidates:
        # also try Baseline folder
        candidates = list(Path(root).rglob(f"*{experiment_id}*/*/Logs/logfile.log"))
    return candidates[0] if candidates else None


def run_experiment(strategy_name, config, seeds, dry_run=False):
    """Run all seeds for a strategy; return per-seed metrics."""
    label = strategy_name.replace("-", "_").replace(" ", "_")
    results = []
    for seed in seeds:
        cmd, experiment_id = build_command(strategy_name, config, seed, out_dir=None)
        log_file = find_log(experiment_id, seed)
        if log_file and log_file.exists():
            print(f"  [SKIP existing] {strategy_name} seed={seed}")
            metrics = extract_metrics_from_log(log_file)
            metrics["seed"] = seed
            results.append(metrics)
            continue

        print(f"  [RUN] {strategy_name} seed={seed}")
        if dry_run:
            print("    " + " ".join(str(c) for c in cmd))
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [ERROR] seed={seed} rc={proc.returncode}")
            print(proc.stderr[-500:])
            continue

        # find the newly created log
        log_file = find_log(experiment_id, seed)
        if log_file:
            metrics = extract_metrics_from_log(log_file)
            metrics["seed"] = seed
            metrics["runtime_sec"] = elapsed
            results.append(metrics)
            print(f"    → net_profit={metrics.get('net_profit','?'):.1f}  "
                  f"total_costs={metrics.get('total_costs','?'):.1f}  "
                  f"({elapsed:.0f}s)")
        else:
            print(f"  [WARN] log not found for seed={seed}")

    return results


def summarize(results, strategy_name):
    """Compute mean ± std and 95% CI."""
    import math, statistics
    if not results:
        return None
    keys = ["net_profit", "total_costs", "quit_rate", "home_delivery",
            "travel_costs", "service_costs", "failure_costs",
            "charge_revenue", "discount_costs", "avg_charge", "avg_discount"]
    summary = {"strategy": strategy_name, "n": len(results)}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = statistics.stdev(vals) if n > 1 else 0.0
        t_crit = {1: 12.706, 2: 4.303, 3: 3.182, 5: 2.571,
                  10: 2.228, 20: 2.086, 29: 2.045, 30: 2.042}.get(n - 1, 2.0)
        ci_half = t_crit * std / math.sqrt(n)
        summary[f"{k}_mean"] = round(mean, 2)
        summary[f"{k}_std"] = round(std, 2)
        summary[f"{k}_ci_half"] = round(ci_half, 2)
    return summary


def print_table(summaries):
    """Print a Table-1-style summary."""
    print("\n" + "="*90)
    print("TABLE 1 UPDATE — Net Profit & Total Costs (all 6 strategies)")
    print("="*90)
    fmt = "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("Strategy", "Home%", "Net profit", "Total costs",
                     "Quit%", "Savings%(cost)", "CI±"))
    print("-"*90)

    only_home_opex = None
    for s in summaries:
        opex = (s.get("travel_costs_mean", 0)
                + s.get("service_costs_mean", 0)
                + s.get("failure_costs_mean", 0))
        if s["strategy"] == "Only-home":
            only_home_opex = opex
        s["_opex"] = opex

    for s in summaries:
        savings = ""
        ci = ""
        if only_home_opex and only_home_opex > 0:
            sv = (only_home_opex - s["_opex"]) / only_home_opex * 100
            savings = f"{sv:.1f}%"
            # CI from net_profit CI (rough approximation)
            ci = f"{s.get('net_profit_ci_half', 0):.1f}"
        print(fmt.format(
            s["strategy"],
            f"{s.get('home_delivery_mean', 0)*100:.1f}%",
            f"{s.get('net_profit_mean', 0):.1f}",
            f"{s.get('total_costs_mean', 0):.1f}",
            f"{s.get('quit_rate_mean', 0):.1f}%",
            savings,
            f"±{ci}",
        ))
    print("="*90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze", action="store_true",
                        help="Only analyze existing results, no running")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"Experiments/analysis/all6_strategies_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    # ─── Run / analyze baseline strategies ───
    for strategy_name, config in STRATEGIES.items():
        print(f"\n{'─'*60}")
        print(f"Strategy: {strategy_name}")
        if args.analyze:
            # Only analyze existing logs
            results = []
            for seed in SEEDS:
                _, experiment_id = build_command(strategy_name, config, seed, None)
                log_file = find_log(experiment_id, seed)
                if log_file:
                    m = extract_metrics_from_log(log_file)
                    m["seed"] = seed
                    results.append(m)
                    print(f"  Found seed={seed}: net_profit={m.get('net_profit','?')}")
                else:
                    print(f"  Missing seed={seed}")
        else:
            results = run_experiment(strategy_name, config, SEEDS, dry_run=args.dry_run)

        if results:
            s = summarize(results, strategy_name)
            all_summaries.append(s)
            print(f"  Summary: net_profit={s.get('net_profit_mean','?'):.1f} "
                  f"± {s.get('net_profit_ci_half','?'):.1f}  |  "
                  f"total_costs={s.get('total_costs_mean','?'):.1f}")

    # ─── Load existing DSPO / DRPO results ───
    dspo_csv = Path("Experiments/analysis/rc_full12_algo_compare_20260319_225317/compare_summary.csv")
    if dspo_csv.exists():
        print(f"\n{'─'*60}")
        print("Loading existing DSPO/DRPO results...")
        with open(dspo_csv) as f:
            for row in csv.DictReader(f):
                label_map = {"DSPO": "DSPO", "DSPO_plus_SPO": "DRPO"}
                display = label_map.get(row["label"], row["label"])
                # Get detailed metrics from raw CSV
                raw_csv = Path("Experiments/analysis/rc_full12_algo_compare_20260319_225317/compare_raw.csv")
                extra = {}
                if raw_csv.exists():
                    with open(raw_csv) as rf:
                        raw_rows = [r for r in csv.DictReader(rf) if r["label"] == row["label"]]
                    if raw_rows:
                        for k in ["net_profit", "total_costs", "quit_rate"]:
                            vals = [float(r[k]) for r in raw_rows if k in r]
                            import statistics, math
                            if vals:
                                n = len(vals)
                                mean = sum(vals)/n
                                std = statistics.stdev(vals) if n > 1 else 0.0
                                t_crit = 2.045  # t(0.975, 29)
                                extra[f"{k}_mean"] = round(mean, 2)
                                extra[f"{k}_std"] = round(std, 2)
                                extra[f"{k}_ci_half"] = round(t_crit * std / math.sqrt(n), 2)

                s = {
                    "strategy": display,
                    "n": int(row["n_runs"]),
                    "net_profit_mean": round(float(row["net_profit_mean"]), 2),
                    "net_profit_std": round(float(row["net_profit_std"]), 2),
                    "net_profit_ci_half": round(
                        2.045 * float(row["net_profit_std"]) / (float(row["n_runs"])**0.5), 2),
                    "total_costs_mean": round(float(row["total_costs_mean"]), 2),
                    "quit_rate_mean": round(float(row["quit_rate_mean"]), 2),
                }
                s.update(extra)
                all_summaries.append(s)
                print(f"  {display}: net_profit={s['net_profit_mean']:.1f} ± {s['net_profit_ci_half']:.1f}")

    # ─── Order: Only-home first, then others ───
    order = ["Only-home", "Only-meeting-points", "No-pricing",
             "Static-pricing", "DSPO", "DRPO"]
    all_summaries.sort(key=lambda x: order.index(x["strategy"]) if x["strategy"] in order else 99)

    # ─── Print table ───
    if all_summaries:
        print_table(all_summaries)

    # ─── Save summary JSON ───
    summary_path = out_dir / "all6_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved to: {summary_path}")

    return all_summaries


if __name__ == "__main__":
    main()
