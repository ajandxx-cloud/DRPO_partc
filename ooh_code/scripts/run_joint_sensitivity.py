#!/usr/bin/env python3
"""
Joint sensitivity analysis: grid search over (beta_price, U0, alpha_home).

Runs DSPO and DRPO on a 3x3x3 parameter grid (27 combinations),
3 seeds each = 162 runs total. Reports DRPO vs DSPO win rate per cell.

Parameter grid:
  beta_price  (incentive_sens): -0.15 (weak), -0.25 (default), -0.35 (strong)
  U0          (outside_option_util): -0.5 (strong outside), -1.0 (default), -1.5 (weak outside)
  alpha_home  (home_util): 1.0 (weak home pref), 1.4 (default), 1.8 (strong home pref)

Usage:
    python scripts/run_joint_sensitivity.py              # run all 162 runs
    python scripts/run_joint_sensitivity.py --dry_run    # print commands only
    python scripts/run_joint_sensitivity.py --analyze    # analyze existing results
    python scripts/run_joint_sensitivity.py --seeds 40 67  # subset of seeds
"""
import argparse
import csv
import itertools
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

BASE_CONFIG = {
    "instance": "RC",
    "data_seed": 0,
    "data_seed_test": 1,
    "fuel_cost": 0.6,
    "home_failure": 0.1,
    "k": 10,
    "learning_rate": 0.001,
    "revenue": 50.0,
    "spo_loss_weight": 0.7,
    "spo_rampup_episodes": 10,
    "spo_warmup_episodes": 5,
    "max_episodes": 200,
    "save_count": 1,
    "batch_size": 256,
    "gpu": 0,
}

SEEDS = [40, 67, 97]

# 3x3x3 parameter grid
PARAM_GRID = {
    "incentive_sens":     [-0.15, -0.25, -0.35],   # beta_price (weak→strong)
    "outside_option_util": [-0.5, -1.0, -1.5],     # U0 (strong→weak outside option)
    "home_util":          [1.0, 1.4, 1.8],          # alpha_home (weak→strong home pref)
}

PATTERNS = {
    "net_profit":    re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_costs":   re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "quit_rate":     re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?)%"),
    "home_delivery": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?)"),
}


def canonical_algo_name(name):
    return "DRPO" if name == "DSPO_plus_SPO" else name


def param_tag(incentive_sens, outside_option_util, home_util):
    """Short tag for experiment ID."""
    return (f"b{abs(incentive_sens)*100:.0f}"
            f"_u{abs(outside_option_util)*10:.0f}"
            f"_h{home_util*10:.0f}")


def build_command(algo_name, seed, incentive_sens, outside_option_util, home_util):
    tag = param_tag(incentive_sens, outside_option_util, home_util)
    experiment_id = f"JTSENS_{algo_name}_{tag}_{seed}"
    cfg = dict(BASE_CONFIG)
    spo_weight = cfg["spo_loss_weight"] if canonical_algo_name(algo_name) == "DRPO" else 0.0

    cmd = [
        sys.executable, "run.py",
        "--algo_name", algo_name,
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
        "--home_util", str(home_util),
        "--incentive_sens", str(incentive_sens),
        "--k", str(cfg["k"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--outside_option_util", str(outside_option_util),
        "--revenue", str(cfg["revenue"]),
        "--spo_loss_weight", str(spo_weight),
        "--spo_rampup_episodes", str(cfg["spo_rampup_episodes"]),
        "--spo_warmup_episodes", str(cfg["spo_warmup_episodes"]),
        "--experiment", experiment_id,
        "--folder_suffix", "_jtsens",
    ]
    return cmd, experiment_id


def find_log(experiment_id, seed):
    root = ROOT / "Experiments/Parcelpoint_py/pricing"
    candidates = list(root.rglob(f"*{experiment_id}*/{seed}/Logs/logfile.log"))
    return candidates[0] if candidates else None


def extract_metrics(log_path):
    metrics = {}
    try:
        text = Path(log_path).read_text(errors="replace")
        for key, pat in PATTERNS.items():
            m = pat.search(text)
            if m:
                metrics[key] = float(m.group(1))
    except Exception as e:
        print(f"  Warning: {e}")
    return metrics


def run_one(algo_name, seed, incentive_sens, outside_option_util, home_util, dry_run=False):
    cmd, exp_id = build_command(algo_name, seed, incentive_sens, outside_option_util, home_util)
    log_file = find_log(exp_id, seed)
    if log_file and log_file.exists():
        m = extract_metrics(log_file)
        if "net_profit" in m and "total_costs" in m:
            m.update({"seed": seed, "algo": canonical_algo_name(algo_name),
                      "incentive_sens": incentive_sens,
                      "outside_option_util": outside_option_util,
                      "home_util": home_util})
            return m, True  # (result, was_cached)
        print(f"  [RERUN] {algo_name} seed={seed} (incomplete log)")

    if dry_run:
        print("    " + " ".join(str(c) for c in cmd))
        return None, False

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  [ERROR] {algo_name} seed={seed} rc={proc.returncode}")
        print(proc.stderr[-300:])
        return None, False

    log_file = find_log(exp_id, seed)
    if log_file:
        m = extract_metrics(log_file)
        m.update({"seed": seed, "algo": canonical_algo_name(algo_name),
                  "incentive_sens": incentive_sens,
                  "outside_option_util": outside_option_util,
                  "home_util": home_util,
                  "runtime_sec": elapsed})
        return m, False
    return None, False


def save_csv(results, path):
    if not results:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {path}")


def analyze_results(all_results):
    """For each parameter combination, compute DRPO vs DSPO win rate."""
    combos = list(itertools.product(
        PARAM_GRID["incentive_sens"],
        PARAM_GRID["outside_option_util"],
        PARAM_GRID["home_util"],
    ))

    grid_summary = []
    for (beta, u0, alpha) in combos:
        dspo_rows = [r for r in all_results
                     if canonical_algo_name(r["algo"]) == "DSPO"
                     and abs(r["incentive_sens"] - beta) < 1e-6
                     and abs(r["outside_option_util"] - u0) < 1e-6
                     and abs(r["home_util"] - alpha) < 1e-6]
        drpo_rows = [r for r in all_results
                     if canonical_algo_name(r["algo"]) == "DRPO"
                     and abs(r["incentive_sens"] - beta) < 1e-6
                     and abs(r["outside_option_util"] - u0) < 1e-6
                     and abs(r["home_util"] - alpha) < 1e-6]

        dspo_by_seed = {r["seed"]: r for r in dspo_rows}
        drpo_by_seed = {r["seed"]: r for r in drpo_rows}
        common = sorted(set(dspo_by_seed) & set(drpo_by_seed))

        wins_profit = 0
        wins_cost = 0
        profit_diffs = []
        for seed in common:
            dp = drpo_by_seed[seed].get("net_profit", 0) - dspo_by_seed[seed].get("net_profit", 0)
            dc = dspo_by_seed[seed].get("total_costs", 0) - drpo_by_seed[seed].get("total_costs", 0)
            profit_diffs.append(dp)
            if dp > 0:
                wins_profit += 1
            if dc > 0:
                wins_cost += 1

        n = len(common)
        mean_diff = sum(profit_diffs) / n if n > 0 else float("nan")
        dspo_profit_mean = (sum(r.get("net_profit", 0) for r in dspo_rows) / len(dspo_rows)
                            if dspo_rows else float("nan"))
        pct_diff = mean_diff / dspo_profit_mean * 100 if dspo_profit_mean else float("nan")

        grid_summary.append({
            "incentive_sens": beta,
            "outside_option_util": u0,
            "home_util": alpha,
            "n_paired": n,
            "drpo_wins_profit": wins_profit,
            "drpo_wins_cost": wins_cost,
            "profit_diff_mean": round(mean_diff, 2),
            "profit_diff_pct": round(pct_diff, 2),
            "win_rate": f"{wins_profit}/{n}" if n > 0 else "?",
        })

    return grid_summary


def print_heatmap(grid_summary):
    """Print ASCII heatmap of win rates."""
    betas = PARAM_GRID["incentive_sens"]
    u0s = PARAM_GRID["outside_option_util"]
    alphas = PARAM_GRID["home_util"]

    by_key = {(r["incentive_sens"], r["outside_option_util"], r["home_util"]): r
              for r in grid_summary}

    for alpha in alphas:
        print(f"\n  alpha_home = {alpha}")
        print(f"  {'':12s}", end="")
        for beta in betas:
            print(f"  beta={beta:+.2f}", end="")
        print()
        for u0 in u0s:
            print(f"  U0={u0:+.1f}    ", end="")
            for beta in betas:
                cell = by_key.get((beta, u0, alpha), {})
                wr = cell.get("win_rate", "?")
                pct = cell.get("profit_diff_pct", float("nan"))
                if not math.isnan(pct):
                    print(f"  {wr}({pct:+.1f}%)", end="")
                else:
                    print(f"  {wr}(  ?  )", end="")
            print()


def main():
    parser = argparse.ArgumentParser(description="Joint sensitivity analysis")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS
    out_dir = ROOT / "Experiments" / "analysis" / "joint_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(
        PARAM_GRID["incentive_sens"],
        PARAM_GRID["outside_option_util"],
        PARAM_GRID["home_util"],
    ))
    total_runs = len(combos) * 2 * len(seeds)  # DSPO + DRPO per combo per seed

    print(f"\n{'='*65}")
    print(f"Joint Sensitivity Analysis")
    print(f"  Grid: {len(combos)} combinations x 2 algos x {len(seeds)} seeds = {total_runs} runs")
    print(f"{'='*65}")

    all_results = []
    csv_path = out_dir / "joint_sensitivity_results.csv"

    if args.analyze:
        if csv_path.exists():
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = {}
                    for k, v in row.items():
                        try:
                            r[k] = float(v)
                        except (ValueError, TypeError):
                            r[k] = v
                    all_results.append(r)
            print(f"  Loaded {len(all_results)} rows from {csv_path}")
        else:
            print(f"  No results found at {csv_path}. Run without --analyze first.")
            return
    else:
        run_count = 0
        skip_count = 0
        for (beta, u0, alpha) in combos:
            tag = param_tag(beta, u0, alpha)
            for algo in ["DSPO", "DRPO"]:
                for seed in seeds:
                    result, cached = run_one(algo, seed, beta, u0, alpha, dry_run=args.dry_run)
                    if cached:
                        skip_count += 1
                    else:
                        run_count += 1
                    if result:
                        all_results.append(result)

        if not args.dry_run:
            save_csv(all_results, csv_path)
            print(f"\n  Ran {run_count} new, skipped {skip_count} cached.")

    if args.dry_run or not all_results:
        return

    # ── Analysis ─────────────────────────────────────────────────────────────
    grid_summary = analyze_results(all_results)

    total_cells = len(grid_summary)
    cells_with_data = [c for c in grid_summary if c["n_paired"] > 0]
    drpo_wins_all = sum(c["drpo_wins_profit"] for c in cells_with_data)
    total_paired = sum(c["n_paired"] for c in cells_with_data)
    cells_drpo_majority = sum(1 for c in cells_with_data
                               if c["drpo_wins_profit"] > c["n_paired"] / 2)

    print(f"\n{'='*65}")
    print("JOINT SENSITIVITY SUMMARY")
    print(f"{'='*65}")
    print(f"  Total parameter combinations: {total_cells}")
    print(f"  Combinations with data: {len(cells_with_data)}")
    print(f"  DRPO wins (profit) overall: {drpo_wins_all}/{total_paired} paired comparisons")
    print(f"  Combinations where DRPO majority wins: {cells_drpo_majority}/{len(cells_with_data)}")

    print_heatmap(grid_summary)

    # Save grid summary
    save_csv(grid_summary, out_dir / "grid_summary.csv")
    with open(out_dir / "grid_summary.json", "w") as f:
        json.dump(grid_summary, f, indent=2)

    # ── LaTeX table snippet ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("LaTeX TABLE SNIPPET")
    print(f"{'='*65}")
    print(r"""\begin{table}[t]
\centering
\caption{Joint sensitivity: DRPO vs.\ DSPO win rate across parameter combinations
(3 seeds per cell; win rate = seeds where DRPO net profit $>$ DSPO net profit).}
\label{tab:joint-sensitivity}
\small
\begin{tabular}{ccc|cc}
\toprule
$\beta_{\text{price}}$ & $U_0$ & $\alpha_{\text{home}}$ & Win rate & Profit diff (\%) \\
\midrule""")
    for c in grid_summary:
        print(f"{c['incentive_sens']:+.2f} & {c['outside_option_util']:+.1f} & "
              f"{c['home_util']:.1f} & "
              f"{c['win_rate']} & "
              f"{c['profit_diff_pct']:+.1f}\\% \\\\")
    print(r"""\bottomrule
\end{tabular}
\end{table}""")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
