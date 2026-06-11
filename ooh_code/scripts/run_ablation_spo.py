#!/usr/bin/env python3
"""
Ablation experiment: DSPO-ablation (Huber-only) vs DRPO (Huber+SPO+).

Both use the DRPO entry point with identical network/hyperparameters.
The legacy implementation class remains DSPO_plus_SPO for compatibility.
The ONLY difference is spo_loss_weight: 0.0 (ablation) vs 0.7 (DRPO).

This directly answers the reviewer question: "Is the DRPO improvement
attributable to SPO+ or to other training/implementation differences?"

Usage:
    python scripts/run_ablation_spo.py              # run all 30 seeds
    python scripts/run_ablation_spo.py --dry_run    # print commands only
    python scripts/run_ablation_spo.py --analyze    # analyze existing results
    python scripts/run_ablation_spo.py --seeds 40 67 97  # run subset
"""
import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Identical to main experiment BASE_CONFIG ──────────────────────────────────
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
    "spo_rampup_episodes": 10,
    "spo_warmup_episodes": 5,
    "spo_buffer_size": 1000,
    "spo_batch_size": 48,
    "max_episodes": 200,
    "save_count": 1,
    "batch_size": 256,
    "gpu": 0,
}

# Same 30 seeds as main experiment
SEEDS = [40, 67, 97, 52, 29, 20, 17, 88, 63, 79, 60, 62, 7, 48, 56, 15, 66, 53,
         90, 70, 24, 74, 80, 28, 2, 95, 92, 26, 39, 82]

PATTERNS = {
    "net_profit":    re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_costs":   re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "quit_rate":     re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?)%"),
    "home_delivery": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?)"),
    "served_demand": re.compile(r"Accepted customers:\s*(\d+)"),
    "total_demand":  re.compile(r"Total customers:\s*(\d+)"),
}

ROOT = Path(__file__).parent.parent


def build_command(seed, spo_weight, experiment_prefix):
    cfg = dict(BASE_CONFIG)
    experiment_id = f"{experiment_prefix}_{seed}"
    cmd = [
        sys.executable, "run.py",
        "--algo_name", "DRPO",
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
        "--spo_loss_weight", str(spo_weight),
        "--spo_rampup_episodes", str(cfg["spo_rampup_episodes"]),
        "--spo_warmup_episodes", str(cfg["spo_warmup_episodes"]),
        "--spo_buffer_size", str(cfg["spo_buffer_size"]),
        "--spo_batch_size", str(cfg["spo_batch_size"]),
        "--experiment", experiment_id,
        "--folder_suffix", "_ablation",
    ]
    return cmd, experiment_id


def find_log(experiment_id, seed, root="Experiments/Parcelpoint_py/pricing"):
    candidates = list((ROOT / root).rglob(f"*{experiment_id}*/{seed}/Logs/logfile.log"))
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
        print(f"  Warning: could not read {log_path}: {e}")
    return metrics


def run_seeds(seeds, spo_weight, label, dry_run=False):
    prefix = f"ABLATION_{label}"
    results = []
    for seed in seeds:
        cmd, exp_id = build_command(seed, spo_weight, prefix)
        log_file = find_log(exp_id, seed)
        if log_file and log_file.exists():
            m = extract_metrics(log_file)
            if "net_profit" in m and "total_costs" in m:
                print(f"  [SKIP] {label} seed={seed}")
                m["seed"] = seed
                results.append(m)
                continue
            print(f"  [RERUN] {label} seed={seed} (incomplete log)")

        print(f"  [RUN] {label} seed={seed}  spo_weight={spo_weight}")
        if dry_run:
            print("    " + " ".join(str(c) for c in cmd))
            continue

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [ERROR] seed={seed} rc={proc.returncode}")
            print(proc.stderr[-500:])
            continue

        log_file = find_log(exp_id, seed)
        if log_file:
            m = extract_metrics(log_file)
            m["seed"] = seed
            m["runtime_sec"] = elapsed
            results.append(m)
            print(f"    → net_profit={m.get('net_profit','?'):.1f}  "
                  f"total_costs={m.get('total_costs','?'):.1f}  ({elapsed:.0f}s)")
        else:
            print(f"  [WARN] log not found for seed={seed}")
    return results


def summarize(results, label):
    if not results:
        return {}
    keys = ["net_profit", "total_costs", "quit_rate", "home_delivery"]
    s = {"label": label, "n": len(results)}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = statistics.stdev(vals) if n > 1 else 0.0
        t_crit = {29: 2.045, 30: 2.042}.get(n - 1, 2.0)
        ci = t_crit * std / math.sqrt(n)
        s[f"{k}_mean"] = round(mean, 2)
        s[f"{k}_std"] = round(std, 2)
        s[f"{k}_ci"] = round(ci, 2)
    return s


def paired_analysis(ablation_results, drpo_results):
    """Paired comparison: DRPO vs DSPO-ablation per seed."""
    ablation_by_seed = {r["seed"]: r for r in ablation_results}
    drpo_by_seed = {r["seed"]: r for r in drpo_results}
    common = sorted(set(ablation_by_seed) & set(drpo_by_seed))

    profit_diffs = []
    cost_diffs = []
    drpo_wins_profit = 0
    drpo_wins_cost = 0

    for seed in common:
        a = ablation_by_seed[seed]
        d = drpo_by_seed[seed]
        dp = d.get("net_profit", 0) - a.get("net_profit", 0)
        dc = a.get("total_costs", 0) - d.get("total_costs", 0)  # positive = DRPO cheaper
        profit_diffs.append(dp)
        cost_diffs.append(dc)
        if dp > 0:
            drpo_wins_profit += 1
        if dc > 0:
            drpo_wins_cost += 1

    n = len(common)
    if n == 0:
        return {}

    def ci(vals):
        mean = sum(vals) / len(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        t = {29: 2.045, 30: 2.042}.get(len(vals) - 1, 2.0)
        return mean, std, t * std / math.sqrt(len(vals))

    pm, ps, pc = ci(profit_diffs)
    cm, cs, cc = ci(cost_diffs)

    return {
        "n_paired": n,
        "profit_diff_mean": round(pm, 2),
        "profit_diff_std": round(ps, 2),
        "profit_diff_ci": round(pc, 2),
        "profit_diff_pct": round(pm / (sum(r.get("net_profit", 1) for r in ablation_results) / len(ablation_results)) * 100, 2),
        "drpo_wins_profit": drpo_wins_profit,
        "cost_diff_mean": round(cm, 2),
        "cost_diff_std": round(cs, 2),
        "cost_diff_ci": round(cc, 2),
        "drpo_wins_cost": drpo_wins_cost,
    }


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


def load_drpo_results(seeds):
    """Load DRPO baseline results from the synthetic RC main experiment logs."""
    bases = [
        ROOT / "Experiments/Parcelpoint_py/pricing/DRPO",
        ROOT / "Experiments/Parcelpoint_py/pricing/DSPO_plus_SPO",
    ]
    preferred_folders = [
        "RC_FULL12_DSPO_VS_DRPO_DRPO_cmp",
        "RC_FULL12_DSPO_VS_DSPO_PLUS_SPO_DSPO_plus_SPO_cmp",
    ]
    results = []

    for seed in seeds:
        log = None

        for base in bases:
            for folder in preferred_folders:
                preferred = base / folder / str(seed) / "Logs" / "logfile.log"
                if preferred.exists():
                    log = preferred
                    break
            if log is not None:
                break

        if log is None:
            # Backward-compatible fallback for older naming (still RC main experiment only).
            candidates = []
            for base in bases:
                if base.exists():
                    candidates.extend(base.rglob(f"*DRPO*{seed}*/{seed}/Logs/logfile.log"))
                    candidates.extend(base.rglob(f"*DSPO_plus_SPO*{seed}*/{seed}/Logs/logfile.log"))
            candidates = sorted(candidates)
            if candidates:
                log = candidates[-1]

        if log and log.exists():
            m = extract_metrics(log)
            m["seed"] = seed
            results.append(m)
        else:
            print(f"  [WARN] DRPO log not found for seed={seed}")

    return results



def main():
    parser = argparse.ArgumentParser(description="SPO ablation experiment")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="Only analyze existing results")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS
    out_dir = ROOT / "Experiments" / "analysis" / "ablation_spo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SPO Ablation Experiment  ({len(seeds)} seeds)")
    print(f"{'='*60}")

    # ── Run DSPO-ablation (Huber-only, spo_weight=0.0) ──────────────────────
    if not args.analyze:
        print("\n[1/2] Running DSPO-ablation (Huber-only, spo_weight=0.0)...")
        ablation_results = run_seeds(seeds, spo_weight=0.0,
                                     label="DSPO_huber_only", dry_run=args.dry_run)
        if ablation_results and not args.dry_run:
            save_csv(ablation_results, out_dir / "dspo_ablation_results.csv")
    else:
        # Load from CSV if exists
        csv_path = out_dir / "dspo_ablation_results.csv"
        if csv_path.exists():
            with open(csv_path) as f:
                ablation_results = [{k: float(v) if v.replace(".", "").replace("-", "").isdigit()
                                     else v for k, v in row.items()}
                                    for row in csv.DictReader(f)]
            print(f"  Loaded {len(ablation_results)} ablation results from {csv_path}")
        else:
            print(f"  No ablation results found at {csv_path}. Run without --analyze first.")
            return

    # ── Load DRPO results ────────────────────────────────────────────────────
    print("\n[2/2] Loading DRPO results from main experiment...")
    drpo_results = load_drpo_results(seeds)
    if not drpo_results:
        print("  WARNING: Could not find DRPO results. Run main experiment first.")
        print("  Skipping paired analysis.")
    else:
        print(f"  Found {len(drpo_results)} DRPO seeds.")

    if args.dry_run:
        return

    # ── Summary statistics ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*60}")

    abl_sum = summarize(ablation_results, "DSPO-ablation (Huber-only)")
    drpo_sum = summarize(drpo_results, "DRPO (Huber+SPO+)") if drpo_results else {}

    for s in [abl_sum, drpo_sum]:
        if not s:
            continue
        print(f"\n{s['label']}  (n={s['n']})")
        pm = s.get('net_profit_mean', float('nan'))
        ps = s.get('net_profit_std', float('nan'))
        pc = s.get('net_profit_ci', float('nan'))
        cm = s.get('total_costs_mean', float('nan'))
        cs = s.get('total_costs_std', float('nan'))
        cc = s.get('total_costs_ci', float('nan'))
        qm = s.get('quit_rate_mean', float('nan'))
        hm = s.get('home_delivery_mean', float('nan'))
        print(f"  Net profit:  {pm:.2f} ± {ps:.2f}  (95% CI ±{pc:.2f})")
        print(f"  Total costs: {cm:.2f} ± {cs:.2f}  (95% CI ±{cc:.2f})")
        print(f"  Quit rate:   {qm:.2f}%")
        print(f"  Home share:  {hm*100:.1f}%")

    # ── Paired analysis ──────────────────────────────────────────────────────
    if drpo_results:
        print(f"\n{'='*60}")
        print("PAIRED ANALYSIS: DRPO vs DSPO-ablation")
        print(f"{'='*60}")
        pa = paired_analysis(ablation_results, drpo_results)
        if pa:
            n = pa["n_paired"]
            print(f"  Paired seeds: {n}")
            print(f"  Net profit gain (DRPO - ablation):")
            print(f"    Mean: {pa['profit_diff_mean']:+.2f}  ({pa['profit_diff_pct']:+.2f}%)")
            print(f"    Std:  {pa['profit_diff_std']:.2f}")
            print(f"    95% CI: ±{pa['profit_diff_ci']:.2f}")
            print(f"    DRPO wins: {pa['drpo_wins_profit']}/{n} seeds")
            print(f"  Cost reduction (ablation - DRPO, positive = DRPO cheaper):")
            print(f"    Mean: {pa['cost_diff_mean']:+.2f}")
            print(f"    DRPO cheaper: {pa['drpo_wins_cost']}/{n} seeds")

            # Save paired analysis
            with open(out_dir / "paired_analysis.json", "w") as f:
                json.dump(pa, f, indent=2)
            print(f"\n  Saved: {out_dir / 'paired_analysis.json'}")

    # ── LaTeX table snippet ──────────────────────────────────────────────────
    if drpo_results and abl_sum and drpo_sum:
        print(f"\n{'='*60}")
        print("LaTeX TABLE SNIPPET (copy into manuscript.tex)")
        print(f"{'='*60}")
        pa = paired_analysis(ablation_results, drpo_results)
        n = pa.get("n_paired", len(ablation_results))
        print(r"""
\begin{table}[t]
\centering
\caption{Ablation study: effect of SPO+ loss on the synthetic benchmark"""
              f" ({n} seeds; mean $\\pm$ std)." + r"""}
\label{tab:ablation}
\small
\begin{tabular}{lcccc}
\toprule
Method & Home-pickup (\%) & Quit rate (\%) & Total costs & Net profit \\
\midrule"""
              f"\nDSPO-ablation (Huber only) & "
              f"${abl_sum.get('home_delivery_mean',0)*100:.2f}\\pm{abl_sum.get('home_delivery_std',0)*100:.2f}$ & "
              f"${abl_sum.get('quit_rate_mean',0):.2f}\\pm{abl_sum.get('quit_rate_std',0):.2f}$ & "
              f"${abl_sum.get('total_costs_mean',0):.2f}\\pm{abl_sum.get('total_costs_std',0):.2f}$ & "
              f"${abl_sum.get('net_profit_mean',0):.2f}\\pm{abl_sum.get('net_profit_std',0):.2f}$ \\\\"
              f"\nDRPO (Huber + SPO+) & "
              f"${drpo_sum.get('home_delivery_mean',0)*100:.2f}\\pm{drpo_sum.get('home_delivery_std',0)*100:.2f}$ & "
              f"${drpo_sum.get('quit_rate_mean',0):.2f}\\pm{drpo_sum.get('quit_rate_std',0):.2f}$ & "
              f"${drpo_sum.get('total_costs_mean',0):.2f}\\pm{drpo_sum.get('total_costs_std',0):.2f}$ & "
              f"${drpo_sum.get('net_profit_mean',0):.2f}\\pm{drpo_sum.get('net_profit_std',0):.2f}$ \\\\"
              + r"""
\bottomrule
\end{tabular}
\end{table}""")

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
