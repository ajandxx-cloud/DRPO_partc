#!/usr/bin/env python
"""
Generate paper-ready tables from Yanjiao experiment results.

Usage:
  python scripts/generate_yanjiao_table.py                               # auto-find latest results
  python scripts/generate_yanjiao_table.py --input Experiments/analysis/yanjiao_full_20260510_170333
  python scripts/generate_yanjiao_table.py --input yanjiao_raw.csv --main_scale 400
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser(description="Generate Yanjiao experiment tables")
    p.add_argument("--input", required=True, help="Path to yanjiao_raw.csv or analysis directory")
    p.add_argument("--main_scale", type=int, default=400, help="Main experiment scale")
    p.add_argument("--output_dir", default=None, help="Output directory (default: same as input)")
    return p.parse_args()


def to_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def mean_std(vals: List[float]) -> tuple:
    if not vals:
        return float("nan"), float("nan")
    n = len(vals)
    m = sum(vals) / n
    if n <= 1:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, s


def fmt(val: float, decimals: int = 2) -> str:
    if math.isnan(val):
        return "---"
    return f"{val:.{decimals}f}"


def load_raw(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def group_by(rows: List[Dict], key_fn) -> Dict[Any, List[Dict]]:
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    return dict(groups)


def compute_summary(rows: List[Dict], metrics: List[str]) -> Dict[str, Any]:
    result = {"n_runs": len(rows)}
    for m in metrics:
        vals = [to_float(r.get(m)) for r in rows]
        vals = [v for v in vals if v is not None]
        m_val, s_val = mean_std(vals)
        result[f"{m}_mean"] = m_val
        result[f"{m}_std"] = s_val
    return result


def paired_comparison(rows: List[Dict], label_a: str, label_b: str,
                      n_passengers: int, metrics: List[str]) -> Dict[str, Any]:
    """Compute per-seed paired deltas between two strategies."""
    by_seed = defaultdict(dict)
    for r in rows:
        np_val = int(float(r["n_passengers"]))
        seed = int(float(r["seed"]))
        label = str(r["label"])
        if np_val == n_passengers and label in (label_a, label_b):
            by_seed[seed][label] = r

    deltas = defaultdict(list)
    win_count = defaultdict(int)
    total_pairs = 0

    for seed, strat_map in by_seed.items():
        if label_a not in strat_map or label_b not in strat_map:
            continue
        total_pairs += 1
        for m in metrics:
            a = to_float(strat_map[label_a].get(m))
            b = to_float(strat_map[label_b].get(m))
            if a is not None and b is not None:
                deltas[m].append(b - a)
                if (b - a) > 0:
                    win_count[m] += 1

    result = {"total_pairs": total_pairs}
    for m in metrics:
        if deltas[m]:
            d_mean, d_std = mean_std(deltas[m])
            result[f"{m}_delta_mean"] = d_mean
            result[f"{m}_delta_std"] = d_std
            result[f"{m}_win_rate"] = f"{win_count[m]}/{total_pairs} ({100*win_count[m]/total_pairs:.0f}%)"
        else:
            result[f"{m}_delta_mean"] = float("nan")
            result[f"{m}_delta_std"] = float("nan")
            result[f"{m}_win_rate"] = "---"

    return result


def print_main_table(summaries: Dict[str, Dict], main_scale: int):
    """Print the main results table matching paper format."""
    print(f"\n{'='*90}")
    print(f"Table: Beijing Yanjiao ({main_scale} passengers, 30 seeds)")
    print(f"{'='*90}")
    header = f"{'Strategy':<20} {'Home-pickup%':>14} {'MP%':>14} {'Quit rate%':>14} {'Total costs':>14} {'Net profit':>14}"
    print(header)
    print("-" * 90)

    for label in ["Static", "DSPO", "DRPO"]:
        s = summaries.get(label, {})
        n = s.get("n_runs", 0)
        hp_m = s.get("home_pickup_rate_mean", float("nan"))
        hp_s = s.get("home_pickup_rate_std", float("nan"))
        qr_m = s.get("quit_rate_mean", float("nan"))
        qr_s = s.get("quit_rate_std", float("nan"))
        tc_m = s.get("total_costs_mean", float("nan"))
        tc_s = s.get("total_costs_std", float("nan"))
        np_m = s.get("net_profit_mean", float("nan"))
        np_s = s.get("net_profit_std", float("nan"))

        mp_m = 100 - hp_m * 100 - qr_m if not any(math.isnan(v) for v in [hp_m, qr_m]) else float("nan")

        print(f"{label:<20} {hp_m*100:6.2f} +/- {hp_s*100:5.2f}  "
              f"{mp_m:6.2f}          "
              f"{qr_m:6.2f} +/- {qr_s:5.2f}  "
              f"{tc_m:8.2f} +/- {tc_s:6.2f}  "
              f"{np_m:8.2f} +/- {np_s:6.2f}")

    print(f"{'='*90}")


def print_sensitivity_table(summaries: Dict[tuple, Dict], scales: List[int]):
    """Print sensitivity comparison table."""
    print(f"\n{'='*100}")
    print("Table: Sensitivity across scales (DSPO vs DRPO)")
    print(f"{'='*100}")
    header = f"{'Scale':>8} {'Strategy':>10} {'Net profit':>14} {'Total costs':>14} {'Quit rate%':>14} {'Home-pickup%':>14}"
    print(header)
    print("-" * 100)

    for np_val in scales:
        for label in ["DSPO", "DRPO"]:
            s = summaries.get((label, np_val), {})
            np_m = s.get("net_profit_mean", float("nan"))
            tc_m = s.get("total_costs_mean", float("nan"))
            qr_m = s.get("quit_rate_mean", float("nan"))
            hp_m = s.get("home_pickup_rate_mean", float("nan"))

            print(f"{np_val:>8} {label:>10} {np_m:14.2f} {tc_m:14.2f} "
                  f"{qr_m:14.2f} {hp_m*100 if not math.isnan(hp_m) else 0:14.2f}")
        print()

    print(f"{'='*100}")


def print_paired_table(paired_results: Dict[tuple, Dict], metrics: List[str]):
    """Print paired comparison results."""
    print(f"\n{'='*80}")
    print("Paired Comparison: DRPO vs DSPO")
    print(f"{'='*80}")
    header = f"{'Scale':>8} {'Metric':>16} {'Delta mean':>14} {'Delta std':>14} {'Win rate':>20}"
    print(header)
    print("-" * 80)

    for key, result in sorted(paired_results.items()):
        if isinstance(key, tuple):
            scale = key[1] if len(key) > 1 else key[0]
        else:
            scale = key
        for m in metrics:
            d_mean = result.get(f"{m}_delta_mean", float("nan"))
            d_std = result.get(f"{m}_delta_std", float("nan"))
            wr = result.get(f"{m}_win_rate", "---")
            if not math.isnan(d_mean):
                print(f"{scale:>8} {m:>16} {d_mean:14.4f} {d_std:14.4f} {wr:>20}")

    print(f"{'='*80}")


def save_latex_table(summaries: Dict, main_scale: int, output_dir: Path):
    """Save results as a simple LaTeX-formatted file."""
    path = output_dir / "yanjiao_table.tex"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Beijing Yanjiao case study results (400 passengers, 30 seeds)}",
        r"\label{tab:yanjiao_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Strategy & Home-pickup (\%) & Meeting-point (\%) & Quit rate (\%) & Net profit \\",
        r"\midrule",
    ]

    for label in ["Static", "DSPO", "DRPO"]:
        s = summaries.get(label, {})
        hp_m = s.get("home_pickup_rate_mean", 0) * 100
        hp_s = s.get("home_pickup_rate_std", 0) * 100
        qr_m = s.get("quit_rate_mean", 0)
        qr_s = s.get("quit_rate_std", 0)
        np_m = s.get("net_profit_mean", 0)
        np_s = s.get("net_profit_std", 0)
        mp = 100 - hp_m - qr_m

        lines.append(
            f"{label} & ${hp_m:.1f} \\pm {hp_s:.1f}$ & "
            f"${mp:.1f}$ & "
            f"${qr_m:.2f} \\pm {qr_s:.2f}$ & "
            f"${np_m:.1f} \\pm {np_s:.1f}$ \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaTeX table saved to: {path}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path

    # Resolve raw CSV
    if input_path.is_dir():
        raw_csv = input_path / "yanjiao_raw.csv"
    elif input_path.suffix == ".csv":
        raw_csv = input_path
        if output_dir == input_path:
            output_dir = input_path.parent
    else:
        print(f"ERROR: Invalid input: {input_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_raw(raw_csv)
    print(f"Loaded {len(rows)} rows from {raw_csv}")

    if not rows:
        print("No data to analyze.")
        sys.exit(1)

    # Report unique configurations
    labels = sorted({str(r["label"]) for r in rows})
    scales = sorted({int(float(r["n_passengers"])) for r in rows})
    seeds = sorted({int(float(r["seed"])) for r in rows})
    statuses = {str(r.get("status", "?")) for r in rows}
    print(f"Strategies: {labels}")
    print(f"Scales: {scales}")
    print(f"Seeds: {len(seeds)} unique ({seeds[0]}..{seeds[-1]})")
    print(f"Statuses: {statuses}")

    completed = [r for r in rows if str(r.get("status")) in ("completed", "cached")]
    print(f"Completed: {len(completed)}/{len(rows)}")

    # Key metrics
    metrics = ["net_profit", "total_costs", "quit_rate", "home_pickup_rate",
               "travel_costs", "service_costs", "failure_costs"]

    # ─── Main table (400 passengers) ───
    main_summaries = {}
    for label in labels:
        main_rows = [r for r in completed
                     if str(r["label"]) == label and int(float(r["n_passengers"])) == args.main_scale]
        if main_rows:
            main_summaries[label] = compute_summary(main_rows, metrics)

    if main_summaries:
        print_main_table(main_summaries, args.main_scale)

    # ─── Sensitivity table ───
    sens_scales = [s for s in scales if s != args.main_scale]
    if sens_scales:
        all_sens_summaries = {}
        for label in ["DSPO", "DRPO"]:
            for np_val in scales:
                s_rows = [r for r in completed
                          if str(r["label"]) == label and int(float(r["n_passengers"])) == np_val]
                if s_rows:
                    all_sens_summaries[(label, np_val)] = compute_summary(s_rows, metrics)

        print_sensitivity_table(all_sens_summaries, scales)

    # ─── Paired comparison ───
    paired_results = {}
    for np_val in scales:
        # Only compute paired if both DSPO and DRPO exist
        dspo_rows = [r for r in completed
                     if str(r["label"]) == "DSPO" and int(float(r["n_passengers"])) == np_val]
        drpo_rows = [r for r in completed
                     if str(r["label"]) == "DRPO" and int(float(r["n_passengers"])) == np_val]
        if dspo_rows and drpo_rows:
            paired_results[("DSPO_vs_DRPO", np_val)] = paired_comparison(
                completed, "DSPO", "DRPO", np_val, metrics)

    if paired_results:
        print_paired_table(paired_results, ["net_profit", "total_costs", "quit_rate"])

    # ─── Save outputs ───
    if main_summaries:
        save_latex_table(main_summaries, args.main_scale, output_dir)

    # Save summary CSV
    summary_rows = []
    for label in labels:
        for np_val in scales:
            s_rows = [r for r in completed
                      if str(r["label"]) == label and int(float(r["n_passengers"])) == np_val]
            if s_rows:
                s = compute_summary(s_rows, metrics)
                s["label"] = label
                s["n_passengers"] = np_val
                summary_rows.append(s)

    if summary_rows:
        out_csv = output_dir / "yanjiao_table_summary.csv"
        fields = ["label", "n_passengers", "n_runs"]
        for m in metrics:
            fields.extend([f"{m}_mean", f"{m}_std"])
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\nSummary CSV: {out_csv}")


if __name__ == "__main__":
    main()
