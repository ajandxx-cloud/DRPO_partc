#!/usr/bin/env python
"""Run a paired Yanjiao DSPO/DRPO grid_dim sweep.

The script is orchestration-only: each grid_dim is evaluated by
run_yanjiao_experiments.py with matched DSPO and DRPO runs, then the paired
DRPO-DSPO deltas are summarized across seeds.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
METRICS = [
    "net_profit",
    "total_costs",
    "quit_rate",
    "served_demand",
    "served_rate",
    "home_pickup_rate",
    "travel_costs",
    "service_costs",
    "failure_costs",
    "avg_charge",
    "avg_discount",
    "charge_revenue",
    "discount_costs",
    "base_revenue",
]
SPO_WEIGHT_REGEX = "[SPO+ debug] spo_weight became positive:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep CNN grid_dim for Yanjiao DRPO vs DSPO.")
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--grid_dims", nargs="+", type=int, default=[11, 15, 21])
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 67, 97])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    parser.add_argument("--n_passengers", type=int, default=400)
    parser.add_argument("--hgs_reopt_time", type=float, default=0.2)
    parser.add_argument("--hgs_final_time", type=float, default=0.2)
    parser.add_argument("--spo_loss_weight", type=float, default=0.05)
    parser.add_argument("--spo_batch_size", type=int, default=4)
    parser.add_argument("--spo_warmup", type=int, default=5)
    parser.add_argument("--spo_rampup", type=int, default=10)
    parser.add_argument("--initial_phase_epochs", type=int, default=50)
    parser.add_argument("--buffer_size", type=int, default=500)
    parser.add_argument("--run_timeout_sec", type=int, default=7200)
    parser.add_argument("--max_retries", type=int, default=0)
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    return parser.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def parse_spo_health(log_path: str, label: str) -> Dict[str, Any]:
    log = Path(log_path)
    args_yaml = log.parents[2] / "args.yaml" if len(log.parents) >= 3 else None
    args_text = ""
    if args_yaml is not None and args_yaml.exists():
        args_text = args_yaml.read_text(encoding="utf-8", errors="ignore")
    if not log.exists():
        return {
            f"{label}_log_exists": False,
            f"{label}_cuda_used": False,
            f"{label}_spo_loss_weight_zero": "spo_loss_weight: 0.0" in args_text,
            f"{label}_spo_training_data_populated": False,
            f"{label}_spo_weight_positive": False,
            f"{label}_spo_warning_count": "",
        }
    text = log.read_text(encoding="utf-8", errors="ignore")
    if label == "dspo":
        positive = SPO_WEIGHT_REGEX in text
        populated = "[SPO+ debug] spo_training_data populated" in text
    else:
        positive = SPO_WEIGHT_REGEX in text
        populated = "[SPO+ debug] spo_training_data populated" in text
    return {
        f"{label}_log_exists": True,
        f"{label}_cuda_used": "Using GPU device: cuda" in text,
        f"{label}_spo_loss_weight_zero": "spo_loss_weight: 0.0" in args_text,
        f"{label}_spo_training_data_populated": populated,
        f"{label}_spo_weight_positive": positive,
        f"{label}_spo_warning_count": text.count("[SPO+ warning]"),
    }


def build_grid_cmd(args: argparse.Namespace, output_dir: Path, grid_dim: int) -> List[str]:
    cmd = [
        args.python_executable,
        "scripts/run_yanjiao_experiments.py",
        "--python_executable",
        args.python_executable,
        "--gpu",
        str(args.gpu),
        "--phase",
        "main",
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        args.route_label_mode,
        "--strategies",
        "DSPO",
        "DRPO",
        "--run_prefix",
        f"YJ_GRID{grid_dim}",
        "--folder_suffix",
        f"_yj_grid{grid_dim}",
        "--output_dir",
        str(output_dir / f"grid_{grid_dim}"),
        "--allow_existing_output_dir",
        "--persist_every_n",
        "1",
        "--max_retries",
        str(args.max_retries),
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--dspo_spo_loss_weight",
        "0.0",
        "--drpo_spo_loss_weight",
        repr(float(args.spo_loss_weight)),
        "--n_passengers_override",
        str(int(args.n_passengers)),
        "--grid_dim_override",
        str(int(grid_dim)),
        "--hgs_reopt_time_override",
        repr(float(args.hgs_reopt_time)),
        "--hgs_final_time_override",
        repr(float(args.hgs_final_time)),
        "--initial_phase_epochs_override",
        str(int(args.initial_phase_epochs)),
        "--buffer_size_override",
        str(int(args.buffer_size)),
        "--spo_warmup_episodes_override",
        str(int(args.spo_warmup)),
        "--spo_rampup_episodes_override",
        str(int(args.spo_rampup)),
        "--spo_batch_size_override",
        str(int(args.spo_batch_size)),
    ]
    if args.allow_cpu:
        cmd.append("--allow_cpu")
    if not args.skip_existing:
        cmd.append("--no_skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def run_grid(args: argparse.Namespace, output_dir: Path, grid_dim: int) -> int:
    grid_dir = output_dir / f"grid_{grid_dim}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_grid_cmd(args, output_dir, grid_dim)
    (grid_dir / "grid_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    print(f"[RUN] grid_dim={grid_dim}", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    t0 = time.time()
    cp = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    (grid_dir / "grid_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
    print(f"[DONE] grid_dim={grid_dim} rc={cp.returncode} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    return cp.returncode


def paired_rows_for_grid(args: argparse.Namespace, output_dir: Path, grid_dim: int) -> List[Dict[str, Any]]:
    raw_path = output_dir / f"grid_{grid_dim}" / "yanjiao_raw.csv"
    rows = read_csv(raw_path)
    keyed = {
        (row.get("label"), int(float(row.get("seed", -1)))): row
        for row in rows
        if row.get("label") in {"DSPO", "DRPO"}
    }
    out: List[Dict[str, Any]] = []
    for seed in args.seeds:
        dspo = keyed.get(("DSPO", seed))
        drpo = keyed.get(("DRPO", seed))
        if dspo is None or drpo is None:
            continue
        row: Dict[str, Any] = {
            "grid_dim": int(grid_dim),
            "seed": int(seed),
            "n_passengers": int(args.n_passengers),
            "episodes": int(args.episodes),
            "dspo_status": dspo.get("status", ""),
            "drpo_status": drpo.get("status", ""),
            "dspo_runtime_sec": dspo.get("runtime_sec", ""),
            "drpo_runtime_sec": drpo.get("runtime_sec", ""),
            "dspo_log_path": dspo.get("log_path", ""),
            "drpo_log_path": drpo.get("log_path", ""),
        }
        for metric in METRICS:
            dspo_val = to_float(dspo.get(metric))
            drpo_val = to_float(drpo.get(metric))
            row[f"dspo_{metric}"] = dspo_val if dspo_val is not None else ""
            row[f"drpo_{metric}"] = drpo_val if drpo_val is not None else ""
            row[f"delta_{metric}"] = (
                drpo_val - dspo_val if dspo_val is not None and drpo_val is not None else ""
            )
        row.update(parse_spo_health(dspo.get("log_path", ""), "dspo"))
        row.update(parse_spo_health(drpo.get("log_path", ""), "drpo"))
        out.append(row)
    return out


def summarize(raw_rows: List[Dict[str, Any]], n_seeds: int) -> List[Dict[str, Any]]:
    by_grid: Dict[int, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        by_grid.setdefault(int(row["grid_dim"]), []).append(row)
    summary: List[Dict[str, Any]] = []
    for grid_dim, rows in sorted(by_grid.items()):
        item: Dict[str, Any] = {
            "grid_dim": grid_dim,
            "n_runs": len(rows),
            "complete": len(rows) == n_seeds,
            "wins_net_profit": sum((to_float(r.get("delta_net_profit")) or -1e18) > 0 for r in rows),
            "drpo_spo_ok_all_runs": all(
                str(r.get("drpo_spo_weight_positive")).lower() == "true"
                and str(r.get("drpo_spo_training_data_populated")).lower() == "true"
                for r in rows
            ),
            "dspo_spo_disabled_all_runs": all(
                str(r.get("dspo_spo_loss_weight_zero")).lower() == "true"
                and str(r.get("dspo_spo_weight_positive")).lower() != "true"
                for r in rows
            ),
            "spo_warning_count_total": sum(int(float(r.get("drpo_spo_warning_count") or 0)) for r in rows),
        }
        for metric in METRICS:
            vals = [to_float(r.get(f"delta_{metric}")) for r in rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                item[f"mean_delta_{metric}"] = ""
                item[f"std_delta_{metric}"] = ""
                item[f"min_delta_{metric}"] = ""
                item[f"max_delta_{metric}"] = ""
                continue
            item[f"mean_delta_{metric}"] = mean(vals)
            item[f"std_delta_{metric}"] = std(vals)
            item[f"min_delta_{metric}"] = min(vals)
            item[f"max_delta_{metric}"] = max(vals)
        item.update(score_grid(item, n_seeds))
        summary.append(item)
    return summary


def score_grid(row: Dict[str, Any], n_seeds: int) -> Dict[str, Any]:
    mean_profit = to_float(row.get("mean_delta_net_profit")) or -1e9
    std_profit = to_float(row.get("std_delta_net_profit")) or 0.0
    wins = int(float(row.get("wins_net_profit") or 0))
    mean_quit = abs(to_float(row.get("mean_delta_quit_rate")) or 0.0)
    warnings = int(float(row.get("spo_warning_count_total") or 0))
    drpo_ok = str(row.get("drpo_spo_ok_all_runs")).lower() == "true"
    dspo_disabled = str(row.get("dspo_spo_disabled_all_runs")).lower() == "true"
    complete = str(row.get("complete")).lower() == "true"
    score = mean_profit + 5.0 * wins - 0.25 * std_profit - 20.0 * mean_quit
    if wins < max(1, math.ceil(n_seeds / 2)):
        score -= 25.0
    if not complete:
        score -= 100.0
    if not drpo_ok:
        score -= 1000.0
    if not dspo_disabled:
        score -= 1000.0
    score -= 20.0 * warnings
    return {
        "score": score,
        "mean_abs_delta_quit_rate": mean_quit,
    }


def rank_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r.get("score") or -1e9),
            int(float(r.get("wins_net_profit") or 0)),
            float(r.get("mean_delta_net_profit") or -1e9),
            -float(r.get("std_delta_net_profit") or 1e9),
        ),
        reverse=True,
    )


def persist_analysis(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []
    for grid_dim in args.grid_dims:
        raw_rows.extend(paired_rows_for_grid(args, output_dir, int(grid_dim)))
    if raw_rows:
        write_csv(output_dir / "grid_dim_raw.csv", raw_rows)
    summary = summarize(raw_rows, len(args.seeds))
    if summary:
        ranked = rank_summary(summary)
        write_csv(output_dir / "grid_dim_summary.csv", ranked)
        (output_dir / "selected_grid_dim.json").write_text(
            json.dumps(ranked[0], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return ranked
    return []


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(
        args.output_dir
        or f"Experiments/analysis/yanjiao_drpo_grid_dim_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "selection_rule": "score = mean_delta_net_profit + 5*wins - 0.25*std - 20*abs(mean_delta_quit_rate), with penalties for low wins, incomplete runs, SPO health, and warnings",
        "expected_jobs": len(args.grid_dims) * len(args.seeds) * 2,
    }
    (output_dir / "grid_sweep_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.analyze_only:
        for grid_dim in args.grid_dims:
            grid_dir = output_dir / f"grid_{grid_dim}"
            raw_path = grid_dir / "yanjiao_raw.csv"
            rows = read_csv(raw_path)
            expected_rows = len(args.seeds) * 2
            if args.skip_existing and len(rows) >= expected_rows:
                print(f"[CACHE] grid_dim={grid_dim}", flush=True)
            else:
                rc = run_grid(args, output_dir, int(grid_dim))
                if rc != 0:
                    raise RuntimeError(f"grid_dim={grid_dim} failed with return code {rc}")
            ranked = persist_analysis(args, output_dir)
            if ranked:
                print(
                    f"[BEST SO FAR] grid_dim={ranked[0]['grid_dim']} "
                    f"score={ranked[0]['score']:.3f} "
                    f"mean_delta={ranked[0].get('mean_delta_net_profit')}",
                    flush=True,
                )
    else:
        ranked = persist_analysis(args, output_dir)
        if not ranked:
            raise RuntimeError(f"No paired grid results found under {output_dir}")

    print(f"[DONE] raw={output_dir / 'grid_dim_raw.csv'}", flush=True)
    print(f"[DONE] summary={output_dir / 'grid_dim_summary.csv'}", flush=True)
    print(f"[DONE] selected={output_dir / 'selected_grid_dim.json'}", flush=True)


if __name__ == "__main__":
    main()
