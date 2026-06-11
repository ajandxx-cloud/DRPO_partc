#!/usr/bin/env python
"""Yanjiao DRPO SPO-weight sweep.

This runner is intentionally orchestration-only: it reuses the existing
Yanjiao experiment runner for candidate runs, then compares each DRPO weight
against either an existing DSPO baseline CSV or a freshly matched DSPO baseline
run with the same fast-screening parameters.
"""

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_RAW = (
    "Experiments/analysis/yanjiao_drpo_w050_ep30_20260517_195254/yanjiao_raw.csv"
)
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
SPO_WEIGHT_REGEX = re.compile(
    r"\[SPO\+ debug\] spo_weight became positive:\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep DRPO SPO weights on Beijing_Yanjiao and compare with an existing DSPO baseline."
    )
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[40, 67, 97])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    parser.add_argument("--n_passengers", type=int, default=400)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--revenue", type=float, default=None)
    parser.add_argument("--home_util", type=float, default=None)
    parser.add_argument("--outside_option_util", type=float, default=None)
    parser.add_argument("--max_price", type=float, default=None)
    parser.add_argument("--min_price", type=float, default=None)
    parser.add_argument("--incentive_sens", type=float, default=None)
    parser.add_argument("--hgs_reopt_time", type=float, default=None)
    parser.add_argument("--hgs_final_time", type=float, default=None)
    parser.add_argument("--initial_phase_epochs", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--spo_warmup", type=int, default=None)
    parser.add_argument("--spo_rampup", type=int, default=None)
    parser.add_argument("--spo_label_sample_size", type=int, default=None)
    parser.add_argument("--spo_batch_size", type=int, default=None)
    parser.add_argument("--baseline_raw", default=DEFAULT_BASELINE_RAW)
    parser.add_argument(
        "--run_matched_baseline",
        action="store_true",
        help="Run DSPO together with each DRPO candidate instead of using --baseline_raw.",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_prefix", default="YJ_DRPO_LIGHT_SPO")
    parser.add_argument("--folder_suffix", default="_yj_light_spo")
    parser.add_argument("--persist_every_n", type=int, default=1)
    parser.add_argument("--run_timeout_sec", type=int, default=0)
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
    return p if p.is_absolute() else (ROOT / p)


def weight_tag(weight: float) -> str:
    return f"w{int(round(weight * 1000)):04d}"


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
        seen = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
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
        return float(text)
    except ValueError:
        return None


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def load_dspo_baseline(path: Path, n_passengers: int, seeds: Sequence[int]) -> Dict[int, Dict[str, str]]:
    rows = read_csv(path)
    out: Dict[int, Dict[str, str]] = {}
    seed_set = set(seeds)
    for row in rows:
        if row.get("label") != "DSPO":
            continue
        seed = int(float(row["seed"]))
        if seed not in seed_set:
            continue
        if int(float(row.get("n_passengers", 0))) != n_passengers:
            continue
        out[seed] = row
    missing = sorted(seed_set - set(out))
    if missing:
        raise RuntimeError(f"DSPO baseline missing seeds {missing} in {path}")
    return out


def candidate_dspo_baseline(rows: List[Dict[str, str]], n_passengers: int) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if row.get("label") != "DSPO":
            continue
        if int(float(row.get("n_passengers", 0))) != n_passengers:
            continue
        out[int(float(row["seed"]))] = row
    return out


def parse_spo_health(log_path: str) -> Dict[str, Any]:
    log = Path(log_path)
    if not log.exists():
        return {
            "drpo_log_exists": False,
            "cuda_used": False,
            "spo_training_data_populated": False,
            "spo_weight_positive": False,
            "first_positive_spo_weight": "",
            "spo_warning_count": "",
        }
    text = log.read_text(encoding="utf-8", errors="ignore")
    weights = [float(x) for x in SPO_WEIGHT_REGEX.findall(text)]
    return {
        "drpo_log_exists": True,
        "cuda_used": "Using GPU device: cuda" in text,
        "spo_training_data_populated": "[SPO+ debug] spo_training_data populated" in text,
        "spo_weight_positive": any(w > 0 for w in weights),
        "first_positive_spo_weight": weights[0] if weights else "",
        "spo_warning_count": text.count("[SPO+ warning]"),
    }


def build_runner_cmd(
    args: argparse.Namespace,
    candidate_dir: Path,
    tag: str,
    weight: float,
    strategies: Optional[Sequence[str]] = None,
) -> List[str]:
    if strategies is None:
        strategies = ["DRPO"]
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
        str(args.route_label_mode),
        "--strategies",
        *strategies,
        "--run_prefix",
        f"{args.run_prefix}_{tag}",
        "--folder_suffix",
        f"{args.folder_suffix}_{tag}",
        "--output_dir",
        str(candidate_dir),
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
        repr(float(weight)),
        "--n_passengers_override",
        str(int(args.n_passengers)),
    ]
    if args.hgs_reopt_time is not None:
        cmd.extend(["--hgs_reopt_time_override", repr(float(args.hgs_reopt_time))])
    if args.hgs_final_time is not None:
        cmd.extend(["--hgs_final_time_override", repr(float(args.hgs_final_time))])
    if args.k is not None:
        cmd.extend(["--k_override", str(int(args.k))])
    if args.revenue is not None:
        cmd.extend(["--revenue_override", repr(float(args.revenue))])
    if args.home_util is not None:
        cmd.extend(["--home_util_override", repr(float(args.home_util))])
    if args.outside_option_util is not None:
        cmd.extend(["--outside_option_util_override", repr(float(args.outside_option_util))])
    if args.max_price is not None:
        cmd.extend(["--max_price_override", repr(float(args.max_price))])
    if args.min_price is not None:
        cmd.extend(["--min_price_override", repr(float(args.min_price))])
    if args.incentive_sens is not None:
        cmd.extend(["--incentive_sens_override", repr(float(args.incentive_sens))])
    if args.initial_phase_epochs is not None:
        cmd.extend(["--initial_phase_epochs_override", str(int(args.initial_phase_epochs))])
    if args.buffer_size is not None:
        cmd.extend(["--buffer_size_override", str(int(args.buffer_size))])
    if args.spo_warmup is not None:
        cmd.extend(["--spo_warmup_episodes_override", str(int(args.spo_warmup))])
    if args.spo_rampup is not None:
        cmd.extend(["--spo_rampup_episodes_override", str(int(args.spo_rampup))])
    if args.spo_label_sample_size is not None:
        cmd.extend(["--spo_label_sample_size_override", str(int(args.spo_label_sample_size))])
    if args.spo_batch_size is not None:
        cmd.extend(["--spo_batch_size_override", str(int(args.spo_batch_size))])
    if args.allow_cpu:
        cmd.append("--allow_cpu")
    if not args.skip_existing:
        cmd.append("--no_skip_existing")
    return cmd


def collect_candidate_rows(
    args: argparse.Namespace,
    output_dir: Path,
    baseline: Optional[Dict[int, Dict[str, str]]],
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for weight in args.weights:
        tag = weight_tag(weight)
        raw_path = output_dir / f"candidate_{tag}" / "yanjiao_raw.csv"
        rows = read_csv(raw_path)
        candidate_baseline = baseline or candidate_dspo_baseline(rows, args.n_passengers)
        for drpo in rows:
            if drpo.get("label") != "DRPO":
                continue
            seed = int(float(drpo["seed"]))
            if seed not in candidate_baseline:
                continue
            dspo = candidate_baseline[seed]
            row: Dict[str, Any] = {
                "candidate": tag,
                "spo_loss_weight": float(weight),
                "seed": seed,
                "n_passengers": int(float(drpo["n_passengers"])),
                "episodes": int(float(drpo["episodes"])),
                "dspo_status": dspo.get("status", ""),
                "dspo_runtime_sec": dspo.get("runtime_sec", ""),
                "dspo_log_path": dspo.get("log_path", ""),
                "drpo_status": drpo.get("status", ""),
                "drpo_runtime_sec": drpo.get("runtime_sec", ""),
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
            row.update(parse_spo_health(drpo.get("log_path", "")))
            all_rows.append(row)
    all_rows.sort(key=lambda r: (float(r["spo_loss_weight"]), int(r["seed"])))
    return all_rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)

    summary: List[Dict[str, Any]] = []
    for candidate, group in sorted(by_candidate.items(), key=lambda item: float(item[1][0]["spo_loss_weight"])):
        item: Dict[str, Any] = {
            "candidate": candidate,
            "spo_loss_weight": group[0]["spo_loss_weight"],
            "n_runs": len(group),
            "wins_net_profit": sum(float(r["delta_net_profit"]) > 0 for r in group),
            "spo_ok_all_runs": all(
                str(r.get("spo_weight_positive")).lower() == "true"
                and str(r.get("spo_training_data_populated")).lower() == "true"
                for r in group
            ),
            "spo_warning_count_total": sum(int(float(r.get("spo_warning_count") or 0)) for r in group),
        }
        for metric in METRICS:
            vals = [to_float(r.get(f"delta_{metric}")) for r in group]
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
        summary.append(item)
    return summary


def persist_analysis(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    write_csv(output_dir / "candidate_raw.csv", rows)
    summary = summarize(rows)
    if summary:
        write_csv(output_dir / "candidate_summary.csv", summary)
        selected = sorted(
            summary,
            key=lambda r: (
                float(r.get("mean_delta_net_profit") or -1e18),
                -float(r.get("mean_delta_discount_costs") or 1e18),
            ),
            reverse=True,
        )[0]
        (output_dir / "selected_config.json").write_text(
            json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(
        args.output_dir
        or f"Experiments/analysis/yanjiao_spo_weight_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_matched_baseline:
        baseline_path = output_dir / "matched_baseline" / "yanjiao_raw.csv"
        baseline = (
            load_dspo_baseline(baseline_path, args.n_passengers, args.seeds)
            if baseline_path.exists()
            else None
        )
    else:
        baseline_path = resolve_path(args.baseline_raw)
        baseline = load_dspo_baseline(baseline_path, args.n_passengers, args.seeds)

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": args.seeds,
        "weights": args.weights,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "route_label_mode": args.route_label_mode,
        "n_passengers": args.n_passengers,
        "hgs_reopt_time": args.hgs_reopt_time,
        "hgs_final_time": args.hgs_final_time,
        "initial_phase_epochs": args.initial_phase_epochs,
        "buffer_size": args.buffer_size,
        "spo_warmup": args.spo_warmup,
        "spo_rampup": args.spo_rampup,
        "spo_label_sample_size": args.spo_label_sample_size,
        "spo_batch_size": args.spo_batch_size,
        "baseline_mode": "matched_run" if args.run_matched_baseline else "existing_csv",
        "baseline_raw": "" if args.run_matched_baseline else str(baseline_path),
        "run_prefix": args.run_prefix,
        "folder_suffix": args.folder_suffix,
        "selection_rule": "highest mean_delta_net_profit, tie-break lower mean_delta_discount_costs",
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.dry_run:
        if args.run_matched_baseline:
            baseline_dir = output_dir / "matched_baseline"
            print(
                " ".join(build_runner_cmd(args, baseline_dir, "dspo_base", 0.0, ["DSPO"])),
                flush=True,
            )
        for weight in args.weights:
            tag = weight_tag(weight)
            candidate_dir = output_dir / f"candidate_{tag}"
            print(" ".join(build_runner_cmd(args, candidate_dir, tag, weight)), flush=True)
        print(f"[DRY-RUN] output_dir={output_dir}", flush=True)
        return

    if not args.analyze_only:
        if args.run_matched_baseline:
            baseline_dir = output_dir / "matched_baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            if baseline is None:
                cmd = build_runner_cmd(args, baseline_dir, "dspo_base", 0.0, ["DSPO"])
                print("[RUN] matched DSPO baseline", flush=True)
                cp = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                (baseline_dir / "runner_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
                if cp.returncode != 0:
                    raise RuntimeError(
                        f"Yanjiao runner failed for matched DSPO baseline with code {cp.returncode}. "
                        f"See {baseline_dir / 'runner_stdout.log'}"
                    )
                baseline = load_dspo_baseline(baseline_path, args.n_passengers, args.seeds)
                print("[DONE] matched DSPO baseline", flush=True)
            else:
                print("[CACHE] matched DSPO baseline", flush=True)
        for weight in args.weights:
            tag = weight_tag(weight)
            candidate_dir = output_dir / f"candidate_{tag}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_runner_cmd(args, candidate_dir, tag, weight)
            print(f"[RUN] {tag} spo_loss_weight={weight}", flush=True)
            cp = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            (candidate_dir / "runner_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
            rows = collect_candidate_rows(args, output_dir, baseline)
            persist_analysis(output_dir, rows)
            if cp.returncode != 0:
                raise RuntimeError(
                    f"Yanjiao runner failed for {tag} with code {cp.returncode}. "
                    f"See {candidate_dir / 'runner_stdout.log'}"
                )
            print(f"[DONE] {tag}; aggregate rows={len(rows)}", flush=True)

    rows = collect_candidate_rows(args, output_dir, baseline)
    persist_analysis(output_dir, rows)
    print(f"[DONE] raw={output_dir / 'candidate_raw.csv'}", flush=True)
    print(f"[DONE] summary={output_dir / 'candidate_summary.csv'}", flush=True)
    print(f"[DONE] selected={output_dir / 'selected_config.json'}", flush=True)


if __name__ == "__main__":
    main()
