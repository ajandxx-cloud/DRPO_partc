#!/usr/bin/env python
"""Auto-tune DRPO hyperparameters on metric-projection Yanjiao data.

This runner keeps the DSPO baseline fixed and fair, then searches DRPO's SPO
training parameters for a larger paired advantage.  It supports a two-stage
workflow:

1. Screen candidates on one or more seeds.
2. Validate the top screened candidates on the full seed set.

The actual training is delegated to scripts/run_yanjiao_experiments.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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

SCENARIOS = {
    "metric35": {
        "description": "Metric Yanjiao, 35 vehicles, HGS 0.2/0.2.",
        "baseline_raw": "Experiments/analysis/yanjiao_metric_projection_400_20260520/yanjiao_raw.csv",
        "n_vehicles": 35,
        "hgs_reopt_time": 0.2,
        "hgs_final_time": 0.2,
        "home_util": 1.4,
        "outside_option_util": -1.0,
    },
    "metric40": {
        "description": "Metric Yanjiao, 40 vehicles, HGS 0.2/0.2.",
        "baseline_raw": "Experiments/analysis/yanjiao_metric_diagnostics_20260520/fleet40_hgs02/yanjiao_raw.csv",
        "n_vehicles": 40,
        "hgs_reopt_time": 0.2,
        "hgs_final_time": 0.2,
        "home_util": 1.4,
        "outside_option_util": -1.0,
    },
}

SPO_WEIGHT_REGEX = re.compile(
    r"\[SPO\+ debug\] spo_weight became positive:\s*"
    r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric Yanjiao DRPO auto-tuner")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--scenario", choices=sorted(SCENARIOS), default="metric40")
    p.add_argument("--baseline_raw", default=None)
    p.add_argument("--stage", choices=["screen", "validate", "all", "analyze"], default="all")
    p.add_argument("--screen_seeds", nargs="+", type=int, default=[40, 67])
    p.add_argument("--validate_seeds", nargs="+", type=int, default=[40, 67, 97])
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    p.add_argument("--n_passengers", type=int, default=400)
    p.add_argument("--max_steps_r", type=int, default=400)
    p.add_argument("--yanjiao_prefix", default="yanjiao_metric_{n_passengers}_{seed}")
    p.add_argument("--grid_dim", type=int, default=11)
    p.add_argument("--weights", nargs="+", type=float, default=[0.02, 0.035, 0.05, 0.07, 0.1, 0.15])
    p.add_argument("--spo_batch_sizes", nargs="+", type=int, default=[2, 4, 8, 12])
    p.add_argument("--spo_warmups", nargs="+", type=int, default=[0, 3, 5, 8])
    p.add_argument("--spo_rampups", nargs="+", type=int, default=[5, 10, 15])
    p.add_argument("--spo_label_sample_sizes", nargs="+", type=int, default=[0, 4, 8])
    p.add_argument("--initial_phase_epochs", nargs="+", type=int, default=[30, 50, 80])
    p.add_argument("--buffer_sizes", nargs="+", type=int, default=[500, 1000])
    p.add_argument("--max_candidates", type=int, default=12)
    p.add_argument("--mode", choices=["focused", "grid"], default="focused")
    p.add_argument("--run_timeout_sec", type=int, default=10800)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def tag_float(prefix: str, value: float, scale: int = 1000) -> str:
    return f"{prefix}{int(round(value * scale)):04d}"


def candidate_id(c: Dict[str, Any]) -> str:
    return "_".join([
        tag_float("w", float(c["spo_loss_weight"])),
        f"b{int(c['spo_batch_size']):02d}",
        f"wu{int(c['spo_warmup'])}",
        f"ru{int(c['spo_rampup'])}",
        f"ls{int(c['spo_label_sample_size'])}",
        f"init{int(c['initial_phase_epochs'])}",
        f"buf{int(c['buffer_size'])}",
    ])


def candidate_key(c: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        round(float(c["spo_loss_weight"]), 10),
        int(c["spo_batch_size"]),
        int(c["spo_warmup"]),
        int(c["spo_rampup"]),
        int(c["spo_label_sample_size"]),
        int(c["initial_phase_epochs"]),
        int(c["buffer_size"]),
    )


def generate_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    combos = [
        {
            "spo_loss_weight": float(weight),
            "spo_batch_size": int(batch),
            "spo_warmup": int(warmup),
            "spo_rampup": int(rampup),
            "spo_label_sample_size": int(label_size),
            "initial_phase_epochs": int(init_ep),
            "buffer_size": int(buffer_size),
        }
        for weight, batch, warmup, rampup, label_size, init_ep, buffer_size in itertools.product(
            args.weights,
            args.spo_batch_sizes,
            args.spo_warmups,
            args.spo_rampups,
            args.spo_label_sample_sizes,
            args.initial_phase_epochs,
            args.buffer_sizes,
        )
    ]
    if args.mode == "grid":
        return combos[: args.max_candidates]

    by_key = {candidate_key(c): c for c in combos}

    def nearest(values: Sequence[Any], target: float) -> Any:
        return min(values, key=lambda v: (abs(float(v) - target), float(v)))

    center = {
        "spo_loss_weight": float(nearest(args.weights, 0.05)),
        "spo_batch_size": int(nearest(args.spo_batch_sizes, 4)),
        "spo_warmup": int(nearest(args.spo_warmups, 5)),
        "spo_rampup": int(nearest(args.spo_rampups, 10)),
        "spo_label_sample_size": int(nearest(args.spo_label_sample_sizes, 0)),
        "initial_phase_epochs": int(nearest(args.initial_phase_epochs, 50)),
        "buffer_size": int(nearest(args.buffer_sizes, 500)),
    }
    selected: List[Dict[str, Any]] = []
    seen = set()

    def add(**overrides: Any) -> None:
        c = dict(center)
        c.update(overrides)
        actual = by_key.get(candidate_key(c))
        if actual is None:
            return
        cid = candidate_id(actual)
        if cid in seen:
            return
        seen.add(cid)
        selected.append(actual)

    add()
    for weight in sorted(args.weights, key=lambda v: (abs(float(v) - center["spo_loss_weight"]), float(v))):
        add(spo_loss_weight=float(weight))
    for batch in sorted(args.spo_batch_sizes, key=lambda v: (abs(int(v) - center["spo_batch_size"]), int(v))):
        add(spo_batch_size=int(batch))
    for warmup in sorted(args.spo_warmups, key=lambda v: (abs(int(v) - center["spo_warmup"]), int(v))):
        add(spo_warmup=int(warmup))
    for rampup in sorted(args.spo_rampups, key=lambda v: (abs(int(v) - center["spo_rampup"]), int(v))):
        add(spo_rampup=int(rampup))
    for label_size in sorted(args.spo_label_sample_sizes, key=lambda v: (abs(int(v) - center["spo_label_sample_size"]), int(v))):
        add(spo_label_sample_size=int(label_size))
    for init_ep in sorted(args.initial_phase_epochs, key=lambda v: (abs(int(v) - center["initial_phase_epochs"]), int(v))):
        add(initial_phase_epochs=int(init_ep))
    for buffer_size in sorted(args.buffer_sizes, key=lambda v: (abs(int(v) - center["buffer_size"]), int(v))):
        add(buffer_size=int(buffer_size))

    def priority(c: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            abs(float(c["spo_loss_weight"]) - 0.05),
            abs(int(c["spo_batch_size"]) - 4) + abs(int(c["spo_warmup"]) - 5) + abs(int(c["spo_rampup"]) - 10),
            abs(int(c["initial_phase_epochs"]) - 50) + abs(int(c["buffer_size"]) - 500) / 500.0,
        )

    for c in sorted(combos, key=priority):
        if len(selected) >= args.max_candidates:
            break
        cid = candidate_id(c)
        if cid not in seen:
            seen.add(cid)
            selected.append(c)
    return selected[: args.max_candidates]


def scenario_params(args: argparse.Namespace) -> Dict[str, Any]:
    params = dict(SCENARIOS[args.scenario])
    if args.baseline_raw:
        params["baseline_raw"] = args.baseline_raw
    return params


def command_for_candidate(
    args: argparse.Namespace,
    stage: str,
    out_dir: Path,
    c: Dict[str, Any],
    seeds: Sequence[int],
) -> List[str]:
    params = scenario_params(args)
    cid = candidate_id(c)
    run_prefix = f"YJ_METRIC_TUNE_{args.scenario.upper()}_{stage.upper()}_{cid}"
    return [
        args.python_executable,
        "scripts/run_yanjiao_experiments.py",
        "--python_executable",
        args.python_executable,
        "--gpu",
        str(args.gpu),
        "--phase",
        "main",
        "--strategies",
        "DRPO",
        "--seeds",
        *[str(seed) for seed in seeds],
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        args.route_label_mode,
        "--run_prefix",
        run_prefix,
        "--folder_suffix",
        f"_yj_metric_tune_{args.scenario}_{stage}_{cid}",
        "--output_dir",
        str(out_dir / stage / cid),
        "--allow_existing_output_dir",
        "--persist_every_n",
        "1",
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--max_retries",
        str(args.max_retries),
        "--allow_cpu" if args.allow_cpu else "__NO_ALLOW_CPU__",
        "--n_passengers_override",
        str(args.n_passengers),
        "--n_vehicles_override",
        str(int(params["n_vehicles"])),
        "--max_steps_r_override",
        str(args.max_steps_r),
        "--yanjiao_prefix",
        args.yanjiao_prefix,
        "--grid_dim_override",
        str(args.grid_dim),
        "--hgs_reopt_time_override",
        repr(float(params["hgs_reopt_time"])),
        "--hgs_final_time_override",
        repr(float(params["hgs_final_time"])),
        "--home_util_override",
        repr(float(params["home_util"])),
        "--outside_option_util_override",
        repr(float(params["outside_option_util"])),
        "--dspo_spo_loss_weight",
        "0.0",
        "--drpo_spo_loss_weight",
        repr(float(c["spo_loss_weight"])),
        "--spo_batch_size_override",
        str(int(c["spo_batch_size"])),
        "--spo_warmup_episodes_override",
        str(int(c["spo_warmup"])),
        "--spo_rampup_episodes_override",
        str(int(c["spo_rampup"])),
        "--spo_label_sample_size_override",
        str(int(c["spo_label_sample_size"])),
        "--initial_phase_epochs_override",
        str(int(c["initial_phase_epochs"])),
        "--buffer_size_override",
        str(int(c["buffer_size"])),
    ]


def clean_cmd(cmd: Sequence[str]) -> List[str]:
    return [part for part in cmd if part != "__NO_ALLOW_CPU__"]


def run_candidate(args: argparse.Namespace, stage: str, out_dir: Path, c: Dict[str, Any], seeds: Sequence[int]) -> int:
    cid = candidate_id(c)
    run_dir = out_dir / stage / cid
    raw_path = run_dir / "yanjiao_raw.csv"
    if args.skip_existing and raw_path.exists():
        existing = read_csv(raw_path)
        done_seeds = {int(float(r["seed"])) for r in existing if r.get("label") == "DRPO" and r.get("status") == "completed"}
        if set(seeds).issubset(done_seeds):
            print(f"[CACHE] {stage} {cid}", flush=True)
            return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = clean_cmd(command_for_candidate(args, stage, out_dir, c, seeds))
    (run_dir / "autotune_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    print(f"[RUN] {stage} {cid} seeds={list(seeds)}", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    t0 = time.time()
    with (run_dir / "autotune_stdout.log").open("w", encoding="utf-8") as out:
        cp = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    print(f"[DONE] {stage} {cid} rc={cp.returncode} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    return int(cp.returncode)


def load_dspo_baseline(path: Path, n_passengers: int, seeds: Sequence[int]) -> Dict[int, Dict[str, str]]:
    rows = read_csv(path)
    seed_set = set(int(s) for s in seeds)
    out: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if row.get("label") != "DSPO":
            continue
        seed = int(float(row.get("seed", -1)))
        if seed not in seed_set:
            continue
        if int(float(row.get("n_passengers", 0))) != int(n_passengers):
            continue
        out[seed] = row
    missing = sorted(seed_set - set(out))
    if missing:
        raise RuntimeError(f"DSPO baseline missing seeds {missing}: {path}")
    return out


def args_yaml_for_log(log_path: Path) -> Path:
    return log_path.parents[2] / "args.yaml"


def parse_health(row: Dict[str, str], expected_weight: float) -> Dict[str, Any]:
    log_text = ""
    args_text = ""
    log_path_text = row.get("log_path", "")
    if log_path_text:
        log_path = Path(log_path_text)
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            args_path = args_yaml_for_log(log_path)
            if args_path.exists():
                args_text = args_path.read_text(encoding="utf-8", errors="ignore")
    weights = [float(x) for x in SPO_WEIGHT_REGEX.findall(log_text)]
    warning_count = len(re.findall(r"(?im)^.*spo.*warn.*$", log_text))
    expected_text = f"spo_loss_weight: {expected_weight:g}"
    return {
        "drpo_loaded": "Src.Algorithms.DRPO.DRPO" in log_text,
        "cuda_used": "Using GPU device: cuda" in log_text,
        "spo_populated": "spo_training_data populated" in log_text,
        "spo_weight_positive": any(w > 0 for w in weights),
        "first_positive_spo_weight": weights[0] if weights else "",
        "spo_warning_count": warning_count,
        "expected_weight_in_args": expected_text in args_text,
    }


def collect_candidate_pairs(
    args: argparse.Namespace,
    stage: str,
    out_dir: Path,
    candidates: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    params = scenario_params(args)
    baseline = load_dspo_baseline(resolve_path(str(params["baseline_raw"])), args.n_passengers, seeds)
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        cid = candidate_id(c)
        raw_path = out_dir / stage / cid / "yanjiao_raw.csv"
        drpo_rows = [
            r for r in read_csv(raw_path)
            if r.get("label") == "DRPO" and int(float(r.get("seed", -1))) in set(seeds)
        ]
        for drpo in drpo_rows:
            seed = int(float(drpo["seed"]))
            dspo = baseline.get(seed)
            if not dspo:
                continue
            row: Dict[str, Any] = {
                "stage": stage,
                "scenario": args.scenario,
                "candidate_id": cid,
                "seed": seed,
                "n_passengers": args.n_passengers,
                "n_vehicles": params["n_vehicles"],
                "hgs_reopt_time": params["hgs_reopt_time"],
                "hgs_final_time": params["hgs_final_time"],
                "home_util": params["home_util"],
                "outside_option_util": params["outside_option_util"],
                "baseline_raw": str(params["baseline_raw"]),
                "candidate_raw": str(raw_path),
                "drpo_log_path": drpo.get("log_path", ""),
                **c,
            }
            for metric in METRICS:
                a = to_float(dspo.get(metric))
                b = to_float(drpo.get(metric))
                row[f"DSPO_{metric}"] = a if a is not None else ""
                row[f"DRPO_{metric}"] = b if b is not None else ""
                row[f"delta_{metric}"] = (b - a) if a is not None and b is not None else ""
            row.update(parse_health(drpo, float(c["spo_loss_weight"])))
            rows.append(row)
    return sorted(rows, key=lambda r: (str(r["candidate_id"]), int(r["seed"])))


def summarize_pairs(rows: List[Dict[str, Any]], n_seeds: int) -> List[Dict[str, Any]]:
    by_candidate: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate_id"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for cid, group in by_candidate.items():
        first = group[0]
        item: Dict[str, Any] = {
            "stage": first["stage"],
            "scenario": first["scenario"],
            "candidate_id": cid,
            "n_pairs": len(group),
            "wins_net_profit": sum((to_float(r.get("delta_net_profit")) or 0.0) > 0 for r in group),
            "drpo_spo_ok_all_runs": all(
                bool(r.get("drpo_loaded"))
                and bool(r.get("spo_populated"))
                and bool(r.get("spo_weight_positive"))
                and int(to_float(r.get("spo_warning_count")) or 0) == 0
                for r in group
            ),
            "spo_warning_count_total": sum(int(to_float(r.get("spo_warning_count")) or 0) for r in group),
        }
        for key in [
            "spo_loss_weight",
            "spo_batch_size",
            "spo_warmup",
            "spo_rampup",
            "spo_label_sample_size",
            "initial_phase_epochs",
            "buffer_size",
            "n_vehicles",
            "hgs_reopt_time",
            "hgs_final_time",
            "home_util",
            "outside_option_util",
        ]:
            item[key] = first.get(key)
        for metric in METRICS:
            vals = [to_float(r.get(f"delta_{metric}")) for r in group]
            vals = [v for v in vals if v is not None]
            if vals:
                item[f"mean_delta_{metric}"] = mean(vals)
                item[f"std_delta_{metric}"] = std(vals)
                item[f"min_delta_{metric}"] = min(vals)
                item[f"max_delta_{metric}"] = max(vals)
        mean_net = float(item.get("mean_delta_net_profit", -1e9) or -1e9)
        std_net = float(item.get("std_delta_net_profit", 0.0) or 0.0)
        wins = int(item.get("wins_net_profit") or 0)
        mean_quit = abs(float(item.get("mean_delta_quit_rate", 0.0) or 0.0))
        score = mean_net + 5.0 * wins - 0.25 * std_net - 20.0 * mean_quit
        if len(group) < n_seeds:
            score -= 100.0
        if wins < max(1, math.ceil(n_seeds / 2)):
            score -= 25.0
        if not item["drpo_spo_ok_all_runs"]:
            score -= 1000.0
        score -= 20.0 * int(item["spo_warning_count_total"])
        item["score"] = score
        out.append(item)
    return sorted(
        out,
        key=lambda r: (
            float(r.get("score") or -1e9),
            int(r.get("wins_net_profit") or 0),
            float(r.get("mean_delta_net_profit") or -1e9),
            -float(r.get("std_delta_net_profit") or 1e9),
        ),
        reverse=True,
    )


def persist_stage(
    args: argparse.Namespace,
    stage: str,
    out_dir: Path,
    candidates: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = collect_candidate_pairs(args, stage, out_dir, candidates, seeds)
    if rows:
        write_csv(out_dir / f"{stage}_raw.csv", rows)
    summary = summarize_pairs(rows, len(seeds)) if rows else []
    if summary:
        write_csv(out_dir / f"{stage}_summary.csv", summary)
        (out_dir / f"{stage}_best_config.json").write_text(
            json.dumps(summary[0], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def load_candidates_from_summary(path: Path, top_k: int) -> List[Dict[str, Any]]:
    rows = read_csv(path)
    out: List[Dict[str, Any]] = []
    for row in rows[:top_k]:
        out.append({
            "spo_loss_weight": float(row["spo_loss_weight"]),
            "spo_batch_size": int(float(row["spo_batch_size"])),
            "spo_warmup": int(float(row["spo_warmup"])),
            "spo_rampup": int(float(row["spo_rampup"])),
            "spo_label_sample_size": int(float(row["spo_label_sample_size"])),
            "initial_phase_epochs": int(float(row["initial_phase_epochs"])),
            "buffer_size": int(float(row["buffer_size"])),
        })
    return out


def run_stage(args: argparse.Namespace, stage: str, out_dir: Path, candidates: Sequence[Dict[str, Any]], seeds: Sequence[int]) -> List[Dict[str, Any]]:
    if not args.dry_run and args.stage == "analyze":
        return persist_stage(args, stage, out_dir, candidates, seeds)
    for c in candidates:
        rc = run_candidate(args, stage, out_dir, c, seeds)
        summary = persist_stage(args, stage, out_dir, candidates, seeds)
        if rc != 0:
            raise RuntimeError(f"{stage} candidate {candidate_id(c)} failed with rc={rc}")
    return persist_stage(args, stage, out_dir, candidates, seeds)


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(
        args.output_dir
        or f"Experiments/analysis/yanjiao_metric_drpo_autotune_{args.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(args)
    params = scenario_params(args)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "scenario_params": params,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "selection_rule": "score = mean_delta_net_profit + 5*wins - 0.25*std - 20*abs(mean_delta_quit_rate), with penalties for incomplete runs, low wins, and SPO warnings",
    }
    (out_dir / "autotune_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.stage in {"screen", "all"}:
        screen_summary = run_stage(args, "screen", out_dir, candidates, args.screen_seeds)
    elif args.stage == "analyze":
        screen_summary = persist_stage(args, "screen", out_dir, candidates, args.screen_seeds)
    else:
        screen_summary = []

    validate_candidates: List[Dict[str, Any]] = []
    if args.stage in {"validate", "all", "analyze"}:
        screen_summary_path = out_dir / "screen_summary.csv"
        if screen_summary:
            validate_candidates = [
                {
                    "spo_loss_weight": float(r["spo_loss_weight"]),
                    "spo_batch_size": int(float(r["spo_batch_size"])),
                    "spo_warmup": int(float(r["spo_warmup"])),
                    "spo_rampup": int(float(r["spo_rampup"])),
                    "spo_label_sample_size": int(float(r["spo_label_sample_size"])),
                    "initial_phase_epochs": int(float(r["initial_phase_epochs"])),
                    "buffer_size": int(float(r["buffer_size"])),
                }
                for r in screen_summary[: args.top_k]
            ]
        elif screen_summary_path.exists():
            validate_candidates = load_candidates_from_summary(screen_summary_path, args.top_k)
        else:
            validate_candidates = candidates[: args.top_k]

        validate_summary = run_stage(args, "validate", out_dir, validate_candidates, args.validate_seeds)
        if validate_summary:
            (out_dir / "best_config.json").write_text(
                json.dumps(validate_summary[0], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    elif screen_summary:
        (out_dir / "best_config.json").write_text(
            json.dumps(screen_summary[0], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[DONE] output_dir={out_dir}", flush=True)
    for name in ["screen_summary.csv", "validate_summary.csv", "best_config.json"]:
        p = out_dir / name
        if p.exists():
            print(f"[DONE] {name}={p}", flush=True)


if __name__ == "__main__":
    main()
