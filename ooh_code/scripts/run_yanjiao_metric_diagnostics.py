#!/usr/bin/env python
"""Run metric-projection Yanjiao diagnostic DSPO/DRPO experiments.

The script keeps the fixed metric-projection dataset prefix and evaluates a
small set of scenario changes that may explain limited DRPO lift:

- stronger HGS labels
- fleet stress
- behavior sensitivity

Each scenario runs matched DSPO/DRPO with the tuned DRPO SPO parameters and
writes paired deltas plus a ranked summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    source_raw: Optional[str] = None


def default_scenarios(baseline_raw: str) -> List[Scenario]:
    return [
        Scenario(
            name="base_hgs02_v35",
            description="Existing metric baseline: HGS 0.2/0.2, 35 vehicles.",
            source_raw=baseline_raw,
        ),
        Scenario(
            name="hgs05_v35",
            description="Check whether stronger HGS labels stabilize DRPO.",
            overrides={"hgs_reopt_time": 0.5, "hgs_final_time": 0.5, "n_vehicles": 35},
        ),
        Scenario(
            name="fleet30_hgs02",
            description="Check whether tighter fleet capacity creates more room for DRPO.",
            overrides={"n_vehicles": 30, "hgs_reopt_time": 0.2, "hgs_final_time": 0.2},
        ),
        Scenario(
            name="fleet40_hgs02",
            description="Check whether looser fleet capacity suppresses DRPO lift.",
            overrides={"n_vehicles": 40, "hgs_reopt_time": 0.2, "hgs_final_time": 0.2},
        ),
        Scenario(
            name="outside_m05_hgs02",
            description="Check behavior sensitivity with a more attractive outside option.",
            overrides={"outside_option_util": -0.5, "n_vehicles": 35, "hgs_reopt_time": 0.2, "hgs_final_time": 0.2},
        ),
        Scenario(
            name="home12_hgs02",
            description="Check behavior sensitivity with lower home-pickup utility.",
            overrides={"home_util": 1.2, "n_vehicles": 35, "hgs_reopt_time": 0.2, "hgs_final_time": 0.2},
        ),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Metric Yanjiao diagnostic scenario runner.")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--baseline_raw", default="Experiments/analysis/yanjiao_metric_projection_400_20260520/yanjiao_raw.csv")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 67, 97])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    p.add_argument("--n_passengers", type=int, default=400)
    p.add_argument("--max_steps_r", type=int, default=400)
    p.add_argument("--yanjiao_prefix", default="yanjiao_metric_{n_passengers}_{seed}")
    p.add_argument("--grid_dim", type=int, default=11)
    p.add_argument("--hgs_reopt_time", type=float, default=0.2)
    p.add_argument("--hgs_final_time", type=float, default=0.2)
    p.add_argument("--n_vehicles", type=int, default=35)
    p.add_argument("--home_util", type=float, default=1.4)
    p.add_argument("--outside_option_util", type=float, default=-1.0)
    p.add_argument("--spo_loss_weight", type=float, default=0.05)
    p.add_argument("--spo_batch_size", type=int, default=4)
    p.add_argument("--spo_warmup", type=int, default=5)
    p.add_argument("--spo_rampup", type=int, default=10)
    p.add_argument("--initial_phase_epochs", type=int, default=50)
    p.add_argument("--buffer_size", type=int, default=500)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--run_timeout_sec", type=int, default=10800)
    p.add_argument("--persist_every_n", type=int, default=1)
    p.add_argument("--run_prefix", default="YJMD")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--analyze_only", action="store_true")
    p.add_argument("--no_skip_completed", action="store_true")
    return p.parse_args()


def scenario_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").upper()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals: Sequence[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def selected_scenarios(args: argparse.Namespace) -> List[Scenario]:
    scenarios = default_scenarios(args.baseline_raw)
    if not args.scenarios:
        return scenarios
    wanted = set(args.scenarios)
    out = [s for s in scenarios if s.name in wanted]
    missing = wanted - {s.name for s in out}
    if missing:
        raise ValueError("Unknown scenarios: " + ", ".join(sorted(missing)))
    return out


def scenario_params(args: argparse.Namespace, scenario: Scenario) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "n_vehicles": args.n_vehicles,
        "hgs_reopt_time": args.hgs_reopt_time,
        "hgs_final_time": args.hgs_final_time,
        "home_util": args.home_util,
        "outside_option_util": args.outside_option_util,
    }
    params.update(scenario.overrides)
    return params


def build_cmd(args: argparse.Namespace, scenario: Scenario, scenario_dir: Path) -> List[str]:
    params = scenario_params(args, scenario)
    tag = scenario_tag(scenario.name)
    return [
        args.python_executable,
        "scripts/run_yanjiao_experiments.py",
        "--phase",
        "main",
        "--strategies",
        "DSPO",
        "DRPO",
        "--seeds",
        *[str(s) for s in args.seeds],
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        args.route_label_mode,
        "--run_prefix",
        f"{args.run_prefix}_{tag}",
        "--folder_suffix",
        f"_yj_metric_diag_{scenario.name}",
        "--output_dir",
        str(scenario_dir),
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
        "--drpo_spo_loss_weight",
        repr(float(args.spo_loss_weight)),
        "--spo_batch_size_override",
        str(args.spo_batch_size),
        "--spo_warmup_episodes_override",
        str(args.spo_warmup),
        "--spo_rampup_episodes_override",
        str(args.spo_rampup),
        "--initial_phase_epochs_override",
        str(args.initial_phase_epochs),
        "--buffer_size_override",
        str(args.buffer_size),
        "--gpu",
        str(args.gpu),
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--persist_every_n",
        str(args.persist_every_n),
    ] + (["--allow_cpu"] if args.allow_cpu else [])


def scenario_completed(raw_path: Path, seeds: Sequence[int]) -> bool:
    rows = read_csv(raw_path)
    if len(rows) < len(seeds) * 2:
        return False
    done = {
        (row.get("label"), int(float(row.get("seed", -1))))
        for row in rows
        if row.get("status") == "completed"
    }
    return all(("DSPO", s) in done and ("DRPO", s) in done for s in seeds)


def run_scenario(args: argparse.Namespace, root: Path, out_dir: Path, scenario: Scenario) -> int:
    scenario_dir = out_dir / scenario.name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(args, scenario, scenario_dir)
    cmd_path = scenario_dir / "command.txt"
    cmd_path.write_text(" ".join(cmd), encoding="utf-8")

    if args.dry_run:
        print(f"\n[DRY-RUN] {scenario.name}: {' '.join(cmd)}", flush=True)
        return 0

    stdout_path = scenario_dir / "driver_stdout.log"
    stderr_path = scenario_dir / "driver_stderr.log"
    t0 = time.time()
    print(f"[RUN] {scenario.name}", flush=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        cp = subprocess.run(
            cmd,
            cwd=root,
            stdout=out,
            stderr=err,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=args.run_timeout_sec * 2 if args.run_timeout_sec else None,
        )
    elapsed = (time.time() - t0) / 60.0
    print(f"[DONE] {scenario.name} rc={cp.returncode} elapsed={elapsed:.1f}min", flush=True)
    return int(cp.returncode)


def args_yaml_for_log(log_path: Path) -> Path:
    return log_path.parents[2] / "args.yaml"


def parse_health(row: Dict[str, Any]) -> Dict[str, Any]:
    label = str(row.get("label", ""))
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

    warning_count = len(re.findall(r"(?im)^.*spo.*warn.*$", log_text))
    out: Dict[str, Any] = {
        "cuda_used": "Using GPU device: cuda" in log_text,
        "spo_warning_count": warning_count,
    }
    if label == "DRPO":
        out.update({
            "drpo_loaded": "Src.Algorithms.DRPO.DRPO" in log_text,
            "spo_populated": "spo_training_data populated" in log_text,
            "spo_weight_positive": "spo_weight became positive" in log_text,
            "drpo_spo_ok": (
                "Src.Algorithms.DRPO.DRPO" in log_text
                and "spo_training_data populated" in log_text
                and "spo_weight became positive" in log_text
                and warning_count == 0
            ),
            "drpo_positive_spo_weight_arg": "spo_loss_weight: 0.05" in args_text,
        })
    elif label == "DSPO":
        out.update({
            "dspo_spo_disabled": "spo_loss_weight: 0.0" in args_text,
        })
    return out


def raw_for_scenario(out_dir: Path, scenario: Scenario) -> Path:
    if scenario.source_raw:
        return Path(scenario.source_raw)
    return out_dir / scenario.name / "yanjiao_raw.csv"


def paired_rows_for_scenario(args: argparse.Namespace, out_dir: Path, scenario: Scenario) -> List[Dict[str, Any]]:
    raw_path = raw_for_scenario(out_dir, scenario)
    rows = read_csv(raw_path)
    keyed = {
        (str(r.get("label")), int(float(r.get("seed", -1)))): r
        for r in rows
        if r.get("label") in {"DSPO", "DRPO"} and r.get("seed")
    }

    params = scenario_params(args, scenario)
    out: List[Dict[str, Any]] = []
    for seed in args.seeds:
        dspo = keyed.get(("DSPO", seed))
        drpo = keyed.get(("DRPO", seed))
        if not dspo or not drpo:
            continue
        row: Dict[str, Any] = {
            "scenario": scenario.name,
            "description": scenario.description,
            "seed": seed,
            "n_passengers": args.n_passengers,
            "n_vehicles": params["n_vehicles"],
            "hgs_reopt_time": params["hgs_reopt_time"],
            "hgs_final_time": params["hgs_final_time"],
            "home_util": params["home_util"],
            "outside_option_util": params["outside_option_util"],
            "source_raw": str(raw_path),
        }
        for metric in METRICS:
            a = to_float(dspo.get(metric))
            b = to_float(drpo.get(metric))
            row[f"DSPO_{metric}"] = a
            row[f"DRPO_{metric}"] = b
            row[f"delta_{metric}"] = (b - a) if a is not None and b is not None else ""
        dspo_health = parse_health(dspo)
        drpo_health = parse_health(drpo)
        row.update({f"dspo_{k}": v for k, v in dspo_health.items()})
        row.update({f"drpo_{k}": v for k, v in drpo_health.items()})
        out.append(row)
    return out


def summarize_pairs(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in pairs:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for scenario, rows in by_scenario.items():
        net = [to_float(r.get("delta_net_profit")) for r in rows]
        net = [v for v in net if v is not None]
        quit_delta = [to_float(r.get("delta_quit_rate")) for r in rows]
        quit_delta = [v for v in quit_delta if v is not None]
        summary: Dict[str, Any] = {
            "scenario": scenario,
            "description": rows[0].get("description", ""),
            "n_pairs": len(rows),
            "wins_net_profit": sum(1 for v in net if v > 0),
            "drpo_spo_ok_all_runs": all(str(r.get("drpo_drpo_spo_ok", "")).lower() == "true" or r.get("drpo_drpo_spo_ok") is True for r in rows),
            "dspo_spo_disabled_all_runs": all(str(r.get("dspo_dspo_spo_disabled", "")).lower() == "true" or r.get("dspo_dspo_spo_disabled") is True for r in rows),
            "spo_warning_count_total": sum(int(to_float(r.get("drpo_spo_warning_count")) or 0) for r in rows),
            "n_vehicles": rows[0].get("n_vehicles"),
            "hgs_reopt_time": rows[0].get("hgs_reopt_time"),
            "hgs_final_time": rows[0].get("hgs_final_time"),
            "home_util": rows[0].get("home_util"),
            "outside_option_util": rows[0].get("outside_option_util"),
        }
        for metric in METRICS:
            vals = [to_float(r.get(f"delta_{metric}")) for r in rows]
            vals = [v for v in vals if v is not None]
            if vals:
                summary[f"mean_delta_{metric}"] = mean(vals)
                summary[f"std_delta_{metric}"] = std(vals)
                summary[f"min_delta_{metric}"] = min(vals)
                summary[f"max_delta_{metric}"] = max(vals)
        mean_net = float(summary.get("mean_delta_net_profit", 0.0) or 0.0)
        std_net = float(summary.get("std_delta_net_profit", 0.0) or 0.0)
        wins = int(summary.get("wins_net_profit", 0) or 0)
        mean_quit = mean(quit_delta) if quit_delta else 0.0
        penalty = 0.0
        if not summary["drpo_spo_ok_all_runs"]:
            penalty += 50.0
        if not summary["dspo_spo_disabled_all_runs"]:
            penalty += 50.0
        penalty += 10.0 * int(summary["spo_warning_count_total"])
        summary["score"] = mean_net + 5.0 * wins - 0.25 * std_net - 20.0 * abs(mean_quit) - penalty
        summaries.append(summary)

    return sorted(summaries, key=lambda r: float(r.get("score", -1e18)), reverse=True)


def analyze(args: argparse.Namespace, out_dir: Path, scenarios: Sequence[Scenario]) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        raw_rows.extend(paired_rows_for_scenario(args, out_dir, scenario))
    write_csv(out_dir / "diagnostic_raw.csv", raw_rows)
    summary = summarize_pairs(raw_rows)
    write_csv(out_dir / "diagnostic_summary.csv", summary)
    if summary:
        (out_dir / "selected_diagnostic_scenario.json").write_text(
            json.dumps(summary[0], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return summary


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(
        args.output_dir
        or f"Experiments/analysis/yanjiao_metric_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = selected_scenarios(args)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "expected_new_runs": sum(0 if s.source_raw else len(args.seeds) * 2 for s in scenarios),
        "scenarios": [
            {"name": s.name, "description": s.description, "overrides": s.overrides, "source_raw": s.source_raw}
            for s in scenarios
        ],
        "args": vars(args),
    }
    (out_dir / "metric_diagnostics_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.dry_run:
        print(f"[DRY-RUN] output_dir={out_dir.resolve()}", flush=True)
        print(f"[DRY-RUN] scenarios={', '.join(s.name for s in scenarios)}", flush=True)
        for scenario in scenarios:
            if scenario.source_raw:
                print(f"\n[DRY-RUN] {scenario.name}: reuse {scenario.source_raw}", flush=True)
            else:
                run_scenario(args, root, out_dir, scenario)
        print(f"[DRY-RUN] expected_new_runs={meta['expected_new_runs']}", flush=True)
        return

    if not args.analyze_only:
        for scenario in scenarios:
            if scenario.source_raw:
                print(f"[REUSE] {scenario.name}: {scenario.source_raw}", flush=True)
                continue
            raw_path = raw_for_scenario(out_dir, scenario)
            if not args.no_skip_completed and scenario_completed(raw_path, args.seeds):
                print(f"[CACHE] {scenario.name}", flush=True)
            else:
                rc = run_scenario(args, root, out_dir, scenario)
                if rc != 0:
                    raise RuntimeError(f"Scenario {scenario.name} failed with return code {rc}")
            summary = analyze(args, out_dir, scenarios)
            if summary:
                best = summary[0]
                print(
                    f"[BEST SO FAR] {best['scenario']} "
                    f"score={float(best.get('score', 0.0)):.3f} "
                    f"mean_delta={float(best.get('mean_delta_net_profit', 0.0)):.3f} "
                    f"wins={best.get('wins_net_profit')}/3",
                    flush=True,
                )
    else:
        print("[ANALYZE ONLY]", flush=True)

    summary = analyze(args, out_dir, scenarios)
    print(f"[DONE] raw={out_dir / 'diagnostic_raw.csv'}", flush=True)
    print(f"[DONE] summary={out_dir / 'diagnostic_summary.csv'}", flush=True)
    if summary:
        print(f"[DONE] selected={out_dir / 'selected_diagnostic_scenario.json'}", flush=True)


if __name__ == "__main__":
    main()
