#!/usr/bin/env python
"""Search Yanjiao settings where DRPO clearly beats DSPO.

The script implements a staged experiment:

1. DSPO-only pressure calibration.
2. Paired DSPO/DRPO search across revenue, price window, and k scenarios.
3. Multi-seed validation of the best candidates.

It delegates actual training to scripts/run_yanjiao_experiments.py and writes
compact CSV summaries for each stage.
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

PRESSURE_CANDIDATES = [
    {"pressure_id": "A1", "outside_option_util": -0.8, "home_util": 1.4, "k": 10},
    {"pressure_id": "A2", "outside_option_util": -0.6, "home_util": 1.4, "k": 10},
    {"pressure_id": "A3", "outside_option_util": -0.4, "home_util": 1.4, "k": 10},
    {"pressure_id": "A4", "outside_option_util": -0.8, "home_util": 1.3, "k": 10},
    {"pressure_id": "A5", "outside_option_util": -0.6, "home_util": 1.3, "k": 10},
    {"pressure_id": "A6", "outside_option_util": -0.4, "home_util": 1.3, "k": 10},
    {"pressure_id": "A7", "outside_option_util": -0.6, "home_util": 1.2, "k": 10},
    {"pressure_id": "A8", "outside_option_util": -0.4, "home_util": 1.2, "k": 10},
    {"pressure_id": "A9", "outside_option_util": -0.6, "home_util": 1.4, "k": 8},
    {"pressure_id": "A10", "outside_option_util": -0.4, "home_util": 1.4, "k": 8},
]

SEARCH_SCENARIOS = [
    {
        "scenario_id": "S0",
        "description": "Pressure parameters with default revenue and price window.",
        "revenue": 50.0,
        "min_price": None,
        "max_price": None,
        "k": 10,
    },
    {
        "scenario_id": "S1",
        "description": "Tighter discount room and higher positive charge.",
        "revenue": 50.0,
        "min_price": -3.5,
        "max_price": 5.0,
        "k": 10,
    },
    {
        "scenario_id": "S2",
        "description": "Smaller candidate set to increase routing and choice pressure.",
        "revenue": 50.0,
        "min_price": None,
        "max_price": None,
        "k": 8,
    },
    {
        "scenario_id": "S3",
        "description": "Lower base revenue with a conservative price window.",
        "revenue": 40.0,
        "min_price": -3.5,
        "max_price": 4.0,
        "k": 8,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged Yanjiao DRPO advantage search")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument(
        "--stage",
        choices=["all", "pressure", "search", "validation", "analyze"],
        default="all",
        help="Run all stages, a single stage, or only re-analyze existing CSVs.",
    )
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--route_label_mode", default="hep", choices=["hgs", "hep"])
    p.add_argument("--pressure_seed", type=int, default=67)
    p.add_argument("--validation_seeds", nargs="+", type=int, default=[40, 67, 97])
    p.add_argument("--search_weights", nargs="+", type=float, default=[0.5, 0.9])
    p.add_argument("--neighbor_weight", type=float, default=0.7)
    p.add_argument("--top_pressure", type=int, default=2)
    p.add_argument("--top_validation", type=int, default=2)
    p.add_argument("--target_profit_delta", type=float, default=30.0)
    p.add_argument("--target_quit_min", type=float, default=2.0)
    p.add_argument("--target_quit_max", type=float, default=5.0)
    p.add_argument("--fallback_quit_min", type=float, default=1.5)
    p.add_argument("--fallback_quit_max", type=float, default=6.0)
    p.add_argument("--max_abs_quit_delta", type=float, default=0.3)
    p.add_argument("--max_abs_served_delta", type=float, default=2.0)
    p.add_argument("--n_passengers", type=int, default=400)
    p.add_argument("--n_vehicles", type=int, default=35)
    p.add_argument("--max_steps_r", type=int, default=400)
    p.add_argument("--hgs_reopt_time", type=float, default=0.2)
    p.add_argument("--hgs_final_time", type=float, default=0.2)
    p.add_argument("--spo_label_sample_size", type=int, default=4)
    p.add_argument("--spo_batch_size", type=int, default=8)
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--yanjiao_prefix", default=None)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def resolve_output_dir(value: Optional[str]) -> Path:
    if value:
        p = Path(value)
        return p if p.is_absolute() else ROOT / p
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "Experiments" / "analysis" / f"yanjiao_drpo_advantage_search_{stamp}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = collect_fields(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def collect_fields(rows: Iterable[Dict[str, Any]]) -> List[str]:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def safe_delta(a: Any, b: Any) -> Optional[float]:
    av = to_float(a)
    bv = to_float(b)
    if av is None or bv is None:
        return None
    return bv - av


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def tag_float(value: Optional[float]) -> str:
    if value is None:
        return "default"
    text = f"{value:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    return text.rstrip("0").rstrip("p")


def build_runner_cmd(
    args: argparse.Namespace,
    run_dir: Path,
    run_prefix: str,
    folder_suffix: str,
    seeds: Sequence[int],
    strategies: Sequence[str],
    overrides: Dict[str, Any],
    drpo_weight: Optional[float] = None,
) -> List[str]:
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
        *[str(s) for s in seeds],
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        args.route_label_mode,
        "--strategies",
        *strategies,
        "--run_prefix",
        run_prefix,
        "--folder_suffix",
        folder_suffix,
        "--output_dir",
        rel_path(run_dir),
        "--allow_existing_output_dir",
        "--persist_every_n",
        "1",
        "--max_retries",
        str(args.max_retries),
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--dspo_spo_loss_weight",
        "0.0",
        "--n_passengers_override",
        str(int(args.n_passengers)),
        "--n_vehicles_override",
        str(int(args.n_vehicles)),
        "--max_steps_r_override",
        str(int(args.max_steps_r)),
        "--hgs_reopt_time_override",
        repr(float(args.hgs_reopt_time)),
        "--hgs_final_time_override",
        repr(float(args.hgs_final_time)),
        "--spo_label_sample_size_override",
        str(int(args.spo_label_sample_size)),
        "--spo_batch_size_override",
        str(int(args.spo_batch_size)),
    ]
    if drpo_weight is not None:
        cmd.extend(["--drpo_spo_loss_weight", repr(float(drpo_weight))])
    if args.yanjiao_prefix:
        cmd.extend(["--yanjiao_prefix", args.yanjiao_prefix])

    override_to_arg = {
        "k": "--k_override",
        "revenue": "--revenue_override",
        "home_util": "--home_util_override",
        "outside_option_util": "--outside_option_util_override",
        "min_price": "--min_price_override",
        "max_price": "--max_price_override",
    }
    for key, cli_arg in override_to_arg.items():
        if key in overrides and overrides[key] is not None:
            cmd.extend([cli_arg, str(overrides[key])])

    if args.allow_cpu:
        cmd.append("--allow_cpu")
    if args.dry_run:
        cmd.append("--dry_run")
    if not args.skip_existing:
        cmd.append("--no_skip_existing")
    return cmd


def run_command(cmd: List[str], run_dir: Path, dry_run: bool) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
    print("[RUN] " + " ".join(cmd), flush=True)
    if dry_run:
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
    elapsed = time.time() - t0
    (run_dir / "runner_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
    print(f"[DONE] rc={cp.returncode} elapsed={elapsed/60:.1f}min dir={run_dir}", flush=True)
    return cp.returncode


def first_row(raw_path: Path, label: str) -> Optional[Dict[str, str]]:
    for row in read_csv(raw_path):
        if row.get("label") == label:
            return row
    return None


def build_overrides(pressure: Dict[str, Any], scenario: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    overrides = {
        "outside_option_util": pressure["outside_option_util"],
        "home_util": pressure["home_util"],
        "k": pressure["k"],
        "revenue": 50.0,
        "min_price": None,
        "max_price": None,
    }
    if scenario:
        for key in ("k", "revenue", "min_price", "max_price"):
            overrides[key] = scenario.get(key)
    return overrides


def pressure_run_dir(output_dir: Path, pressure: Dict[str, Any]) -> Path:
    return output_dir / "stage1_pressure" / pressure["pressure_id"]


def search_run_dir(
    output_dir: Path,
    pressure: Dict[str, Any],
    scenario: Dict[str, Any],
    label: str,
    weight: Optional[float] = None,
) -> Path:
    parts = [pressure["pressure_id"], scenario["scenario_id"], label]
    if weight is not None:
        parts.append(f"w{int(round(weight * 100)):03d}")
    return output_dir / "stage2_search" / "_".join(parts)


def validation_run_dir(output_dir: Path, selected_id: str, label: str, weight: Optional[float] = None) -> Path:
    parts = [selected_id, label]
    if weight is not None:
        parts.append(f"w{int(round(weight * 100)):03d}")
    return output_dir / "stage3_validation" / "_".join(parts)


def run_pressure_stage(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pressure in PRESSURE_CANDIDATES:
        run_dir = pressure_run_dir(output_dir, pressure)
        overrides = build_overrides(pressure)
        run_prefix = f"YJ_PRESS_{pressure['pressure_id']}"
        cmd = build_runner_cmd(
            args,
            run_dir,
            run_prefix=run_prefix,
            folder_suffix=f"_yj_press_{pressure['pressure_id']}",
            seeds=[args.pressure_seed],
            strategies=["DSPO"],
            overrides=overrides,
        )
        rc = run_command(cmd, run_dir, args.dry_run)
        raw = first_row(run_dir / "yanjiao_raw.csv", "DSPO")
        row = dict(pressure)
        row.update({
            "stage": "pressure",
            "seed": args.pressure_seed,
            "run_dir": rel_path(run_dir),
            "returncode": rc,
            "status": raw.get("status") if raw else ("dry_run" if args.dry_run else "missing_raw"),
            "selected_primary": False,
            "selected_fallback": False,
        })
        if raw:
            for metric in METRICS:
                row[f"dspo_{metric}"] = to_float(raw.get(metric))
            quit_rate = to_float(raw.get("quit_rate"))
            row["quit_distance_to_3pct"] = abs((quit_rate or 0.0) - 3.0) if quit_rate is not None else ""
            row["primary_quit_window"] = (
                quit_rate is not None
                and args.target_quit_min <= quit_rate <= args.target_quit_max
            )
            row["fallback_quit_window"] = (
                quit_rate is not None
                and args.fallback_quit_min <= quit_rate <= args.fallback_quit_max
            )
        rows.append(row)
        write_csv(output_dir / "pressure_summary.csv", rows)
    selected = select_pressure_rows(rows, args)
    selected_ids = {r["pressure_id"] for r in selected}
    for row in rows:
        row["selected_for_search"] = row["pressure_id"] in selected_ids
    write_csv(output_dir / "pressure_summary.csv", rows)
    (output_dir / "selected_pressure.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return rows


def select_pressure_rows(rows: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> Tuple[float, float]:
        distance = to_float(row.get("quit_distance_to_3pct"))
        profit = to_float(row.get("dspo_net_profit"))
        return (distance if distance is not None else 1e9, -(profit if profit is not None else -1e9))

    primary = [r for r in rows if r.get("primary_quit_window")]
    pool = primary
    if not pool:
        pool = [r for r in rows if r.get("fallback_quit_window")]
    if not pool:
        pool = [r for r in rows if to_float(r.get("dspo_quit_rate")) is not None]
    if not pool:
        pool = rows
    return sorted(pool, key=sort_key)[: args.top_pressure]


def load_selected_pressure(output_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_path = output_dir / "selected_pressure.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))
    return select_pressure_rows(read_csv(output_dir / "pressure_summary.csv"), args)


def summarize_pair(
    pressure: Dict[str, Any],
    scenario: Dict[str, Any],
    weight: float,
    dspo_row: Optional[Dict[str, str]],
    drpo_row: Optional[Dict[str, str]],
    dspo_dir: Path,
    drpo_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "stage": "search",
        "pressure_id": pressure["pressure_id"],
        "scenario_id": scenario["scenario_id"],
        "description": scenario["description"],
        "spo_loss_weight": weight,
        "outside_option_util": pressure["outside_option_util"],
        "home_util": pressure["home_util"],
        "k": scenario["k"],
        "revenue": scenario["revenue"],
        "min_price": "" if scenario["min_price"] is None else scenario["min_price"],
        "max_price": "" if scenario["max_price"] is None else scenario["max_price"],
        "dspo_dir": rel_path(dspo_dir),
        "drpo_dir": rel_path(drpo_dir),
        "status": "completed_pair" if dspo_row and drpo_row else "missing_pair",
    }
    if not dspo_row or not drpo_row:
        return out
    for metric in METRICS:
        dspo_val = to_float(dspo_row.get(metric))
        drpo_val = to_float(drpo_row.get(metric))
        out[f"dspo_{metric}"] = dspo_val
        out[f"drpo_{metric}"] = drpo_val
        out[f"delta_{metric}"] = (
            drpo_val - dspo_val if dspo_val is not None and drpo_val is not None else ""
        )
    delta_profit = to_float(out.get("delta_net_profit"))
    delta_quit = to_float(out.get("delta_quit_rate"))
    delta_served = to_float(out.get("delta_served_demand"))
    delta_base = to_float(out.get("delta_base_revenue")) or 0.0
    ex_base = (delta_profit - delta_base) if delta_profit is not None else None
    out["price_cost_delta_ex_base_revenue"] = ex_base
    out["profit_target_ok"] = delta_profit is not None and delta_profit >= args.target_profit_delta
    out["same_demand_like"] = (
        delta_quit is not None
        and delta_served is not None
        and abs(delta_quit) <= args.max_abs_quit_delta
        and abs(delta_served) <= args.max_abs_served_delta
    )
    out["structure_gain_like"] = (
        bool(out["profit_target_ok"])
        and bool(out["same_demand_like"])
        and ex_base is not None
        and ex_base > 0.0
    )
    if out["structure_gain_like"]:
        mechanism = "price_cost_structure"
    elif bool(out["profit_target_ok"]):
        mechanism = "demand_or_mixed_gain"
    elif delta_profit is not None and delta_profit > 0:
        mechanism = "small_positive_gain"
    else:
        mechanism = "no_drpo_advantage"
    out["mechanism_type"] = mechanism
    out["hard_fail"] = delta_profit is not None and delta_profit < -200.0
    return out


def run_search_stage(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    selected_pressure = load_selected_pressure(output_dir, args)
    if not selected_pressure:
        raise RuntimeError("No selected pressure rows found. Run --stage pressure first.")

    summaries: List[Dict[str, Any]] = []
    for pressure in selected_pressure:
        for scenario in SEARCH_SCENARIOS:
            overrides = build_overrides(pressure, scenario)
            dspo_dir = search_run_dir(output_dir, pressure, scenario, "DSPO")
            dspo_cmd = build_runner_cmd(
                args,
                dspo_dir,
                run_prefix=f"YJ_SEARCH_{pressure['pressure_id']}_{scenario['scenario_id']}_DSPO",
                folder_suffix=f"_yj_search_{pressure['pressure_id']}_{scenario['scenario_id']}_dspo",
                seeds=[args.pressure_seed],
                strategies=["DSPO"],
                overrides=overrides,
            )
            run_command(dspo_cmd, dspo_dir, args.dry_run)
            dspo_row = first_row(dspo_dir / "yanjiao_raw.csv", "DSPO")

            for weight in args.search_weights:
                drpo_dir = search_run_dir(output_dir, pressure, scenario, "DRPO", weight)
                drpo_cmd = build_runner_cmd(
                    args,
                    drpo_dir,
                    run_prefix=f"YJ_SEARCH_{pressure['pressure_id']}_{scenario['scenario_id']}_DRPO_w{int(round(weight * 100)):03d}",
                    folder_suffix=f"_yj_search_{pressure['pressure_id']}_{scenario['scenario_id']}_drpo_w{int(round(weight * 100)):03d}",
                    seeds=[args.pressure_seed],
                    strategies=["DRPO"],
                    overrides=overrides,
                    drpo_weight=weight,
                )
                run_command(drpo_cmd, drpo_dir, args.dry_run)
                drpo_row = first_row(drpo_dir / "yanjiao_raw.csv", "DRPO")
                summaries.append(
                    summarize_pair(pressure, scenario, weight, dspo_row, drpo_row, dspo_dir, drpo_dir, args)
                )
                write_csv(output_dir / "search_summary.csv", summaries)

    ranked = sorted(summaries, key=search_sort_key, reverse=True)
    selected = ranked[: args.top_validation]
    for row in summaries:
        row["selected_for_validation"] = row in selected
    write_csv(output_dir / "search_summary.csv", summaries)
    (output_dir / "selected_search_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summaries


def search_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
    profit_ok = 1 if as_bool(row.get("profit_target_ok")) else 0
    structure_ok = 1 if as_bool(row.get("structure_gain_like")) else 0
    same_demand = 1 if as_bool(row.get("same_demand_like")) else 0
    delta_profit = to_float(row.get("delta_net_profit")) or -1e9
    ex_base = to_float(row.get("price_cost_delta_ex_base_revenue")) or -1e9
    hard_fail = 1 if as_bool(row.get("hard_fail")) else 0
    return (-hard_fail, profit_ok, structure_ok, same_demand, delta_profit + 0.01 * ex_base)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_selected_search(output_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    selected_path = output_dir / "selected_search_candidates.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text(encoding="utf-8"))
    rows = read_csv(output_dir / "search_summary.csv")
    return sorted(rows, key=search_sort_key, reverse=True)[: args.top_validation]


def reconstruct_pressure(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pressure_id": row["pressure_id"],
        "outside_option_util": float(row["outside_option_util"]),
        "home_util": float(row["home_util"]),
        "k": int(float(row["k"])),
    }


def reconstruct_scenario(row: Dict[str, Any]) -> Dict[str, Any]:
    def optional_float(key: str) -> Optional[float]:
        value = row.get(key)
        return None if value in (None, "") else float(value)

    return {
        "scenario_id": row["scenario_id"],
        "description": row.get("description", ""),
        "revenue": float(row["revenue"]),
        "min_price": optional_float("min_price"),
        "max_price": optional_float("max_price"),
        "k": int(float(row["k"])),
    }


def summarize_validation_candidate(
    selected_id: str,
    source: Dict[str, Any],
    weight: float,
    dspo_rows: Dict[int, Dict[str, str]],
    drpo_rows: Dict[int, Dict[str, str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    for seed in args.validation_seeds:
        dspo = dspo_rows.get(seed)
        drpo = drpo_rows.get(seed)
        seed_row: Dict[str, Any] = {"seed": seed, "status": "completed_pair" if dspo and drpo else "missing_pair"}
        if dspo and drpo:
            for metric in METRICS:
                seed_row[f"delta_{metric}"] = safe_delta(dspo.get(metric), drpo.get(metric))
            seed_row["win"] = (seed_row.get("delta_net_profit") or -1e9) > 0
        per_seed.append(seed_row)

    def mean_delta(metric: str) -> Optional[float]:
        return mean(to_float(row.get(f"delta_{metric}")) for row in per_seed)

    delta_profit = mean_delta("net_profit")
    delta_quit = mean_delta("quit_rate")
    delta_served = mean_delta("served_demand")
    delta_base = mean_delta("base_revenue") or 0.0
    ex_base = (delta_profit - delta_base) if delta_profit is not None else None
    wins = sum(1 for row in per_seed if row.get("win"))
    complete = all(row["status"] == "completed_pair" for row in per_seed)
    structure_like = (
        delta_profit is not None
        and delta_profit >= args.target_profit_delta
        and delta_quit is not None
        and abs(delta_quit) <= args.max_abs_quit_delta
        and delta_served is not None
        and abs(delta_served) <= args.max_abs_served_delta
        and ex_base is not None
        and ex_base > 0.0
    )
    return {
        "stage": "validation",
        "selected_id": selected_id,
        "source_pressure_id": source["pressure_id"],
        "source_scenario_id": source["scenario_id"],
        "spo_loss_weight": weight,
        "outside_option_util": source["outside_option_util"],
        "home_util": source["home_util"],
        "k": source["k"],
        "revenue": source["revenue"],
        "min_price": source.get("min_price", ""),
        "max_price": source.get("max_price", ""),
        "complete": complete,
        "win_count": wins,
        "n_seeds": len(args.validation_seeds),
        "mean_delta_net_profit": delta_profit,
        "mean_delta_quit_rate": delta_quit,
        "mean_delta_served_demand": delta_served,
        "mean_price_cost_delta_ex_base_revenue": ex_base,
        "validation_pass": (
            complete
            and wins >= 2
            and delta_profit is not None
            and delta_profit >= args.target_profit_delta
        ),
        "structure_validation_pass": structure_like,
        "per_seed": json.dumps(per_seed, ensure_ascii=False),
    }


def rows_by_seed(raw_path: Path, label: str) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    for row in read_csv(raw_path):
        if row.get("label") == label:
            seed = int(float(row["seed"]))
            out[seed] = row
    return out


def run_validation_stage(args: argparse.Namespace, output_dir: Path) -> List[Dict[str, Any]]:
    selected_rows = load_selected_search(output_dir, args)
    if not selected_rows:
        raise RuntimeError("No selected search candidates found. Run --stage search first.")

    summaries: List[Dict[str, Any]] = []
    for idx, source in enumerate(selected_rows, 1):
        pressure = reconstruct_pressure(source)
        scenario = reconstruct_scenario(source)
        overrides = build_overrides(pressure, scenario)
        selected_id = f"C{idx}_{pressure['pressure_id']}_{scenario['scenario_id']}"

        dspo_dir = validation_run_dir(output_dir, selected_id, "DSPO")
        dspo_cmd = build_runner_cmd(
            args,
            dspo_dir,
            run_prefix=f"YJ_VAL_{selected_id}_DSPO",
            folder_suffix=f"_yj_val_{selected_id}_dspo",
            seeds=args.validation_seeds,
            strategies=["DSPO"],
            overrides=overrides,
        )
        run_command(dspo_cmd, dspo_dir, args.dry_run)
        dspo_rows = rows_by_seed(dspo_dir / "yanjiao_raw.csv", "DSPO")

        best_weight = float(source["spo_loss_weight"])
        validation_weights = []
        for weight in [best_weight, args.neighbor_weight]:
            if weight not in validation_weights:
                validation_weights.append(weight)

        for weight in validation_weights:
            drpo_dir = validation_run_dir(output_dir, selected_id, "DRPO", weight)
            drpo_cmd = build_runner_cmd(
                args,
                drpo_dir,
                run_prefix=f"YJ_VAL_{selected_id}_DRPO_w{int(round(weight * 100)):03d}",
                folder_suffix=f"_yj_val_{selected_id}_drpo_w{int(round(weight * 100)):03d}",
                seeds=args.validation_seeds,
                strategies=["DRPO"],
                overrides=overrides,
                drpo_weight=weight,
            )
            run_command(drpo_cmd, drpo_dir, args.dry_run)
            drpo_rows = rows_by_seed(drpo_dir / "yanjiao_raw.csv", "DRPO")
            summaries.append(summarize_validation_candidate(selected_id, source, weight, dspo_rows, drpo_rows, args))
            write_csv(output_dir / "validation_summary.csv", summaries)
    write_csv(output_dir / "validation_summary.csv", summaries)
    return summaries


def write_final_report(output_dir: Path) -> None:
    validation = read_csv(output_dir / "validation_summary.csv")
    search = read_csv(output_dir / "search_summary.csv")
    pressure = read_csv(output_dir / "pressure_summary.csv")
    best_validation = sorted(validation, key=validation_sort_key, reverse=True)[:5]
    best_search = sorted(search, key=search_sort_key, reverse=True)[:5]
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_validation": best_validation,
        "best_search": best_search,
        "selected_pressure": [r for r in pressure if str(r.get("selected_for_search")).lower() == "true"],
        "files": {
            "pressure_summary": rel_path(output_dir / "pressure_summary.csv"),
            "search_summary": rel_path(output_dir / "search_summary.csv"),
            "validation_summary": rel_path(output_dir / "validation_summary.csv"),
        },
    }
    (output_dir / "final_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def validation_sort_key(row: Dict[str, Any]) -> Tuple[int, int, float, float]:
    validation_pass = 1 if str(row.get("validation_pass")).lower() == "true" else 0
    structure_pass = 1 if str(row.get("structure_validation_pass")).lower() == "true" else 0
    wins = int(float(row.get("win_count") or 0))
    delta_profit = to_float(row.get("mean_delta_net_profit")) or -1e9
    return (validation_pass, structure_pass, wins, delta_profit)


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advantage_search_meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "args": vars(args),
                "pressure_candidates": PRESSURE_CANDIDATES,
                "search_scenarios": SEARCH_SCENARIOS,
                "success_criteria": {
                    "target_profit_delta": args.target_profit_delta,
                    "target_quit_window": [args.target_quit_min, args.target_quit_max],
                    "fallback_quit_window": [args.fallback_quit_min, args.fallback_quit_max],
                    "max_abs_quit_delta": args.max_abs_quit_delta,
                    "max_abs_served_delta": args.max_abs_served_delta,
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[INFO] output_dir={output_dir}", flush=True)

    if args.stage in ("all", "pressure"):
        run_pressure_stage(args, output_dir)
    if args.stage in ("all", "search"):
        if args.stage == "all" and args.dry_run:
            print("[INFO] Dry-run all: search stage uses selected_pressure from existing results only.", flush=True)
        run_search_stage(args, output_dir)
    if args.stage in ("all", "validation"):
        run_validation_stage(args, output_dir)
    if args.stage in ("all", "analyze", "pressure", "search", "validation"):
        write_final_report(output_dir)
        print(f"[DONE] report={output_dir / 'final_report.json'}", flush=True)


if __name__ == "__main__":
    main()
