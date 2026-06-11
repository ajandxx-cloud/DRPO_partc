#!/usr/bin/env python
"""Probe whether Yanjiao DRPO gains can come from price/cost structure.

The probe runs paired DSPO/DRPO jobs for a few small scenario variants and
summarizes whether net-profit gains are driven by served demand / quit-rate
changes or by pricing and operating-cost decomposition.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


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

SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "base_w09",
        "description": "Current Yanjiao fast setting, DRPO weight 0.9.",
        "overrides": {},
    },
    {
        "name": "price_window_tight_discount",
        "description": "Reduce discount room and allow a higher positive charge.",
        "overrides": {"min_price": -3.5, "max_price": 5.0},
    },
    {
        "name": "lower_revenue_40",
        "description": "Reduce base fare revenue so served-demand changes dominate less.",
        "overrides": {"revenue": 40.0},
    },
    {
        "name": "stable_acceptance_home16",
        "description": "Raise home utility to make DSPO/DRPO serve nearly the same demand.",
        "overrides": {"home_util": 1.6},
    },
    {
        "name": "k5_choice_pressure",
        "description": "Use fewer meeting-point candidates to amplify routing/choice structure.",
        "overrides": {"k": 5},
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yanjiao paired mechanism probe")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=67)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--spo_loss_weight", type=float, default=0.9)
    p.add_argument("--spo_label_sample_size", type=int, default=4)
    p.add_argument("--spo_batch_size", type=int, default=8)
    p.add_argument("--hgs_reopt_time", type=float, default=0.2)
    p.add_argument("--hgs_final_time", type=float, default=0.2)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--only", nargs="*", default=None, help="Optional scenario names to run")
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def build_cmd(args: argparse.Namespace, scenario: Dict[str, Any], out_dir: Path) -> List[str]:
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
        str(args.seed),
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        "hep",
        "--strategies",
        "DSPO",
        "DRPO",
        "--run_prefix",
        f"YJ_MECH_{scenario['name']}",
        "--folder_suffix",
        f"_yj_mech_{scenario['name']}",
        "--output_dir",
        str(out_dir),
        "--allow_existing_output_dir",
        "--persist_every_n",
        "1",
        "--max_retries",
        "0",
        "--run_timeout_sec",
        "0",
        "--dspo_spo_loss_weight",
        "0.0",
        "--drpo_spo_loss_weight",
        repr(float(args.spo_loss_weight)),
        "--hgs_reopt_time_override",
        repr(float(args.hgs_reopt_time)),
        "--hgs_final_time_override",
        repr(float(args.hgs_final_time)),
        "--spo_label_sample_size_override",
        str(int(args.spo_label_sample_size)),
        "--spo_batch_size_override",
        str(int(args.spo_batch_size)),
    ]
    if args.allow_cpu:
        cmd.append("--allow_cpu")

    override_to_arg = {
        "k": "--k_override",
        "revenue": "--revenue_override",
        "home_util": "--home_util_override",
        "outside_option_util": "--outside_option_util_override",
        "min_price": "--min_price_override",
        "max_price": "--max_price_override",
        "incentive_sens": "--incentive_sens_override",
    }
    for key, value in scenario.get("overrides", {}).items():
        cmd.extend([override_to_arg[key], str(value)])
    return cmd


def summarize_scenario(scenario: Dict[str, Any], raw_path: Path) -> Dict[str, Any]:
    rows = read_csv(raw_path)
    dspo = next((r for r in rows if r.get("label") == "DSPO"), None)
    drpo = next((r for r in rows if r.get("label") == "DRPO"), None)
    out: Dict[str, Any] = {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "overrides": json.dumps(scenario.get("overrides", {}), sort_keys=True),
        "status": "completed" if dspo and drpo else "missing_pair",
    }
    if not dspo or not drpo:
        return out

    for metric in METRICS:
        d_val = to_float(dspo.get(metric))
        r_val = to_float(drpo.get(metric))
        out[f"dspo_{metric}"] = d_val
        out[f"drpo_{metric}"] = r_val
        out[f"delta_{metric}"] = r_val - d_val

    out["price_cost_delta_ex_base_revenue"] = (
        out["delta_net_profit"] - out["delta_base_revenue"]
    )
    out["same_demand_like"] = (
        abs(out["delta_served_demand"]) <= 1.0 and abs(out["delta_quit_rate"]) <= 0.05
    )
    out["structure_gain_like"] = (
        out["delta_net_profit"] > 0.0
        and abs(out["delta_served_demand"]) <= 1.0
        and out["price_cost_delta_ex_base_revenue"] > 0.0
    )
    return out


def main() -> None:
    args = parse_args()
    selected = [s for s in SCENARIOS if args.only is None or s["name"] in set(args.only)]
    if not selected:
        raise RuntimeError("No scenarios selected")

    out_root = ROOT / (
        args.output_dir
        or f"Experiments/analysis/yanjiao_mechanism_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "probe_meta.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "seed": args.seed,
                "episodes": args.episodes,
                "eval_episodes": args.eval_episodes,
                "spo_loss_weight": args.spo_loss_weight,
                "spo_label_sample_size": args.spo_label_sample_size,
                "spo_batch_size": args.spo_batch_size,
                "hgs_reopt_time": args.hgs_reopt_time,
                "hgs_final_time": args.hgs_final_time,
                "scenarios": selected,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary_rows: List[Dict[str, Any]] = []
    for scenario in selected:
        scenario_dir = out_root / scenario["name"]
        scenario_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_cmd(args, scenario, scenario_dir)
        (scenario_dir / "command.txt").write_text(" ".join(cmd), encoding="utf-8")
        print(f"[RUN] {scenario['name']}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue
        cp = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        (scenario_dir / "runner_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
        row = summarize_scenario(scenario, scenario_dir / "yanjiao_raw.csv")
        row["returncode"] = cp.returncode
        summary_rows.append(row)
        write_csv(out_root / "mechanism_probe_summary.csv", summary_rows)
        print(f"[DONE] {scenario['name']} rc={cp.returncode}", flush=True)
        if cp.returncode != 0:
            raise RuntimeError(f"Scenario failed: {scenario['name']} ({scenario_dir / 'runner_stdout.log'})")

    if summary_rows:
        write_csv(out_root / "mechanism_probe_summary.csv", summary_rows)
    print(f"[DONE] summary={out_root / 'mechanism_probe_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
