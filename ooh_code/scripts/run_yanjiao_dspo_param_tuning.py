#!/usr/bin/env python
"""Run DSPO parameter tuning for the old Yanjiao paper batch."""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


SEEDS = [40, 67]
BASELINE_DSPO = {40: 9732.229, 67: 10675.1685}
DRPO_REF = {40: 9882.309000000001, 67: 10838.173}
STATIC_2_5 = {40: 11058.959999999997, 67: 12217.799999999997}
STATIC_3_5 = {40: 11290.32, 67: 12606.6}

PARAM_SETS = [
    ("baseline_repro", {}),
    ("maxep300", {"max_episodes": 300}),
    ("init100", {"initial_phase_epochs": 100}),
    ("buffer1000", {"buffer_size": 1000}),
    ("lr0p0005", {"learning_rate": 0.0005}),
    ("maxprice5", {"max_price": 5.0}),
]

METRIC_PATTERNS = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "home_pickup_rate": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
}


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def parse_metrics(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    out = {k: extract_last(p, text) for k, p in METRIC_PATTERNS.items()}
    if out["net_profit"] is None or out["total_costs"] is None or out["quit_rate"] is None:
        return None
    return {k: v for k, v in out.items() if v is not None}


def run_id(label: str, seed: int) -> str:
    return f"YJ_DSPO_PARAM_{label}_seed{seed}"


def log_path(root: Path, label: str, seed: int, suffix: str) -> Path:
    return (
        root
        / "Experiments"
        / "Parcelpoint_py"
        / "pricing"
        / "DSPO"
        / f"{run_id(label, seed)}{suffix}"
        / str(seed)
        / "Logs"
        / "logfile.log"
    )


def cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def build_cmd(py: str, label: str, seed: int, suffix: str, overrides: Dict[str, object]) -> List[str]:
    params: Dict[str, object] = {
        "max_episodes": 150,
        "initial_phase_epochs": 50,
        "buffer_size": 500,
        "batch_size": 256,
        "learning_rate": 0.001,
        "init_theta_cnn": 0.75,
        "cool_theta_cnn": 1.0 / 850.0,
        "max_price": 3.5,
        "min_price": -5.0,
    }
    params.update(overrides)

    cmd = [
        py,
        "run.py",
        "--algo_name",
        "DSPO",
        "--instance",
        "Beijing_Yanjiao",
        "--seed",
        str(seed),
        "--data_seed",
        "0",
        "--data_seed_test",
        "1",
        "--save_count",
        "1",
        "--log_output",
        "file",
        "--debug",
        "False",
        "--gpu",
        "0",
        "--max_steps_r",
        "400",
        "--max_steps_p",
        "0.5",
        "--n_passengers",
        "400",
        "--n_vehicles",
        "35",
        "--veh_capacity",
        "12",
        "--k",
        "10",
        "--pricing",
        "True",
        "--home_util",
        "1.4",
        "--base_util",
        "-1.0",
        "--outside_option_util",
        "-1.0",
        "--incentive_sens",
        "-0.25",
        "--revenue",
        "50",
        "--fuel_cost",
        "0.6",
        "--driver_wage",
        "30",
        "--home_failure",
        "0.1",
        "--failure_cost",
        "20.0",
        "--l0_home",
        "2.5",
        "--l_mp",
        "0.75",
        "--hgs_reopt_time",
        "1.1",
        "--hgs_final_time",
        "1.5",
        "--spo_warmup_episodes",
        "5",
        "--spo_rampup_episodes",
        "10",
        "--spo_loss_weight",
        "0.0",
    ]
    for key in sorted(params):
        cmd.extend([f"--{key}", cli_value(params[key])])
    cmd.extend(["--experiment", run_id(label, seed), "--folder_suffix", suffix])
    return cmd


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "seed",
        "net_profit",
        "total_costs",
        "quit_rate",
        "home_pickup_rate",
        "baseline_dspo_net_profit",
        "delta_vs_baseline_dspo",
        "drpo_ref_net_profit",
        "delta_vs_drpo_ref",
        "static_2_5_net_profit",
        "static_3_5_net_profit",
        "status",
        "runtime_sec",
        "log_path",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    labels = [label for label, _ in PARAM_SETS]
    summary = []
    for label in labels:
        group = [r for r in rows if r["label"] == label]
        if len(group) != len(SEEDS):
            continue
        mean_profit = sum(float(r["net_profit"]) for r in group) / len(group)
        mean_delta = sum(float(r["delta_vs_baseline_dspo"]) for r in group) / len(group)
        both_improve = all(float(r["delta_vs_baseline_dspo"]) > 0 for r in group)
        summary.append(
            {
                "label": label,
                "mean_net_profit": mean_profit,
                "mean_delta_vs_baseline_dspo": mean_delta,
                "seed40_net_profit": next(float(r["net_profit"]) for r in group if int(r["seed"]) == 40),
                "seed67_net_profit": next(float(r["net_profit"]) for r in group if int(r["seed"]) == 67),
                "seed40_delta": next(float(r["delta_vs_baseline_dspo"]) for r in group if int(r["seed"]) == 40),
                "seed67_delta": next(float(r["delta_vs_baseline_dspo"]) for r in group if int(r["seed"]) == 67),
                "mean_quit_rate": sum(float(r["quit_rate"]) for r in group) / len(group),
                "both_improve": both_improve,
            }
        )
    summary.sort(key=lambda r: (not bool(r["both_improve"]), -float(r["mean_net_profit"])))
    if not summary:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--folder_suffix", default="_yanjiao_paper_param")
    parser.add_argument("--output_dir", default="Experiments/analysis/dspo_param_tuning_yanjiao_paper")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output_dir = root / args.output_dir
    raw_csv = output_dir / "dspo_param_raw.csv"
    summary_csv = output_dir / "dspo_param_summary.csv"
    rows: List[Dict[str, object]] = []

    for label, overrides in PARAM_SETS:
        for seed in SEEDS:
            log = log_path(root, label, seed, args.folder_suffix)
            cmd = build_cmd(args.python_executable, label, seed, args.folder_suffix, overrides)
            t0 = time.time()
            status = "cached"
            metrics = parse_metrics(log) if args.skip_existing else None
            if metrics is None:
                print(f"[RUN] label={label} seed={seed}", flush=True)
                cp = subprocess.run(
                    cmd,
                    cwd=root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
                if cp.returncode != 0:
                    raise RuntimeError(f"Run failed label={label} seed={seed}: {(cp.stdout or '')[-2500:]}")
                metrics = parse_metrics(log)
                if metrics is None:
                    raise RuntimeError(f"Metrics missing after run: {log}")
                status = "completed"
            runtime = time.time() - t0
            row: Dict[str, object] = {
                "label": label,
                "seed": seed,
                "baseline_dspo_net_profit": BASELINE_DSPO[seed],
                "delta_vs_baseline_dspo": float(metrics["net_profit"]) - BASELINE_DSPO[seed],
                "drpo_ref_net_profit": DRPO_REF[seed],
                "delta_vs_drpo_ref": float(metrics["net_profit"]) - DRPO_REF[seed],
                "static_2_5_net_profit": STATIC_2_5[seed],
                "static_3_5_net_profit": STATIC_3_5[seed],
                "status": status,
                "runtime_sec": runtime,
                "log_path": str(log),
                "command": " ".join(cmd),
            }
            row.update(metrics)
            rows.append(row)
            write_csv(raw_csv, rows)
            write_summary(summary_csv, rows)
            print(
                f"[OK] label={label} seed={seed} "
                f"net_profit={metrics['net_profit']:.3f} "
                f"delta={row['delta_vs_baseline_dspo']:.3f}",
                flush=True,
            )

    print(f"[DONE] Raw: {raw_csv}", flush=True)
    print(f"[DONE] Summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
