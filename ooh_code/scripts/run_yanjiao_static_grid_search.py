#!/usr/bin/env python
"""Run fixed static-pricing grid for the old Yanjiao paper batch."""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SEEDS = [40, 67]
DSPO_NET_PROFIT = {40: 9732.229, 67: 10675.1685}
DRPO_NET_PROFIT = {40: 9882.309000000001, 67: 10838.173}
PRICE_GRID = [
    (0.0, 0.0),
    (0.5, -0.5),
    (0.5, -1.0),
    (1.0, -1.0),
    (1.0, -1.5),
    (1.0, -2.0),
    (1.5, -1.5),
    (1.5, -2.0),
    (2.0, -2.0),
    (2.0, -3.0),
]

METRIC_PATTERNS = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "home_pickup_rate": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
}


def price_token(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def run_id(home: float, pp: float, seed: int) -> str:
    return f"YJ_STATIC_GRID_h{price_token(home)}_p{price_token(pp)}_seed{seed}"


def log_path(root: Path, home: float, pp: float, seed: int, suffix: str) -> Path:
    return (
        root
        / "Experiments"
        / "Parcelpoint_py"
        / "pricing"
        / "Baseline"
        / f"{run_id(home, pp, seed)}{suffix}"
        / str(seed)
        / "Logs"
        / "logfile.log"
    )


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


def build_cmd(py: str, home: float, pp: float, seed: int, suffix: str) -> List[str]:
    return [
        py,
        "run.py",
        "--algo_name",
        "Baseline",
        "--instance",
        "Beijing_Yanjiao",
        "--seed",
        str(seed),
        "--data_seed",
        "0",
        "--data_seed_test",
        "1",
        "--max_episodes",
        "150",
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
        "--learning_rate",
        "0.001",
        "--batch_size",
        "256",
        "--buffer_size",
        "500",
        "--init_theta_cnn",
        "0.75",
        "--cool_theta_cnn",
        "0.001176470588235294",
        "--spo_warmup_episodes",
        "5",
        "--spo_rampup_episodes",
        "10",
        "--spo_loss_weight",
        "0.7",
        "--price_home",
        str(home),
        "--price_pp",
        str(pp),
        "--experiment",
        run_id(home, pp, seed),
        "--folder_suffix",
        suffix,
    ]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "price_home",
        "price_pp",
        "seed",
        "net_profit",
        "total_costs",
        "quit_rate",
        "home_pickup_rate",
        "dspo_net_profit",
        "drpo_net_profit",
        "satisfies_drpo_gt_dspo_gt_static",
        "status",
        "runtime_sec",
        "log_path",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for home, pp in PRICE_GRID:
        group = [r for r in rows if float(r["price_home"]) == home and float(r["price_pp"]) == pp]
        if len(group) != len(SEEDS):
            continue
        ok = all(str(r["satisfies_drpo_gt_dspo_gt_static"]).lower() == "true" for r in group)
        mean_profit = sum(float(r["net_profit"]) for r in group) / len(group)
        out.append(
            {
                "price_home": home,
                "price_pp": pp,
                "mean_net_profit": mean_profit,
                "seed40_net_profit": next(float(r["net_profit"]) for r in group if int(r["seed"]) == 40),
                "seed67_net_profit": next(float(r["net_profit"]) for r in group if int(r["seed"]) == 67),
                "satisfies_both": ok,
            }
        )
    out.sort(key=lambda r: (not bool(r["satisfies_both"]), -float(r["mean_net_profit"])))
    return out


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    summary = summarize(rows)
    if not summary:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_executable", default=sys.executable)
    parser.add_argument("--folder_suffix", default="_yanjiao_paper_static_grid")
    parser.add_argument("--output_dir", default="Experiments/analysis/yanjiao_paper_static_grid_search")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    output_dir = root / args.output_dir
    raw_csv = output_dir / "static_grid_raw.csv"
    summary_csv = output_dir / "static_grid_summary.csv"
    rows: List[Dict[str, object]] = []

    for home, pp in PRICE_GRID:
        for seed in SEEDS:
            log = log_path(root, home, pp, seed, args.folder_suffix)
            cmd = build_cmd(args.python_executable, home, pp, seed, args.folder_suffix)
            t0 = time.time()
            status = "cached"
            metrics = parse_metrics(log) if args.skip_existing else None
            if metrics is None:
                print(f"[RUN] h={home} pp={pp} seed={seed}", flush=True)
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
                    raise RuntimeError(f"Run failed h={home} pp={pp} seed={seed}: {(cp.stdout or '')[-2000:]}")
                metrics = parse_metrics(log)
                if metrics is None:
                    raise RuntimeError(f"Metrics missing after run: {log}")
                status = "completed"
            runtime = time.time() - t0
            row: Dict[str, object] = {
                "price_home": home,
                "price_pp": pp,
                "seed": seed,
                "dspo_net_profit": DSPO_NET_PROFIT[seed],
                "drpo_net_profit": DRPO_NET_PROFIT[seed],
                "satisfies_drpo_gt_dspo_gt_static": (
                    DRPO_NET_PROFIT[seed] > DSPO_NET_PROFIT[seed] > float(metrics["net_profit"])
                ),
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
                f"[OK] h={home} pp={pp} seed={seed} "
                f"net_profit={metrics['net_profit']:.3f} "
                f"ok={row['satisfies_drpo_gt_dspo_gt_static']}",
                flush=True,
            )

    print(f"[DONE] Raw: {raw_csv}", flush=True)
    print(f"[DONE] Summary: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
