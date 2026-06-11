#!/usr/bin/env python
import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FACTORS = {
    "outside_option_util": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "incentive_sens": [-0.35, -0.30, -0.25, -0.20, -0.15],
    "home_util": [1.0, 1.2, 1.4, 1.6, 1.8],
    "k": [5, 7, 10, 12, 15],
}

DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10,
}

DEFAULT_SEEDS = [0, 21, 42]

METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
}


@dataclass
class RunRecord:
    stage: str
    factor: str
    value: float
    seed: int
    episodes: int
    run_id: str
    status: str
    runtime_sec: float
    net_profit: float
    total_costs: float
    quit_rate: float
    log_path: str
    command: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OAT sensitivity analysis for DRPO on RC.")
    parser.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus"])
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--data_seed_test", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run_prefix", default="SENS_DRPO")
    parser.add_argument("--folder_suffix", default="_sens")
    parser.add_argument("--python_executable", default=sys.executable, help="Python executable used for subprocess runs.")
    parser.add_argument("--stage1_episodes", type=int, default=80)
    parser.add_argument("--stage2_episodes", type=int, default=200)
    parser.add_argument("--save_count", type=int, default=20)
    parser.add_argument("--spo_warmup_episodes", type=int, default=5)
    parser.add_argument("--spo_rampup_episodes", type=int, default=10)
    parser.add_argument("--spo_loss_weight", type=float, default=0.7)
    parser.add_argument("--skip_existing", action="store_true", help="Skip runs with complete existing logs.")
    parser.add_argument("--run_smoke_validation", action="store_true")
    parser.add_argument("--only_smoke", action="store_true", help="Run smoke validation only, then exit.")
    parser.add_argument("--smoke_seed", type=int, default=0)
    parser.add_argument("--smoke_episodes", type=int, default=20)
    parser.add_argument("--allow_smoke_failure", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="Experiments/analysis/drpo_sensitivity",
        help="Directory for CSV summaries, plots, and validation reports.",
    )
    return parser.parse_args()


def _value_token(value: float) -> str:
    token = str(value)
    token = token.replace("-", "m").replace(".", "p")
    return token


def _run_log_path(root: Path, run_id: str, folder_suffix: str, seed: int) -> Path:
    exp_dir = root / "Experiments" / "Parcelpoint_py" / "pricing" / "DRPO" / f"{run_id}{folder_suffix}"
    return exp_dir / str(seed) / "Logs" / "logfile.log"


def _extract_last_float(pattern: re.Pattern, text: str) -> Optional[float]:
    matches = pattern.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def parse_metrics_from_log(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    net_profit = _extract_last_float(METRIC_REGEX["net_profit"], text)
    total_costs = _extract_last_float(METRIC_REGEX["total_costs"], text)
    quit_rate = _extract_last_float(METRIC_REGEX["quit_rate"], text)
    if net_profit is None or total_costs is None or quit_rate is None:
        return None
    return {
        "net_profit": net_profit,
        "total_costs": total_costs,
        "quit_rate": quit_rate,
    }


def parse_smoke_signals(log_path: Path) -> Dict[str, bool]:
    result = {
        "spo_result_constructor_error": False,
        "spo_training_data_populated": False,
        "spo_weight_positive": False,
    }
    if not log_path.exists():
        return result

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    result["spo_result_constructor_error"] = "SPOExperimentResult() takes no arguments" in text
    result["spo_training_data_populated"] = "[SPO+ debug] spo_training_data populated" in text
    result["spo_weight_positive"] = "[SPO+ debug] spo_weight became positive" in text
    return result


def build_run_command(
    args: argparse.Namespace,
    root: Path,
    run_id: str,
    seed: int,
    episodes: int,
    config: Dict[str, float],
    extra_overrides: Optional[Dict[str, float]] = None,
) -> List[str]:
    cfg = dict(config)
    if extra_overrides:
        cfg.update(extra_overrides)

    cmd = [
        args.python_executable,
        "run.py",
        "--algo_name",
        "DRPO",
        "--instance",
        args.instance,
        "--seed",
        str(seed),
        "--data_seed",
        str(args.data_seed),
        "--data_seed_test",
        str(args.data_seed_test),
        "--max_episodes",
        str(episodes),
        "--save_count",
        str(args.save_count),
        "--log_output",
        "file",
        "--debug",
        "False",
        "--outside_option_util",
        str(cfg["outside_option_util"]),
        "--incentive_sens",
        str(cfg["incentive_sens"]),
        "--home_util",
        str(cfg["home_util"]),
        "--k",
        str(int(cfg["k"])),
        "--spo_warmup_episodes",
        str(int(cfg.get("spo_warmup_episodes", args.spo_warmup_episodes))),
        "--spo_rampup_episodes",
        str(int(cfg.get("spo_rampup_episodes", args.spo_rampup_episodes))),
        "--spo_loss_weight",
        str(cfg.get("spo_loss_weight", args.spo_loss_weight)),
        "--experiment",
        run_id,
        "--folder_suffix",
        args.folder_suffix,
    ]
    return cmd


def execute_single_run(
    args: argparse.Namespace,
    root: Path,
    stage: str,
    factor: str,
    value: float,
    seed: int,
    episodes: int,
    config: Dict[str, float],
    extra_overrides: Optional[Dict[str, float]] = None,
) -> RunRecord:
    value_token = _value_token(value)
    run_id = f"{args.run_prefix}_{stage}_{factor}_{value_token}"
    log_path = _run_log_path(root, run_id, args.folder_suffix, seed)
    cmd = build_run_command(args, root, run_id, seed, episodes, config, extra_overrides=extra_overrides)

    if args.skip_existing:
        cached_metrics = parse_metrics_from_log(log_path)
        if cached_metrics is not None:
            return RunRecord(
                stage=stage,
                factor=factor,
                value=float(value),
                seed=seed,
                episodes=episodes,
                run_id=run_id,
                status="cached",
                runtime_sec=0.0,
                net_profit=cached_metrics["net_profit"],
                total_costs=cached_metrics["total_costs"],
                quit_rate=cached_metrics["quit_rate"],
                log_path=str(log_path),
                command=" ".join(cmd),
            )

    t0 = time.time()
    completed = subprocess.run(
        cmd,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    runtime = time.time() - t0

    metrics = parse_metrics_from_log(log_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Run failed (return code {completed.returncode}) for {run_id}, seed={seed}. "
            f"Last output:\n{completed.stdout[-2000:]}"
        )
    if metrics is None:
        raise RuntimeError(f"Could not parse metrics from log: {log_path}")

    return RunRecord(
        stage=stage,
        factor=factor,
        value=float(value),
        seed=seed,
        episodes=episodes,
        run_id=run_id,
        status="completed",
        runtime_sec=runtime,
        net_profit=metrics["net_profit"],
        total_costs=metrics["total_costs"],
        quit_rate=metrics["quit_rate"],
        log_path=str(log_path),
        command=" ".join(cmd),
    )


def summarize_records(records: List[RunRecord]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, float], List[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.factor, rec.value), []).append(rec)

    summaries = []
    for (factor, value), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        net = np.array([r.net_profit for r in recs], dtype=float)
        costs = np.array([r.total_costs for r in recs], dtype=float)
        quit_rates = np.array([r.quit_rate for r in recs], dtype=float)

        summaries.append(
            {
                "factor": factor,
                "value": value,
                "n_runs": len(recs),
                "net_profit_mean": float(np.mean(net)),
                "net_profit_std": float(np.std(net, ddof=1)) if len(net) > 1 else 0.0,
                "total_costs_mean": float(np.mean(costs)),
                "total_costs_std": float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
                "quit_rate_mean": float(np.mean(quit_rates)),
                "quit_rate_std": float(np.std(quit_rates, ddof=1)) if len(quit_rates) > 1 else 0.0,
            }
        )
    return summaries


def choose_stage2_candidates(stage1_summary: List[Dict[str, float]]) -> Dict[str, float]:
    candidates: Dict[str, float] = {}
    for factor in DEFAULT_FACTORS:
        rows = [r for r in stage1_summary if r["factor"] == factor]
        if not rows:
            continue
        default_value = DEFAULT_CONFIG[factor]
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r["net_profit_mean"],
                -abs(r["value"] - default_value),
                -r["value"],
            ),
            reverse=True,
        )
        candidates[factor] = float(rows_sorted[0]["value"])
    return candidates


def compute_sensitivity_scores(stage1_summary: List[Dict[str, float]]) -> List[Dict[str, float]]:
    scores = []
    for factor in DEFAULT_FACTORS:
        rows = sorted([r for r in stage1_summary if r["factor"] == factor], key=lambda x: x["value"])
        if len(rows) < 2:
            continue

        x = np.array([r["value"] for r in rows], dtype=float)
        y = np.array([r["net_profit_mean"] for r in rows], dtype=float)

        default_value = DEFAULT_CONFIG[factor]
        idx = int(np.where(np.isclose(x, default_value))[0][0])

        if idx == 0:
            local_slope = (y[1] - y[0]) / (x[1] - x[0])
        elif idx == len(x) - 1:
            local_slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            local_slope = (y[idx + 1] - y[idx - 1]) / (x[idx + 1] - x[idx - 1])

        range_diff = float(np.max(y) - np.min(y))
        scores.append(
            {
                "factor": factor,
                "default_value": default_value,
                "default_net_profit": float(y[idx]),
                "local_slope": float(local_slope),
                "abs_local_slope": float(abs(local_slope)),
                "range_max_diff": range_diff,
            }
        )
    return scores


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_stage1_curves(stage1_summary: List[Dict[str, float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("net_profit_mean", "net_profit_std", "Net Profit"),
        ("total_costs_mean", "total_costs_std", "Total Costs"),
        ("quit_rate_mean", "quit_rate_std", "Quit Rate (%)"),
    ]

    for factor in DEFAULT_FACTORS:
        rows = sorted([r for r in stage1_summary if r["factor"] == factor], key=lambda x: x["value"])
        if not rows:
            continue

        x = np.array([r["value"] for r in rows], dtype=float)
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        for ax, (mean_key, std_key, label) in zip(axes, metrics):
            mean = np.array([r[mean_key] for r in rows], dtype=float)
            std = np.array([r[std_key] for r in rows], dtype=float)
            ax.plot(x, mean, marker="o", linewidth=2)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25)
            ax.axvline(DEFAULT_CONFIG[factor], color="gray", linestyle="--", linewidth=1)

        axes[-1].set_xlabel(factor)
        fig.suptitle(f"Stage1 OAT Sensitivity - {factor}")
        fig.tight_layout()
        fig.savefig(output_dir / f"stage1_{factor}.png", dpi=200)
        plt.close(fig)


def run_smoke_validation(args: argparse.Namespace, root: Path, output_dir: Path) -> None:
    stage = "smoke"
    factor = "smoke_config"
    value = DEFAULT_CONFIG["outside_option_util"]
    config = dict(DEFAULT_CONFIG)
    smoke_overrides = {
        "spo_warmup_episodes": 1,
        "spo_rampup_episodes": 2,
        "spo_loss_weight": args.spo_loss_weight,
    }

    record = execute_single_run(
        args=args,
        root=root,
        stage=stage,
        factor=factor,
        value=value,
        seed=args.smoke_seed,
        episodes=args.smoke_episodes,
        config=config,
        extra_overrides=smoke_overrides,
    )

    signals = parse_smoke_signals(Path(record.log_path))
    report_lines = [
        f"smoke_log={record.log_path}",
        f"spo_result_constructor_error={signals['spo_result_constructor_error']}",
        f"spo_training_data_populated={signals['spo_training_data_populated']}",
        f"spo_weight_positive={signals['spo_weight_positive']}",
        f"net_profit={record.net_profit}",
        f"total_costs={record.total_costs}",
        f"quit_rate={record.quit_rate}",
    ]
    report_path = output_dir / "smoke_validation.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    smoke_ok = (
        (not signals["spo_result_constructor_error"])
        and signals["spo_training_data_populated"]
        and signals["spo_weight_positive"]
    )
    if (not smoke_ok) and (not args.allow_smoke_failure):
        raise RuntimeError(
            "Smoke validation failed. Check "
            f"{report_path} and log {record.log_path}"
        )


def build_stage_jobs(seeds: List[int], stage: str, stage2_candidates: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, int, Dict[str, float]]]:
    jobs = []
    if stage == "stage1":
        for factor, values in DEFAULT_FACTORS.items():
            for value in values:
                for seed in seeds:
                    cfg = dict(DEFAULT_CONFIG)
                    cfg[factor] = value
                    jobs.append((factor, float(value), seed, cfg))
    elif stage == "stage2":
        if not stage2_candidates:
            return jobs
        for factor, value in stage2_candidates.items():
            for seed in seeds:
                cfg = dict(DEFAULT_CONFIG)
                cfg[factor] = value
                jobs.append((factor, float(value), seed, cfg))
    return jobs


def to_record_dict(rec: RunRecord) -> Dict:
    return {
        "stage": rec.stage,
        "factor": rec.factor,
        "value": rec.value,
        "seed": rec.seed,
        "episodes": rec.episodes,
        "run_id": rec.run_id,
        "status": rec.status,
        "runtime_sec": rec.runtime_sec,
        "net_profit": rec.net_profit,
        "total_costs": rec.total_costs,
        "quit_rate": rec.quit_rate,
        "log_path": rec.log_path,
        "command": rec.command,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Starting DRPO OAT sensitivity analysis")
    print(f"[INFO] Output dir: {output_dir}")

    if args.run_smoke_validation:
        print("[INFO] Running smoke validation first...")
        run_smoke_validation(args, root, output_dir)
        print("[INFO] Smoke validation completed.")
        if args.only_smoke:
            print("[INFO] --only_smoke is set. Exiting after smoke validation.")
            return

    stage1_jobs = build_stage_jobs(args.seeds, stage="stage1")
    stage1_records: List[RunRecord] = []
    print(f"[INFO] Stage1 jobs: {len(stage1_jobs)}")
    for idx, (factor, value, seed, config) in enumerate(stage1_jobs, start=1):
        print(
            f"[Stage1 {idx}/{len(stage1_jobs)}] factor={factor}, value={value}, seed={seed}, "
            f"episodes={args.stage1_episodes}"
        )
        rec = execute_single_run(
            args=args,
            root=root,
            stage="stage1",
            factor=factor,
            value=value,
            seed=seed,
            episodes=args.stage1_episodes,
            config=config,
        )
        stage1_records.append(rec)

    stage1_summary = summarize_records(stage1_records)
    stage2_candidates = choose_stage2_candidates(stage1_summary)
    stage2_jobs = build_stage_jobs(args.seeds, stage="stage2", stage2_candidates=stage2_candidates)

    stage2_records: List[RunRecord] = []
    print(f"[INFO] Stage2 jobs: {len(stage2_jobs)}")
    for idx, (factor, value, seed, config) in enumerate(stage2_jobs, start=1):
        print(
            f"[Stage2 {idx}/{len(stage2_jobs)}] factor={factor}, value={value}, seed={seed}, "
            f"episodes={args.stage2_episodes}"
        )
        rec = execute_single_run(
            args=args,
            root=root,
            stage="stage2",
            factor=factor,
            value=value,
            seed=seed,
            episodes=args.stage2_episodes,
            config=config,
        )
        stage2_records.append(rec)

    stage2_summary = summarize_records(stage2_records)
    sensitivity_scores = compute_sensitivity_scores(stage1_summary)

    stage1_raw_rows = [to_record_dict(r) for r in stage1_records]
    stage2_raw_rows = [to_record_dict(r) for r in stage2_records]

    write_csv(
        output_dir / "stage1_raw.csv",
        stage1_raw_rows,
        fieldnames=[
            "stage",
            "factor",
            "value",
            "seed",
            "episodes",
            "run_id",
            "status",
            "runtime_sec",
            "net_profit",
            "total_costs",
            "quit_rate",
            "log_path",
            "command",
        ],
    )
    write_csv(
        output_dir / "stage2_raw.csv",
        stage2_raw_rows,
        fieldnames=[
            "stage",
            "factor",
            "value",
            "seed",
            "episodes",
            "run_id",
            "status",
            "runtime_sec",
            "net_profit",
            "total_costs",
            "quit_rate",
            "log_path",
            "command",
        ],
    )
    write_csv(
        output_dir / "stage1_summary.csv",
        stage1_summary,
        fieldnames=[
            "factor",
            "value",
            "n_runs",
            "net_profit_mean",
            "net_profit_std",
            "total_costs_mean",
            "total_costs_std",
            "quit_rate_mean",
            "quit_rate_std",
        ],
    )
    write_csv(
        output_dir / "stage2_summary.csv",
        stage2_summary,
        fieldnames=[
            "factor",
            "value",
            "n_runs",
            "net_profit_mean",
            "net_profit_std",
            "total_costs_mean",
            "total_costs_std",
            "quit_rate_mean",
            "quit_rate_std",
        ],
    )
    write_csv(
        output_dir / "sensitivity_scores.csv",
        sensitivity_scores,
        fieldnames=[
            "factor",
            "default_value",
            "default_net_profit",
            "local_slope",
            "abs_local_slope",
            "range_max_diff",
        ],
    )

    candidate_rows = [{"factor": k, "stage2_best_value": v} for k, v in stage2_candidates.items()]
    write_csv(output_dir / "stage2_candidates.csv", candidate_rows, fieldnames=["factor", "stage2_best_value"])

    plot_stage1_curves(stage1_summary, output_dir / "plots")

    expected_stage1 = len(DEFAULT_FACTORS) * 5 * len(args.seeds)
    expected_stage2 = len(DEFAULT_FACTORS) * len(args.seeds)
    validation_report = [
        f"expected_stage1_runs={expected_stage1}",
        f"actual_stage1_runs={len(stage1_records)}",
        f"expected_stage2_runs={expected_stage2}",
        f"actual_stage2_runs={len(stage2_records)}",
    ]
    (output_dir / "validation_report.txt").write_text("\n".join(validation_report), encoding="utf-8")

    print("[INFO] Analysis finished.")
    print(f"[INFO] Stage1 summary: {output_dir / 'stage1_summary.csv'}")
    print(f"[INFO] Stage2 summary: {output_dir / 'stage2_summary.csv'}")
    print(f"[INFO] Sensitivity scores: {output_dir / 'sensitivity_scores.csv'}")
    print(f"[INFO] Plots dir: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
