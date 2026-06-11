#!/usr/bin/env python
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10.0,
}

DEFAULT_LEVELS = {
    "outside_option_util": [-2.0, -1.0, 2.0],
    "incentive_sens": [-0.35, -0.25, -0.15],
    "home_util": [1.0, 1.4, 1.8],
    "k": [5.0, 10.0, 15.0],
}

METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Medium OAT quick-check for DRPO (RC).")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--instance", default="RC")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 21])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_count", type=int, default=30)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--spo_warmup_episodes", type=int, default=5)
    p.add_argument("--spo_rampup_episodes", type=int, default=10)
    p.add_argument("--spo_loss_weight", type=float, default=0.7)
    p.add_argument("--run_timeout_sec", type=int, default=3600)
    p.add_argument("--output_dir", default="Experiments/analysis/drpo_medium_oat_check_3_9")
    p.add_argument("--folder_suffix", default="_probe_mid")
    p.add_argument("--experiment_prefix", default="MID_OAT_PROBE")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def token(v: float) -> str:
    return str(v).replace("-", "m").replace(".", "p")


def run_log_path(root: Path, run_id: str, seed: int, folder_suffix: str) -> Path:
    return root / "Experiments" / "Parcelpoint_py" / "pricing" / "DRPO" / f"{run_id}{folder_suffix}" / str(seed) / "Logs" / "logfile.log"


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.findall(text)
    return float(m[-1]) if m else None


def parse_metrics(log: Path) -> Dict[str, Optional[float]]:
    if not log.exists():
        return {"net_profit": None, "total_costs": None, "quit_rate": None}
    txt = log.read_text(encoding="utf-8", errors="ignore")
    return {
        "net_profit": extract_last(METRIC_REGEX["net_profit"], txt),
        "total_costs": extract_last(METRIC_REGEX["total_costs"], txt),
        "quit_rate": extract_last(METRIC_REGEX["quit_rate"], txt),
    }


def build_cmd(args: argparse.Namespace, run_id: str, seed: int, cfg: Dict[str, float]) -> List[str]:
    return [
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
        str(args.episodes),
        "--save_count",
        str(args.save_count),
        "--log_output",
        "file",
        "--debug",
        "False",
        "--gpu",
        str(args.gpu),
        "--outside_option_util",
        str(cfg["outside_option_util"]),
        "--incentive_sens",
        str(cfg["incentive_sens"]),
        "--home_util",
        str(cfg["home_util"]),
        "--k",
        str(int(cfg["k"])),
        "--spo_warmup_episodes",
        str(args.spo_warmup_episodes),
        "--spo_rampup_episodes",
        str(args.spo_rampup_episodes),
        "--spo_loss_weight",
        str(args.spo_loss_weight),
        "--experiment",
        run_id,
        "--folder_suffix",
        args.folder_suffix,
    ]


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def as_bool(x: object) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y"}


def summarize(rows: List[Dict]) -> List[Dict]:
    out = []
    for factor, vals in DEFAULT_LEVELS.items():
        for v in vals:
            rs = [
                r
                for r in rows
                if r["status"] == "completed"
                and r["factor"] == factor
                and r.get("net_profit") not in (None, "")
                and r.get("total_costs") not in (None, "")
                and r.get("quit_rate") not in (None, "")
                and np.isclose(float(r["value"]), float(v))
            ]
            if not rs:
                continue
            net = np.array([float(r["net_profit"]) for r in rs], dtype=float)
            cst = np.array([float(r["total_costs"]) for r in rs], dtype=float)
            qrt = np.array([float(r["quit_rate"]) for r in rs], dtype=float)
            out.append(
                {
                    "factor": factor,
                    "value": float(v),
                    "n_runs": len(rs),
                    "net_profit_mean": float(np.mean(net)),
                    "net_profit_std": float(np.std(net, ddof=1)) if len(net) > 1 else 0.0,
                    "total_costs_mean": float(np.mean(cst)),
                    "total_costs_std": float(np.std(cst, ddof=1)) if len(cst) > 1 else 0.0,
                    "quit_rate_mean": float(np.mean(qrt)),
                    "quit_rate_std": float(np.std(qrt, ddof=1)) if len(qrt) > 1 else 0.0,
                }
            )
    return out


def sensitivity_scores(summary_rows: List[Dict]) -> List[Dict]:
    scores = []
    for factor in DEFAULT_LEVELS:
        rs = sorted([r for r in summary_rows if r["factor"] == factor], key=lambda x: x["value"])
        if len(rs) < 2:
            continue
        x = np.array([float(r["value"]) for r in rs], dtype=float)
        y = np.array([float(r["net_profit_mean"]) for r in rs], dtype=float)
        dv = float(DEFAULT_CONFIG[factor])
        idx = np.where(np.isclose(x, dv))[0]
        if len(idx) == 0:
            continue
        i = int(idx[0])
        if i == 0:
            slope = (y[1] - y[0]) / (x[1] - x[0])
        elif i == len(x) - 1:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            slope = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        scores.append(
            {
                "factor": factor,
                "default_value": dv,
                "default_net_profit": float(y[i]),
                "local_slope": float(slope),
                "abs_local_slope": float(abs(slope)),
                "range_max_diff": float(np.max(y) - np.min(y)),
            }
        )
    return scores


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "medium_probe_raw.csv"
    summary_path = out_dir / "medium_probe_summary.csv"
    score_path = out_dir / "medium_probe_sensitivity_scores.csv"
    report_path = out_dir / "medium_probe_report.txt"

    rows: List[Dict] = []
    done_keys = set()
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r.get("status") == "completed":
                done_keys.add((r.get("factor"), float(r.get("value")), int(r.get("seed"))))

    jobs: List[Tuple[str, float, int]] = []
    for factor, values in DEFAULT_LEVELS.items():
        for value in values:
            for seed in args.seeds:
                jobs.append((factor, float(value), int(seed)))

    print(f"[INFO] medium OAT jobs={len(jobs)}, episodes={args.episodes}, seeds={args.seeds}")
    for idx, (factor, value, seed) in enumerate(jobs, 1):
        key = (factor, float(value), int(seed))
        if args.skip_existing and key in done_keys:
            print(f"[SKIP {idx}/{len(jobs)}] factor={factor}, value={value}, seed={seed}")
            continue

        cfg = dict(DEFAULT_CONFIG)
        cfg[factor] = float(value)
        run_id = f"{args.experiment_prefix}_{factor}_{token(value)}_e{args.episodes}_s{seed}"
        log_path = run_log_path(root, run_id, seed, args.folder_suffix)
        cmd = build_cmd(args, run_id, seed, cfg)

        print(f"[RUN {idx}/{len(jobs)}] factor={factor}, value={value}, seed={seed}")
        t0 = time.time()
        timed_out = False
        rc = None
        stdout_txt = ""
        try:
            cp = subprocess.run(
                cmd,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=args.run_timeout_sec if args.run_timeout_sec > 0 else None,
            )
            rc = cp.returncode
            stdout_txt = cp.stdout or ""
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout_txt = e.stdout or ""
        runtime_sec = time.time() - t0

        log_txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        metrics = parse_metrics(log_path)
        status = "timeout" if timed_out else ("completed" if rc == 0 else f"failed_rc_{rc}")
        row = {
            "factor": factor,
            "value": float(value),
            "seed": seed,
            "episodes": args.episodes,
            "status": status,
            "runtime_sec": runtime_sec,
            "net_profit": metrics["net_profit"],
            "total_costs": metrics["total_costs"],
            "quit_rate": metrics["quit_rate"],
            "spo_result_constructor_error": "SPOExperimentResult() takes no arguments" in log_txt,
            "spo_training_data_populated": "[SPO+ debug] spo_training_data populated" in log_txt,
            "spo_weight_positive": "[SPO+ debug] spo_weight became positive" in log_txt,
            "gpu_marker": "Using GPU device: cuda" in log_txt,
            "log_path": str(log_path),
            "command": " ".join(cmd),
            "tail": stdout_txt[-1200:],
        }
        rows.append(row)
        if status == "completed":
            done_keys.add(key)

        write_csv(
            raw_path,
            rows,
            [
                "factor",
                "value",
                "seed",
                "episodes",
                "status",
                "runtime_sec",
                "net_profit",
                "total_costs",
                "quit_rate",
                "spo_result_constructor_error",
                "spo_training_data_populated",
                "spo_weight_positive",
                "gpu_marker",
                "log_path",
                "command",
                "tail",
            ],
        )
        completed = [r for r in rows if r["status"] == "completed" and r.get("net_profit") not in (None, "")]
        sm = summarize(completed)
        sc = sensitivity_scores(sm)
        write_csv(
            summary_path,
            sm,
            [
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
            score_path,
            sc,
            [
                "factor",
                "default_value",
                "default_net_profit",
                "local_slope",
                "abs_local_slope",
                "range_max_diff",
            ],
        )
        print(
            f"  -> status={status}, runtime={runtime_sec:.1f}s, "
            f"net_profit={metrics['net_profit']}, quit={metrics['quit_rate']}, "
            f"gpu={'yes' if row['gpu_marker'] else 'no'}"
        )

    completed = [r for r in rows if r["status"] == "completed" and r.get("net_profit") not in (None, "")]
    total_jobs = len(jobs)
    completed_jobs = len(completed)
    missing = total_jobs - completed_jobs
    report_lines = [
        f"total_jobs={total_jobs}",
        f"completed_jobs={completed_jobs}",
        f"missing_jobs={missing}",
        f"no_spo_constructor_error_all={all(not as_bool(r['spo_result_constructor_error']) for r in completed) if completed else False}",
        f"spo_training_data_populated_count={sum(1 for r in completed if as_bool(r['spo_training_data_populated']))}/{completed_jobs}",
        f"spo_weight_positive_count={sum(1 for r in completed if as_bool(r['spo_weight_positive']))}/{completed_jobs}",
        f"gpu_marker_count={sum(1 for r in completed if as_bool(r['gpu_marker']))}/{completed_jobs}",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("[INFO] medium OAT check done.")
    print(f"[INFO] raw={raw_path}")
    print(f"[INFO] summary={summary_path}")
    print(f"[INFO] scores={score_path}")
    print(f"[INFO] report={report_path}")


if __name__ == "__main__":
    main()
