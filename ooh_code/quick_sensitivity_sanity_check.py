#!/usr/bin/env python
import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10,
}
OUTSIDE_VALUES = [-2.0, -1.0, 0.0, 1.0, 2.0]
K_VALUES = [5, 10, 15]
METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick sensitivity sanity check for DRPO (RC).")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--instance", default="RC")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_count", type=int, default=10)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--spo_warmup_episodes", type=int, default=5)
    p.add_argument("--spo_rampup_episodes", type=int, default=10)
    p.add_argument("--spo_loss_weight", type=float, default=0.7)
    p.add_argument("--run_timeout_sec", type=int, default=2400)
    p.add_argument("--outside_values", nargs="*", type=float, default=OUTSIDE_VALUES)
    p.add_argument("--k_values", nargs="*", type=float, default=K_VALUES)
    p.add_argument("--output_dir", default="Experiments/analysis/drpo_quick_sanity")
    return p.parse_args()


def token(v: float) -> str:
    return str(v).replace("-", "m").replace(".", "p")


def run_log_path(root: Path, run_id: str, seed: int, folder_suffix: str) -> Path:
    return (
        root
        / "Experiments"
        / "Parcelpoint_py"
        / "pricing"
        / "DRPO"
        / f"{run_id}{folder_suffix}"
        / str(seed)
        / "Logs"
        / "logfile.log"
    )


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.findall(text)
    return float(m[-1]) if m else None


def parse_metrics(log_path: Path) -> Dict[str, Optional[float]]:
    if not log_path.exists():
        return {"net_profit": None, "total_costs": None, "quit_rate": None}
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return {
        "net_profit": extract_last(METRIC_REGEX["net_profit"], txt),
        "total_costs": extract_last(METRIC_REGEX["total_costs"], txt),
        "quit_rate": extract_last(METRIC_REGEX["quit_rate"], txt),
    }


def build_cmd(args: argparse.Namespace, run_id: str, cfg: Dict[str, float], folder_suffix: str) -> List[str]:
    return [
        args.python_executable,
        "run.py",
        "--algo_name",
        "DRPO",
        "--instance",
        args.instance,
        "--seed",
        str(args.seed),
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
        folder_suffix,
    ]


def run_one(
    args: argparse.Namespace,
    root: Path,
    factor: str,
    value: float,
    folder_suffix: str,
) -> Dict[str, object]:
    cfg = dict(DEFAULT_CONFIG)
    cfg[factor] = value
    run_id = f"QUICK_SENS_{factor}_{token(value)}_e{args.episodes}_s{args.seed}"
    log_path = run_log_path(root, run_id, args.seed, folder_suffix)
    cmd = build_cmd(args, run_id, cfg, folder_suffix)
    t0 = time.time()
    cp_returncode: Optional[int] = None
    cp_stdout = ""
    timed_out = False
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
        cp_returncode = cp.returncode
        cp_stdout = cp.stdout or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        cp_returncode = None
        cp_stdout = (e.stdout or "")
    runtime_sec = time.time() - t0

    txt = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    metrics = parse_metrics(log_path)
    row: Dict[str, object] = {
        "factor": factor,
        "value": value,
        "seed": args.seed,
        "episodes": args.episodes,
        "status": "timeout" if timed_out else ("completed" if cp_returncode == 0 else f"failed_rc_{cp_returncode}"),
        "runtime_sec": runtime_sec,
        "net_profit": metrics["net_profit"],
        "total_costs": metrics["total_costs"],
        "quit_rate": metrics["quit_rate"],
        "spo_result_constructor_error": "SPOExperimentResult() takes no arguments" in txt,
        "spo_training_data_populated": "[SPO+ debug] spo_training_data populated" in txt,
        "spo_weight_positive": "[SPO+ debug] spo_weight became positive" in txt,
        "gpu_marker": "Using GPU device: cuda" in txt,
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }
    if timed_out or cp_returncode != 0:
        row["tail"] = cp_stdout[-1200:]
    else:
        row["tail"] = ""
    return row


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def cred_metric(ok_rows: List[Dict[str, object]], key: str, expect: bool = True) -> str:
    if not ok_rows:
        return "N/A"
    return str(all(bool(r[key]) == expect for r in ok_rows))


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    folder_suffix = "_quick_sanity"

    jobs = [("outside_option_util", v) for v in args.outside_values] + [("k", float(v)) for v in args.k_values]
    if not jobs:
        raise RuntimeError("No jobs requested. Provide --outside_values and/or --k_values.")
    rows: List[Dict[str, object]] = []
    print(f"[INFO] quick sanity jobs={len(jobs)}, seed={args.seed}, episodes={args.episodes}")

    for i, (factor, value) in enumerate(jobs, 1):
        print(f"[RUN {i}/{len(jobs)}] factor={factor}, value={value}")
        row = run_one(args, root, factor, value, folder_suffix)
        rows.append(row)
        write_csv(
            out_dir / "quick_sanity_results.csv",
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

    ok_rows = [r for r in rows if r["status"] == "completed" and r["net_profit"] is not None]
    outside = sorted([r for r in ok_rows if r["factor"] == "outside_option_util"], key=lambda x: float(x["value"]))
    kvals = sorted([r for r in ok_rows if r["factor"] == "k"], key=lambda x: float(x["value"]))

    outside_quit = [float(r["quit_rate"]) for r in outside]
    outside_profit = [float(r["net_profit"]) for r in outside]
    k_profit = [float(r["net_profit"]) for r in kvals]

    quit_non_decreasing = all(outside_quit[i + 1] >= outside_quit[i] - 1e-9 for i in range(len(outside_quit) - 1))
    profit_drop_from_low_to_high = False
    if len(outside) >= 2:
        profit_drop_from_low_to_high = float(outside[-1]["net_profit"]) < float(outside[0]["net_profit"])

    summary_lines = [
        f"jobs_total={len(jobs)}",
        f"jobs_completed={len(ok_rows)}",
        f"credibility_no_spo_constructor_error={cred_metric(ok_rows, 'spo_result_constructor_error', expect=False)}",
        f"credibility_spo_training_data_seen={cred_metric(ok_rows, 'spo_training_data_populated', expect=True)}",
        f"credibility_spo_weight_positive_seen={cred_metric(ok_rows, 'spo_weight_positive', expect=True)}",
        f"credibility_gpu_marker_seen={cred_metric(ok_rows, 'gpu_marker', expect=True)}",
        f"outside_option_quit_non_decreasing={quit_non_decreasing}",
        f"outside_option_profit_drop_low_to_high={profit_drop_from_low_to_high}",
        f"outside_option_net_profit_range={max(outside_profit) - min(outside_profit) if outside_profit else 0.0}",
        f"outside_option_quit_rate_range={max(outside_quit) - min(outside_quit) if outside_quit else 0.0}",
        f"k_net_profit_range={max(k_profit) - min(k_profit) if k_profit else 0.0}",
        f"outside_option_mean_runtime_sec={mean([float(r['runtime_sec']) for r in outside]) if outside else 0.0}",
        f"k_mean_runtime_sec={mean([float(r['runtime_sec']) for r in kvals]) if kvals else 0.0}",
    ]
    (out_dir / "quick_sanity_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("[INFO] quick sanity completed.")
    print(f"[INFO] results: {out_dir / 'quick_sanity_results.csv'}")
    print(f"[INFO] summary: {out_dir / 'quick_sanity_summary.txt'}")


if __name__ == "__main__":
    main()
