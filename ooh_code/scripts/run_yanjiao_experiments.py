#!/usr/bin/env python
"""
Beijing Yanjiao full experiment runner.

Runs DSPO/DRPO/Static-pricing across multiple scales with 30 seeds each.
Based on compare_algorithms_dspo_vs_plus.py architecture.

Usage:
  python scripts/run_yanjiao_experiments.py --phase main --dry_run       # verify commands
  python scripts/run_yanjiao_experiments.py --phase main --seeds 40      # smoke test (1 seed)
  python scripts/run_yanjiao_experiments.py --phase main                 # full main experiment
  python scripts/run_yanjiao_experiments.py --phase sensitivity          # sensitivity runs
  python scripts/run_yanjiao_experiments.py --phase all                  # everything
"""
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ─── Metric regex patterns (matching eval2 output in run.py) ───

METRIC_REGEX = {
    "net_profit":       re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs":      re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate":        re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "home_pickup_rate": re.compile(r"percentage home delivery:\s+([\d.]+)"),
    "travel_costs":     re.compile(r"travel costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "service_costs":    re.compile(r"service costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "failure_costs":    re.compile(r"failure costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "avg_charge":       re.compile(r"Avg\. Charge:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "avg_discount":     re.compile(r"Avg\. Discount:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "charge_revenue":   re.compile(r"Charge revenue:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "discount_costs":   re.compile(r"Discount costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "base_revenue":     re.compile(r"Base revenue:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "served_demand":    re.compile(r"Accepted customers:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_demand":     re.compile(r"Total customers:\s*([+-]?\d+(?:\.\d+)?)"),
}

T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

INT_PARAMS = {"k", "batch_size", "grid_dim", "n_passengers", "n_vehicles", "max_steps_r",
               "initial_phase_epochs", "buffer_size",
              "spo_warmup_episodes", "spo_rampup_episodes",
              "spo_buffer_size", "spo_batch_size"}
METRICS = ["net_profit", "total_costs", "quit_rate", "home_pickup_rate",
           "served_rate", "served_demand", "total_demand",
           "travel_costs", "service_costs", "failure_costs",
           "avg_charge", "avg_discount", "charge_revenue", "discount_costs"]

RECOMMENDED_SEEDS = [
    40, 67, 97, 52, 29, 20, 17, 88, 63, 79,
    60, 62,  7, 48, 56, 15, 66, 53, 90, 70,
    24, 74, 80, 28,  2, 95, 92, 26, 39, 82,
]

# ─── Seed split (Phase 1 protocol: random_state=42 shuffle, 10/10/10) ───
# Mirrors .planning/01-SEED-SPLIT.md — do NOT modify independently.

SEED_SPLIT = {
    "tuning":     [7, 15, 17, 20, 56, 60, 62, 70, 80, 92],
    "validation": [26, 48, 66, 67, 74, 79, 82, 90, 95, 97],
    "test":       [2, 24, 28, 29, 39, 40, 52, 53, 63, 88],
}

# ─── Experiment configurations ───

STRATEGIES = {
    "Only-home": {
        "algo_name": "Baseline",
        "price_home": 0.0,
        "price_pp": 100.0,
    },
    "Only-meeting-points": {
        "algo_name": "Baseline",
        "price_home": 100.0,
        "price_pp": 0.0,
    },
    "No-pricing": {
        "algo_name": "Baseline",
        "price_home": 0.0,
        "price_pp": 0.0,
    },
    "Static-pricing": {
        "algo_name": "Baseline",
        "price_home": 0.5,
        "price_pp": -1.0,
    },
    "Static": {
        "algo_name": "Baseline",
        "price_home": 0.5,
        "price_pp": -1.0,
    },
    "DSPO": {
        "algo_name": "DSPO",
    },
    "DRPO": {
        "algo_name": "DRPO",
    },
}

SCALES = {
    "main": [
        {"n_passengers": 400, "n_vehicles": 35, "max_steps_r": 400},
    ],
    "sensitivity": [
        {"n_passengers": 150, "n_vehicles": 15, "max_steps_r": 150},
        {"n_passengers": 300, "n_vehicles": 25, "max_steps_r": 300},
        {"n_passengers": 415, "n_vehicles": 40, "max_steps_r": 415},
    ],
}

COMMON_PARAMS = {
    "instance": "Beijing_Yanjiao",
    "k": 10,
    "veh_capacity": 12,
    "max_steps_p": 0.5,
    "data_seed": 0,
    "data_seed_test": 1,
    "pricing": True,
    "max_price": 3.5,
    "min_price": -5.0,
    "home_util": 1.4,
    "base_util": -1.0,
    "outside_option_util": -0.75,
    "travel_time_weight": -0.0002,
    "walk_distance_weight": -0.0015,
    "use_travel_time_prediction": True,
    "travel_time_learning_rate": 0.001,
    "incentive_sens": -0.15,
    "revenue": 50,
    "fuel_cost": 0.6,
    "driver_wage": 30,
    "home_failure": 0.1,
    "failure_cost": 20.0,
    "l0_home": 2.5,
    "l_mp": 0.75,
    "hgs_reopt_time": 1.1,
    "hgs_final_time": 1.5,
    "grid_dim": 11,
    "learning_rate": 0.001,
    "initial_phase_epochs": 50,
    "batch_size": 256,
    "buffer_size": 500,
    "init_theta_cnn": 0.75,
    "cool_theta_cnn": 1.0 / 850,
    "spo_warmup_episodes": 5,
    "spo_rampup_episodes": 10,
    "spo_loss_weight": 0.85,
    "spo_buffer_size": 1000,
    "spo_batch_size": 64,
    "spo_label_sample_size": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Beijing Yanjiao full experiment runner")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--phase", default="all", choices=["main", "sensitivity", "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=RECOMMENDED_SEEDS)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--eval_episodes", type=int, default=20)
    p.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    p.add_argument("--save_count", type=int, default=1)
    p.add_argument("--folder_suffix", default="_yanjiao_paper")
    p.add_argument("--run_prefix", default="YJ",
                   help="Prefix for experiment run IDs; use a unique value for parameter sweeps.")
    p.add_argument("--strategies", nargs="+", default=None,
                   help="Override strategies: e.g. Static DSPO DRPO")
    p.add_argument("--sensitivity_strategies", nargs="+", default=["DSPO", "DRPO"],
                   help="Strategies for sensitivity phase (default: DSPO DRPO)")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_existing_output_dir", action="store_true")
    p.add_argument("--persist_every_n", type=int, default=2)
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--final_yanjiao_mode", action="store_true",
                   help="Pass strict final Yanjiao utility/data guards to run.py.")
    p.add_argument("--allow_derived_choice_utility", action="store_true",
                   help="Allow audited derived choice utility matrices in final Yanjiao mode.")
    p.add_argument("--seed_split", default=None,
                   choices=["tuning", "validation", "test"],
                   help="Use predefined seed split instead of --seeds. "
                        "tuning=Phases 2-4, validation=Phases 5-6, test=Phase 7.")
    p.add_argument("--analyze_only", action="store_true",
                   help="Only re-read existing raw CSV and regenerate summary/paired outputs.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--yanjiao_prefix", default=None,
                   help="Optional Beijing_Yanjiao data-file prefix template passed to run.py.")
    p.add_argument("--n_passengers_override", type=int, default=None,
                   help="Override main-phase Beijing_Yanjiao passenger count.")
    p.add_argument("--n_vehicles_override", type=int, default=None,
                   help="Override main-phase vehicle count.")
    p.add_argument("--max_steps_r_override", type=int, default=None,
                   help="Override main-phase max_steps_r.")
    p.add_argument("--incentive_sens_override", type=float, default=None,
                   help="Override COMMON_PARAMS['incentive_sens'] for a run.")
    p.add_argument("--k_override", type=int, default=None,
                   help="Override COMMON_PARAMS['k'] for a run.")
    p.add_argument("--revenue_override", type=float, default=None,
                   help="Override COMMON_PARAMS['revenue'] for a run.")
    p.add_argument("--home_util_override", type=float, default=None,
                   help="Override COMMON_PARAMS['home_util'] for a run.")
    p.add_argument("--outside_option_util_override", type=float, default=None,
                   help="Override COMMON_PARAMS['outside_option_util'] for a run.")
    p.add_argument("--travel_time_weight_override", type=float, default=None,
                   help="Override COMMON_PARAMS['travel_time_weight'] for a run.")
    p.add_argument("--walk_distance_weight_override", type=float, default=None,
                   help="Override COMMON_PARAMS['walk_distance_weight'] for a run.")
    p.add_argument("--use_travel_time_prediction_override", type=str, default=None,
                   choices=["True", "False", "true", "false", "1", "0", "yes", "no"],
                   help="Override COMMON_PARAMS['use_travel_time_prediction'] for a run.")
    p.add_argument("--max_price_override", type=float, default=None,
                   help="Override COMMON_PARAMS['max_price'] for a run.")
    p.add_argument("--min_price_override", type=float, default=None,
                   help="Override COMMON_PARAMS['min_price'] for a run.")
    p.add_argument("--initial_phase_epochs_override", type=int, default=None,
                   help="Override COMMON_PARAMS['initial_phase_epochs'] for a run.")
    p.add_argument("--buffer_size_override", type=int, default=None,
                   help="Override COMMON_PARAMS['buffer_size'] for a run.")
    p.add_argument("--init_theta_cnn_override", type=float, default=None,
                   help="Override COMMON_PARAMS['init_theta_cnn'] for a run.")
    p.add_argument("--cool_theta_cnn_override", type=float, default=None,
                   help="Override COMMON_PARAMS['cool_theta_cnn'] for a run.")
    p.add_argument("--grid_dim_override", type=int, default=None,
                   help="Override COMMON_PARAMS['grid_dim'] for a run.")
    p.add_argument("--hgs_reopt_time_override", type=float, default=None,
                   help="Override COMMON_PARAMS['hgs_reopt_time'] for a run.")
    p.add_argument("--hgs_final_time_override", type=float, default=None,
                   help="Override COMMON_PARAMS['hgs_final_time'] for a run.")
    p.add_argument("--spo_warmup_episodes_override", type=int, default=None,
                   help="Override COMMON_PARAMS['spo_warmup_episodes'] for a run.")
    p.add_argument("--spo_rampup_episodes_override", type=int, default=None,
                   help="Override COMMON_PARAMS['spo_rampup_episodes'] for a run.")
    p.add_argument("--spo_label_sample_size_override", type=int, default=None,
                   help="Override COMMON_PARAMS['spo_label_sample_size'] for a run.")
    p.add_argument("--spo_batch_size_override", type=int, default=None,
                   help="Override COMMON_PARAMS['spo_batch_size'] for a run.")
    p.add_argument("--static_price_home", type=float, default=None,
                   help="Override STRATEGIES['Static']['price_home'].")
    p.add_argument("--static_price_pp", type=float, default=None,
                   help="Override STRATEGIES['Static']['price_pp'].")
    p.add_argument("--dspo_spo_loss_weight", type=float, default=0.0,
                   help="SPO loss weight passed to the DSPO baseline; default keeps DSPO SPO-free.")
    p.add_argument("--drpo_spo_loss_weight", type=float, default=None,
                   help="Override COMMON_PARAMS['spo_loss_weight'] for DRPO runs.")
    p.add_argument("--drop_params", nargs="+", default=None,
                   help="Remove these keys from COMMON_PARAMS so run.py uses its own defaults.")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


# ─── Utility functions (from compare_algorithms_dspo_vs_plus.py) ───

def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96 if df > 30 else 2.0


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.findall(text)
    return float(m[-1]) if m else None


def parse_metrics(log: Path) -> Optional[Dict[str, Optional[float]]]:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="ignore")
    result = {}
    all_found = True
    for key in ["net_profit", "total_costs", "quit_rate"]:
        val = extract_last(METRIC_REGEX[key], txt)
        if val is None:
            all_found = False
        result[key] = val

    if not all_found:
        return None

    for key in ["home_pickup_rate", "travel_costs", "service_costs", "failure_costs",
                "avg_charge", "avg_discount", "charge_revenue", "discount_costs",
                "base_revenue", "served_demand", "total_demand"]:
        result[key] = extract_last(METRIC_REGEX[key], txt)

    sd = result.get("served_demand")
    td = result.get("total_demand")
    result["served_rate"] = (sd / td) if (sd is not None and td is not None and td > 0) else None
    return result


def has_gpu_marker(log: Path) -> bool:
    return log.exists() and ("Using GPU device: cuda" in log.read_text(encoding="utf-8", errors="ignore"))


def probe_runtime(pyexe: str) -> Dict[str, Any]:
    code = (
        "import json,torch;print(json.dumps({'torch_version':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
    )
    cp = subprocess.run([pyexe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="ignore")
    if cp.returncode != 0:
        raise RuntimeError("Torch probe failed:\n" + cp.stderr)
    return json.loads(cp.stdout.strip())


def run_log_path(root: Path, algo_name: str, run_id: str, suffix: str, seed: int) -> Path:
    return (root / "Experiments" / "Parcelpoint_py" / "pricing" / algo_name
            / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log")


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def build_cmd(
    pyexe: str, gpu: int, algo_name: str, run_id: str, seed: int,
    episodes: int, eval_episodes: int, route_label_mode: str, save_count: int, folder_suffix: str,
    scale_params: Dict[str, Any], strategy_params: Dict[str, Any],
) -> List[str]:
    cmd = [
        pyexe, "run.py",
        "--algo_name", algo_name,
        "--instance", str(COMMON_PARAMS["instance"]),
        "--seed", str(seed),
        "--data_seed", str(COMMON_PARAMS["data_seed"]),
        "--data_seed_test", str(COMMON_PARAMS["data_seed_test"]),
        "--max_episodes", str(episodes),
        "--eval_episodes", str(eval_episodes),
        "--route_label_mode", str(route_label_mode),
        "--save_count", str(save_count),
        "--log_output", "file",
        "--debug", "False",
        "--gpu", str(gpu),
    ]
    for k, v in sorted({**COMMON_PARAMS, **scale_params, **strategy_params}.items()):
        if k in ("instance", "data_seed", "data_seed_test"):
            continue
        if k == "yanjiao_prefix" and not v:
            continue
        cmd.extend([f"--{k}", cli_value(k, v)])
    cmd.extend(["--experiment", run_id, "--folder_suffix", folder_suffix])
    return cmd


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def collect_fields(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = ["label", "algo_name", "run_id", "seed", "n_passengers", "grid_dim", "episodes",
                 "status", "runtime_sec", "log_path", "command"]
    seen = set(preferred)
    extra = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return preferred + sorted(extra)


def row_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(row["label"]), int(float(row["seed"])), int(float(row["n_passengers"])))


# ─── Core execution ───

def run_single(
    args: argparse.Namespace, root: Path,
    label: str, algo_name: str, run_id: str, seed: int,
    scale_params: Dict[str, Any], strategy_params: Dict[str, Any],
) -> Dict[str, Any]:
    log = run_log_path(root, algo_name, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args.python_executable, args.gpu, algo_name, run_id, seed,
                    args.episodes, args.eval_episodes, args.route_label_mode, args.save_count, args.folder_suffix,
                    scale_params, strategy_params)
    needs_gpu_marker = algo_name != "Baseline"

    if args.skip_existing:
        m = parse_metrics(log)
        if m is not None:
            if args.allow_cpu or (not needs_gpu_marker) or has_gpu_marker(log):
                row: Dict[str, Any] = {
                    "label": label, "algo_name": algo_name, "run_id": run_id,
                    "seed": seed, "n_passengers": scale_params["n_passengers"],
                    "grid_dim": COMMON_PARAMS.get("grid_dim", ""),
                    "episodes": args.episodes, "status": "cached",
                    "runtime_sec": 0.0, "log_path": str(log),
                    "command": " ".join(cmd),
                }
                row.update(m)
                return row

    timeout = None if args.run_timeout_sec <= 0 else args.run_timeout_sec
    attempts = max(1, args.max_retries + 1)
    last_error = ""

    for att in range(1, attempts + 1):
        t0 = time.time()
        try:
            cp = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="ignore", timeout=timeout)
        except subprocess.TimeoutExpired as e:
            tail = (e.stdout or "")[-1500:] if e.stdout else ""
            last_error = f"Timeout {att}/{attempts} for {label} seed={seed}. tail={tail}"
            if att < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        rt = time.time() - t0
        m = parse_metrics(log)
        if cp.returncode != 0:
            last_error = f"Return code {cp.returncode} {att}/{attempts} for {label} seed={seed}. tail={(cp.stdout or '')[-1500:]}"
        elif m is None:
            last_error = f"Metrics missing {att}/{attempts} for {label} seed={seed}. log={log}"
        elif (not args.allow_cpu) and needs_gpu_marker and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing {att}/{attempts} for {label} seed={seed}. log={log}"
        else:
            status = "completed" if att == 1 else f"completed_retry_{att}"
            row = {
                "label": label, "algo_name": algo_name, "run_id": run_id,
                "seed": seed, "n_passengers": scale_params["n_passengers"],
                "grid_dim": COMMON_PARAMS.get("grid_dim", ""),
                "episodes": args.episodes, "status": status,
                "runtime_sec": rt, "log_path": str(log),
                "command": " ".join(cmd),
            }
            row.update(m)
            return row

        if att < attempts:
            time.sleep(args.retry_backoff_sec)

    raise RuntimeError(last_error)


# ─── Statistics ───

def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (str(r["label"]), int(float(r["n_passengers"])))
        by_key.setdefault(key, []).append(r)

    out = []
    for key in sorted(by_key.keys()):
        rs = by_key[key]
        row: Dict[str, Any] = {
            "label": key[0], "n_passengers": key[1], "n_runs": len(rs),
        }
        for m in METRICS + ["runtime_sec"]:
            vals = [to_float(r.get(m)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
                continue
            n = len(vals)
            mean_v = sum(vals) / n
            std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
            ci95 = t_critical_95(n - 1) * std_v / math.sqrt(n) if n > 1 else 0.0
            row[f"{m}_mean"] = round(mean_v, 4)
            row[f"{m}_std"] = round(std_v, 4)
            row[f"{m}_ci95_halfwidth"] = round(ci95, 4)
        out.append(row)
    return out


def summarize_paired_deltas(paired_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not paired_rows:
        return []
    metrics = [k for k in paired_rows[0].keys() if k.startswith("delta_")]
    out = []
    for metric in metrics:
        vals = [to_float(r.get(metric)) for r in paired_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        n = len(vals)
        mean_v = sum(vals) / n
        std_v = math.sqrt(sum((v - mean_v) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        ci95 = t_critical_95(n - 1) * std_v / math.sqrt(n) if n > 1 else 0.0
        out.append({
            "metric": metric,
            "n_pairs": n,
            "mean_delta": round(mean_v, 4),
            "std_delta": round(std_v, 4),
            "ci95_halfwidth": round(ci95, 4),
            "positive_pairs": sum(1 for v in vals if v > 0),
            "negative_pairs": sum(1 for v in vals if v < 0),
            "zero_pairs": sum(1 for v in vals if v == 0),
        })
    return out


def paired_deltas(rows: List[Dict[str, Any]], label_a: str, label_b: str,
                  n_passengers: int) -> List[Dict[str, Any]]:
    keyed = {}
    for r in rows:
        k = (str(r["label"]), int(float(r["n_passengers"])), int(float(r["seed"])))
        keyed[k] = r

    seeds = sorted({int(float(r["seed"])) for r in rows
                    if int(float(r["n_passengers"])) == n_passengers})
    out = []
    for seed in seeds:
        a = keyed.get((label_a, n_passengers, seed))
        b = keyed.get((label_b, n_passengers, seed))
        if a is None or b is None:
            continue
        row: Dict[str, Any] = {"seed": seed, "n_passengers": n_passengers}
        for metric in METRICS:
            av = to_float(a.get(metric))
            bv = to_float(b.get(metric))
            row[f"{label_a}_{metric}"] = av
            row[f"{label_b}_{metric}"] = bv
            row[f"delta_{metric}"] = (bv - av) if (av is not None and bv is not None) else ""
        out.append(row)
    return out


def persist(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    raw_fields = collect_fields(rows)
    write_csv(output_dir / "yanjiao_raw.csv", rows, raw_fields)

    summary_rows = summarize(rows)
    if summary_rows:
        write_csv(output_dir / "yanjiao_summary.csv", summary_rows, summary_rows[0].keys())

    n_values = sorted({
        int(float(r["n_passengers"])) for r in rows
        if str(r.get("label")) in ("DSPO", "DRPO")
    })
    paired_rows: List[Dict[str, Any]] = []
    for n_passengers in n_values:
        paired_rows.extend(paired_deltas(rows, "DSPO", "DRPO", n_passengers))
    if paired_rows:
        write_csv(output_dir / "yanjiao_paired_dspo_drpo.csv",
                  paired_rows, paired_rows[0].keys())
        paired_summary = summarize_paired_deltas(paired_rows)
        if paired_summary:
            write_csv(output_dir / "yanjiao_paired_dspo_drpo_summary.csv",
                      paired_summary, paired_summary[0].keys())


# ─── Report table generation (Phase 1 protocol: RPT-01 to RPT-04) ───

MAIN_STRATEGIES = ["No-pricing", "Static-pricing", "DSPO", "DRPO"]
AUX_STRATEGIES = ["Only-meeting-points", "Only-home"]

# Canonical label aliases for paired comparisons
PAIR_COMPARISONS = [
    ("DRPO", "DSPO"),
    ("DSPO", "Static-pricing"),
    ("Static-pricing", "No-pricing"),
]


def _fmt(v: Any, decimals: int = 4) -> str:
    """Format a numeric value for CSV/Markdown output."""
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    """Render a list of dicts as a GitHub-Flavored Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for r in rows:
        cells = []
        for c in columns:
            val = r.get(c, "")
            cells.append(str(val) if val is not None else "")
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body_lines)


def _label_canonical(label: str) -> str:
    """Map script labels to canonical report names."""
    mapping = {"Static": "Static-pricing"}
    return mapping.get(label, label)


def _quit_rate_ratio(raw_value: Any) -> Optional[float]:
    """Convert quit_rate from percentage (eval2 output) to 0-1 ratio.

    eval2 always outputs quit_rate as a percentage (e.g., 16.33 means 16.33%,
    0.8 means 0.8%). The regex captures the number before the '%' sign.
    Always divide by 100 to get the ratio.
    """
    v = to_float(raw_value)
    if v is None:
        return None
    return v / 100.0


def generate_report_tables(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    """Generate RPT-01 through RPT-04 report tables as CSV + Markdown.

    Reads from the raw result rows (same data as yanjiao_raw.csv).
    Outputs are written to output_dir alongside the existing CSV files.
    """
    if not rows:
        print("[RPT] No rows to report.", flush=True)
        return

    # Canonicalize labels
    for r in rows:
        r["_label"] = _label_canonical(str(r.get("label", "")))

    # ─── RPT-01: Main Policy Comparison ───
    main_rows = [r for r in rows if r["_label"] in MAIN_STRATEGIES]
    rpt01_rows: List[Dict[str, Any]] = []
    if main_rows:
        summaries = summarize(main_rows)
        for s in summaries:
            label = s.get("label", "")
            qr_mean = _quit_rate_ratio(s.get("quit_rate_mean"))
            rpt01_rows.append({
                "policy": label,
                "net_profit_mean": _fmt(s.get("net_profit_mean")),
                "net_profit_std": _fmt(s.get("net_profit_std")),
                "net_profit_ci95": _fmt(s.get("net_profit_ci95_halfwidth")),
                "total_costs_mean": _fmt(s.get("total_costs_mean")),
                "charge_revenue_mean": _fmt(s.get("charge_revenue_mean")),
                "discount_costs_mean": _fmt(s.get("discount_costs_mean")),
                "quit_rate_mean": _fmt(qr_mean),
                "quit_rate_std": _fmt(_quit_rate_ratio(s.get("quit_rate_std"))),
                "home_pickup_rate_mean": _fmt(s.get("home_pickup_rate_mean")),
                "home_pickup_rate_std": _fmt(s.get("home_pickup_rate_std")),
                "served_rate_mean": _fmt(s.get("served_rate_mean")),
                "n_runs": s.get("n_runs", ""),
            })
        # Sort by net_profit_mean descending
        rpt01_rows.sort(key=lambda x: to_float(x.get("net_profit_mean", "")) or 0, reverse=True)

    # ─── RPT-02: Paired Net-Profit Deltas ───
    rpt02_rows: List[Dict[str, Any]] = []
    n_passengers_set = sorted({int(float(r["n_passengers"])) for r in main_rows})
    for label_a, label_b in PAIR_COMPARISONS:
        for n_p in n_passengers_set:
            paired = paired_deltas(main_rows, label_a, label_b, n_p)
            if not paired:
                continue
            # Compute summary for net_profit deltas
            deltas = [to_float(r.get("delta_net_profit")) for r in paired]
            deltas = [d for d in deltas if d is not None]
            if not deltas:
                continue
            n = len(deltas)
            mean_d = sum(deltas) / n
            std_d = math.sqrt(sum((d - mean_d) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
            ci95 = t_critical_95(n - 1) * std_d / math.sqrt(n) if n > 1 else 0.0
            pos = sum(1 for d in deltas if d > 0)
            # Approximate significance
            if n > 1 and abs(mean_d) > 0:
                t_stat = abs(mean_d) / (std_d / math.sqrt(n)) if std_d > 0 else float("inf")
                if t_stat > 3.25:
                    p_str = "p < 0.01"
                elif t_stat > 2.26:
                    p_str = "p < 0.05"
                else:
                    p_str = "n.s."
            else:
                p_str = "n.s."
            rpt02_rows.append({
                "comparison": f"{label_b} − {label_a}",
                "mean_delta": _fmt(mean_d),
                "std_delta": _fmt(std_d),
                "ci95_halfwidth": _fmt(ci95),
                "positive_seed_count": pos,
                "total_seed_count": n,
                "p_value_approx": p_str,
            })

    # ─── RPT-03: Behavioral Realism Check ───
    rpt03_rows: List[Dict[str, Any]] = []
    if main_rows:
        summaries = summarize(main_rows)
        for s in summaries:
            label = s.get("label", "")
            qr_mean = _quit_rate_ratio(s.get("quit_rate_mean"))
            hpr_mean = to_float(s.get("home_pickup_rate_mean"))
            hpr_std = to_float(s.get("home_pickup_rate_std"))
            qr_std_raw = to_float(s.get("quit_rate_std"))

            in_range = hpr_mean is not None and 0.50 <= hpr_mean <= 0.70
            not_excessive = qr_mean is not None and qr_mean <= 0.20

            flags = []
            if not in_range and hpr_mean is not None:
                flags.append(f"home_pickup {hpr_mean:.1%} outside [50%, 70%]")
            if not not_excessive and qr_mean is not None:
                flags.append(f"quit_rate {qr_mean:.1%} excessive (>20%)")
            status = "acceptable" if (in_range and not_excessive) else "flagged: " + "; ".join(flags)

            mpr_mean = (1.0 - hpr_mean) if hpr_mean is not None else None

            rpt03_rows.append({
                "policy": label,
                "quit_rate_mean": _fmt(qr_mean),
                "quit_rate_std": _fmt(qr_std_raw / 100.0 if qr_std_raw is not None else None),
                "home_pickup_rate_mean": _fmt(hpr_mean),
                "home_pickup_rate_std": _fmt(hpr_std),
                "meeting_point_rate_mean": _fmt(mpr_mean),
                "home_pickup_in_range": str(in_range),
                "quit_not_excessive": str(not_excessive),
                "status": status,
            })

    # ─── RPT-04: Auxiliary Cost-Bound Policies ───
    aux_rows = [r for r in rows if r["_label"] in AUX_STRATEGIES]
    rpt04_rows: List[Dict[str, Any]] = []
    if aux_rows:
        summaries = summarize(aux_rows)
        notes = {
            "Only-meeting-points": "Artificial lower bound on operational cost (forces all pickups at meeting points)",
            "Only-home": "Artificial upper bound on operational cost (forces all pickups at home)",
        }
        for s in summaries:
            label = s.get("label", "")
            qr_mean = _quit_rate_ratio(s.get("quit_rate_mean"))
            rpt04_rows.append({
                "policy": label,
                "net_profit_mean": _fmt(s.get("net_profit_mean")),
                "net_profit_std": _fmt(s.get("net_profit_std")),
                "total_costs_mean": _fmt(s.get("total_costs_mean")),
                "quit_rate_mean": _fmt(qr_mean),
                "home_pickup_rate_mean": _fmt(s.get("home_pickup_rate_mean")),
                "note": notes.get(label, ""),
            })

    # ─── Write all report tables ───
    reports = [
        ("rpt_01_main_comparison", rpt01_rows),
        ("rpt_02_paired_deltas", rpt02_rows),
        ("rpt_03_behavioral_realism", rpt03_rows),
        ("rpt_04_auxiliary_bounds", rpt04_rows),
    ]
    for basename, rpt_rows in reports:
        if not rpt_rows:
            print(f"[RPT] {basename}: no data — skipped.", flush=True)
            continue
        cols = list(rpt_rows[0].keys())
        # CSV
        csv_path = output_dir / f"{basename}.csv"
        write_csv(csv_path, rpt_rows, cols)
        # Markdown
        md_path = output_dir / f"{basename}.md"
        title = basename.replace("_", " ").replace("rpt ", "RPT-").upper()
        # Title formatting: rpt_01_main_comparison → RPT-01: Main Comparison
        parts = basename.split("_", 2)  # ["rpt", "01", "main_comparison"]
        title = f"RPT-{parts[1]}: {parts[2].replace('_', ' ').title()}"
        md_content = f"## {title}\n\n{_md_table(rpt_rows, cols)}\n"
        md_path.write_text(md_content, encoding="utf-8")
        print(f"[RPT] {basename}: {len(rpt_rows)} rows → {csv_path.name}, {md_path.name}", flush=True)

    # Clean up temporary _label field
    for r in rows:
        r.pop("_label", None)


# ─── Job builder ───

def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Build the full job list based on phase."""
    jobs = []

    # Determine which scales and strategies to use
    scale_list = []
    strat_list = []

    if args.phase in ("main", "all"):
        scale_list.extend(SCALES["main"])
        main_strats = args.strategies if args.strategies else ["Static", "DSPO", "DRPO"]
        strat_list.append(("main", main_strats))

    if args.phase in ("sensitivity", "all"):
        scale_list.extend(SCALES["sensitivity"])
        strat_list.append(("sensitivity", args.sensitivity_strategies))

    # Build flat job list
    seen = set()
    for phase_name, strats in strat_list:
        for scale in scale_list:
            np_val = scale["n_passengers"]
            for strat_name in strats:
                strat = STRATEGIES[strat_name]
                algo_name = strat["algo_name"]
                for seed in args.seeds:
                    run_id = f"{args.run_prefix}_{np_val}_{strat_name}_seed{seed}"
                    job_key = (strat_name, seed, np_val)
                    if job_key not in seen:
                        seen.add(job_key)
                        jobs.append({
                            "label": strat_name,
                            "algo_name": algo_name,
                            "run_id": run_id,
                            "seed": seed,
                            "scale_params": scale,
                            "strategy_params": {k: v for k, v in strat.items() if k != "algo_name"},
                            "n_passengers": np_val,
                        })
    return jobs


# ─── Main ───

def main() -> None:
    args = parse_args()
    if args.persist_every_n <= 0:
        raise RuntimeError("--persist_every_n must be >= 1")

    # Phase 1 protocol: override --seeds with predefined split
    if args.seed_split is not None:
        args.seeds = SEED_SPLIT[args.seed_split]

    if args.incentive_sens_override is not None:
        COMMON_PARAMS["incentive_sens"] = args.incentive_sens_override
    if args.yanjiao_prefix is not None:
        COMMON_PARAMS["yanjiao_prefix"] = args.yanjiao_prefix
    if args.n_passengers_override is not None:
        SCALES["main"][0]["n_passengers"] = args.n_passengers_override
    if args.n_vehicles_override is not None:
        SCALES["main"][0]["n_vehicles"] = args.n_vehicles_override
    if args.max_steps_r_override is not None:
        SCALES["main"][0]["max_steps_r"] = args.max_steps_r_override
    for arg_name, param_name in [
        ("k_override", "k"),
        ("revenue_override", "revenue"),
        ("home_util_override", "home_util"),
        ("outside_option_util_override", "outside_option_util"),
        ("travel_time_weight_override", "travel_time_weight"),
        ("walk_distance_weight_override", "walk_distance_weight"),
        ("max_price_override", "max_price"),
        ("min_price_override", "min_price"),
        ("initial_phase_epochs_override", "initial_phase_epochs"),
        ("buffer_size_override", "buffer_size"),
        ("init_theta_cnn_override", "init_theta_cnn"),
        ("cool_theta_cnn_override", "cool_theta_cnn"),
        ("grid_dim_override", "grid_dim"),
        ("hgs_reopt_time_override", "hgs_reopt_time"),
        ("hgs_final_time_override", "hgs_final_time"),
        ("spo_warmup_episodes_override", "spo_warmup_episodes"),
        ("spo_rampup_episodes_override", "spo_rampup_episodes"),
        ("spo_label_sample_size_override", "spo_label_sample_size"),
        ("spo_batch_size_override", "spo_batch_size"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            COMMON_PARAMS[param_name] = value
    if args.use_travel_time_prediction_override is not None:
        COMMON_PARAMS["use_travel_time_prediction"] = (
            args.use_travel_time_prediction_override.lower() in ("true", "1", "yes")
        )
    COMMON_PARAMS["final_yanjiao_mode"] = bool(args.final_yanjiao_mode)
    COMMON_PARAMS["allow_derived_choice_utility"] = bool(args.allow_derived_choice_utility)
    if args.static_price_home is not None:
        STRATEGIES["Static"]["price_home"] = args.static_price_home
        STRATEGIES["Static-pricing"]["price_home"] = args.static_price_home
    if args.static_price_pp is not None:
        STRATEGIES["Static"]["price_pp"] = args.static_price_pp
        STRATEGIES["Static-pricing"]["price_pp"] = args.static_price_pp
    STRATEGIES["DSPO"]["spo_loss_weight"] = args.dspo_spo_loss_weight
    if args.drpo_spo_loss_weight is not None:
        STRATEGIES["DRPO"]["spo_loss_weight"] = args.drpo_spo_loss_weight
    if args.drop_params:
        for key in args.drop_params:
            COMMON_PARAMS.pop(key, None)

    root = Path(__file__).resolve().parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir is None:
        args.output_dir = f"Experiments/analysis/yanjiao_full_{ts}"
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        raw_path = output_dir / "yanjiao_raw.csv"
        rows = read_csv(raw_path)
        if not rows:
            raise RuntimeError(f"--analyze_only requested but no rows found in {raw_path}")
        persist(output_dir, [dict(r) for r in rows])
        print(f"[ANALYZE] Rebuilt outputs from {raw_path}", flush=True)
        print(f"[ANALYZE] Summary: {output_dir / 'yanjiao_summary.csv'}", flush=True)
        paired = output_dir / "yanjiao_paired_dspo_drpo.csv"
        if paired.exists():
            print(f"[ANALYZE] Paired: {paired}", flush=True)
        generate_report_tables(output_dir, [dict(r) for r in rows])
        return

    jobs = build_jobs(args)
    print(f"[INFO] Phase={args.phase}, Total jobs={len(jobs)}, Seeds={len(args.seeds)}", flush=True)

    # Probe runtime
    runtime = probe_runtime(args.python_executable)
    print(f"[INFO] torch={runtime['torch_version']}, "
          f"cuda={runtime['cuda_available']}, gpu_count={runtime['cuda_count']}", flush=True)
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable. Pass --allow_cpu to continue on CPU.")

    # Save run metadata
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": args.phase,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "eval_episodes": args.eval_episodes,
        "route_label_mode": args.route_label_mode,
        "run_prefix": args.run_prefix,
        "common_params": COMMON_PARAMS,
        "strategies": {k: v for k, v in STRATEGIES.items()},
        "scales": {k: v for k, v in SCALES.items()},
    }
    (output_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dry run: print sample commands
    if args.dry_run:
        print("\n[DRY-RUN] Sample commands (first 3 jobs):", flush=True)
        for job in jobs[:3]:
            cmd = build_cmd(args.python_executable, args.gpu,
                           job["algo_name"], job["run_id"], job["seed"],
                           args.episodes, args.eval_episodes, args.route_label_mode, args.save_count, args.folder_suffix,
                           job["scale_params"], job["strategy_params"])
            print(f"\n[{job['label']} p={job['n_passengers']} seed={job['seed']}]")
            print(" ".join(cmd), flush=True)
        print(f"\n[DRY-RUN] Total jobs: {len(jobs)}")
        print(f"[DRY-RUN] Output dir: {output_dir}")
        return

    # Load existing progress
    raw_path = output_dir / "yanjiao_raw.csv"
    existing_rows = read_csv(raw_path)
    rows: List[Dict[str, Any]] = [dict(r) for r in existing_rows]
    done = {row_key(r) for r in rows}

    remaining = sum(1 for j in jobs if (j["label"], j["seed"], j["n_passengers"]) not in done)
    print(f"[INFO] existing={len(rows)}, remaining={remaining}/{len(jobs)}", flush=True)

    if remaining == 0:
        print("[INFO] All jobs already completed.", flush=True)
        persist(output_dir, rows)
        generate_report_tables(output_dir, rows)
        return

    # Execute
    new_count = 0
    t_start = time.time()
    for idx, job in enumerate(jobs, 1):
        key = (job["label"], job["seed"], job["n_passengers"])
        if key in done:
            continue

        print(f"\n[{idx}/{len(jobs)}] {job['label']} p={job['n_passengers']} seed={job['seed']}",
              flush=True, end=" ")

        row = run_single(
            args=args, root=root,
            label=job["label"], algo_name=job["algo_name"],
            run_id=job["run_id"], seed=job["seed"],
            scale_params=job["scale_params"],
            strategy_params=job["strategy_params"],
        )
        rows.append(row)
        done.add(key)
        new_count += 1

        elapsed = time.time() - t_start
        avg_per_run = elapsed / new_count
        eta = avg_per_run * (remaining - new_count)
        print(f"-> {row['status']} ({row['runtime_sec']:.0f}s) "
              f"ETA: {eta/3600:.1f}h", flush=True)

        if new_count % args.persist_every_n == 0:
            persist(output_dir, rows)
            print(f"[INFO] Persisted after {new_count} new runs.", flush=True)

    # Final persist
    persist(output_dir, rows)
    total_time = time.time() - t_start
    print(f"\n[DONE] {new_count} runs in {total_time/3600:.1f}h", flush=True)
    print(f"[DONE] Raw: {output_dir / 'yanjiao_raw.csv'}", flush=True)
    print(f"[DONE] Summary: {output_dir / 'yanjiao_summary.csv'}", flush=True)

    # Generate RPT-01 to RPT-04 report tables
    generate_report_tables(output_dir, rows)


if __name__ == "__main__":
    main()
