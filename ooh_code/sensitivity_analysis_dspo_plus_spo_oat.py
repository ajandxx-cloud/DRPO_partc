#!/usr/bin/env python
import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "served_demand": re.compile(r"Accepted customers:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_demand": re.compile(r"Total customers:\s*([+-]?\d+(?:\.\d+)?)"),
}

SUMMARY_METRICS = ["net_profit", "total_costs", "quit_rate", "served_demand", "total_demand", "served_rate"]
BASIC_STAGE_METRICS = ["net_profit", "total_costs", "quit_rate", "served_rate"]

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

INT_PARAMS = {"k", "batch_size", "spo_warmup_episodes", "spo_rampup_episodes"}
PRIMARY_DIRECTION = {
    "net_profit": "max",
    "served_rate": "max",
    "total_costs": "min",
    "quit_rate": "min",
}


LEGACY4_FACTORS = {
    "outside_option_util": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "incentive_sens": [-0.35, -0.30, -0.25, -0.20, -0.15],
    "home_util": [1.0, 1.2, 1.4, 1.6, 1.8],
    "k": [5, 7, 10, 12, 15],
}
LEGACY4_DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10,
}

RC_FULL12_FACTORS = {
    "outside_option_util": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "incentive_sens": [-0.35, -0.30, -0.25, -0.20, -0.15],
    "home_util": [1.0, 1.2, 1.4, 1.6, 1.8],
    "k": [5, 7, 10, 12, 15],
    "revenue": [40, 45, 50, 55, 60],
    "fuel_cost": [0.40, 0.50, 0.60, 0.70, 0.80],
    "home_failure": [0.02, 0.06, 0.10, 0.14, 0.18],
    "learning_rate": [3e-4, 6e-4, 1e-3, 1.5e-3, 2e-3],
    "batch_size": [128, 192, 256, 320, 384],
    "spo_warmup_episodes": [0, 3, 5, 8, 12],
    "spo_rampup_episodes": [5, 8, 10, 15, 20],
    "spo_loss_weight": [0.20, 0.40, 0.70, 0.90, 1.00],
}
RC_FULL12_DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10,
    "revenue": 50.0,
    "fuel_cost": 0.6,
    "home_failure": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "spo_warmup_episodes": 5,
    "spo_rampup_episodes": 10,
    "spo_loss_weight": 0.7,
}

PROFILE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "legacy4": {
        "default_config": LEGACY4_DEFAULT_CONFIG,
        "factors": LEGACY4_FACTORS,
        "stage1_seeds_default": [0, 21, 42],
        "stage2_seeds_default": [0, 21, 42],
        "run_smoke_default": False,
        "disable_cache_default": False,
    },
    "rc_full12": {
        "default_config": RC_FULL12_DEFAULT_CONFIG,
        "factors": RC_FULL12_FACTORS,
        "stage1_seeds_default": [0, 21, 42, 63, 84],
        "stage2_seeds_default": [0, 7, 14, 21, 28, 35, 42, 49, 56, 63],
        "run_smoke_default": True,
        "disable_cache_default": True,
    },
}

ALL_PROFILE_FACTORS = sorted({f for prof in PROFILE_LIBRARY.values() for f in prof["factors"].keys()})


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
    served_demand: Optional[float]
    total_demand: Optional[float]
    served_rate: Optional[float]
    log_path: str
    command: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DRPO OAT scan with profiles and stage guardrails.")
    p.add_argument("--profile", default="legacy4", choices=sorted(PROFILE_LIBRARY.keys()))
    p.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus"])
    p.add_argument("--factors", nargs="+", default=None, help="Optional subset of factors to scan.")
    p.add_argument(
        "--factor_grid_json",
        default=None,
        help="Optional JSON file mapping factor names to explicit level lists. Unspecified factors keep profile defaults.",
    )
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)

    p.add_argument("--seeds", nargs="+", type=int, default=None, help="Shared seeds for both stages when stage seeds are not specified.")
    p.add_argument("--stage1_seeds", nargs="+", type=int, default=None)
    p.add_argument("--stage2_seeds", nargs="+", type=int, default=None)

    p.add_argument("--run_prefix", default="SENS_DRPO_OAT")
    p.add_argument("--folder_suffix", default="_sens")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")

    p.add_argument("--stage1_episodes", type=int, default=80)
    p.add_argument("--stage2_episodes", type=int, default=200)
    p.add_argument("--save_count", type=int, default=20)
    p.add_argument("--skip_stage2", action="store_true", help="Run only Stage 1 and skip Stage 2 candidate evaluation.")

    p.add_argument("--skip_existing", action="store_true", help="Reuse complete existing runs if available.")
    p.add_argument("--disable_cache", dest="disable_cache", action="store_true", help="Force fresh rerun and ignore cache.")
    p.add_argument("--allow_cache", dest="disable_cache", action="store_false")
    p.set_defaults(disable_cache=None)
    p.add_argument("--resume_missing_only", action="store_true", help="Resume by scheduling only missing stage jobs.")
    p.add_argument("--persist_every_n", type=int, default=1, help="Persist intermediate CSVs every N new runs.")
    p.add_argument(
        "--resume_trust_existing_raw",
        dest="resume_trust_existing_raw",
        action="store_true",
        help="When resuming, trust existing stage*_raw.csv to rebuild completed run records.",
    )
    p.add_argument(
        "--resume_no_trust_existing_raw",
        dest="resume_trust_existing_raw",
        action="store_false",
        help="When resuming, do not preload existing stage*_raw.csv records.",
    )
    p.set_defaults(resume_trust_existing_raw=True)

    p.add_argument("--run_timeout_sec", type=int, default=3600)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)

    p.add_argument("--run_smoke_validation", dest="run_smoke_validation", action="store_true")
    p.add_argument("--no_smoke_validation", dest="run_smoke_validation", action="store_false")
    p.set_defaults(run_smoke_validation=None)
    p.add_argument("--only_smoke", action="store_true")
    p.add_argument("--smoke_seed", type=int, default=0)
    p.add_argument("--smoke_episodes", type=int, default=20)
    p.add_argument("--allow_smoke_failure", action="store_true")

    p.add_argument("--primary_metric", default="net_profit", choices=sorted(PRIMARY_DIRECTION.keys()))
    p.add_argument("--guardrail_quit_delta_pp", type=float, default=2.0)
    p.add_argument("--guardrail_served_rate_delta", type=float, default=-0.02)

    p.add_argument("--max_factors", type=int, default=0, help="Debug only: limit number of scanned factors (0 = all).")
    p.add_argument("--max_values_per_factor", type=int, default=0, help="Debug only: cap levels per factor while preserving default.")
    p.add_argument("--e2e_smoke_small", action="store_true", help="Small end-to-end run: default+1 level per factor, single seed.")

    p.add_argument("--diagnose_factor", choices=ALL_PROFILE_FACTORS, default=None)
    p.add_argument("--diagnose_value", type=float, default=None)
    p.add_argument("--diagnose_seed", type=int, default=42)
    p.add_argument("--diagnose_episodes", type=int, default=10)
    p.add_argument("--continue_after_diagnose", action="store_true")

    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_existing_output_dir", action="store_true")
    return p.parse_args()


def dedupe_ints(vals: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in vals:
        i = int(x)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def factor_dtype(name: str) -> str:
    return "int" if name in INT_PARAMS else "float"


def to_numeric(name: str, value: float) -> float:
    if name in INT_PARAMS:
        return int(round(float(value)))
    return float(value)


def select_values_with_default(values: Sequence[float], default_value: float, max_n: int) -> List[float]:
    ordered = [float(v) for v in values]
    if max_n <= 0 or max_n >= len(ordered):
        return ordered

    keep: List[float] = []
    if any(np.isclose(v, default_value) for v in ordered):
        keep.append(default_value)
    else:
        keep.append(ordered[0])

    candidates = sorted(ordered, key=lambda v: (abs(v - default_value), v))
    for v in candidates:
        if any(np.isclose(v, k) for k in keep):
            continue
        keep.append(v)
        if len(keep) >= max_n:
            break
    return sorted(keep)


def load_factor_grid_override(path_text: str, factor_grid: Dict[str, List[float]], default_config: Dict[str, float]) -> Dict[str, List[float]]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise RuntimeError(f"factor_grid_json not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse factor_grid_json {path}: {e}") from e

    if not isinstance(payload, dict):
        raise RuntimeError(f"factor_grid_json must be a JSON object mapping factor -> levels, got: {type(payload).__name__}")

    updated = {k: list(v) for k, v in factor_grid.items()}
    for factor, raw_levels in payload.items():
        if factor not in updated:
            raise RuntimeError(f"factor_grid_json contains unknown factor '{factor}'. Known factors: {sorted(updated.keys())}")
        if not isinstance(raw_levels, list) or len(raw_levels) == 0:
            raise RuntimeError(f"factor_grid_json factor '{factor}' must map to a non-empty list of numeric levels.")

        levels: List[float] = []
        for raw in raw_levels:
            try:
                levels.append(to_numeric(factor, float(raw)))
            except Exception as e:
                raise RuntimeError(f"Invalid level '{raw}' for factor '{factor}' in {path}: {e}") from e

        default_value = to_numeric(factor, float(default_config[factor]))
        if not any(np.isclose(float(v), float(default_value)) for v in levels):
            levels.append(default_value)
        updated[factor] = sorted({to_numeric(factor, float(v)) for v in levels})

    return updated


def resolve_runtime_profile(args: argparse.Namespace) -> None:
    profile = PROFILE_LIBRARY[args.profile]
    default_config = {k: to_numeric(k, v) for k, v in profile["default_config"].items()}
    factor_grid = {k: [to_numeric(k, v) for v in vals] for k, vals in profile["factors"].items()}

    if args.factor_grid_json:
        factor_grid = load_factor_grid_override(args.factor_grid_json, factor_grid, default_config)

    if args.factors:
        selected: List[str] = []
        seen = set()
        for f in args.factors:
            if f not in seen:
                selected.append(f)
                seen.add(f)
        unknown = [f for f in selected if f not in factor_grid]
        if unknown:
            raise RuntimeError(f"Unknown factors for profile '{args.profile}': {unknown}")
        factor_grid = {f: factor_grid[f] for f in selected}

    stage1_default = dedupe_ints(profile["stage1_seeds_default"])
    stage2_default = dedupe_ints(profile["stage2_seeds_default"])

    if args.seeds is not None:
        shared = dedupe_ints(args.seeds)
        if args.stage1_seeds is None:
            args.stage1_seeds = shared
        if args.stage2_seeds is None:
            args.stage2_seeds = shared

    args.stage1_seeds = dedupe_ints(args.stage1_seeds or stage1_default)
    args.stage2_seeds = dedupe_ints(args.stage2_seeds or stage2_default)

    if args.run_smoke_validation is None:
        args.run_smoke_validation = bool(profile["run_smoke_default"])
    if args.disable_cache is None:
        args.disable_cache = bool(profile["disable_cache_default"])
    if args.disable_cache:
        args.skip_existing = False

    if args.e2e_smoke_small:
        args.max_values_per_factor = 2
        args.stage1_episodes = min(args.stage1_episodes, 20)
        args.stage2_episodes = min(args.stage2_episodes, 30)
        args.stage1_seeds = [args.stage1_seeds[0]]
        args.stage2_seeds = [args.stage2_seeds[0]]

    if args.max_factors > 0:
        keep_factors = list(factor_grid.keys())[: args.max_factors]
        factor_grid = {f: factor_grid[f] for f in keep_factors}

    if args.max_values_per_factor > 0:
        trimmed: Dict[str, List[float]] = {}
        for f, vals in factor_grid.items():
            trimmed[f] = select_values_with_default(vals, float(default_config[f]), args.max_values_per_factor)
        factor_grid = trimmed

    if args.e2e_smoke_small:
        compact: Dict[str, List[float]] = {}
        for f, vals in factor_grid.items():
            dv = float(default_config[f])
            alt = None
            for v in vals:
                if not np.isclose(v, dv):
                    alt = v
                    break
            compact[f] = [dv] if alt is None else sorted([dv, alt])
        factor_grid = compact

    ts = datetime.now().strftime("%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"Experiments/analysis/drpo_sensitivity_oat_{args.profile}_{ts}"

    if args.persist_every_n <= 0:
        raise RuntimeError("--persist_every_n must be >= 1")

    args.default_config = default_config
    args.factor_grid = factor_grid


def token(v: float) -> str:
    return str(v).replace("-", "m").replace(".", "p")


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    return 1.96 if df > 30 else 2.0


def as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def run_log_path(root: Path, run_id: str, suffix: str, seed: int) -> Path:
    return root / "Experiments" / "Parcelpoint_py" / "pricing" / "DRPO" / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log"


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.findall(text)
    return float(m[-1]) if m else None


def parse_metrics(log: Path) -> Optional[Dict[str, Optional[float]]]:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="ignore")
    net_profit = extract_last(METRIC_REGEX["net_profit"], txt)
    total_costs = extract_last(METRIC_REGEX["total_costs"], txt)
    quit_rate = extract_last(METRIC_REGEX["quit_rate"], txt)
    served_demand = extract_last(METRIC_REGEX["served_demand"], txt)
    total_demand = extract_last(METRIC_REGEX["total_demand"], txt)
    served_rate = None
    if served_demand is not None and total_demand is not None and total_demand > 0:
        served_rate = served_demand / total_demand

    if net_profit is None or total_costs is None or quit_rate is None:
        return None

    return {
        "net_profit": net_profit,
        "total_costs": total_costs,
        "quit_rate": quit_rate,
        "served_demand": served_demand,
        "total_demand": total_demand,
        "served_rate": served_rate,
    }


def has_gpu_marker(log: Path) -> bool:
    return log.exists() and ("Using GPU device: cuda" in log.read_text(encoding="utf-8", errors="ignore"))


def probe_torch(pyexe: str) -> Dict[str, object]:
    code = "import json,torch;print(json.dumps({'torch_version':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
    cp = subprocess.run([pyexe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if cp.returncode != 0:
        raise RuntimeError(f"Torch probe failed for {pyexe}:\n{cp.stderr}")
    return json.loads(cp.stdout.strip())


def validate_runtime(args: argparse.Namespace) -> None:
    info = probe_torch(args.python_executable)
    print(
        f"[INFO] Runtime probe: python={args.python_executable}, torch={info['torch_version']}, "
        f"cuda_available={info['cuda_available']}, cuda_count={info['cuda_count']}"
    )
    if (not args.allow_cpu) and (not info["cuda_available"]):
        raise RuntimeError("CUDA unavailable in selected runtime. Install CUDA torch or pass --allow_cpu.")


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def build_cmd(args: argparse.Namespace, run_id: str, seed: int, episodes: int, cfg: Dict[str, float]) -> List[str]:
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
        "--gpu",
        str(args.gpu),
    ]

    for k in sorted(cfg.keys()):
        cmd.extend([f"--{k}", cli_value(k, cfg[k])])

    cmd.extend(["--experiment", run_id, "--folder_suffix", args.folder_suffix])
    return cmd


def run_once(args: argparse.Namespace, root: Path, stage: str, factor: str, value: float, seed: int, episodes: int, cfg: Dict[str, float]) -> RunRecord:
    run_id = f"{args.run_prefix}_{stage}_{factor}_{token(value)}"
    log = run_log_path(root, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args, run_id, seed, episodes, cfg)

    if args.skip_existing and (not args.disable_cache):
        m = parse_metrics(log)
        if m is not None:
            if (not args.allow_cpu) and (not has_gpu_marker(log)):
                raise RuntimeError(f"Cached run has no GPU marker: {log}")
            return RunRecord(
                stage,
                factor,
                float(value),
                seed,
                episodes,
                run_id,
                "cached",
                0.0,
                float(m["net_profit"]),
                float(m["total_costs"]),
                float(m["quit_rate"]),
                as_float(m.get("served_demand")),
                as_float(m.get("total_demand")),
                as_float(m.get("served_rate")),
                str(log),
                " ".join(cmd),
            )

    timeout = None if args.run_timeout_sec <= 0 else args.run_timeout_sec
    attempts = max(1, args.max_retries + 1)
    last_error = ""
    for att in range(1, attempts + 1):
        t0 = time.time()
        try:
            cp = subprocess.run(
                cmd,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            last_error = f"Timeout attempt {att}/{attempts} for {run_id}, seed={seed}. tail={(e.stdout or '')[-1000:] if e.stdout else ''}"
            if att < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        rt = time.time() - t0
        m = parse_metrics(log)
        if cp.returncode != 0:
            last_error = f"Return code {cp.returncode} attempt {att}/{attempts} for {run_id}, seed={seed}. tail={(cp.stdout or '')[-1500:]}"
        elif m is None:
            last_error = f"Metrics missing attempt {att}/{attempts}: {log}"
        elif (not args.allow_cpu) and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing attempt {att}/{attempts}: {log}"
        else:
            status = "completed" if att == 1 else f"completed_retry_{att}"
            return RunRecord(
                stage,
                factor,
                float(value),
                seed,
                episodes,
                run_id,
                status,
                rt,
                float(m["net_profit"]),
                float(m["total_costs"]),
                float(m["quit_rate"]),
                as_float(m.get("served_demand")),
                as_float(m.get("total_demand")),
                as_float(m.get("served_rate")),
                str(log),
                " ".join(cmd),
            )

        if att < attempts:
            time.sleep(args.retry_backoff_sec)
    raise RuntimeError(last_error)


def summarize(records: List[RunRecord]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float], List[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.factor, r.value), []).append(r)

    out: List[Dict[str, Any]] = []
    for (factor, value), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        row: Dict[str, Any] = {"factor": factor, "value": value, "n_runs": len(rs)}
        for metric in SUMMARY_METRICS:
            vals = [as_float(getattr(r, metric)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
                row[f"{metric}_ci95_low"] = ""
                row[f"{metric}_ci95_high"] = ""
                continue
            arr = np.array(vals, dtype=float)
            mean_v = float(np.mean(arr))
            std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            t = t_critical_95(len(arr) - 1)
            half = t * std_v / math.sqrt(len(arr)) if len(arr) > 0 else float("nan")
            row[f"{metric}_mean"] = mean_v
            row[f"{metric}_std"] = std_v
            row[f"{metric}_ci95_low"] = mean_v - half
            row[f"{metric}_ci95_high"] = mean_v + half
        out.append(row)
    return out


def basic_summary(enhanced: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for r in enhanced:
        row = {"factor": r["factor"], "value": r["value"], "n_runs": r["n_runs"]}
        for metric in BASIC_STAGE_METRICS:
            row[f"{metric}_mean"] = r.get(f"{metric}_mean", "")
            row[f"{metric}_std"] = r.get(f"{metric}_std", "")
        rows.append(row)
    return rows


def metric_gain(delta_raw: float, primary_metric: str) -> float:
    return delta_raw if PRIMARY_DIRECTION[primary_metric] == "max" else -delta_raw


def choose_stage2_candidates(stage1_summary_enhanced: List[Dict[str, Any]], factor_grid: Dict[str, List[float]], default_config: Dict[str, float], primary_metric: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    mk = f"{primary_metric}_mean"
    maximize = PRIMARY_DIRECTION[primary_metric] == "max"

    for factor in factor_grid:
        rows = [r for r in stage1_summary_enhanced if r["factor"] == factor and as_float(r.get(mk)) is not None]
        if not rows:
            out[factor] = float(default_config[factor])
            continue

        dv = float(default_config[factor])

        def sort_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
            score = float(r[mk])
            if not maximize:
                score = -score
            return (score, -abs(float(r["value"]) - dv), -float(r["value"]))

        best = sorted(rows, key=sort_key, reverse=True)[0]
        out[factor] = float(best["value"])
    return out


def stage2_values(candidates: Dict[str, float], factor_grid: Dict[str, List[float]], default_config: Dict[str, float]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for factor in factor_grid:
        dv = float(default_config[factor])
        cv = float(candidates.get(factor, dv))
        uniq = sorted({dv, cv})
        out[factor] = uniq
    return out


def sensitivity_scores(stage1_summary_enhanced: List[Dict[str, Any]], factor_grid: Dict[str, List[float]], default_config: Dict[str, float], primary_metric: str) -> List[Dict[str, Any]]:
    mk = f"{primary_metric}_mean"
    out: List[Dict[str, Any]] = []
    for factor in factor_grid:
        rows = sorted(
            [r for r in stage1_summary_enhanced if r["factor"] == factor and as_float(r.get(mk)) is not None],
            key=lambda x: float(x["value"]),
        )
        if len(rows) < 2:
            continue
        x = np.array([float(r["value"]) for r in rows], dtype=float)
        y = np.array([float(r[mk]) for r in rows], dtype=float)
        dv = float(default_config[factor])
        idxs = np.where(np.isclose(x, dv))[0]
        if len(idxs) == 0:
            continue
        i = int(idxs[0])
        if i == 0:
            slope = (y[1] - y[0]) / (x[1] - x[0])
        elif i == len(x) - 1:
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        else:
            slope = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
        out.append(
            {
                "factor": factor,
                "primary_metric": primary_metric,
                "default_value": dv,
                "default_primary_mean": float(y[i]),
                "local_slope": float(slope),
                "abs_local_slope": float(abs(slope)),
                "range_max_diff": float(np.max(y) - np.min(y)),
            }
        )
    return out


def build_parameter_catalog(profile: str, factor_grid: Dict[str, List[float]], default_config: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for factor, vals in factor_grid.items():
        rows.append(
            {
                "profile": profile,
                "factor": factor,
                "dtype": factor_dtype(factor),
                "default_value": default_config[factor],
                "n_levels": len(vals),
                "levels_csv": ",".join(str(v) for v in vals),
            }
        )
    return rows


def compute_stage2_guardrail_and_recommendations(
    stage2_summary_enhanced: List[Dict[str, Any]],
    candidates: Dict[str, float],
    factor_grid: Dict[str, List[float]],
    default_config: Dict[str, float],
    primary_metric: str,
    guardrail_quit_delta_pp: float,
    guardrail_served_rate_delta: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mk = f"{primary_metric}_mean"
    ranking_rows: List[Dict[str, Any]] = []
    rec_rows: List[Dict[str, Any]] = []

    by_factor: Dict[str, Dict[float, Dict[str, Any]]] = {}
    for r in stage2_summary_enhanced:
        by_factor.setdefault(str(r["factor"]), {})[float(r["value"])] = r

    for factor in factor_grid:
        dv = float(default_config[factor])
        cv = float(candidates.get(factor, dv))
        table = by_factor.get(factor, {})
        def_row = table.get(dv)
        cand_row = table.get(cv)

        default_primary = as_float(def_row.get(mk)) if def_row else None
        candidate_primary = as_float(cand_row.get(mk)) if cand_row else None
        default_quit = as_float(def_row.get("quit_rate_mean")) if def_row else None
        candidate_quit = as_float(cand_row.get("quit_rate_mean")) if cand_row else None
        default_served = as_float(def_row.get("served_rate_mean")) if def_row else None
        candidate_served = as_float(cand_row.get("served_rate_mean")) if cand_row else None

        primary_delta = (candidate_primary - default_primary) if (candidate_primary is not None and default_primary is not None) else float("nan")
        gain = metric_gain(primary_delta, primary_metric) if not math.isnan(primary_delta) else float("nan")
        quit_delta = (candidate_quit - default_quit) if (candidate_quit is not None and default_quit is not None) else float("nan")
        served_delta = (candidate_served - default_served) if (candidate_served is not None and default_served is not None) else float("nan")

        guardrail_pass = bool(
            not math.isnan(quit_delta)
            and not math.isnan(served_delta)
            and (quit_delta <= guardrail_quit_delta_pp)
            and (served_delta >= guardrail_served_rate_delta)
        )

        if np.isclose(cv, dv):
            recommended_value = dv
            recommendation_type = "default"
            risk_flag = ""
            selected_guardrail_pass = True
        else:
            if guardrail_pass:
                recommended_value = cv
                recommendation_type = "guardrail_pass"
                risk_flag = ""
                selected_guardrail_pass = True
            else:
                if (not math.isnan(gain)) and gain > 0:
                    recommended_value = cv
                    recommendation_type = "fallback_risky"
                    risk_flag = "RED"
                    selected_guardrail_pass = False
                else:
                    recommended_value = dv
                    recommendation_type = "default_preferred"
                    risk_flag = "candidate_guardrail_fail"
                    selected_guardrail_pass = False

        ranking_rows.append(
            {
                "factor": factor,
                "primary_metric": primary_metric,
                "default_value": dv,
                "candidate_value": cv,
                "default_primary_mean": default_primary,
                "candidate_primary_mean": candidate_primary,
                "primary_delta_candidate_minus_default": primary_delta,
                "primary_gain_for_ranking": gain,
                "default_quit_rate_mean": default_quit,
                "candidate_quit_rate_mean": candidate_quit,
                "quit_rate_delta_candidate_minus_default": quit_delta,
                "default_served_rate_mean": default_served,
                "candidate_served_rate_mean": candidate_served,
                "served_rate_delta_candidate_minus_default": served_delta,
                "guardrail_quit_threshold_pp": guardrail_quit_delta_pp,
                "guardrail_served_rate_threshold": guardrail_served_rate_delta,
                "guardrail_pass": guardrail_pass,
                "recommended_value": recommended_value,
                "recommendation_type": recommendation_type,
                "risk_flag": risk_flag,
            }
        )

        rec_rows.append(
            {
                "factor": factor,
                "recommended_value": recommended_value,
                "recommendation_type": recommendation_type,
                "risk_flag": risk_flag,
                "default_value": dv,
                "candidate_value": cv,
                "primary_metric": primary_metric,
                "primary_delta_candidate_minus_default": primary_delta,
                "guardrail_pass": selected_guardrail_pass,
            }
        )

    def _rank_key(r: Dict[str, Any]) -> Tuple[float, str]:
        gain_v = as_float(r.get("primary_gain_for_ranking"))
        if gain_v is None or math.isnan(gain_v):
            return (1e30, str(r["factor"]))
        return (-gain_v, str(r["factor"]))

    ranking_rows = sorted(ranking_rows, key=_rank_key)
    for i, r in enumerate(ranking_rows, 1):
        r["rank_by_primary_gain"] = i

    rec_rows = sorted(rec_rows, key=lambda r: str(r["factor"]))
    return ranking_rows, rec_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_dict(r: RunRecord) -> Dict[str, Any]:
    return {
        "stage": r.stage,
        "factor": r.factor,
        "value": r.value,
        "seed": r.seed,
        "episodes": r.episodes,
        "run_id": r.run_id,
        "status": r.status,
        "runtime_sec": r.runtime_sec,
        "net_profit": r.net_profit,
        "total_costs": r.total_costs,
        "quit_rate": r.quit_rate,
        "served_demand": r.served_demand,
        "total_demand": r.total_demand,
        "served_rate": r.served_rate,
        "log_path": r.log_path,
        "command": r.command,
    }


def run_key(stage: str, factor: str, value: float, seed: int) -> Tuple[str, str, float, int]:
    return (str(stage), str(factor), float(value), int(seed))


def run_key_from_record(r: RunRecord) -> Tuple[str, str, float, int]:
    return run_key(r.stage, r.factor, r.value, r.seed)


def dedupe_run_records(records: List[RunRecord]) -> List[RunRecord]:
    latest_by_key: Dict[Tuple[str, str, float, int], RunRecord] = {}
    for rec in records:
        latest_by_key[run_key_from_record(rec)] = rec
    ordered_keys = sorted(latest_by_key.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
    return [latest_by_key[k] for k in ordered_keys]


def parse_raw_run_record(row: Dict[str, Any], expected_stage: str, run_prefix: str) -> Optional[RunRecord]:
    stage = str(row.get("stage", "")).strip()
    if stage != expected_stage:
        return None

    factor = str(row.get("factor", "")).strip()
    value = as_float(row.get("value"))
    seed_f = as_float(row.get("seed"))
    run_id = str(row.get("run_id", "")).strip()
    if factor == "" or value is None or seed_f is None or run_id == "":
        return None
    if run_prefix and (not run_id.startswith(f"{run_prefix}_{expected_stage}_")):
        return None

    net_profit = as_float(row.get("net_profit"))
    total_costs = as_float(row.get("total_costs"))
    quit_rate = as_float(row.get("quit_rate"))
    if net_profit is None or total_costs is None or quit_rate is None:
        return None

    episodes_f = as_float(row.get("episodes"))
    runtime_f = as_float(row.get("runtime_sec"))
    episodes = int(round(episodes_f)) if episodes_f is not None else 0
    runtime_sec = float(runtime_f) if runtime_f is not None else 0.0

    return RunRecord(
        stage=stage,
        factor=factor,
        value=float(value),
        seed=int(round(seed_f)),
        episodes=episodes,
        run_id=run_id,
        status=str(row.get("status", "loaded")).strip() or "loaded",
        runtime_sec=runtime_sec,
        net_profit=float(net_profit),
        total_costs=float(total_costs),
        quit_rate=float(quit_rate),
        served_demand=as_float(row.get("served_demand")),
        total_demand=as_float(row.get("total_demand")),
        served_rate=as_float(row.get("served_rate")),
        log_path=str(row.get("log_path", "")).strip(),
        command=str(row.get("command", "")).strip(),
    )


def load_existing_raw_records(raw_path: Path, expected_stage: str, run_prefix: str) -> List[RunRecord]:
    if not raw_path.exists():
        return []
    try:
        with raw_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"[WARN] Failed to read existing raw file: {raw_path}. error={e}")
        return []

    parsed: List[RunRecord] = []
    skipped = 0
    for row in rows:
        rec = parse_raw_run_record(row, expected_stage=expected_stage, run_prefix=run_prefix)
        if rec is None:
            skipped += 1
            continue
        parsed.append(rec)

    deduped = dedupe_run_records(parsed)
    print(
        f"[INFO] Loaded existing {expected_stage} records: raw_rows={len(rows)}, "
        f"parsed={len(parsed)}, deduped={len(deduped)}, skipped={skipped}"
    )
    return deduped


def filter_records_by_expected(
    records: List[RunRecord], expected_keys: Set[Tuple[str, str, float, int]]
) -> Tuple[List[RunRecord], int]:
    kept = [r for r in records if run_key_from_record(r) in expected_keys]
    dropped = len(records) - len(kept)
    return dedupe_run_records(kept), dropped


def filter_missing_jobs(
    jobs: List[Tuple[str, float, int, Dict[str, float]]],
    stage: str,
    existing_keys: Set[Tuple[str, str, float, int]],
) -> Tuple[List[Tuple[str, float, int, Dict[str, float]]], int]:
    pending: List[Tuple[str, float, int, Dict[str, float]]] = []
    skipped = 0
    for factor, value, seed, cfg in jobs:
        if run_key(stage, factor, float(value), int(seed)) in existing_keys:
            skipped += 1
            continue
        pending.append((factor, value, seed, cfg))
    return pending, skipped


def persist(
    output_dir: Path,
    args: argparse.Namespace,
    parameter_catalog: List[Dict[str, Any]],
    stage1_records: List[RunRecord],
    stage2_records: List[RunRecord],
    candidates: Optional[Dict[str, float]],
) -> None:
    s1_raw = [to_dict(r) for r in stage1_records]
    s2_raw = [to_dict(r) for r in stage2_records]

    s1_enh = summarize(stage1_records) if stage1_records else []
    s2_enh = summarize(stage2_records) if stage2_records else []
    s1_basic = basic_summary(s1_enh)
    s2_basic = basic_summary(s2_enh)
    sens = sensitivity_scores(s1_enh, args.factor_grid, args.default_config, args.primary_metric) if s1_enh else []

    write_csv(
        output_dir / "stage1_raw.csv",
        s1_raw,
        [
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
            "served_demand",
            "total_demand",
            "served_rate",
            "log_path",
            "command",
        ],
    )
    write_csv(
        output_dir / "stage2_raw.csv",
        s2_raw,
        [
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
            "served_demand",
            "total_demand",
            "served_rate",
            "log_path",
            "command",
        ],
    )

    s_basic_fields = ["factor", "value", "n_runs"] + [f"{m}_{suf}" for m in BASIC_STAGE_METRICS for suf in ("mean", "std")]
    write_csv(output_dir / "stage1_summary.csv", s1_basic, s_basic_fields)
    write_csv(output_dir / "stage2_summary.csv", s2_basic, s_basic_fields)

    s_enh_fields = ["factor", "value", "n_runs"] + [f"{m}_{suf}" for m in SUMMARY_METRICS for suf in ("mean", "std", "ci95_low", "ci95_high")]
    write_csv(output_dir / "stage1_summary_enhanced.csv", s1_enh, s_enh_fields)
    write_csv(output_dir / "stage2_summary_enhanced.csv", s2_enh, s_enh_fields)

    write_csv(
        output_dir / "sensitivity_scores.csv",
        sens,
        ["factor", "primary_metric", "default_value", "default_primary_mean", "local_slope", "abs_local_slope", "range_max_diff"],
    )

    write_csv(
        output_dir / "parameter_catalog.csv",
        parameter_catalog,
        ["profile", "factor", "dtype", "default_value", "n_levels", "levels_csv"],
    )

    if candidates is not None:
        s2vals = stage2_values(candidates, args.factor_grid, args.default_config)
        crows = [
            {
                "factor": f,
                "default_value": float(args.default_config[f]),
                "stage1_best_value": float(candidates.get(f, args.default_config[f])),
                "stage2_values": ",".join(str(v) for v in s2vals.get(f, [])),
            }
            for f in args.factor_grid
        ]
        write_csv(output_dir / "stage2_candidates.csv", crows, ["factor", "default_value", "stage1_best_value", "stage2_values"])

        if s2_enh:
            ranking_rows, rec_rows = compute_stage2_guardrail_and_recommendations(
                stage2_summary_enhanced=s2_enh,
                candidates=candidates,
                factor_grid=args.factor_grid,
                default_config=args.default_config,
                primary_metric=args.primary_metric,
                guardrail_quit_delta_pp=args.guardrail_quit_delta_pp,
                guardrail_served_rate_delta=args.guardrail_served_rate_delta,
            )
            write_csv(
                output_dir / "stage2_guardrail_ranking.csv",
                ranking_rows,
                [
                    "rank_by_primary_gain",
                    "factor",
                    "primary_metric",
                    "default_value",
                    "candidate_value",
                    "default_primary_mean",
                    "candidate_primary_mean",
                    "primary_delta_candidate_minus_default",
                    "primary_gain_for_ranking",
                    "default_quit_rate_mean",
                    "candidate_quit_rate_mean",
                    "quit_rate_delta_candidate_minus_default",
                    "default_served_rate_mean",
                    "candidate_served_rate_mean",
                    "served_rate_delta_candidate_minus_default",
                    "guardrail_quit_threshold_pp",
                    "guardrail_served_rate_threshold",
                    "guardrail_pass",
                    "recommended_value",
                    "recommendation_type",
                    "risk_flag",
                ],
            )
            write_csv(
                output_dir / "final_recommendations.csv",
                rec_rows,
                [
                    "factor",
                    "recommended_value",
                    "recommendation_type",
                    "risk_flag",
                    "default_value",
                    "candidate_value",
                    "primary_metric",
                    "primary_delta_candidate_minus_default",
                    "guardrail_pass",
                ],
            )

    long_rows: List[Dict[str, Any]] = []
    for src, rows in [("stage1_summary", s1_basic), ("stage2_summary", s2_basic)]:
        for r in rows:
            for m in [f"{mm}_mean" for mm in BASIC_STAGE_METRICS] + [f"{mm}_std" for mm in BASIC_STAGE_METRICS]:
                long_rows.append(
                    {
                        "source": src,
                        "factor": r["factor"],
                        "value": r["value"],
                        "metric": m,
                        "metric_value": r.get(m, ""),
                        "n_runs": r["n_runs"],
                    }
                )
    for r in sens:
        for m in ["default_value", "default_primary_mean", "local_slope", "abs_local_slope", "range_max_diff"]:
            long_rows.append(
                {
                    "source": "sensitivity_scores",
                    "factor": r["factor"],
                    "value": r["default_value"],
                    "metric": m,
                    "metric_value": r[m],
                    "n_runs": "",
                }
            )
    write_csv(output_dir / "result_3.9.csv", long_rows, ["source", "factor", "value", "metric", "metric_value", "n_runs"])


def plot_stage1(stage1_summary_enhanced: List[Dict[str, Any]], factor_grid: Dict[str, List[float]], default_config: Dict[str, float], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for factor in factor_grid:
        rows = sorted([r for r in stage1_summary_enhanced if r["factor"] == factor], key=lambda x: float(x["value"]))
        if not rows:
            continue

        x = np.array([float(r["value"]) for r in rows], dtype=float)
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        specs = [
            ("net_profit_mean", "net_profit_std", "Net Profit"),
            ("total_costs_mean", "total_costs_std", "Total Costs"),
            ("quit_rate_mean", "quit_rate_std", "Quit Rate (%)"),
        ]
        for ax, (mk, sk, ylabel) in zip(axes, specs):
            y = np.array([float(as_float(r.get(mk)) or 0.0) for r in rows], dtype=float)
            s = np.array([float(as_float(r.get(sk)) or 0.0) for r in rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=2)
            ax.fill_between(x, y - s, y + s, alpha=0.2)
            ax.axvline(float(default_config[factor]), color="gray", linestyle="--", linewidth=1)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel(factor)
        fig.suptitle(f"Stage1 OAT Sensitivity - {factor}")
        fig.tight_layout()
        fig.savefig(fig_dir / f"stage1_{factor}.png", dpi=200)
        plt.close(fig)


def smoke(args: argparse.Namespace, root: Path, output_dir: Path) -> None:
    cfg = dict(args.default_config)
    first_factor = next(iter(args.factor_grid.keys()))
    value = float(cfg[first_factor])

    rec = run_once(args, root, "smoke", "smoke_config", value, args.smoke_seed, args.smoke_episodes, cfg)
    txt = Path(rec.log_path).read_text(encoding="utf-8", errors="ignore")
    ok = {
        "spo_result_constructor_error": "SPOExperimentResult() takes no arguments" in txt,
        "spo_training_data_populated": "[SPO+ debug] spo_training_data populated" in txt,
        "spo_weight_positive": "[SPO+ debug] spo_weight became positive" in txt,
        "gpu_used": "Using GPU device: cuda" in txt,
    }

    (output_dir / "smoke_validation.txt").write_text(
        "\n".join(
            [
                f"smoke_log={rec.log_path}",
                f"spo_result_constructor_error={ok['spo_result_constructor_error']}",
                f"spo_training_data_populated={ok['spo_training_data_populated']}",
                f"spo_weight_positive={ok['spo_weight_positive']}",
                f"gpu_used={ok['gpu_used']}",
                f"net_profit={rec.net_profit}",
                f"total_costs={rec.total_costs}",
                f"quit_rate={rec.quit_rate}",
                f"served_rate={rec.served_rate}",
            ]
        ),
        encoding="utf-8",
    )

    passed = (
        (not ok["spo_result_constructor_error"])
        and ok["spo_training_data_populated"]
        and ok["spo_weight_positive"]
        and (args.allow_cpu or ok["gpu_used"])
    )
    if (not passed) and (not args.allow_smoke_failure):
        raise RuntimeError(f"Smoke validation failed. See {output_dir / 'smoke_validation.txt'}")


def build_stage1_jobs(args: argparse.Namespace) -> List[Tuple[str, float, int, Dict[str, float]]]:
    jobs = []
    for factor, values in args.factor_grid.items():
        for value in values:
            for seed in args.stage1_seeds:
                cfg = dict(args.default_config)
                cfg[factor] = to_numeric(factor, value)
                jobs.append((factor, float(value), int(seed), cfg))
    return jobs


def build_stage2_jobs(args: argparse.Namespace, s2vals: Dict[str, List[float]]) -> List[Tuple[str, float, int, Dict[str, float]]]:
    jobs = []
    for factor in args.factor_grid:
        for value in s2vals[factor]:
            for seed in args.stage2_seeds:
                cfg = dict(args.default_config)
                cfg[factor] = to_numeric(factor, value)
                jobs.append((factor, float(value), int(seed), cfg))
    return jobs


def write_run_meta(args: argparse.Namespace, output_dir: Path, parameter_catalog: List[Dict[str, Any]]) -> None:
    data = {
        "profile": args.profile,
        "instance": args.instance,
        "factor_grid_json": args.factor_grid_json,
        "data_seed": args.data_seed,
        "data_seed_test": args.data_seed_test,
        "stage1_seeds": args.stage1_seeds,
        "stage2_seeds": args.stage2_seeds,
        "stage1_episodes": args.stage1_episodes,
        "stage2_episodes": args.stage2_episodes,
        "skip_stage2": args.skip_stage2,
        "primary_metric": args.primary_metric,
        "guardrail_quit_delta_pp": args.guardrail_quit_delta_pp,
        "guardrail_served_rate_delta": args.guardrail_served_rate_delta,
        "run_smoke_validation": args.run_smoke_validation,
        "disable_cache": args.disable_cache,
        "resume_missing_only": args.resume_missing_only,
        "persist_every_n": args.persist_every_n,
        "resume_trust_existing_raw": args.resume_trust_existing_raw,
        "factor_grid": args.factor_grid,
        "default_config": args.default_config,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "parameter_count": len(parameter_catalog),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolve_runtime_profile(args)

    root = Path(__file__).resolve().parent
    output_dir = (root / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and (not args.allow_existing_output_dir):
        raise RuntimeError(
            f"Output dir is not empty: {output_dir}. Use a unique --output_dir or pass --allow_existing_output_dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    parameter_catalog = build_parameter_catalog(args.profile, args.factor_grid, args.default_config)
    write_csv(
        output_dir / "parameter_catalog.csv",
        parameter_catalog,
        ["profile", "factor", "dtype", "default_value", "n_levels", "levels_csv"],
    )
    write_run_meta(args, output_dir, parameter_catalog)

    validate_runtime(args)

    if args.run_smoke_validation:
        print("[INFO] Running smoke validation...")
        smoke(args, root, output_dir)
        print("[INFO] Smoke validation completed.")
        if args.only_smoke:
            return

    if args.diagnose_factor is not None:
        if args.diagnose_value is None:
            raise RuntimeError("--diagnose_factor requires --diagnose_value")
        if args.diagnose_factor not in args.default_config:
            raise RuntimeError(f"diagnose_factor '{args.diagnose_factor}' is not tunable in profile {args.profile}")
        cfg = dict(args.default_config)
        cfg[args.diagnose_factor] = to_numeric(args.diagnose_factor, float(args.diagnose_value))
        rec = run_once(
            args,
            root,
            "diagnose",
            args.diagnose_factor,
            float(args.diagnose_value),
            args.diagnose_seed,
            args.diagnose_episodes,
            cfg,
        )
        (output_dir / "diagnose_report.txt").write_text(
            "\n".join(
                [
                    f"factor={args.diagnose_factor}",
                    f"value={args.diagnose_value}",
                    f"seed={args.diagnose_seed}",
                    f"episodes={args.diagnose_episodes}",
                    f"status={rec.status}",
                    f"runtime_sec={rec.runtime_sec}",
                    f"log={rec.log_path}",
                ]
            ),
            encoding="utf-8",
        )
        if not args.continue_after_diagnose:
            return

    stage1_records: List[RunRecord] = []
    stage2_records: List[RunRecord] = []
    if args.resume_missing_only and args.resume_trust_existing_raw:
        stage1_records = load_existing_raw_records(output_dir / "stage1_raw.csv", expected_stage="stage1", run_prefix=args.run_prefix)
        if not args.skip_stage2:
            stage2_records = load_existing_raw_records(output_dir / "stage2_raw.csv", expected_stage="stage2", run_prefix=args.run_prefix)
    elif args.resume_missing_only:
        print("[INFO] resume_missing_only is enabled but existing raw preload is disabled.")

    stage1_jobs_all = build_stage1_jobs(args)
    expected_stage1_keys = {run_key("stage1", f, float(v), int(s)) for f, v, s, _ in stage1_jobs_all}
    if args.resume_missing_only and stage1_records:
        stage1_records, dropped_stage1_preload = filter_records_by_expected(stage1_records, expected_stage1_keys)
        if dropped_stage1_preload > 0:
            print(f"[INFO] Dropped preloaded Stage1 records outside current grid: {dropped_stage1_preload}")

    stage1_jobs = stage1_jobs_all
    if args.resume_missing_only:
        existing_stage1_keys = {run_key_from_record(r) for r in stage1_records}
        stage1_jobs, skipped_stage1 = filter_missing_jobs(stage1_jobs_all, stage="stage1", existing_keys=existing_stage1_keys)
        print(
            f"[INFO] Stage1 resume filter: total={len(stage1_jobs_all)}, "
            f"preloaded={len(stage1_records)}, pending={len(stage1_jobs)}, skipped={skipped_stage1}"
        )

    print(
        f"[INFO] Profile={args.profile}, factors={len(args.factor_grid)}, "
        f"Stage1 total={len(stage1_jobs_all)}, pending={len(stage1_jobs)}"
    )
    stage1_seen = {run_key_from_record(r) for r in stage1_records}
    stage1_new = 0
    for i, (factor, value, seed, cfg) in enumerate(stage1_jobs, 1):
        print(f"[Stage1 {i}/{len(stage1_jobs)}] factor={factor}, value={value}, seed={seed}, episodes={args.stage1_episodes}")
        k = run_key("stage1", factor, float(value), int(seed))
        if k in stage1_seen:
            continue
        stage1_records.append(run_once(args, root, "stage1", factor, value, seed, args.stage1_episodes, cfg))
        stage1_seen.add(k)
        stage1_new += 1
        if stage1_new % args.persist_every_n == 0:
            persist(output_dir, args, parameter_catalog, stage1_records, stage2_records, None)

    persist(output_dir, args, parameter_catalog, stage1_records, stage2_records, None)

    s1_enh = summarize(stage1_records)
    candidates = choose_stage2_candidates(s1_enh, args.factor_grid, args.default_config, args.primary_metric)
    s2vals: Dict[str, List[float]] = {f: [] for f in args.factor_grid}
    drows: List[Dict[str, Any]] = []
    s2_enh: List[Dict[str, Any]] = []

    if args.skip_stage2:
        for factor in args.factor_grid:
            dv = float(args.default_config[factor])
            cv = float(candidates.get(factor, dv))
            drows.append(
                {
                    "factor": factor,
                    "default_value": dv,
                    "candidate_value": cv,
                    "stage1_delta_candidate_minus_default": "",
                    "stage2_delta_candidate_minus_default": "",
                    "direction_consistent": "SKIPPED",
                    "note": "stage2 skipped",
                }
            )
        write_csv(
            output_dir / "stage2_direction_check.csv",
            drows,
            [
                "factor",
                "default_value",
                "candidate_value",
                "stage1_delta_candidate_minus_default",
                "stage2_delta_candidate_minus_default",
                "direction_consistent",
                "note",
            ],
        )
    else:
        s2vals = stage2_values(candidates, args.factor_grid, args.default_config)
        stage2_jobs_all = build_stage2_jobs(args, s2vals)
        expected_stage2_keys = {run_key("stage2", f, float(v), int(s)) for f, vals in s2vals.items() for v in vals for s in args.stage2_seeds}
        if args.resume_missing_only and stage2_records:
            stage2_records, dropped_stage2_preload = filter_records_by_expected(stage2_records, expected_stage2_keys)
            if dropped_stage2_preload > 0:
                print(f"[INFO] Dropped preloaded Stage2 records outside current candidate grid: {dropped_stage2_preload}")

        stage2_jobs = stage2_jobs_all
        if args.resume_missing_only:
            existing_stage2_keys = {run_key_from_record(r) for r in stage2_records}
            stage2_jobs, skipped_stage2 = filter_missing_jobs(stage2_jobs_all, stage="stage2", existing_keys=existing_stage2_keys)
            print(
                f"[INFO] Stage2 resume filter: total={len(stage2_jobs_all)}, "
                f"preloaded={len(stage2_records)}, pending={len(stage2_jobs)}, skipped={skipped_stage2}"
            )

        print(f"[INFO] Stage2 total={len(stage2_jobs_all)}, pending={len(stage2_jobs)}")
        stage2_seen = {run_key_from_record(r) for r in stage2_records}
        stage2_new = 0
        for i, (factor, value, seed, cfg) in enumerate(stage2_jobs, 1):
            print(f"[Stage2 {i}/{len(stage2_jobs)}] factor={factor}, value={value}, seed={seed}, episodes={args.stage2_episodes}")
            k = run_key("stage2", factor, float(value), int(seed))
            if k in stage2_seen:
                continue
            stage2_records.append(run_once(args, root, "stage2", factor, value, seed, args.stage2_episodes, cfg))
            stage2_seen.add(k)
            stage2_new += 1
            if stage2_new % args.persist_every_n == 0:
                persist(output_dir, args, parameter_catalog, stage1_records, stage2_records, candidates)

        persist(output_dir, args, parameter_catalog, stage1_records, stage2_records, candidates)

        s1_enh = summarize(stage1_records)
        s2_enh = summarize(stage2_records)

        s1map = {(r["factor"], float(r["value"])): r for r in s1_enh}
        s2map = {(r["factor"], float(r["value"])): r for r in s2_enh}
        for factor in args.factor_grid:
            dv = float(args.default_config[factor])
            cv = float(candidates.get(factor, dv))
            row = {
                "factor": factor,
                "default_value": dv,
                "candidate_value": cv,
                "stage1_delta_candidate_minus_default": "",
                "stage2_delta_candidate_minus_default": "",
                "direction_consistent": "",
                "note": "",
            }
            mk = f"{args.primary_metric}_mean"
            if np.isclose(dv, cv):
                row.update(
                    {
                        "stage1_delta_candidate_minus_default": 0.0,
                        "stage2_delta_candidate_minus_default": 0.0,
                        "direction_consistent": True,
                        "note": "candidate equals default",
                    }
                )
            elif (factor, dv) in s1map and (factor, cv) in s1map and (factor, dv) in s2map and (factor, cv) in s2map:
                s1_cv = as_float(s1map[(factor, cv)].get(mk))
                s1_dv = as_float(s1map[(factor, dv)].get(mk))
                s2_cv = as_float(s2map[(factor, cv)].get(mk))
                s2_dv = as_float(s2map[(factor, dv)].get(mk))
                if None not in (s1_cv, s1_dv, s2_cv, s2_dv):
                    d1 = float(s1_cv - s1_dv)
                    d2 = float(s2_cv - s2_dv)
                    cons = bool(np.sign(d1) == np.sign(d2) or np.sign(d1) == 0 or np.sign(d2) == 0)
                    row.update(
                        {
                            "stage1_delta_candidate_minus_default": d1,
                            "stage2_delta_candidate_minus_default": d2,
                            "direction_consistent": cons,
                            "note": "ok" if cons else "direction mismatch",
                        }
                    )
                else:
                    row.update({"direction_consistent": "UNKNOWN", "note": "missing metric in summary rows"})
            else:
                row.update({"direction_consistent": "UNKNOWN", "note": "missing summary rows"})
            drows.append(row)

        write_csv(
            output_dir / "stage2_direction_check.csv",
            drows,
            [
                "factor",
                "default_value",
                "candidate_value",
                "stage1_delta_candidate_minus_default",
                "stage2_delta_candidate_minus_default",
                "direction_consistent",
                "note",
            ],
        )

    expected_s1 = {(f, float(v), int(s)) for f, vals in args.factor_grid.items() for v in vals for s in args.stage1_seeds}
    actual_s1 = {(r.factor, float(r.value), int(r.seed)) for r in stage1_records}
    expected_s2 = {(f, float(v), int(s)) for f in args.factor_grid for v in s2vals.get(f, []) for s in args.stage2_seeds}
    actual_s2 = {(r.factor, float(r.value), int(r.seed)) for r in stage2_records}
    miss1 = sorted(expected_s1 - actual_s1)
    miss2 = sorted(expected_s2 - actual_s2)

    validation_lines = [
        f"profile={args.profile}",
        f"primary_metric={args.primary_metric}",
        f"skip_stage2={args.skip_stage2}",
        f"expected_stage1_runs={len(expected_s1)}",
        f"actual_stage1_runs={len(stage1_records)}",
        f"missing_stage1_runs={len(miss1)}",
        f"expected_stage2_runs={len(expected_s2)}",
        f"actual_stage2_runs={len(stage2_records)}",
        f"missing_stage2_runs={len(miss2)}",
        f"stage2_direction_mismatch_or_unknown={0 if args.skip_stage2 else sum(1 for r in drows if r['direction_consistent'] not in (True, 'True'))}",
    ]
    if miss1:
        validation_lines.append("missing_stage1_examples=" + ";".join(str(x) for x in miss1[:10]))
    if miss2:
        validation_lines.append("missing_stage2_examples=" + ";".join(str(x) for x in miss2[:10]))

    vpath = output_dir / "validation_report.txt"
    vpath.write_text("\n".join(validation_lines), encoding="utf-8")
    if miss1 or miss2:
        raise RuntimeError(f"Incomplete scan detected. See {vpath}")

    plot_stage1(s1_enh, args.factor_grid, args.default_config, output_dir / "plots")
    persist(output_dir, args, parameter_catalog, stage1_records, stage2_records, candidates)

    print("[INFO] Analysis finished.")
    print(f"[INFO] Stage1 summary: {output_dir / 'stage1_summary.csv'}")
    print(f"[INFO] Stage1 enhanced summary: {output_dir / 'stage1_summary_enhanced.csv'}")
    if args.skip_stage2:
        print("[INFO] Stage2 was skipped for this run.")
    else:
        print(f"[INFO] Stage2 summary: {output_dir / 'stage2_summary.csv'}")
        print(f"[INFO] Guardrail ranking: {output_dir / 'stage2_guardrail_ranking.csv'}")
        print(f"[INFO] Final recommendations: {output_dir / 'final_recommendations.csv'}")
    print(f"[INFO] Sensitivity scores: {output_dir / 'sensitivity_scores.csv'}")
    print(f"[INFO] Validation report: {vpath}")


if __name__ == "__main__":
    main()
