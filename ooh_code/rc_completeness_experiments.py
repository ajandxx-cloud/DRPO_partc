#!/usr/bin/env python
import argparse
import csv
import itertools
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = {
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "home_util": 1.4,
    "k": 10.0,
}
DEFAULT_BASELINE_OAT_DIR = "Experiments/analysis/drpo_sensitivity_oat_3_11_full"
LEGACY_BASELINE_OAT_DIR = "Experiments/analysis/dspo_plus_spo_sensitivity_oat_3_11_full"

OAT_FACTORS = {
    "outside_option_util": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "incentive_sens": [-0.35, -0.30, -0.25, -0.20, -0.15],
    "home_util": [1.0, 1.2, 1.4, 1.6, 1.8],
    "k": [5.0, 7.0, 10.0, 12.0, 15.0],
}

INTERACTION_LEVELS = {
    "outside_option_util": [-2.0, -1.0],
    "incentive_sens": [-0.35, -0.25],
    "home_util": [1.0, 1.4],
    "k": [10.0],
}

METRIC_PATTERNS = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?)%"),
    "home_pickup_rate": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?)"),
    "avg_charge": re.compile(r"Avg\. Charge:\s*([+-]?\d+(?:\.\d+)?)"),
    "avg_discount": re.compile(r"Avg\. Discount:\s*([+-]?\d+(?:\.\d+)?)"),
    "served_demand": re.compile(r"Accepted customers:\s*([+-]?\d+)"),
    "total_demand": re.compile(r"Total customers:\s*([+-]?\d+)"),
}

METRICS_NUMERIC = [
    "net_profit",
    "total_costs",
    "quit_rate",
    "home_pickup_rate",
    "avg_discount",
    "avg_charge",
    "served_demand",
    "total_demand",
    "served_rate",
]

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RC completeness experiment runner (seed=10, interaction, robustness).")
    p.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus"])
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--run_prefix", default="RC_COMPLETENESS")
    p.add_argument("--folder_suffix", default="_sens")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--data_seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--data_seed_test_mode", choices=["fixed1", "sync"], default="fixed1")
    p.add_argument("--fixed_data_seed_test", type=int, default=1)
    p.add_argument("--oat_stage1_episodes", type=int, default=80)
    p.add_argument("--oat_stage2_episodes", type=int, default=200)
    p.add_argument("--interaction_episodes", type=int, default=200)
    p.add_argument("--robustness_episodes", type=int, default=200)
    p.add_argument("--save_count", type=int, default=20)
    p.add_argument("--spo_warmup_episodes", type=int, default=5)
    p.add_argument("--spo_rampup_episodes", type=int, default=10)
    p.add_argument("--spo_loss_weight", type=float, default=0.7)
    p.add_argument("--run_timeout_sec", type=int, default=3600)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--output_dir", default="Experiments/analysis/drpo_rc_seed10_complete")
    p.add_argument("--oat_dir", default=None, help="Existing OAT directory. If set, OAT run is skipped unless --force_oat.")
    p.add_argument("--baseline_oat_dir", default=DEFAULT_BASELINE_OAT_DIR)
    p.add_argument("--top_candidates", type=int, default=3)
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--max_jobs_per_layer", type=int, default=0, help="For debug only; 0 means unlimited.")
    p.add_argument("--force_oat", action="store_true")
    p.add_argument("--no_interaction", action="store_true")
    p.add_argument("--no_robustness", action="store_true")
    return p.parse_args()


def token(v: float) -> str:
    return str(v).replace("-", "m").replace(".", "p")


def sanitize_tag(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("_", "-", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
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


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_last(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.findall(text)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def parse_metrics_from_log(log_path: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {k: None for k in METRIC_PATTERNS}
    out["served_rate"] = None
    if not log_path.exists():
        return out
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    for k, pat in METRIC_PATTERNS.items():
        out[k] = parse_last(pat, txt)
    served = out["served_demand"]
    total = out["total_demand"]
    if served is not None and total is not None and total > 0:
        out["served_rate"] = served / total
    return out


def has_gpu_marker(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    return "Using GPU device: cuda" in txt


def run_log_path(root: Path, run_id: str, suffix: str, seed: int) -> Path:
    return (
        root
        / "Experiments"
        / "Parcelpoint_py"
        / "pricing"
        / "DRPO"
        / f"{run_id}{suffix}"
        / str(seed)
        / "Logs"
        / "logfile.log"
    )


def build_run_cmd(
    args: argparse.Namespace,
    run_id: str,
    seed: int,
    episodes: int,
    cfg: Dict[str, float],
    data_seed: int,
    data_seed_test: int,
) -> List[str]:
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
        str(data_seed),
        "--data_seed_test",
        str(data_seed_test),
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


def build_row(
    stage: str,
    tag: str,
    seed: int,
    episodes: int,
    data_seed: int,
    data_seed_test: int,
    cfg: Dict[str, float],
    status: str,
    runtime_sec: float,
    metrics: Dict[str, Optional[float]],
    log_path: Path,
    cmd: Sequence[str],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "stage": stage,
        "tag": tag,
        "seed": int(seed),
        "episodes": int(episodes),
        "data_seed": int(data_seed),
        "data_seed_test": int(data_seed_test),
        "outside_option_util": float(cfg["outside_option_util"]),
        "incentive_sens": float(cfg["incentive_sens"]),
        "home_util": float(cfg["home_util"]),
        "k": float(cfg["k"]),
        "status": status,
        "runtime_sec": float(runtime_sec),
        "log_path": str(log_path),
        "command": " ".join(cmd),
    }
    for k in METRICS_NUMERIC:
        row[k] = metrics.get(k)
    return row


def run_single_job(
    args: argparse.Namespace,
    root: Path,
    stage: str,
    tag: str,
    seed: int,
    episodes: int,
    cfg: Dict[str, float],
    data_seed: int,
    data_seed_test: int,
) -> Dict[str, object]:
    run_id = f"{args.run_prefix}_{stage}_{sanitize_tag(tag)}"
    log_path = run_log_path(root, run_id, args.folder_suffix, seed)
    cmd = build_run_cmd(args, run_id, seed, episodes, cfg, data_seed, data_seed_test)

    if args.skip_existing:
        m = parse_metrics_from_log(log_path)
        if m["net_profit"] is not None and m["total_costs"] is not None and m["quit_rate"] is not None:
            if args.allow_cpu or has_gpu_marker(log_path):
                return build_row(
                    stage=stage,
                    tag=tag,
                    seed=seed,
                    episodes=episodes,
                    data_seed=data_seed,
                    data_seed_test=data_seed_test,
                    cfg=cfg,
                    status="cached",
                    runtime_sec=0.0,
                    metrics=m,
                    log_path=log_path,
                    cmd=cmd,
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
            tail = (e.stdout or "")[-1000:] if e.stdout else ""
            last_error = f"timeout {att}/{attempts} for {run_id}, seed={seed}, tail={tail}"
            if att < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        rt = time.time() - t0
        m = parse_metrics_from_log(log_path)
        if cp.returncode != 0:
            last_error = f"returncode {cp.returncode} {att}/{attempts} for {run_id}, seed={seed}, tail={(cp.stdout or '')[-1500:]}"
        elif m["net_profit"] is None or m["total_costs"] is None or m["quit_rate"] is None:
            last_error = f"metrics_missing {att}/{attempts} for {run_id}, seed={seed}, log={log_path}"
        elif (not args.allow_cpu) and (not has_gpu_marker(log_path)):
            last_error = f"gpu_marker_missing {att}/{attempts} for {run_id}, seed={seed}, log={log_path}"
        else:
            status = "completed" if att == 1 else f"completed_retry_{att}"
            return build_row(
                stage=stage,
                tag=tag,
                seed=seed,
                episodes=episodes,
                data_seed=data_seed,
                data_seed_test=data_seed_test,
                cfg=cfg,
                status=status,
                runtime_sec=rt,
                metrics=m,
                log_path=log_path,
                cmd=cmd,
            )

        if att < attempts:
            time.sleep(args.retry_backoff_sec)
    raise RuntimeError(last_error)


def row_key(row: Dict[str, object]) -> Tuple[str, str, int, int, int, int]:
    return (
        str(row["stage"]),
        str(row["tag"]),
        int(float(row["seed"])),
        int(float(row["data_seed"])),
        int(float(row["data_seed_test"])),
        int(float(row["episodes"])),
    )


def t_critical_95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in T_CRIT_95:
        return T_CRIT_95[df]
    if df > 30:
        return 1.96
    return 2.0


def summarize_by(rows: List[Dict[str, object]], group_keys: Sequence[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, object]] = []
    for key, rs in sorted(groups.items(), key=lambda x: x[0]):
        row: Dict[str, object] = {k: v for k, v in zip(group_keys, key)}
        row["n_runs"] = len(rs)
        for m in METRICS_NUMERIC:
            vals = [as_float(r.get(m)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
                row[f"{m}_ci95_low"] = ""
                row[f"{m}_ci95_high"] = ""
                continue
            n = len(vals)
            mean_v = sum(vals) / n
            if n <= 1:
                std_v = 0.0
            else:
                var = sum((v - mean_v) ** 2 for v in vals) / (n - 1)
                std_v = math.sqrt(var)
            t = t_critical_95(n - 1)
            half = t * std_v / math.sqrt(n) if n > 0 else float("nan")
            row[f"{m}_mean"] = mean_v
            row[f"{m}_std"] = std_v
            row[f"{m}_ci95_low"] = mean_v - half
            row[f"{m}_ci95_high"] = mean_v + half
        out.append(row)
    return out


def bootstrap_diff_ci(
    values_a: List[float],
    values_b: List[float],
    samples: int,
    seed: int = 20260312,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    if not values_a or not values_b:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(samples):
        sa = [values_a[rng.randrange(len(values_a))] for _ in range(len(values_a))]
        sb = [values_b[rng.randrange(len(values_b))] for _ in range(len(values_b))]
        diffs.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    diffs.sort()
    i_lo = int(0.025 * len(diffs))
    i_hi = max(i_lo, int(0.975 * len(diffs)) - 1)
    mean_diff = sum(diffs) / len(diffs)
    return mean_diff, diffs[i_lo], diffs[i_hi]


def probe_runtime(pyexe: str) -> Dict[str, object]:
    code = "import json,torch;print(json.dumps({'torch_version':torch.__version__,'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
    cp = subprocess.run([pyexe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if cp.returncode != 0:
        raise RuntimeError(f"Torch probe failed:\\n{cp.stderr}")
    import json

    return json.loads(cp.stdout.strip())


def run_oat_layer(args: argparse.Namespace, root: Path, output_root: Path) -> Path:
    oat_dir = output_root / "oat_rc_seed10"
    rel_oat = str(oat_dir.relative_to(root))
    cmd = [
        args.python_executable,
        "-u",
        "sensitivity_analysis_dspo_plus_spo_oat.py",
        "--python_executable",
        args.python_executable,
        "--instance",
        args.instance,
        "--seeds",
        *[str(s) for s in args.seeds],
        "--stage1_episodes",
        str(args.oat_stage1_episodes),
        "--stage2_episodes",
        str(args.oat_stage2_episodes),
        "--gpu",
        str(args.gpu),
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--max_retries",
        str(args.max_retries),
        "--output_dir",
        rel_oat,
        "--run_smoke_validation",
    ]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.allow_cpu:
        cmd.append("--allow_cpu")
    print("[INFO] Running OAT layer")
    print("[INFO] Command:", " ".join(cmd))
    cp = subprocess.run(cmd, cwd=root)
    if cp.returncode != 0:
        raise RuntimeError(f"OAT layer failed with return code {cp.returncode}")
    return oat_dir


def enhance_oat_reports(args: argparse.Namespace, oat_dir: Path) -> None:
    stage1 = read_csv(oat_dir / "stage1_raw.csv")
    stage2 = read_csv(oat_dir / "stage2_raw.csv")
    for rows in (stage1, stage2):
        for r in rows:
            log_path = Path(str(r.get("log_path", "")))
            m = parse_metrics_from_log(log_path)
            for k in METRICS_NUMERIC:
                r[k] = m.get(k)

    enhanced_fields: List[str] = []
    if stage1:
        enhanced_fields.extend(list(stage1[0].keys()))
    if stage2:
        enhanced_fields.extend(list(stage2[0].keys()))
    enhanced_fields.extend(METRICS_NUMERIC)
    enhanced_fields = list(dict.fromkeys(enhanced_fields))
    if stage1:
        write_csv(oat_dir / "stage1_raw_enhanced.csv", stage1, enhanced_fields)
    if stage2:
        write_csv(oat_dir / "stage2_raw_enhanced.csv", stage2, enhanced_fields)

    def _normalize(rows: List[Dict[str, str]], stage_name: str) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for r in rows:
            rr: Dict[str, object] = {
                "stage": stage_name,
                "factor": r.get("factor"),
                "value": as_float(r.get("value")),
                "seed": int(as_float(r.get("seed")) or 0),
            }
            for m in METRICS_NUMERIC:
                rr[m] = as_float(r.get(m))
            out.append(rr)
        return out

    s1_norm = _normalize(stage1, "stage1")
    s2_norm = _normalize(stage2, "stage2")
    s1_summary = summarize_by(s1_norm, ["factor", "value"])
    s2_summary = summarize_by(s2_norm, ["factor", "value"])
    if s1_summary:
        write_csv(oat_dir / "stage1_summary_enhanced.csv", s1_summary, s1_summary[0].keys())
    if s2_summary:
        write_csv(oat_dir / "stage2_summary_enhanced.csv", s2_summary, s2_summary[0].keys())

    by_factor: Dict[str, Dict[float, Dict[str, object]]] = {}
    for r in s1_summary:
        f = str(r["factor"])
        v = float(r["value"])
        by_factor.setdefault(f, {})[v] = r
    effects = []
    for f, table in by_factor.items():
        dv = float(DEFAULT_CONFIG[f])
        if dv not in table:
            continue
        default_np = as_float(table[dv].get("net_profit_mean"))
        for v, row in sorted(table.items(), key=lambda x: x[0]):
            np_mean = as_float(row.get("net_profit_mean"))
            if default_np is None or np_mean is None:
                continue
            delta = np_mean - default_np
            pct = (delta / default_np * 100.0) if default_np != 0 else float("nan")
            effects.append(
                {
                    "factor": f,
                    "value": v,
                    "default_value": dv,
                    "net_profit_mean": np_mean,
                    "default_net_profit_mean": default_np,
                    "delta_vs_default": delta,
                    "effect_size_pct_vs_default": pct,
                }
            )
    if effects:
        write_csv(oat_dir / "stage1_effect_size_vs_default.csv", effects, effects[0].keys())

    keys = [
        ("outside_option_util", -2.0, -1.0),
        ("incentive_sens", -0.35, -0.25),
        ("home_util", 1.0, 1.4),
        ("k", 10.0, 7.0),
    ]
    boot_rows = []
    for factor, a, b in keys:
        va = [
            as_float(r.get("net_profit"))
            for r in stage1
            if r.get("factor") == factor and as_float(r.get("value")) == a and as_float(r.get("net_profit")) is not None
        ]
        vb = [
            as_float(r.get("net_profit"))
            for r in stage1
            if r.get("factor") == factor and as_float(r.get("value")) == b and as_float(r.get("net_profit")) is not None
        ]
        va = [x for x in va if x is not None]
        vb = [x for x in vb if x is not None]
        mean_diff, ci_lo, ci_hi = bootstrap_diff_ci(va, vb, args.bootstrap_samples)
        boot_rows.append(
            {
                "factor": factor,
                "value_a": a,
                "value_b": b,
                "metric": "net_profit",
                "n_a": len(va),
                "n_b": len(vb),
                "bootstrap_samples": args.bootstrap_samples,
                "mean_diff_a_minus_b": mean_diff,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
            }
        )
    if boot_rows:
        write_csv(oat_dir / "stage1_bootstrap_key_diffs.csv", boot_rows, boot_rows[0].keys())

    baseline_dir = Path(args.baseline_oat_dir)
    if not baseline_dir.exists():
        legacy_baseline_dir = Path(LEGACY_BASELINE_OAT_DIR)
        if legacy_baseline_dir.exists():
            baseline_dir = legacy_baseline_dir
    baseline_stage1 = read_csv(baseline_dir / "stage1_summary.csv")
    current_stage1 = read_csv(oat_dir / "stage1_summary.csv")
    ci_rows = []
    for factor, dv in DEFAULT_CONFIG.items():
        base = [r for r in baseline_stage1 if r.get("factor") == factor and as_float(r.get("value")) == float(dv)]
        cur = [r for r in current_stage1 if r.get("factor") == factor and as_float(r.get("value")) == float(dv)]
        if not base or not cur:
            continue
        base = base[0]
        cur = cur[0]
        n0 = int(as_float(base.get("n_runs")) or 0)
        n1 = int(as_float(cur.get("n_runs")) or 0)
        s0 = as_float(base.get("net_profit_std")) or 0.0
        s1 = as_float(cur.get("net_profit_std")) or 0.0
        h0 = t_critical_95(max(1, n0 - 1)) * s0 / math.sqrt(n0) if n0 > 0 else float("nan")
        h1 = t_critical_95(max(1, n1 - 1)) * s1 / math.sqrt(n1) if n1 > 0 else float("nan")
        ratio = h1 / h0 if h0 and not math.isnan(h0) else float("nan")
        ci_rows.append(
            {
                "factor": factor,
                "baseline_n": n0,
                "current_n": n1,
                "baseline_ci_halfwidth": h0,
                "current_ci_halfwidth": h1,
                "ratio_current_over_baseline": ratio,
                "theory_ratio_sqrt_n": math.sqrt(n0 / n1) if n0 > 0 and n1 > 0 else float("nan"),
            }
        )
    if ci_rows:
        write_csv(oat_dir / "ci_shrink_check.csv", ci_rows, ci_rows[0].keys())


def factor_value_tag(cfg: Dict[str, float]) -> str:
    return "u_{u}_s_{s}_h_{h}_k_{k}".format(
        u=token(cfg["outside_option_util"]),
        s=token(cfg["incentive_sens"]),
        h=token(cfg["home_util"]),
        k=token(cfg["k"]),
    )


def run_interaction_layer(args: argparse.Namespace, root: Path, output_root: Path) -> Path:
    out_dir = output_root / "interaction_rc_seed10"
    raw_path = out_dir / "interaction_raw.csv"
    existing = read_csv(raw_path)
    done = {row_key(r) for r in existing}
    rows: List[Dict[str, object]] = [dict(r) for r in existing]

    combos = list(
        itertools.product(
            INTERACTION_LEVELS["outside_option_util"],
            INTERACTION_LEVELS["incentive_sens"],
            INTERACTION_LEVELS["home_util"],
        )
    )
    jobs = []
    for outside, sens, home in combos:
        cfg = dict(DEFAULT_CONFIG)
        cfg["outside_option_util"] = float(outside)
        cfg["incentive_sens"] = float(sens)
        cfg["home_util"] = float(home)
        cfg["k"] = 10.0
        tag = factor_value_tag(cfg)
        for seed in args.seeds:
            jobs.append((tag, seed, cfg))

    if args.max_jobs_per_layer > 0:
        jobs = jobs[: args.max_jobs_per_layer]

    print(f"[INFO] Interaction jobs: {len(jobs)}")
    for idx, (tag, seed, cfg) in enumerate(jobs, 1):
        key = ("interaction", tag, int(seed), 0, 1, int(args.interaction_episodes))
        if key in done:
            continue
        print(f"[Interaction {idx}/{len(jobs)}] {tag}, seed={seed}")
        row = run_single_job(
            args=args,
            root=root,
            stage="interaction",
            tag=tag,
            seed=int(seed),
            episodes=int(args.interaction_episodes),
            cfg=cfg,
            data_seed=0,
            data_seed_test=1,
        )
        rows.append(row)
        done.add(row_key(row))
        write_csv(raw_path, rows, rows[0].keys())

    norm = []
    for r in rows:
        rr = dict(r)
        for k in ["outside_option_util", "incentive_sens", "home_util", "k"]:
            rr[k] = as_float(rr.get(k))
        for m in METRICS_NUMERIC:
            rr[m] = as_float(rr.get(m))
        norm.append(rr)

    summary = summarize_by(norm, ["outside_option_util", "incentive_sens", "home_util", "k"])
    if summary:
        write_csv(out_dir / "interaction_summary.csv", summary, summary[0].keys())

    effects = []
    metric_list = ["net_profit", "total_costs", "quit_rate", "home_pickup_rate"]
    for metric in metric_list:
        vals = [r for r in norm if as_float(r.get(metric)) is not None]
        if not vals:
            continue
        for term in ["A", "B", "C", "AB", "AC", "BC"]:
            pos = []
            neg = []
            for r in vals:
                A = -1 if as_float(r["outside_option_util"]) == -2.0 else 1
                B = -1 if as_float(r["incentive_sens"]) == -0.35 else 1
                C = -1 if as_float(r["home_util"]) == 1.0 else 1
                sign = {
                    "A": A,
                    "B": B,
                    "C": C,
                    "AB": A * B,
                    "AC": A * C,
                    "BC": B * C,
                }[term]
                v = as_float(r.get(metric))
                if v is None:
                    continue
                if sign > 0:
                    pos.append(v)
                else:
                    neg.append(v)
            if not pos or not neg:
                continue
            eff = (sum(pos) / len(pos)) - (sum(neg) / len(neg))
            effects.append(
                {
                    "metric": metric,
                    "term": term,
                    "effect_plus_minus": eff,
                    "mean_plus": sum(pos) / len(pos),
                    "mean_minus": sum(neg) / len(neg),
                    "n_plus": len(pos),
                    "n_minus": len(neg),
                }
            )
    if effects:
        write_csv(out_dir / "interaction_effects.csv", effects, effects[0].keys())
    return out_dir


def load_stage1_summary(oat_dir: Path) -> List[Dict[str, str]]:
    rows = read_csv(oat_dir / "stage1_summary.csv")
    if not rows:
        rows = read_csv(oat_dir / "stage1_summary_enhanced.csv")
    return rows


def load_stage2_candidates(oat_dir: Path) -> List[Dict[str, str]]:
    return read_csv(oat_dir / "stage2_candidates.csv")


def choose_strategies_from_oat(oat_dir: Path, top_n: int) -> List[Dict[str, object]]:
    candidates = load_stage2_candidates(oat_dir)
    stage1 = load_stage1_summary(oat_dir)
    if not candidates or not stage1:
        raise RuntimeError("Need stage2_candidates.csv and stage1_summary.csv before robustness layer.")

    s1_map: Dict[Tuple[str, float], float] = {}
    for r in stage1:
        f = str(r.get("factor"))
        v = as_float(r.get("value"))
        np_mean = as_float(r.get("net_profit_mean"))
        if v is None or np_mean is None:
            continue
        s1_map[(f, v)] = np_mean

    strategies: List[Dict[str, object]] = [
        {"strategy": "default", "factor": "", "value": "", "delta_vs_default": 0.0, "config": dict(DEFAULT_CONFIG)}
    ]
    deltas = []
    for r in candidates:
        factor = str(r.get("factor"))
        cand_v = as_float(r.get("stage1_best_value"))
        if cand_v is None:
            continue
        default_v = float(DEFAULT_CONFIG[factor])
        if abs(cand_v - default_v) < 1e-12:
            continue
        np_def = s1_map.get((factor, default_v))
        np_cand = s1_map.get((factor, float(cand_v)))
        delta = (np_cand - np_def) if (np_cand is not None and np_def is not None) else float("nan")
        deltas.append((factor, float(cand_v), delta))

    deltas.sort(key=lambda x: (x[2] if not math.isnan(x[2]) else -1e18), reverse=True)
    for factor, val, delta in deltas[: max(1, top_n)]:
        cfg = dict(DEFAULT_CONFIG)
        cfg[factor] = val
        strategies.append(
            {
                "strategy": f"cand_{factor}_{token(val)}",
                "factor": factor,
                "value": val,
                "delta_vs_default": delta,
                "config": cfg,
            }
        )
    return strategies


def run_robustness_layer(args: argparse.Namespace, root: Path, output_root: Path, oat_dir: Path) -> Path:
    out_dir = output_root / "robustness_rc_seed10"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = choose_strategies_from_oat(oat_dir, args.top_candidates)
    write_csv(
        out_dir / "robustness_strategies.csv",
        [{k: v for k, v in s.items() if k != "config"} for s in strategies],
        ["strategy", "factor", "value", "delta_vs_default"],
    )

    raw_path = out_dir / "robustness_raw.csv"
    existing = read_csv(raw_path)
    done = {row_key(r) for r in existing}
    rows: List[Dict[str, object]] = [dict(r) for r in existing]

    jobs = []
    for strat in strategies:
        cfg = dict(strat["config"])
        s_name = str(strat["strategy"])
        for data_seed in args.data_seeds:
            data_seed_test = int(data_seed) if args.data_seed_test_mode == "sync" else int(args.fixed_data_seed_test)
            tag = f"{s_name}_d{data_seed}_t{data_seed_test}"
            for seed in args.seeds:
                jobs.append((tag, s_name, int(data_seed), int(data_seed_test), int(seed), cfg))

    if args.max_jobs_per_layer > 0:
        jobs = jobs[: args.max_jobs_per_layer]

    print(f"[INFO] Robustness jobs: {len(jobs)}")
    for idx, (tag, strategy_name, data_seed, data_seed_test, seed, cfg) in enumerate(jobs, 1):
        key = ("robustness", tag, seed, data_seed, data_seed_test, int(args.robustness_episodes))
        if key in done:
            continue
        print(f"[Robustness {idx}/{len(jobs)}] {strategy_name}, data_seed={data_seed}, seed={seed}")
        row = run_single_job(
            args=args,
            root=root,
            stage="robustness",
            tag=tag,
            seed=seed,
            episodes=int(args.robustness_episodes),
            cfg=cfg,
            data_seed=data_seed,
            data_seed_test=data_seed_test,
        )
        row["strategy"] = strategy_name
        rows.append(row)
        done.add(row_key(row))
        write_csv(raw_path, rows, rows[0].keys())

    norm = []
    for r in rows:
        rr = dict(r)
        rr["strategy"] = rr.get("strategy", "")
        rr["data_seed"] = int(as_float(rr.get("data_seed")) or 0)
        for m in METRICS_NUMERIC:
            rr[m] = as_float(rr.get(m))
        norm.append(rr)

    by_seed = summarize_by(norm, ["strategy", "data_seed"])
    by_strategy = summarize_by(norm, ["strategy"])
    if by_seed:
        write_csv(out_dir / "robustness_summary_by_data_seed.csv", by_seed, by_seed[0].keys())
    if by_strategy:
        write_csv(out_dir / "robustness_summary_overall.csv", by_strategy, by_strategy[0].keys())

    direction_rows = []
    default_map = {(r["strategy"], int(r["data_seed"])): r for r in by_seed if r.get("strategy") == "default"}
    for r in by_seed:
        strat = str(r.get("strategy"))
        if strat == "default":
            continue
        ds = int(as_float(r.get("data_seed")) or 0)
        d = default_map.get(("default", ds))
        np_s = as_float(r.get("net_profit_mean"))
        np_d = as_float(d.get("net_profit_mean")) if d else None
        sign = ""
        if np_s is not None and np_d is not None:
            diff = np_s - np_d
            sign = "pos" if diff > 0 else ("neg" if diff < 0 else "zero")
        direction_rows.append(
            {
                "strategy": strat,
                "data_seed": ds,
                "net_profit_mean": np_s,
                "default_net_profit_mean": np_d,
                "delta_vs_default": (np_s - np_d) if (np_s is not None and np_d is not None) else "",
                "direction": sign,
            }
        )
    if direction_rows:
        write_csv(out_dir / "robustness_direction_check.csv", direction_rows, direction_rows[0].keys())
    return out_dir


def split_csv_values(cell: str) -> List[float]:
    if cell is None:
        return []
    s = str(cell).strip().strip('"')
    if not s:
        return []
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def expected_stage2_from_candidates(cand_rows: List[Dict[str, str]], seeds: Sequence[int]) -> int:
    total = 0
    for r in cand_rows:
        vals = split_csv_values(str(r.get("stage2_values", "")))
        total += len(vals) * len(seeds)
    return total


def write_completeness_report(
    output_root: Path,
    args: argparse.Namespace,
    oat_dir: Path,
    interaction_dir: Optional[Path],
    robustness_dir: Optional[Path],
) -> None:
    lines = []
    lines.append("=== Completeness Check ===")

    oat_stage1 = read_csv(oat_dir / "stage1_raw.csv")
    oat_stage2 = read_csv(oat_dir / "stage2_raw.csv")
    cand_rows = read_csv(oat_dir / "stage2_candidates.csv")
    observed_oat_seeds = sorted({int(as_float(r.get("seed")) or 0) for r in oat_stage1})
    effective_oat_seeds = observed_oat_seeds if observed_oat_seeds else list(args.seeds)
    exp_s1 = sum(len(v) for v in OAT_FACTORS.values()) * len(effective_oat_seeds)
    exp_s2 = expected_stage2_from_candidates(cand_rows, effective_oat_seeds) if cand_rows else 0
    lines.append(f"oat_expected_stage1={exp_s1}")
    lines.append(f"oat_actual_stage1={len(oat_stage1)}")
    lines.append(f"oat_expected_stage2={exp_s2}")
    lines.append(f"oat_actual_stage2={len(oat_stage2)}")

    if interaction_dir is not None:
        iraw = read_csv(interaction_dir / "interaction_raw.csv")
        exp_i = 8 * len(args.seeds)
        lines.append(f"interaction_expected={exp_i}")
        lines.append(f"interaction_actual={len(iraw)}")

    if robustness_dir is not None:
        rraw = read_csv(robustness_dir / "robustness_raw.csv")
        strats = read_csv(robustness_dir / "robustness_strategies.csv")
        exp_r = len(strats) * len(args.data_seeds) * len(args.seeds)
        lines.append(f"robustness_expected={exp_r}")
        lines.append(f"robustness_actual={len(rraw)}")

    report_path = output_root / "completeness_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def resolve_dir(root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    output_root = resolve_dir(root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    runtime = probe_runtime(args.python_executable)
    print(
        f"[INFO] Runtime probe: python={args.python_executable}, "
        f"torch={runtime['torch_version']}, cuda_available={runtime['cuda_available']}, cuda_count={runtime['cuda_count']}"
    )
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable. Pass --allow_cpu to continue on CPU.")

    if args.oat_dir is not None:
        oat_dir = resolve_dir(root, args.oat_dir)
    else:
        oat_dir = output_root / "oat_rc_seed10"

    if args.force_oat or not (oat_dir / "stage1_raw.csv").exists() or not (oat_dir / "stage2_raw.csv").exists():
        oat_dir = run_oat_layer(args, root, output_root)
    else:
        print(f"[INFO] Reusing existing OAT results: {oat_dir}")

    enhance_oat_reports(args, oat_dir)

    interaction_dir: Optional[Path] = None
    if not args.no_interaction:
        interaction_dir = run_interaction_layer(args, root, output_root)

    robustness_dir: Optional[Path] = None
    if not args.no_robustness:
        robustness_dir = run_robustness_layer(args, root, output_root, oat_dir)

    write_completeness_report(output_root, args, oat_dir, interaction_dir, robustness_dir)
    print("[INFO] All requested layers finished.")
    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] OAT dir: {oat_dir}")
    if interaction_dir:
        print(f"[INFO] Interaction dir: {interaction_dir}")
    if robustness_dir:
        print(f"[INFO] Robustness dir: {robustness_dir}")
    print(f"[INFO] Completeness report: {output_root / 'completeness_report.txt'}")


if __name__ == "__main__":
    main()
