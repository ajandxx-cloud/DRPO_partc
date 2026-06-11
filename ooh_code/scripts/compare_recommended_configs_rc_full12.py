#!/usr/bin/env python
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


METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "served_demand": re.compile(r"Accepted customers:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_demand": re.compile(r"Total customers:\s*([+-]?\d+(?:\.\d+)?)"),
}

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
STAGE2_DEFAULT_SEEDS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63]
METRICS = ["net_profit", "total_costs", "quit_rate", "served_rate", "served_demand", "total_demand"]
DEFAULT_BASELINE_STAGE2_RAW = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_resume_full_20260313_222441/stage2_raw.csv"
DEFAULT_BASELINE_RUN_META = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_resume_full_20260313_222441/run_meta.json"
LEGACY_BASELINE_STAGE2_RAW = "Experiments/analysis/dspo_plus_spo_sensitivity_oat_rc_full12_resume_full_20260313_222441/stage2_raw.csv"
LEGACY_BASELINE_RUN_META = "Experiments/analysis/dspo_plus_spo_sensitivity_oat_rc_full12_resume_full_20260313_222441/run_meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paired RC full12 reproduction for conservative/aggressive recommendation configs.")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus"])
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--seeds", nargs="+", type=int, default=STAGE2_DEFAULT_SEEDS)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--save_count", type=int, default=1)
    p.add_argument("--folder_suffix", default="_sens")

    p.add_argument("--run_prefix", default="RC_FULL12_REC_COMPARE")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_existing_output_dir", action="store_true")

    p.add_argument("--conservative_json", required=True)
    p.add_argument("--aggressive_json", required=True)

    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--persist_every_n", type=int, default=1)
    p.add_argument("--run_timeout_sec", type=int, default=3600)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--allow_cpu", action="store_true")

    p.add_argument(
        "--baseline_stage2_raw",
        default=DEFAULT_BASELINE_STAGE2_RAW,
    )
    p.add_argument(
        "--baseline_run_meta",
        default=DEFAULT_BASELINE_RUN_META,
    )
    return p.parse_args()


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


def probe_runtime(pyexe: str) -> Dict[str, Any]:
    code = (
        "import json,torch;print(json.dumps({'torch_version':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
    )
    cp = subprocess.run([pyexe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if cp.returncode != 0:
        raise RuntimeError("Torch probe failed:\n" + cp.stderr)
    return json.loads(cp.stdout.strip())


def run_log_path(root: Path, run_id: str, suffix: str, seed: int) -> Path:
    return root / "Experiments" / "Parcelpoint_py" / "pricing" / "DRPO" / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log"


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def build_cmd(args: argparse.Namespace, run_id: str, seed: int, cfg: Dict[str, float]) -> List[str]:
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
        str(args.episodes),
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


def load_config(path: Path) -> Dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    data = payload["config"] if isinstance(payload, dict) and isinstance(payload.get("config"), dict) else payload
    cfg: Dict[str, float] = {}
    for k, v in data.items():
        fv = to_float(v)
        if fv is None:
            raise RuntimeError(f"Config value for {k} is not numeric in {path}")
        cfg[k] = int(round(fv)) if k in INT_PARAMS else float(fv)
    return cfg


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def resolve_existing_path(root: Path, p: str, legacy_p: str) -> Path:
    candidate = resolve_path(root, p)
    if candidate.exists():
        return candidate
    legacy = resolve_path(root, legacy_p)
    if legacy.exists():
        return legacy
    return candidate


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


def row_key(row: Dict[str, Any]) -> Tuple[str, int, int, int, int]:
    return (
        str(row["strategy"]),
        int(float(row["seed"])),
        int(float(row["episodes"])),
        int(float(row["data_seed"])),
        int(float(row["data_seed_test"])),
    )


def run_single(
    args: argparse.Namespace,
    root: Path,
    strategy: str,
    run_id: str,
    seed: int,
    cfg: Dict[str, float],
) -> Dict[str, Any]:
    log = run_log_path(root, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args, run_id, seed, cfg)

    if args.skip_existing:
        m = parse_metrics(log)
        if m is not None:
            if args.allow_cpu or has_gpu_marker(log):
                row: Dict[str, Any] = {
                    "strategy": strategy,
                    "run_id": run_id,
                    "seed": seed,
                    "episodes": args.episodes,
                    "data_seed": args.data_seed,
                    "data_seed_test": args.data_seed_test,
                    "status": "cached",
                    "runtime_sec": 0.0,
                    "log_path": str(log),
                    "command": " ".join(cmd),
                }
                row.update({k: cfg[k] for k in sorted(cfg.keys())})
                row.update(m)
                return row

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
            tail = (e.stdout or "")[-1200:] if e.stdout else ""
            last_error = f"Timeout {att}/{attempts} for {strategy} seed={seed}. tail={tail}"
            if att < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        rt = time.time() - t0
        m = parse_metrics(log)
        if cp.returncode != 0:
            last_error = f"Return code {cp.returncode} {att}/{attempts} for {strategy} seed={seed}. tail={(cp.stdout or '')[-1500:]}"
        elif m is None:
            last_error = f"Metrics missing {att}/{attempts} for {strategy} seed={seed}. log={log}"
        elif (not args.allow_cpu) and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing {att}/{attempts} for {strategy} seed={seed}. log={log}"
        else:
            status = "completed" if att == 1 else f"completed_retry_{att}"
            row = {
                "strategy": strategy,
                "run_id": run_id,
                "seed": seed,
                "episodes": args.episodes,
                "data_seed": args.data_seed,
                "data_seed_test": args.data_seed_test,
                "status": status,
                "runtime_sec": rt,
                "log_path": str(log),
                "command": " ".join(cmd),
            }
            row.update({k: cfg[k] for k in sorted(cfg.keys())})
            row.update(m)
            return row
        if att < attempts:
            time.sleep(args.retry_backoff_sec)
    raise RuntimeError(last_error)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_strategy.setdefault(str(r["strategy"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for strategy in sorted(by_strategy.keys()):
        rs = by_strategy[strategy]
        row: Dict[str, Any] = {"strategy": strategy, "n_runs": len(rs)}
        for m in METRICS + ["runtime_sec"]:
            vals = [to_float(r.get(m)) for r in rs]
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


def paired_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    m: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        m[(str(r["strategy"]), int(float(r["seed"])))] = r

    seeds = sorted({int(float(r["seed"])) for r in rows})
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        c = m.get(("conservative", seed))
        a = m.get(("aggressive", seed))
        if c is None or a is None:
            continue
        row: Dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            cv = to_float(c.get(metric))
            av = to_float(a.get(metric))
            row[f"conservative_{metric}"] = cv
            row[f"aggressive_{metric}"] = av
            row[f"aggressive_minus_conservative_{metric}"] = (av - cv) if (cv is not None and av is not None) else ""
        out.append(row)
    return out


def parse_default_config(run_meta_path: Path) -> Dict[str, float]:
    if not run_meta_path.exists():
        return {}
    payload = json.loads(run_meta_path.read_text(encoding="utf-8-sig"))
    raw = payload.get("default_config", {})
    out: Dict[str, float] = {}
    for k, v in raw.items():
        fv = to_float(v)
        if fv is None:
            continue
        out[k] = float(fv)
    return out


def baseline_by_seed(stage2_raw_path: Path, default_cfg: Dict[str, float]) -> Dict[int, Dict[str, float]]:
    rows = read_csv(stage2_raw_path)
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for r in rows:
        factor = str(r.get("factor", ""))
        fv = to_float(r.get("value"))
        dv = default_cfg.get(factor)
        if fv is None or dv is None or abs(fv - dv) > 1e-12:
            continue
        seed = int(to_float(r.get("seed")) or 0)
        g = grouped.setdefault(seed, {m: [] for m in METRICS})
        for m in METRICS:
            mv = to_float(r.get(m))
            if mv is not None:
                g[m].append(mv)

    out: Dict[int, Dict[str, float]] = {}
    for seed, agg in grouped.items():
        out[seed] = {}
        for m in METRICS:
            vals = agg[m]
            if vals:
                out[seed][m] = sum(vals) / len(vals)
    return out


def compare_with_baseline(rows: List[Dict[str, Any]], baseline_seed_map: Dict[int, Dict[str, float]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    detail: List[Dict[str, Any]] = []
    for r in rows:
        strategy = str(r["strategy"])
        seed = int(float(r["seed"]))
        base = baseline_seed_map.get(seed, {})
        row: Dict[str, Any] = {"strategy": strategy, "seed": seed}
        for m in METRICS:
            sv = to_float(r.get(m))
            bv = base.get(m)
            row[f"{m}"] = sv
            row[f"baseline_{m}"] = bv if bv is not None else ""
            row[f"delta_vs_baseline_{m}"] = (sv - bv) if (sv is not None and bv is not None) else ""
        detail.append(row)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in detail:
        grouped.setdefault(str(r["strategy"]), []).append(r)
    summary_rows: List[Dict[str, Any]] = []
    for strategy in sorted(grouped.keys()):
        rs = grouped[strategy]
        row: Dict[str, Any] = {"strategy": strategy, "n_runs": len(rs)}
        for m in METRICS:
            vals = [to_float(x.get(f"delta_vs_baseline_{m}")) for x in rs]
            vals = [v for v in vals if v is not None]
            row[f"delta_vs_baseline_{m}_mean"] = (sum(vals) / len(vals)) if vals else ""
        summary_rows.append(row)
    return detail, summary_rows


def write_report(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    paired_rows: List[Dict[str, Any]],
    delta_baseline_summary: List[Dict[str, Any]],
) -> None:
    summary_map = {str(r["strategy"]): r for r in summary_rows}
    cons = summary_map.get("conservative", {})
    aggr = summary_map.get("aggressive", {})

    np_c = to_float(cons.get("net_profit_mean"))
    np_a = to_float(aggr.get("net_profit_mean"))
    qr_c = to_float(cons.get("quit_rate_mean"))
    qr_a = to_float(aggr.get("quit_rate_mean"))
    sr_c = to_float(cons.get("served_rate_mean"))
    sr_a = to_float(aggr.get("served_rate_mean"))

    delta_np = (np_a - np_c) if (np_a is not None and np_c is not None) else None
    delta_qr = (qr_a - qr_c) if (qr_a is not None and qr_c is not None) else None
    delta_sr = (sr_a - sr_c) if (sr_a is not None and sr_c is not None) else None

    paired_np = [to_float(r.get("aggressive_minus_conservative_net_profit")) for r in paired_rows]
    paired_np = [v for v in paired_np if v is not None]
    paired_np_mean = (sum(paired_np) / len(paired_np)) if paired_np else None

    guardrail = "unknown"
    if delta_qr is not None and delta_sr is not None:
        guardrail = "pass" if (delta_qr <= 2.0 and delta_sr >= -0.02) else "fail"

    lines = []
    lines.append("=== RC Full12 Recommended Config Compare ===")
    lines.append(f"generated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("[Overall Means]")
    lines.append(f"conservative_net_profit_mean={np_c}")
    lines.append(f"aggressive_net_profit_mean={np_a}")
    lines.append(f"aggressive_minus_conservative_net_profit_mean={delta_np}")
    lines.append(f"paired_seed_net_profit_mean_diff={paired_np_mean}")
    lines.append(f"aggressive_minus_conservative_quit_rate_pp={delta_qr}")
    lines.append(f"aggressive_minus_conservative_served_rate={delta_sr}")
    lines.append(f"guardrail_check_quit<=2pp_and_served>=-0.02 => {guardrail}")
    lines.append("")
    lines.append("[Vs Baseline Means]")
    for r in delta_baseline_summary:
        s = str(r["strategy"])
        lines.append(f"{s}_delta_vs_baseline_net_profit_mean={r.get('delta_vs_baseline_net_profit_mean')}")
        lines.append(f"{s}_delta_vs_baseline_quit_rate_mean={r.get('delta_vs_baseline_quit_rate_mean')}")
        lines.append(f"{s}_delta_vs_baseline_served_rate_mean={r.get('delta_vs_baseline_served_rate_mean')}")
    (output_dir / "compare_report.txt").write_text("\n".join(lines), encoding="utf-8")


def persist(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    baseline_seed_map: Dict[int, Dict[str, float]],
) -> None:
    if not rows:
        return
    raw_fields = list(rows[0].keys())
    write_csv(output_dir / "compare_raw.csv", rows, raw_fields)

    summary_rows = summarize(rows)
    if summary_rows:
        write_csv(output_dir / "compare_summary.csv", summary_rows, summary_rows[0].keys())

    paired_rows = paired_deltas(rows)
    if paired_rows:
        write_csv(output_dir / "compare_paired_seed_delta.csv", paired_rows, paired_rows[0].keys())

    delta_detail, delta_summary = compare_with_baseline(rows, baseline_seed_map)
    if delta_detail:
        write_csv(output_dir / "compare_vs_baseline_by_seed.csv", delta_detail, delta_detail[0].keys())
    if delta_summary:
        write_csv(output_dir / "compare_vs_baseline_summary.csv", delta_summary, delta_summary[0].keys())

    write_report(output_dir, summary_rows, paired_rows, delta_summary)


def main() -> None:
    args = parse_args()
    if args.persist_every_n <= 0:
        raise RuntimeError("--persist_every_n must be >= 1")

    root = Path(__file__).resolve().parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"Experiments/analysis/rc_full12_recommended_compare_{ts}"
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_items = list(output_dir.iterdir())
    if existing_items and (not args.allow_existing_output_dir):
        raise RuntimeError(f"Output directory already contains files: {output_dir}. Re-run with --allow_existing_output_dir.")

    runtime = probe_runtime(args.python_executable)
    print(
        "[INFO] Runtime probe: "
        + f"python={args.python_executable}, torch={runtime['torch_version']}, "
        + f"cuda_available={runtime['cuda_available']}, cuda_count={runtime['cuda_count']}"
    )
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable in selected runtime. Pass --allow_cpu to continue on CPU.")

    conservative_path = resolve_path(root, args.conservative_json)
    aggressive_path = resolve_path(root, args.aggressive_json)
    cfg_conservative = load_config(conservative_path)
    cfg_aggressive = load_config(aggressive_path)

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_executable": args.python_executable,
        "gpu": args.gpu,
        "instance": args.instance,
        "data_seed": args.data_seed,
        "data_seed_test": args.data_seed_test,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "save_count": args.save_count,
        "folder_suffix": args.folder_suffix,
        "run_prefix": args.run_prefix,
        "conservative_json": str(conservative_path),
        "aggressive_json": str(aggressive_path),
        "conservative_config": cfg_conservative,
        "aggressive_config": cfg_aggressive,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    baseline_raw_path = resolve_existing_path(root, args.baseline_stage2_raw, LEGACY_BASELINE_STAGE2_RAW)
    baseline_run_meta_path = resolve_existing_path(root, args.baseline_run_meta, LEGACY_BASELINE_RUN_META)
    baseline_default_cfg = parse_default_config(baseline_run_meta_path)
    baseline_seed_map = baseline_by_seed(baseline_raw_path, baseline_default_cfg) if baseline_default_cfg else {}

    raw_path = output_dir / "compare_raw.csv"
    existing_rows = read_csv(raw_path)
    rows: List[Dict[str, Any]] = [dict(r) for r in existing_rows]
    done = {row_key(r) for r in rows}

    jobs: List[Tuple[str, str, Dict[str, float], int]] = []
    for strategy, cfg in [("conservative", cfg_conservative), ("aggressive", cfg_aggressive)]:
        run_id = f"{args.run_prefix}_{strategy}"
        for seed in args.seeds:
            jobs.append((strategy, run_id, cfg, int(seed)))

    print(f"[INFO] Total jobs={len(jobs)}, existing_rows={len(rows)}, remaining={sum(1 for s, _, _, sd in jobs if (s, sd, args.episodes, args.data_seed, args.data_seed_test) not in done)}")

    new_count = 0
    for idx, (strategy, run_id, cfg, seed) in enumerate(jobs, 1):
        key = (strategy, seed, args.episodes, args.data_seed, args.data_seed_test)
        if key in done:
            continue
        print(f"[RUN {idx}/{len(jobs)}] strategy={strategy}, seed={seed}")
        row = run_single(args=args, root=root, strategy=strategy, run_id=run_id, seed=seed, cfg=cfg)
        rows.append(row)
        done.add(row_key(row))
        new_count += 1
        if new_count % args.persist_every_n == 0:
            persist(output_dir, rows, baseline_seed_map)
            print(f"[INFO] Persisted after {new_count} new runs.")

    persist(output_dir, rows, baseline_seed_map)
    print(f"[DONE] Compare finished. Output dir: {output_dir}")
    print(f"[DONE] Summary: {output_dir / 'compare_summary.csv'}")
    print(f"[DONE] Paired deltas: {output_dir / 'compare_paired_seed_delta.csv'}")
    print(f"[DONE] Report: {output_dir / 'compare_report.txt'}")


if __name__ == "__main__":
    main()
