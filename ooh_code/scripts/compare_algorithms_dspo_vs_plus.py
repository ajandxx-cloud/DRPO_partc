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

INT_PARAMS = {"k", "batch_size", "spo_warmup_episodes", "spo_rampup_episodes", "spo_buffer_size", "spo_batch_size"}
METRICS = ["net_profit", "total_costs", "quit_rate", "served_rate", "served_demand", "total_demand"]
RECOMMENDED_SEEDS = [40, 67, 97, 52, 29, 20, 17, 88, 63, 79, 60, 62, 7, 48, 56, 15, 66, 53, 90, 70, 24, 74, 80, 28, 2, 95, 92, 26, 39, 82]
DEFAULT_RUN_META = "Experiments/analysis/drpo_sensitivity_oat_rc_full12_resume_full_20260313_222441/run_meta.json"
LEGACY_RUN_META = "Experiments/analysis/dspo_plus_spo_sensitivity_oat_rc_full12_resume_full_20260313_222441/run_meta.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired algorithm comparison: DSPO vs DRPO on RC full12.")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus"])
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--seeds", nargs="+", type=int, default=RECOMMENDED_SEEDS)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--save_count", type=int, default=1)
    p.add_argument("--folder_suffix", default="_cmp")

    p.add_argument("--algo_a", default="DSPO")
    p.add_argument("--algo_b", default="DRPO")
    p.add_argument("--label_a", default="DSPO")
    p.add_argument("--label_b", default="DRPO")

    p.add_argument("--run_prefix", default="RC_FULL12_DSPO_VS_DRPO")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_existing_output_dir", action="store_true")

    p.add_argument("--run_meta_json", default=DEFAULT_RUN_META)
    p.add_argument("--config_json", default=None, help="Optional JSON path (dict or {'config': dict}) overriding run_meta default_config.")

    p.add_argument("--persist_every_n", type=int, default=2)
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
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


def run_log_path(root: Path, algo_name: str, run_id: str, suffix: str, seed: int) -> Path:
    return root / "Experiments" / "Parcelpoint_py" / "pricing" / algo_name / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log"


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def build_cmd(
    args: argparse.Namespace,
    algo_name: str,
    run_id: str,
    seed: int,
    cfg: Dict[str, float],
) -> List[str]:
    cmd = [
        args.python_executable,
        "run.py",
        "--algo_name",
        algo_name,
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


def to_numeric_dict(payload: Dict[str, Any], source_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in payload.items():
        fv = to_float(v)
        if fv is None:
            continue
        out[k] = int(round(fv)) if k in INT_PARAMS else float(fv)
    if not out:
        raise RuntimeError(f"No numeric config found in {source_path}")
    return out


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


def load_config(root: Path, args: argparse.Namespace) -> Dict[str, float]:
    if args.config_json:
        cfg_path = resolve_path(root, args.config_json)
        payload = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        raw = payload["config"] if isinstance(payload, dict) and isinstance(payload.get("config"), dict) else payload
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid config JSON format: {cfg_path}")
        return to_numeric_dict(raw, cfg_path)

    run_meta_path = resolve_existing_path(root, args.run_meta_json, LEGACY_RUN_META)
    payload = json.loads(run_meta_path.read_text(encoding="utf-8-sig"))
    raw = payload.get("default_config")
    if not isinstance(raw, dict):
        raise RuntimeError(f"default_config not found in run_meta: {run_meta_path}")
    return to_numeric_dict(raw, run_meta_path)


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
    preferred = [
        "label",
        "algo_name",
        "run_id",
        "seed",
        "episodes",
        "data_seed",
        "data_seed_test",
        "status",
        "runtime_sec",
        "log_path",
        "command",
    ]
    seen = set(preferred)
    extra: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                extra.append(k)
    return preferred + sorted(extra)


def row_key(row: Dict[str, Any]) -> Tuple[str, int, int, int, int]:
    return (
        str(row["label"]),
        int(float(row["seed"])),
        int(float(row["episodes"])),
        int(float(row["data_seed"])),
        int(float(row["data_seed_test"])),
    )


def run_single(
    args: argparse.Namespace,
    root: Path,
    label: str,
    algo_name: str,
    run_id: str,
    seed: int,
    cfg: Dict[str, float],
) -> Dict[str, Any]:
    log = run_log_path(root, algo_name, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args, algo_name, run_id, seed, cfg)

    if args.skip_existing:
        m = parse_metrics(log)
        if m is not None:
            if args.allow_cpu or has_gpu_marker(log):
                row: Dict[str, Any] = {
                    "label": label,
                    "algo_name": algo_name,
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
        elif (not args.allow_cpu) and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing {att}/{attempts} for {label} seed={seed}. log={log}"
        else:
            status = "completed" if att == 1 else f"completed_retry_{att}"
            row = {
                "label": label,
                "algo_name": algo_name,
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
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(str(r["label"]), []).append(r)

    out: List[Dict[str, Any]] = []
    for label in sorted(by_label.keys()):
        rs = by_label[label]
        row: Dict[str, Any] = {"label": label, "n_runs": len(rs)}
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


def paired_deltas(rows: List[Dict[str, Any]], label_a: str, label_b: str) -> List[Dict[str, Any]]:
    keyed: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        keyed[(str(r["label"]), int(float(r["seed"])))] = r

    seeds = sorted({int(float(r["seed"])) for r in rows})
    out: List[Dict[str, Any]] = []
    for seed in seeds:
        a = keyed.get((label_a, seed))
        b = keyed.get((label_b, seed))
        if a is None or b is None:
            continue
        row: Dict[str, Any] = {"seed": seed}
        for metric in METRICS:
            av = to_float(a.get(metric))
            bv = to_float(b.get(metric))
            row[f"{label_a}_{metric}"] = av
            row[f"{label_b}_{metric}"] = bv
            row[f"{label_b}_minus_{label_a}_{metric}"] = (bv - av) if (av is not None and bv is not None) else ""
        out.append(row)
    return out


def write_report(output_dir: Path, summary_rows: List[Dict[str, Any]], paired_rows: List[Dict[str, Any]], label_a: str, label_b: str) -> None:
    summary_map = {str(r["label"]): r for r in summary_rows}
    a = summary_map.get(label_a, {})
    b = summary_map.get(label_b, {})

    np_a = to_float(a.get("net_profit_mean"))
    np_b = to_float(b.get("net_profit_mean"))
    qr_a = to_float(a.get("quit_rate_mean"))
    qr_b = to_float(b.get("quit_rate_mean"))
    sr_a = to_float(a.get("served_rate_mean"))
    sr_b = to_float(b.get("served_rate_mean"))

    delta_np = (np_b - np_a) if (np_a is not None and np_b is not None) else None
    delta_qr = (qr_b - qr_a) if (qr_a is not None and qr_b is not None) else None
    delta_sr = (sr_b - sr_a) if (sr_a is not None and sr_b is not None) else None

    paired_np = [to_float(r.get(f"{label_b}_minus_{label_a}_net_profit")) for r in paired_rows]
    paired_np = [v for v in paired_np if v is not None]
    paired_np_mean = (sum(paired_np) / len(paired_np)) if paired_np else None

    lines: List[str] = []
    lines.append("=== Algorithm Comparison Report ===")
    lines.append(f"generated_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"label_a={label_a}")
    lines.append(f"label_b={label_b}")
    lines.append("")
    lines.append("[Overall Means]")
    lines.append(f"{label_a}_net_profit_mean={np_a}")
    lines.append(f"{label_b}_net_profit_mean={np_b}")
    lines.append(f"{label_b}_minus_{label_a}_net_profit_mean={delta_np}")
    lines.append(f"{label_b}_minus_{label_a}_quit_rate_pp={delta_qr}")
    lines.append(f"{label_b}_minus_{label_a}_served_rate={delta_sr}")
    lines.append(f"paired_seed_net_profit_mean_diff={paired_np_mean}")
    (output_dir / "compare_report.txt").write_text("\n".join(lines), encoding="utf-8")


def persist(output_dir: Path, rows: List[Dict[str, Any]], label_a: str, label_b: str) -> None:
    if not rows:
        return
    raw_fields = collect_fields(rows)
    write_csv(output_dir / "compare_raw.csv", rows, raw_fields)

    summary_rows = summarize(rows)
    if summary_rows:
        write_csv(output_dir / "compare_summary.csv", summary_rows, summary_rows[0].keys())

    paired_rows = paired_deltas(rows, label_a=label_a, label_b=label_b)
    if paired_rows:
        write_csv(output_dir / "compare_paired_seed_delta.csv", paired_rows, paired_rows[0].keys())

    write_report(output_dir, summary_rows=summary_rows, paired_rows=paired_rows, label_a=label_a, label_b=label_b)


def main() -> None:
    args = parse_args()
    if args.persist_every_n <= 0:
        raise RuntimeError("--persist_every_n must be >= 1")
    if len(args.seeds) == 0:
        raise RuntimeError("At least one seed is required.")

    root = Path(__file__).resolve().parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"Experiments/analysis/rc_full12_algo_compare_{ts}"
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_items = list(output_dir.iterdir())
    if existing_items and (not args.allow_existing_output_dir):
        raise RuntimeError(f"Output directory already contains files: {output_dir}. Re-run with --allow_existing_output_dir.")

    runtime = probe_runtime(args.python_executable)
    print(
        "[INFO] Runtime probe: "
        + f"python={args.python_executable}, torch={runtime['torch_version']}, "
        + f"cuda_available={runtime['cuda_available']}, cuda_count={runtime['cuda_count']}",
        flush=True,
    )
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable in selected runtime. Pass --allow_cpu to continue on CPU.")

    cfg = load_config(root=root, args=args)
    run_id_a = f"{args.run_prefix}_{args.label_a}"
    run_id_b = f"{args.run_prefix}_{args.label_b}"

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
        "algo_a": args.algo_a,
        "algo_b": args.algo_b,
        "label_a": args.label_a,
        "label_b": args.label_b,
        "config_source": args.config_json if args.config_json else args.run_meta_json,
        "config": cfg,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("[DRY-RUN] Showing first 2 commands per algorithm:", flush=True)
        for seed in args.seeds[:2]:
            cmd_a = build_cmd(args, algo_name=args.algo_a, run_id=run_id_a, seed=seed, cfg=cfg)
            cmd_b = build_cmd(args, algo_name=args.algo_b, run_id=run_id_b, seed=seed, cfg=cfg)
            print("[A] " + " ".join(cmd_a), flush=True)
            print("[B] " + " ".join(cmd_b), flush=True)
        print(f"[DRY-RUN] Output dir: {output_dir}", flush=True)
        return

    raw_path = output_dir / "compare_raw.csv"
    existing_rows = read_csv(raw_path)
    rows: List[Dict[str, Any]] = [dict(r) for r in existing_rows]
    done = {row_key(r) for r in rows}

    jobs: List[Tuple[str, str, str, int]] = []
    for seed in args.seeds:
        jobs.append((args.label_a, args.algo_a, run_id_a, int(seed)))
        jobs.append((args.label_b, args.algo_b, run_id_b, int(seed)))

    remaining = sum(
        1
        for label, _, _, seed in jobs
        if (label, seed, args.episodes, args.data_seed, args.data_seed_test) not in done
    )
    print(f"[INFO] Total jobs={len(jobs)}, existing_rows={len(rows)}, remaining={remaining}", flush=True)

    new_count = 0
    for idx, (label, algo_name, run_id, seed) in enumerate(jobs, 1):
        key = (label, seed, args.episodes, args.data_seed, args.data_seed_test)
        if key in done:
            continue
        print(f"[RUN {idx}/{len(jobs)}] label={label}, algo={algo_name}, seed={seed}", flush=True)
        row = run_single(
            args=args,
            root=root,
            label=label,
            algo_name=algo_name,
            run_id=run_id,
            seed=seed,
            cfg=cfg,
        )
        rows.append(row)
        done.add(row_key(row))
        new_count += 1
        if new_count % args.persist_every_n == 0:
            persist(output_dir=output_dir, rows=rows, label_a=args.label_a, label_b=args.label_b)
            print(f"[INFO] Persisted after {new_count} new runs.", flush=True)

    persist(output_dir=output_dir, rows=rows, label_a=args.label_a, label_b=args.label_b)
    print(f"[DONE] Compare finished. Output dir: {output_dir}", flush=True)
    print(f"[DONE] Summary: {output_dir / 'compare_summary.csv'}", flush=True)
    print(f"[DONE] Paired deltas: {output_dir / 'compare_paired_seed_delta.csv'}", flush=True)
    print(f"[DONE] Report: {output_dir / 'compare_report.txt'}", flush=True)


if __name__ == "__main__":
    main()
