#!/usr/bin/env python3
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "home_delivery": re.compile(r"percentage home delivery:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "served_demand": re.compile(r"Accepted customers:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_demand": re.compile(r"Total customers:\s*([+-]?\d+(?:\.\d+)?)"),
}

INT_PARAMS = {"k", "batch_size", "spo_warmup_episodes", "spo_rampup_episodes"}
METRICS = ["net_profit", "total_costs", "quit_rate", "home_delivery", "served_rate", "served_demand", "total_demand"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-seed pilot for DSPO, DRPO, and static pricing.")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--instance", default="RC", choices=["RC", "C", "R", "Beijing_bus", "Beijing_Yanjiao"])
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 67, 97])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--save_count", type=int, default=1)
    p.add_argument("--folder_suffix", default="_main3")
    p.add_argument("--run_prefix", default="MAIN3_CALIB")
    p.add_argument("--config_json", default="configs/rc_main_pilot_3seed.json")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--allow_existing_output_dir", action="store_true")
    p.add_argument("--persist_every_n", type=int, default=1)
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)

    p.add_argument("--static_price_home", type=float, default=None)
    p.add_argument("--static_price_pp", type=float, default=None)
    return p.parse_args()


def resolve_path(root: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def numeric_config(payload: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in payload.items():
        fv = to_float(v)
        if fv is None:
            continue
        out[k] = int(round(fv)) if k in INT_PARAMS else float(fv)
    return out


def load_payload(root: Path, args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float], Path]:
    cfg_path = resolve_path(root, args.config_json)
    payload = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    raw_config = payload["config"] if isinstance(payload.get("config"), dict) else payload
    cfg = numeric_config(raw_config)
    if not cfg:
        raise RuntimeError(f"No numeric config found in {cfg_path}")

    raw_static = payload.get("static_pricing", {}) if isinstance(payload, dict) else {}
    static_cfg = numeric_config(raw_static)
    if args.static_price_home is not None:
        static_cfg["price_home"] = float(args.static_price_home)
    if args.static_price_pp is not None:
        static_cfg["price_pp"] = float(args.static_price_pp)
    static_cfg.setdefault("price_home", 2.0)
    static_cfg.setdefault("price_pp", -5.0)
    return cfg, static_cfg, cfg_path


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def probe_runtime(pyexe: str) -> Dict[str, Any]:
    code = (
        "import json,torch,sys;print(json.dumps({'exe':sys.executable,'torch_version':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),'cuda_count':int(torch.cuda.device_count())}))"
    )
    cp = subprocess.run([pyexe, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="ignore")
    if cp.returncode != 0:
        raise RuntimeError("Torch probe failed:\n" + cp.stderr)
    return json.loads(cp.stdout.strip())


def log_path(root: Path, algo_name: str, run_id: str, suffix: str, seed: int) -> Path:
    return root / "Experiments" / "Parcelpoint_py" / "pricing" / algo_name / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log"


def extract_last(pattern: re.Pattern, text: str) -> Optional[float]:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def parse_metrics(log: Path) -> Optional[Dict[str, Optional[float]]]:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="ignore")
    out = {k: extract_last(p, txt) for k, p in METRIC_REGEX.items()}
    if out["net_profit"] is None or out["total_costs"] is None or out["quit_rate"] is None:
        return None
    served = out.get("served_demand")
    total = out.get("total_demand")
    out["served_rate"] = served / total if served is not None and total else None
    return out


def has_gpu_marker(log: Path) -> bool:
    return log.exists() and ("Using GPU device: cuda" in log.read_text(encoding="utf-8", errors="ignore"))


def algo_config(strategy: str, cfg: Dict[str, float]) -> Dict[str, float]:
    out = dict(cfg)
    if strategy == "DSPO":
        out["spo_loss_weight"] = 0.0
    return out


def build_cmd(args: argparse.Namespace, strategy: str, algo_name: str, run_id: str, seed: int, cfg: Dict[str, float], static_cfg: Dict[str, float]) -> List[str]:
    run_cfg = algo_config(strategy, cfg)
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
    for k in sorted(run_cfg.keys()):
        cmd.extend([f"--{k}", cli_value(k, run_cfg[k])])
    if strategy == "Static-pricing":
        cmd.extend(["--price_home", cli_value("price_home", static_cfg["price_home"])])
        cmd.extend(["--price_pp", cli_value("price_pp", static_cfg["price_pp"])])
    cmd.extend(["--experiment", run_id, "--folder_suffix", args.folder_suffix])
    return cmd


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def collect_fields(rows: Iterable[Dict[str, Any]]) -> List[str]:
    preferred = ["strategy", "algo_name", "run_id", "seed", "episodes", "status", "runtime_sec", "log_path", "command"]
    seen = set(preferred)
    extra: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(key)
    return preferred + sorted(extra)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["strategy"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for strategy in sorted(grouped):
        rs = grouped[strategy]
        summary: Dict[str, Any] = {"strategy": strategy, "n_runs": len(rs)}
        for metric in METRICS:
            vals = [to_float(r.get(metric)) for r in rs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            n = len(vals)
            mean_v = sum(vals) / n
            std_v = 0.0 if n <= 1 else math.sqrt(sum((v - mean_v) ** 2 for v in vals) / (n - 1))
            summary[f"{metric}_mean"] = mean_v
            summary[f"{metric}_std"] = std_v
        out.append(summary)
    order = {"Static-pricing": 0, "DSPO": 1, "DRPO": 2}
    out.sort(key=lambda r: order.get(str(r["strategy"]), 99))
    return out


def persist(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    write_csv(output_dir / "pilot_raw.csv", rows, collect_fields(rows))
    summary = summarize(rows)
    if summary:
        write_csv(output_dir / "pilot_summary.csv", summary, collect_fields(summary))


def run_one(
    args: argparse.Namespace,
    root: Path,
    strategy: str,
    algo_name: str,
    run_id: str,
    seed: int,
    cfg: Dict[str, float],
    static_cfg: Dict[str, float],
) -> Dict[str, Any]:
    log = log_path(root, algo_name, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args, strategy, algo_name, run_id, seed, cfg, static_cfg)
    needs_gpu_marker = algo_name != "Baseline"

    if args.skip_existing:
        metrics = parse_metrics(log)
        if metrics is not None and (args.allow_cpu or (not needs_gpu_marker) or has_gpu_marker(log)):
            row = {
                "strategy": strategy,
                "algo_name": algo_name,
                "run_id": run_id,
                "seed": seed,
                "episodes": args.episodes,
                "status": "cached",
                "runtime_sec": 0.0,
                "log_path": str(log),
                "command": " ".join(cmd),
            }
            row.update(algo_config(strategy, cfg))
            if strategy == "Static-pricing":
                row.update(static_cfg)
            row.update(metrics)
            return row

    timeout = None if args.run_timeout_sec <= 0 else args.run_timeout_sec
    attempts = max(1, args.max_retries + 1)
    last_error = ""
    for attempt in range(1, attempts + 1):
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
        except subprocess.TimeoutExpired as exc:
            tail = (exc.stdout or "")[-1500:] if exc.stdout else ""
            last_error = f"Timeout {attempt}/{attempts} for {strategy} seed={seed}. tail={tail}"
            if attempt < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        runtime = time.time() - t0
        metrics = parse_metrics(log)
        if cp.returncode != 0:
            last_error = f"Return code {cp.returncode} for {strategy} seed={seed}. tail={(cp.stdout or '')[-1500:]}"
        elif metrics is None:
            last_error = f"Metrics missing for {strategy} seed={seed}; log={log}"
        elif (not args.allow_cpu) and needs_gpu_marker and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing for {strategy} seed={seed}; log={log}"
        else:
            row = {
                "strategy": strategy,
                "algo_name": algo_name,
                "run_id": run_id,
                "seed": seed,
                "episodes": args.episodes,
                "status": "completed" if attempt == 1 else f"completed_retry_{attempt}",
                "runtime_sec": runtime,
                "log_path": str(log),
                "command": " ".join(cmd),
            }
            row.update(algo_config(strategy, cfg))
            if strategy == "Static-pricing":
                row.update(static_cfg)
            row.update(metrics)
            return row

        if attempt < attempts:
            time.sleep(args.retry_backoff_sec)

    raise RuntimeError(last_error)


def row_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    return (str(row["strategy"]), int(float(row["seed"])), int(float(row["episodes"])))


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 3:
        print(f"[WARN] This is a 3-seed pilot script; received seeds={args.seeds}", flush=True)

    root = Path(__file__).resolve().parent.parent
    cfg, static_cfg, cfg_path = load_payload(root, args)
    runtime = probe_runtime(args.python_executable)
    print(
        "[INFO] Runtime probe: "
        + f"python={runtime['exe']}, torch={runtime['torch_version']}, "
        + f"cuda_available={runtime['cuda_available']}, cuda_count={runtime['cuda_count']}",
        flush=True,
    )
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable in selected runtime. Pass --allow_cpu to continue on CPU.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        args.output_dir = f"Experiments/analysis/main_pilot_3seed_{ts}"
    output_dir = resolve_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.allow_existing_output_dir:
        raise RuntimeError(f"Output directory already contains files: {output_dir}. Use --allow_existing_output_dir.")

    strategies = [
        ("DSPO", "DSPO"),
        ("DRPO", "DRPO"),
        ("Static-pricing", "Baseline"),
    ]
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_source": str(cfg_path),
        "config": cfg,
        "static_pricing": static_cfg,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "gpu": args.gpu,
        "run_prefix": args.run_prefix,
        "folder_suffix": args.folder_suffix,
        "strategies": strategies,
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        for seed in args.seeds:
            for strategy, algo in strategies:
                run_id = f"{args.run_prefix}_{strategy.replace('-', '_')}"
                print(" ".join(build_cmd(args, strategy, algo, run_id, seed, cfg, static_cfg)), flush=True)
        print(f"[DRY-RUN] Output dir: {output_dir}", flush=True)
        return

    rows = [dict(r) for r in read_csv(output_dir / "pilot_raw.csv")]
    done = {row_key(r) for r in rows}
    jobs = [(strategy, algo, f"{args.run_prefix}_{strategy.replace('-', '_')}", int(seed)) for seed in args.seeds for strategy, algo in strategies]
    remaining = [job for job in jobs if (job[0], job[3], args.episodes) not in done]
    print(f"[INFO] Total jobs={len(jobs)}, existing_rows={len(rows)}, remaining={len(remaining)}", flush=True)

    new_count = 0
    for idx, (strategy, algo, run_id, seed) in enumerate(jobs, 1):
        key = (strategy, seed, args.episodes)
        if key in done:
            continue
        print(f"[RUN {idx}/{len(jobs)}] strategy={strategy}, algo={algo}, seed={seed}", flush=True)
        row = run_one(args, root, strategy, algo, run_id, seed, cfg, static_cfg)
        print(
            f"[OK] {strategy} seed={seed} "
            + f"net_profit={row.get('net_profit')} home={row.get('home_delivery')} quit={row.get('quit_rate')}",
            flush=True,
        )
        rows.append(row)
        done.add(row_key(row))
        new_count += 1
        if new_count % args.persist_every_n == 0:
            persist(output_dir, rows)

    persist(output_dir, rows)
    print(f"[DONE] Output dir: {output_dir}", flush=True)
    print(f"[DONE] Summary: {output_dir / 'pilot_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
