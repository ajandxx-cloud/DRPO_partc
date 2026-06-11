#!/usr/bin/env python3
"""Small-sample RC tuning for lifted DRPO.

Runs a DSPO baseline and a fixed SPO-parameter grid for DRPO on
three RC seeds. Outputs candidate_raw.csv, candidate_summary.csv,
selected_config.json, and run_meta.json.
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


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [40, 67, 97]
BASE_CONFIG: Dict[str, Any] = {
    "k": 10,
    "revenue": 50.0,
    "fuel_cost": 0.6,
    "home_failure": 0.1,
    "home_util": 1.4,
    "outside_option_util": -1.0,
    "incentive_sens": -0.25,
    "learning_rate": 0.001,
    "batch_size": 256,
}
SPO_CANDIDATES: Dict[str, Dict[str, Any]] = {
    "w000": {"spo_loss_weight": 0.0, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w002": {"spo_loss_weight": 0.02, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w005": {"spo_loss_weight": 0.05, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w010": {"spo_loss_weight": 0.1, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w030": {"spo_loss_weight": 0.3, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w050": {"spo_loss_weight": 0.5, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w070": {"spo_loss_weight": 0.7, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "w090": {"spo_loss_weight": 0.9, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "default": {"spo_loss_weight": 0.7, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "conservative": {"spo_loss_weight": 0.4, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "strong": {"spo_loss_weight": 0.9, "spo_warmup_episodes": 5, "spo_rampup_episodes": 10},
    "early": {"spo_loss_weight": 0.7, "spo_warmup_episodes": 3, "spo_rampup_episodes": 8},
    "delayed": {"spo_loss_weight": 0.7, "spo_warmup_episodes": 8, "spo_rampup_episodes": 15},
}
INT_PARAMS = {"k", "batch_size", "spo_warmup_episodes", "spo_rampup_episodes"}
METRIC_REGEX = {
    "net_profit": re.compile(r"Net profit:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "total_costs": re.compile(r"total costs:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "quit_rate": re.compile(r"Quit rate:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)%"),
    "served_demand": re.compile(r"Accepted customers:\s*([+-]?\d+(?:\.\d+)?)"),
    "total_demand": re.compile(r"Total customers:\s*([+-]?\d+(?:\.\d+)?)"),
}
SPO_WEIGHT_REGEX = re.compile(r"\[SPO\+ debug\] spo_weight became positive:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
METRICS = ["net_profit", "total_costs", "quit_rate", "served_rate", "served_demand", "total_demand"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune lifted DRPO on a small RC sample.")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--data_seed_test", type=int, default=1)
    p.add_argument("--save_count", type=int, default=1)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--run_prefix", default="RC_DRPO_LIFTED")
    p.add_argument("--folder_suffix", default="_drpo_lifted_smoke")
    p.add_argument("--persist_every_n", type=int, default=1)
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=1)
    p.add_argument("--retry_backoff_sec", type=int, default=10)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--candidates", nargs="+", default=list(SPO_CANDIDATES.keys()),
                   choices=list(SPO_CANDIDATES.keys()))
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def cli_value(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if name in INT_PARAMS:
        return str(int(round(float(value))))
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def build_cmd(args: argparse.Namespace, algo: str, run_id: str, seed: int, cfg: Dict[str, Any]) -> List[str]:
    cmd = [
        args.python_executable, "run.py",
        "--algo_name", algo,
        "--instance", "RC",
        "--seed", str(seed),
        "--data_seed", str(args.data_seed),
        "--data_seed_test", str(args.data_seed_test),
        "--max_episodes", str(args.episodes),
        "--save_count", str(args.save_count),
        "--log_output", "file",
        "--debug", "False",
        "--gpu", str(args.gpu),
    ]
    for key in sorted(cfg):
        cmd.extend([f"--{key}", cli_value(key, cfg[key])])
    cmd.extend(["--experiment", run_id, "--folder_suffix", args.folder_suffix])
    return cmd


def run_log_path(algo: str, run_id: str, suffix: str, seed: int) -> Path:
    return ROOT / "Experiments" / "Parcelpoint_py" / "pricing" / algo / f"{run_id}{suffix}" / str(seed) / "Logs" / "logfile.log"


def parse_metrics(log: Path) -> Optional[Dict[str, float]]:
    if not log.exists():
        return None
    txt = log.read_text(encoding="utf-8", errors="ignore")
    out: Dict[str, float] = {}
    for key, pattern in METRIC_REGEX.items():
        vals = pattern.findall(txt)
        if vals:
            out[key] = float(vals[-1])
    if not {"net_profit", "total_costs", "quit_rate"}.issubset(out):
        return None
    if out.get("total_demand", 0) > 0 and "served_demand" in out:
        out["served_rate"] = out["served_demand"] / out["total_demand"]
    return out


def has_gpu_marker(log: Path) -> bool:
    return log.exists() and "Using GPU device: cuda" in log.read_text(encoding="utf-8", errors="ignore")


def parse_spo_health(log: Path) -> Dict[str, Any]:
    if not log.exists():
        return {
            "spo_weight_positive": False,
            "first_spo_weight": "",
            "max_spo_weight_seen": "",
            "spo_training_data_populated": False,
            "spo_warning_count": "",
        }
    txt = log.read_text(encoding="utf-8", errors="ignore")
    weights = [float(x) for x in SPO_WEIGHT_REGEX.findall(txt)]
    return {
        "spo_weight_positive": any(w > 0 for w in weights),
        "first_spo_weight": weights[0] if weights else "",
        "max_spo_weight_seen": max(weights) if weights else "",
        "spo_training_data_populated": "[SPO+ debug] spo_training_data populated" in txt,
        "spo_warning_count": txt.count("[SPO+ warning]"),
    }


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_fields(rows: List[Dict[str, Any]]) -> List[str]:
    preferred = ["label", "candidate", "algo_name", "run_id", "seed", "status", "runtime_sec", "log_path", "command"]
    seen = set(preferred)
    extra: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(key)
    return preferred + sorted(extra)


def row_key(row: Dict[str, Any]) -> Tuple[str, str, int]:
    return str(row["label"]), str(row["candidate"]), int(float(row["seed"]))


def run_single(args: argparse.Namespace, label: str, candidate: str, algo: str, run_id: str,
               seed: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    log = run_log_path(algo, run_id, args.folder_suffix, seed)
    cmd = build_cmd(args, algo, run_id, seed, cfg)
    if args.skip_existing:
        metrics = parse_metrics(log)
        if metrics and (args.allow_cpu or has_gpu_marker(log)):
            row: Dict[str, Any] = {
                "label": label, "candidate": candidate, "algo_name": algo, "run_id": run_id,
                "seed": seed, "status": "cached", "runtime_sec": 0.0,
                "log_path": str(log), "command": " ".join(cmd),
            }
            row.update(cfg)
            row.update(metrics)
            row.update(parse_spo_health(log))
            return row

    timeout = None if args.run_timeout_sec <= 0 else args.run_timeout_sec
    attempts = max(1, args.max_retries + 1)
    last_error = ""
    for attempt in range(1, attempts + 1):
        start = time.time()
        try:
            cp = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="ignore", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_error = f"Timeout {attempt}/{attempts} for {label}-{candidate} seed={seed}: {(exc.stdout or '')[-1000:]}"
            if attempt < attempts:
                time.sleep(args.retry_backoff_sec)
                continue
            raise RuntimeError(last_error)

        runtime = time.time() - start
        metrics = parse_metrics(log)
        if cp.returncode != 0:
            last_error = f"Return code {cp.returncode} for {label}-{candidate} seed={seed}: {(cp.stdout or '')[-1000:]}"
        elif metrics is None:
            last_error = f"Metrics missing for {label}-{candidate} seed={seed}; log={log}"
        elif (not args.allow_cpu) and (not has_gpu_marker(log)):
            last_error = f"GPU marker missing for {label}-{candidate} seed={seed}; log={log}"
        else:
            row = {
                "label": label, "candidate": candidate, "algo_name": algo, "run_id": run_id,
                "seed": seed, "status": "completed" if attempt == 1 else f"completed_retry_{attempt}",
                "runtime_sec": runtime, "log_path": str(log), "command": " ".join(cmd),
            }
            row.update(cfg)
            row.update(metrics)
            row.update(parse_spo_health(log))
            return row
        if attempt < attempts:
            time.sleep(args.retry_backoff_sec)
    raise RuntimeError(last_error)


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["label"]), str(row["candidate"])), []).append(row)
    out: List[Dict[str, Any]] = []
    for (label, candidate), group in sorted(groups.items()):
        item: Dict[str, Any] = {"label": label, "candidate": candidate, "n_runs": len(group)}
        for metric in METRICS:
            vals = [to_float(row.get(metric)) for row in group]
            vals = [v for v in vals if v is not None]
            if vals:
                mean = sum(vals) / len(vals)
                std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
            else:
                item[f"{metric}_mean"] = ""
                item[f"{metric}_std"] = ""
        out.append(item)
    return out


def select_candidate(summary: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = next((r for r in summary if r["label"] == "DSPO"), None)
    if baseline is None:
        raise RuntimeError("DSPO baseline summary missing.")
    base_np = float(baseline["net_profit_mean"])
    base_cost = float(baseline["total_costs_mean"])
    base_quit = float(baseline["quit_rate_mean"])
    base_served = float(baseline["served_rate_mean"])

    candidates = []
    for row in summary:
        if row["label"] != "DRPO":
            continue
        np_delta = float(row["net_profit_mean"]) - base_np
        cost_delta = float(row["total_costs_mean"]) - base_cost
        quit_delta = float(row["quit_rate_mean"]) - base_quit
        served_delta = float(row["served_rate_mean"]) - base_served
        ok = np_delta > 0 and cost_delta < 0 and quit_delta <= 2.0 and served_delta >= -0.02
        enriched = dict(row)
        enriched.update({
            "baseline_net_profit_mean": base_np,
            "baseline_total_costs_mean": base_cost,
            "baseline_quit_rate_mean": base_quit,
            "baseline_served_rate_mean": base_served,
            "net_profit_delta": np_delta,
            "total_costs_delta": cost_delta,
            "quit_rate_delta_pp": quit_delta,
            "served_rate_delta": served_delta,
            "guardrails_pass": ok,
        })
        candidates.append(enriched)

    if not candidates:
        raise RuntimeError("No DRPO candidates found.")
    feasible = [c for c in candidates if c["guardrails_pass"]]
    pool = feasible if feasible else candidates
    selected = sorted(pool, key=lambda r: (r["net_profit_delta"], -r["total_costs_delta"]), reverse=True)[0]
    selected["selection_status"] = "guardrails_pass" if feasible else "best_available_guardrails_failed"
    selected["config"] = SPO_CANDIDATES[selected["candidate"]]
    return selected


def persist(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    write_csv(output_dir / "candidate_raw.csv", rows, collect_fields(rows))
    summary = summarize(rows)
    if summary:
        write_csv(output_dir / "candidate_summary.csv", summary, summary[0].keys())
        labels = {str(row["label"]) for row in summary}
        if "DSPO" in labels and "DRPO" in labels:
            selected = select_candidate(summary)
            (output_dir / "selected_config.json").write_text(
                json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.output_dir:
        args.output_dir = f"Experiments/analysis/rc_drpo_lifted_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = probe_runtime(args.python_executable)
    print(f"[INFO] Runtime: torch={runtime['torch_version']} cuda={runtime['cuda_available']} count={runtime['cuda_count']}", flush=True)
    if (not args.allow_cpu) and (not runtime["cuda_available"]):
        raise RuntimeError("CUDA unavailable. Use --allow_cpu for a CPU smoke run.")

    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_prefix": args.run_prefix,
        "folder_suffix": args.folder_suffix,
        "seeds": args.seeds,
        "episodes": args.episodes,
        "base_config": BASE_CONFIG,
        "spo_candidates": {k: SPO_CANDIDATES[k] for k in args.candidates},
        "selection_rule": {
            "net_profit_delta": "> 0",
            "total_costs_delta": "< 0",
            "quit_rate_delta_pp": "<= 2.0",
            "served_rate_delta": ">= -0.02",
        },
    }
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    jobs: List[Tuple[str, str, str, str, Dict[str, Any]]] = []
    jobs.append(("DSPO", "baseline", "DSPO", f"{args.run_prefix}_DSPO", dict(BASE_CONFIG)))
    for name in args.candidates:
        cfg = dict(BASE_CONFIG)
        cfg.update(SPO_CANDIDATES[name])
        jobs.append(("DRPO", name, "DRPO", f"{args.run_prefix}_DRPO_{name}", cfg))

    if args.dry_run:
        for label, candidate, algo, run_id, cfg in jobs:
            for seed in args.seeds[:1]:
                print(f"[DRY-RUN] {label}/{candidate}: " + " ".join(build_cmd(args, algo, run_id, seed, cfg)), flush=True)
        print(f"[DRY-RUN] output_dir={output_dir}", flush=True)
        return

    rows = [dict(r) for r in read_csv(output_dir / "candidate_raw.csv")]
    done = {row_key(r) for r in rows}
    new_count = 0
    total = len(jobs) * len(args.seeds)
    idx = 0
    for label, candidate, algo, run_id, cfg in jobs:
        for seed in args.seeds:
            idx += 1
            key = (label, candidate, int(seed))
            if key in done:
                continue
            print(f"[RUN {idx}/{total}] {label}/{candidate} seed={seed}", flush=True)
            row = run_single(args, label, candidate, algo, run_id, int(seed), cfg)
            rows.append(row)
            done.add(row_key(row))
            new_count += 1
            if new_count % args.persist_every_n == 0:
                persist(output_dir, rows)
    persist(output_dir, rows)
    print(f"[DONE] Raw: {output_dir / 'candidate_raw.csv'}", flush=True)
    print(f"[DONE] Summary: {output_dir / 'candidate_summary.csv'}", flush=True)
    print(f"[DONE] Selected: {output_dir / 'selected_config.json'}", flush=True)


if __name__ == "__main__":
    main()
