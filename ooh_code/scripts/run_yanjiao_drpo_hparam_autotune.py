#!/usr/bin/env python
"""Auto-tune Yanjiao DRPO hyperparameters.

This is a conservative orchestration wrapper around run_yanjiao_spo_weight_sweep.py.
It keeps the algorithm code unchanged, reuses an existing matched DSPO baseline
when provided, and ranks DRPO hyperparameter candidates by seed-robust paired
net-profit improvement.
"""

import argparse
import csv
import itertools
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yanjiao DRPO hyperparameter auto-tuner")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--baseline_raw", default="")
    p.add_argument("--run_matched_baseline", action="store_true")
    p.add_argument("--seeds", nargs="+", type=int, default=[40, 67, 97])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--route_label_mode", default="hgs", choices=["hgs", "hep"])
    p.add_argument("--n_passengers", type=int, default=400)
    p.add_argument("--hgs_reopt_time", type=float, default=0.2)
    p.add_argument("--hgs_final_time", type=float, default=0.2)
    p.add_argument("--weights", nargs="+", type=float, default=[0.03, 0.05, 0.07, 0.08])
    p.add_argument("--spo_batch_sizes", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--spo_warmups", nargs="+", type=int, default=[3, 5, 8])
    p.add_argument("--spo_rampups", nargs="+", type=int, default=[8, 10, 15])
    p.add_argument("--initial_phase_epochs", nargs="+", type=int, default=[50])
    p.add_argument("--buffer_sizes", nargs="+", type=int, default=[500])
    p.add_argument("--max_candidates", type=int, default=18)
    p.add_argument("--mode", choices=["focused", "grid"], default="focused")
    p.add_argument("--run_timeout_sec", type=int, default=7200)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--analyze_only", action="store_true")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def tag_float(prefix: str, value: float, scale: int = 1000) -> str:
    return f"{prefix}{int(round(value * scale)):04d}"


def candidate_id(candidate: Dict[str, Any]) -> str:
    parts = [
        tag_float("w", float(candidate["spo_loss_weight"])),
        f"b{int(candidate['spo_batch_size']):02d}",
        f"wu{int(candidate['spo_warmup'])}",
        f"ru{int(candidate['spo_rampup'])}",
    ]
    if int(candidate.get("initial_phase_epochs", 50)) != 50:
        parts.append(f"init{int(candidate['initial_phase_epochs'])}")
    if int(candidate.get("buffer_size", 500)) != 500:
        parts.append(f"buf{int(candidate['buffer_size'])}")
    return "_".join(parts)


def generate_candidates(args: argparse.Namespace) -> List[Dict[str, Any]]:
    combos = []
    for weight, batch, warmup, rampup, init_ep, buffer_size in itertools.product(
        args.weights,
        args.spo_batch_sizes,
        args.spo_warmups,
        args.spo_rampups,
        args.initial_phase_epochs,
        args.buffer_sizes,
    ):
        combos.append(
            {
                "spo_loss_weight": float(weight),
                "spo_batch_size": int(batch),
                "spo_warmup": int(warmup),
                "spo_rampup": int(rampup),
                "initial_phase_epochs": int(init_ep),
                "buffer_size": int(buffer_size),
            }
        )

    if args.mode == "grid":
        return combos[: args.max_candidates]

    # Focused design around the empirically best w=0.05. Start with anchor
    # weight variants, then one-change-at-a-time training hyperparameter variants.
    def priority(c: Dict[str, Any]) -> Tuple[float, int, int, int]:
        weight_penalty = abs(float(c["spo_loss_weight"]) - 0.05)
        batch_penalty = 0 if int(c["spo_batch_size"]) == 8 else 1
        warmup_penalty = abs(int(c["spo_warmup"]) - 5)
        rampup_penalty = abs(int(c["spo_rampup"]) - 10)
        return (weight_penalty, batch_penalty + warmup_penalty + rampup_penalty, warmup_penalty, rampup_penalty)

    unique: List[Dict[str, Any]] = []
    seen = set()

    def nearest(values: Sequence[Any], target: float) -> Any:
        return min(values, key=lambda v: (abs(float(v) - target), float(v)))

    center = {
        "spo_loss_weight": float(nearest(args.weights, 0.05)),
        "spo_batch_size": int(nearest(args.spo_batch_sizes, 8)),
        "spo_warmup": int(nearest(args.spo_warmups, 5)),
        "spo_rampup": int(nearest(args.spo_rampups, 10)),
        "initial_phase_epochs": int(nearest(args.initial_phase_epochs, 50)),
        "buffer_size": int(nearest(args.buffer_sizes, 500)),
    }

    combo_by_key = {
        (
            round(float(c["spo_loss_weight"]), 10),
            int(c["spo_batch_size"]),
            int(c["spo_warmup"]),
            int(c["spo_rampup"]),
            int(c["initial_phase_epochs"]),
            int(c["buffer_size"]),
        ): c
        for c in combos
    }

    def append_candidate(**overrides: Any) -> None:
        candidate = dict(center)
        candidate.update(overrides)
        key = (
            round(float(candidate["spo_loss_weight"]), 10),
            int(candidate["spo_batch_size"]),
            int(candidate["spo_warmup"]),
            int(candidate["spo_rampup"]),
            int(candidate["initial_phase_epochs"]),
            int(candidate["buffer_size"]),
        )
        c = combo_by_key.get(key)
        if c is None:
            return
        cid = candidate_id(c)
        if cid in seen:
            return
        seen.add(cid)
        unique.append(c)

    append_candidate()
    for weight in sorted(args.weights, key=lambda w: (abs(float(w) - center["spo_loss_weight"]), float(w))):
        append_candidate(spo_loss_weight=float(weight))
        if len(unique) >= args.max_candidates:
            return unique
    for batch in sorted(args.spo_batch_sizes, key=lambda b: (abs(int(b) - center["spo_batch_size"]), int(b))):
        append_candidate(spo_batch_size=int(batch))
        if len(unique) >= args.max_candidates:
            return unique
    for warmup in sorted(args.spo_warmups, key=lambda w: (abs(int(w) - center["spo_warmup"]), int(w))):
        append_candidate(spo_warmup=int(warmup))
        if len(unique) >= args.max_candidates:
            return unique
    for rampup in sorted(args.spo_rampups, key=lambda r: (abs(int(r) - center["spo_rampup"]), int(r))):
        append_candidate(spo_rampup=int(rampup))
        if len(unique) >= args.max_candidates:
            return unique
    for init_ep in sorted(args.initial_phase_epochs, key=lambda e: (abs(int(e) - center["initial_phase_epochs"]), int(e))):
        append_candidate(initial_phase_epochs=int(init_ep))
        if len(unique) >= args.max_candidates:
            return unique
    for buffer_size in sorted(args.buffer_sizes, key=lambda b: (abs(int(b) - center["buffer_size"]), int(b))):
        append_candidate(buffer_size=int(buffer_size))
        if len(unique) >= args.max_candidates:
            return unique

    ordered = sorted(combos, key=priority)
    for c in ordered:
        cid = candidate_id(c)
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(c)
        if len(unique) >= args.max_candidates:
            break
    return unique


def build_sweep_cmd(args: argparse.Namespace, output_dir: Path, c: Dict[str, Any]) -> List[str]:
    cid = candidate_id(c)
    cmd = [
        args.python_executable,
        "scripts/run_yanjiao_spo_weight_sweep.py",
        "--output_dir",
        str(output_dir / cid),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--weights",
        repr(float(c["spo_loss_weight"])),
        "--episodes",
        str(args.episodes),
        "--eval_episodes",
        str(args.eval_episodes),
        "--route_label_mode",
        args.route_label_mode,
        "--n_passengers",
        str(args.n_passengers),
        "--hgs_reopt_time",
        repr(float(args.hgs_reopt_time)),
        "--hgs_final_time",
        repr(float(args.hgs_final_time)),
        "--spo_batch_size",
        str(int(c["spo_batch_size"])),
        "--spo_warmup",
        str(int(c["spo_warmup"])),
        "--spo_rampup",
        str(int(c["spo_rampup"])),
        "--initial_phase_epochs",
        str(int(c["initial_phase_epochs"])),
        "--buffer_size",
        str(int(c["buffer_size"])),
        "--run_prefix",
        f"YJ_HPARAM_{cid}",
        "--folder_suffix",
        f"_yj_hparam_{cid}",
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--max_retries",
        str(args.max_retries),
    ]
    if args.run_matched_baseline:
        cmd.append("--run_matched_baseline")
    elif args.baseline_raw:
        cmd.extend(["--baseline_raw", args.baseline_raw])
    if args.allow_cpu:
        cmd.append("--allow_cpu")
    if not args.skip_existing:
        cmd.append("--no_skip_existing")
    if args.dry_run:
        cmd.append("--dry_run")
    return cmd


def run_candidate(args: argparse.Namespace, output_dir: Path, c: Dict[str, Any]) -> int:
    cid = candidate_id(c)
    run_dir = output_dir / cid
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_sweep_cmd(args, output_dir, c)
    (run_dir / "autotune_command.txt").write_text(" ".join(cmd), encoding="utf-8")
    print(f"[RUN] {cid}", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    t0 = time.time()
    cp = subprocess.run(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    (run_dir / "autotune_stdout.log").write_text(cp.stdout or "", encoding="utf-8")
    print(f"[DONE] {cid} rc={cp.returncode} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    return cp.returncode


def score_summary(summary: Dict[str, str], n_seeds: int) -> Dict[str, Any]:
    mean_profit = to_float(summary.get("mean_delta_net_profit")) or -1e9
    std_profit = to_float(summary.get("std_delta_net_profit")) or 0.0
    wins = int(float(summary.get("wins_net_profit") or 0))
    spo_ok = str(summary.get("spo_ok_all_runs")).strip().lower() == "true"
    warnings = int(float(summary.get("spo_warning_count_total") or 0))
    mean_quit = abs(to_float(summary.get("mean_delta_quit_rate")) or 0.0)
    score = mean_profit + 5.0 * wins - 0.25 * std_profit - 20.0 * mean_quit
    if wins < max(1, math.ceil(n_seeds / 2)):
        score -= 25.0
    if not spo_ok:
        score -= 1000.0
    score -= 20.0 * warnings
    return {
        "score": score,
        "mean_delta_net_profit": mean_profit,
        "std_delta_net_profit": std_profit,
        "wins_net_profit": wins,
        "spo_ok_all_runs": spo_ok,
        "spo_warning_count_total": warnings,
        "mean_abs_delta_quit_rate": mean_quit,
    }


def collect_result(args: argparse.Namespace, output_dir: Path, c: Dict[str, Any], returncode: int) -> Dict[str, Any]:
    cid = candidate_id(c)
    summary_path = output_dir / cid / "candidate_summary.csv"
    selected_path = output_dir / cid / "selected_config.json"
    rows = read_csv(summary_path)
    summary = rows[0] if rows else {}
    score = score_summary(summary, len(args.seeds)) if summary else {
        "score": -1e9,
        "mean_delta_net_profit": "",
        "std_delta_net_profit": "",
        "wins_net_profit": 0,
        "spo_ok_all_runs": False,
        "spo_warning_count_total": "",
        "mean_abs_delta_quit_rate": "",
    }
    result: Dict[str, Any] = {
        "candidate_id": cid,
        "returncode": returncode,
        **c,
        **score,
        "summary_path": str(summary_path),
        "selected_path": str(selected_path),
    }
    for key, value in summary.items():
        if key not in result:
            result[key] = value
    return result


def rank_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r.get("score") or -1e9),
            int(float(r.get("wins_net_profit") or 0)),
            float(r.get("mean_delta_net_profit") or -1e9),
            -float(r.get("std_delta_net_profit") or 1e9),
        ),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    if not args.run_matched_baseline and not args.baseline_raw:
        raise SystemExit("Provide --baseline_raw or use --run_matched_baseline.")

    output_dir = resolve_path(
        args.output_dir
        or f"Experiments/analysis/yanjiao_drpo_hparam_autotune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(args)
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "selection_rule": "score = mean_delta_net_profit + 5*wins - 0.25*std - 20*abs(mean_delta_quit_rate), with penalties for low wins/SPO warnings",
    }
    (output_dir / "autotune_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    results: List[Dict[str, Any]] = []
    if not args.analyze_only:
        for c in candidates:
            cid = candidate_id(c)
            selected_path = output_dir / cid / "selected_config.json"
            if args.skip_existing and selected_path.exists():
                rc = 0
                print(f"[CACHE] {cid}", flush=True)
            else:
                rc = run_candidate(args, output_dir, c)
            result = collect_result(args, output_dir, c, rc)
            results.append(result)
            ranked = rank_results(results)
            write_csv(output_dir / "autotune_results.csv", ranked)
            (output_dir / "best_config.json").write_text(
                json.dumps(ranked[0], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            if rc != 0:
                raise RuntimeError(f"Candidate {cid} failed with return code {rc}.")
    else:
        for c in candidates:
            cid = candidate_id(c)
            if (output_dir / cid / "candidate_summary.csv").exists():
                results.append(collect_result(args, output_dir, c, 0))
        if not results:
            raise RuntimeError(f"No existing candidate summaries found under {output_dir}")
        ranked = rank_results(results)
        write_csv(output_dir / "autotune_results.csv", ranked)
        (output_dir / "best_config.json").write_text(
            json.dumps(ranked[0], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[DONE] results={output_dir / 'autotune_results.csv'}", flush=True)
    print(f"[DONE] best={output_dir / 'best_config.json'}", flush=True)


if __name__ == "__main__":
    main()
