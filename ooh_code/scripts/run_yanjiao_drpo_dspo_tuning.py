#!/usr/bin/env python
"""Tune and validate Yanjiao DSPO/DRPO pricing parameters.

This runner keeps the algorithm code unchanged. It orchestrates the existing
Yanjiao runner across a small candidate grid, checks the target ordering
DRPO > DSPO > Static, then validates the selected setting on three seeds.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TUNING_SEEDS = [40, 67]
VALIDATION_SEEDS = [40, 67, 97]
STRATEGIES = ["Static", "DSPO", "DRPO"]

STATIC_PRICE_HOME = 0.5
STATIC_PRICE_PP = -1.0

FIRST_ROUND = [
    ("A_mp3p5_beta25", 3.5, -0.25),
    ("B_mp5_beta25", 5.0, -0.25),
    ("C_mp5_beta30", 5.0, -0.30),
    ("D_mp5p5_beta25", 5.5, -0.25),
]

SECOND_ROUND = [
    (f"R2_mp{str(mp).replace('.', 'p')}_beta{str(abs(beta)).replace('.', 'p')}", mp, beta)
    for mp in [4.5, 5.0, 5.5]
    for beta in [-0.25, -0.275, -0.30]
]

METRIC_REGEX = {
    "huber_loss": re.compile(r"Huber loss:?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "epoch_huber_loss": re.compile(r"Epoch\s+\d+\s+Huber loss::\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
    "spo_weight": re.compile(r"\[SPO\+ debug\] spo_weight became positive:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yanjiao DRPO/DSPO tuning and 3-seed validation")
    p.add_argument("--python_executable", default=sys.executable)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--output_dir", default="Experiments/analysis/yanjiao_drpo_dspo_tuning")
    p.add_argument("--validation_output_dir", default="Experiments/analysis/yanjiao_final_maxprice5_3seed")
    p.add_argument("--folder_suffix", default="_yanjiao_param_tuning")
    p.add_argument("--validation_folder_suffix", default="_yanjiao_final_3seed")
    p.add_argument("--run_timeout_sec", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--allow_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_existing", dest="skip_existing", action="store_true")
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--force_second_round", action="store_true")
    p.add_argument("--skip_validation", action="store_true")
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def rel_or_abs(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def run_id(label: str, strategy: str, seed: int, phase: str) -> str:
    return f"YJ_{phase}_{label}_400_{strategy}_seed{seed}"


def log_path(root: Path, strategy: str, label: str, seed: int, suffix: str, phase: str) -> Path:
    algo = {"Static": "Baseline", "DSPO": "DSPO", "DRPO": "DRPO"}[strategy]
    preferred = (
        root
        / "Experiments"
        / "Parcelpoint_py"
        / "pricing"
        / algo
        / f"{run_id(label, strategy, seed, phase)}{suffix}"
        / str(seed)
        / "Logs"
        / "logfile.log"
    )
    if strategy == "DRPO" and not preferred.exists():
        legacy = (
            root
            / "Experiments"
            / "Parcelpoint_py"
            / "pricing"
            / "DSPO_plus_SPO"
            / f"{run_id(label, strategy, seed, phase)}{suffix}"
            / str(seed)
            / "Logs"
            / "logfile.log"
        )
        if legacy.exists():
            return legacy
    return preferred


def build_runner_cmd(
    args: argparse.Namespace,
    label: str,
    max_price: float,
    incentive_sens: float,
    seeds: Sequence[int],
    output_dir: str,
    folder_suffix: str,
    phase_tag: str,
) -> List[str]:
    prefix = f"YJ_{phase_tag}_{label}"
    return [
        args.python_executable,
        "scripts/run_yanjiao_experiments.py",
        "--python_executable",
        args.python_executable,
        "--gpu",
        str(args.gpu),
        "--phase",
        "main",
        "--seeds",
        *[str(s) for s in seeds],
        "--episodes",
        str(args.episodes),
        "--strategies",
        *STRATEGIES,
        "--output_dir",
        output_dir,
        "--allow_existing_output_dir",
        "--folder_suffix",
        folder_suffix,
        "--run_prefix",
        prefix,
        "--persist_every_n",
        "1",
        "--max_retries",
        str(args.max_retries),
        "--run_timeout_sec",
        str(args.run_timeout_sec),
        "--incentive_sens_override",
        repr(incentive_sens),
        "--max_price_override",
        repr(max_price),
        "--min_price_override",
        "-5.0",
        "--static_price_home",
        repr(STATIC_PRICE_HOME),
        "--static_price_pp",
        repr(STATIC_PRICE_PP),
        "--dspo_spo_loss_weight",
        "0.0",
    ] + (["--allow_cpu"] if args.allow_cpu else []) + (["--dry_run"] if args.dry_run else []) + (["--no_skip_existing"] if not args.skip_existing else [])


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
        w.writerows(rows)


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def row_index(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for row in rows:
        try:
            out[(str(row["label"]), int(float(row["seed"])))] = row
        except (KeyError, ValueError):
            continue
    return out


def log_health(log: Path, strategy: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "log_exists": log.exists(),
        "spo_weight_positive": False,
        "spo_warning_count": 0,
        "has_nan_or_inf_loss": False,
        "last_huber_loss": "",
    }
    if not log.exists():
        return result

    text = log.read_text(encoding="utf-8", errors="ignore")
    result["spo_warning_count"] = text.count("[SPO+ warning]")
    weights = [float(x) for x in METRIC_REGEX["spo_weight"].findall(text)]
    result["spo_weight_positive"] = any(w > 0 for w in weights)

    losses = [float(x) for x in METRIC_REGEX["huber_loss"].findall(text)]
    losses.extend(float(x) for x in METRIC_REGEX["epoch_huber_loss"].findall(text))
    finite_losses = [x for x in losses if math.isfinite(x)]
    if finite_losses:
        result["last_huber_loss"] = finite_losses[-1]
    if losses and len(finite_losses) != len(losses):
        result["has_nan_or_inf_loss"] = True
    if re.search(r"\b(?:nan|inf|-inf)\b", text, flags=re.IGNORECASE):
        result["has_nan_or_inf_loss"] = True

    if strategy != "DRPO":
        result["spo_weight_positive"] = ""
        result["spo_warning_count"] = ""
    return result


def evaluate_candidate(
    root: Path,
    rows: List[Dict[str, str]],
    label: str,
    max_price: float,
    incentive_sens: float,
    seeds: Sequence[int],
    folder_suffix: str,
    phase_tag: str,
) -> Dict[str, Any]:
    idx = row_index(rows)
    per_seed = []
    for seed in seeds:
        static = idx.get(("Static", seed))
        dspo = idx.get(("DSPO", seed))
        drpo = idx.get(("DRPO", seed))
        seed_ok = False
        if static and dspo and drpo:
            s_np = to_float(static.get("net_profit"))
            d_np = to_float(dspo.get("net_profit"))
            r_np = to_float(drpo.get("net_profit"))
            seed_ok = None not in (s_np, d_np, r_np) and r_np > d_np > s_np
        else:
            s_np = d_np = r_np = None
        per_seed.append({"seed": seed, "static_np": s_np, "dspo_np": d_np, "drpo_np": r_np, "ordering_ok": seed_ok})

    dspo_rows = [idx[("DSPO", seed)] for seed in seeds if ("DSPO", seed) in idx]
    drpo_rows = [idx[("DRPO", seed)] for seed in seeds if ("DRPO", seed) in idx]
    static_rows = [idx[("Static", seed)] for seed in seeds if ("Static", seed) in idx]

    def mean_metric(rs: List[Dict[str, str]], metric: str) -> Optional[float]:
        vals = [to_float(r.get(metric)) for r in rs]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    drpo_logs = [
        log_health(log_path(root, "DRPO", label, seed, folder_suffix, phase_tag), "DRPO")
        for seed in seeds
    ]
    dspo_logs = [
        log_health(log_path(root, "DSPO", label, seed, folder_suffix, phase_tag), "DSPO")
        for seed in seeds
    ]

    dspo_quit = mean_metric(dspo_rows, "quit_rate")
    drpo_quit = mean_metric(drpo_rows, "quit_rate")
    static_home = mean_metric(static_rows, "home_pickup_rate")
    dspo_home = mean_metric(dspo_rows, "home_pickup_rate")
    drpo_home = mean_metric(drpo_rows, "home_pickup_rate")
    dspo_profit = mean_metric(dspo_rows, "net_profit")
    drpo_profit = mean_metric(drpo_rows, "net_profit")
    static_profit = mean_metric(static_rows, "net_profit")

    all_complete = all(static is not None and dspo is not None and drpo is not None for static, dspo, drpo in [
        (idx.get(("Static", seed)), idx.get(("DSPO", seed)), idx.get(("DRPO", seed))) for seed in seeds
    ])
    ordering_all = all(bool(r["ordering_ok"]) for r in per_seed)
    ordering_count = sum(1 for r in per_seed if r["ordering_ok"])
    quit_ok = (
        dspo_quit is not None and drpo_quit is not None
        and dspo_quit <= 3.0 and drpo_quit <= 3.0
    )
    steering_ok = (
        static_home is not None and dspo_home is not None and drpo_home is not None
        and 0.15 <= dspo_home < static_home
        and 0.15 <= drpo_home < static_home
    )
    spo_ok = all(
        bool(h.get("spo_weight_positive"))
        and not bool(h.get("has_nan_or_inf_loss"))
        and int(h.get("spo_warning_count") or 0) <= 3
        for h in drpo_logs
    )

    return {
        "label": label,
        "max_price": max_price,
        "incentive_sens": incentive_sens,
        "all_complete": all_complete,
        "ordering_all": ordering_all,
        "ordering_count": ordering_count,
        "quit_ok": quit_ok,
        "steering_ok": steering_ok,
        "spo_ok": spo_ok,
        "candidate_ok": all_complete and ordering_all and quit_ok and steering_ok and spo_ok,
        "static_profit_mean": static_profit,
        "dspo_profit_mean": dspo_profit,
        "drpo_profit_mean": drpo_profit,
        "drpo_minus_dspo_mean": (
            drpo_profit - dspo_profit if drpo_profit is not None and dspo_profit is not None else None
        ),
        "dspo_minus_static_mean": (
            dspo_profit - static_profit if dspo_profit is not None and static_profit is not None else None
        ),
        "dspo_quit_mean": dspo_quit,
        "drpo_quit_mean": drpo_quit,
        "static_home_mean": static_home,
        "dspo_home_mean": dspo_home,
        "drpo_home_mean": drpo_home,
        "per_seed": json.dumps(per_seed, ensure_ascii=False),
        "drpo_log_health": json.dumps(drpo_logs, ensure_ascii=False),
        "dspo_log_health": json.dumps(dspo_logs, ensure_ascii=False),
    }


def candidate_sort_key(row: Dict[str, Any]) -> Tuple[int, int, float, float]:
    ok = 1 if row.get("candidate_ok") else 0
    ordering = int(row.get("ordering_count") or 0)
    drpo_margin = float(row.get("drpo_minus_dspo_mean") or -1e18)
    dspo_margin = float(row.get("dspo_minus_static_mean") or -1e18)
    return (ok, ordering, drpo_margin, dspo_margin)


def run_candidate(
    args: argparse.Namespace,
    root: Path,
    label: str,
    max_price: float,
    incentive_sens: float,
    seeds: Sequence[int],
    output_dir: Path,
    folder_suffix: str,
    phase_tag: str,
) -> Dict[str, Any]:
    cmd = build_runner_cmd(
        args=args,
        label=label,
        max_price=max_price,
        incentive_sens=incentive_sens,
        seeds=seeds,
        output_dir=str(output_dir.relative_to(root)),
        folder_suffix=folder_suffix,
        phase_tag=phase_tag,
    )

    print(f"[RUN] {phase_tag} {label}: max_price={max_price}, beta={incentive_sens}", flush=True)
    print(" ".join(cmd), flush=True)
    if args.dry_run:
        rows: List[Dict[str, str]] = []
    else:
        cp = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="ignore")
        if cp.returncode != 0:
            raise RuntimeError(f"Candidate {label} failed:\n{(cp.stdout or '')[-3000:]}")
        rows = read_csv(output_dir / "yanjiao_raw.csv")

    summary = evaluate_candidate(root, rows, label, max_price, incentive_sens, seeds, folder_suffix, phase_tag)
    summary["command"] = " ".join(cmd)
    return summary


def persist_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "label", "max_price", "incentive_sens", "candidate_ok", "ordering_all", "ordering_count",
        "quit_ok", "steering_ok", "spo_ok", "all_complete", "static_profit_mean",
        "dspo_profit_mean", "drpo_profit_mean", "drpo_minus_dspo_mean", "dspo_minus_static_mean",
        "dspo_quit_mean", "drpo_quit_mean", "static_home_mean", "dspo_home_mean",
        "drpo_home_mean", "per_seed", "drpo_log_health", "dspo_log_health", "command",
    ]
    write_csv(path, rows, fields)


def main() -> None:
    args = parse_args()
    root = root_dir()
    output_root = rel_or_abs(root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tuning_seeds": TUNING_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "static_pricing": {"price_home": STATIC_PRICE_HOME, "price_pp": STATIC_PRICE_PP},
        "episodes": args.episodes,
        "selection_rule": "candidate_ok first, then ordering count and mean DRPO-DSPO margin",
        "first_round": FIRST_ROUND,
        "second_round": SECOND_ROUND,
    }
    (output_root / "tuning_meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summaries: List[Dict[str, Any]] = []
    for label, max_price, beta in FIRST_ROUND:
        candidate_dir = output_root / label
        summaries.append(
            run_candidate(args, root, label, max_price, beta, TUNING_SEEDS, candidate_dir, args.folder_suffix, "TUNE")
        )
        persist_summary(output_root / "tuning_summary.csv", summaries)

    first_ok = [r for r in summaries if r.get("candidate_ok") and r["label"].startswith(("B_", "C_"))]
    if args.force_second_round or not first_ok:
        print("[INFO] Running second-round fine grid.", flush=True)
        existing_labels = {r["label"] for r in summaries}
        for label, max_price, beta in SECOND_ROUND:
            if label in existing_labels:
                continue
            candidate_dir = output_root / label
            summaries.append(
                run_candidate(args, root, label, max_price, beta, TUNING_SEEDS, candidate_dir, args.folder_suffix, "TUNE")
            )
            persist_summary(output_root / "tuning_summary.csv", summaries)

    ranked = sorted(summaries, key=candidate_sort_key, reverse=True)
    if not ranked:
        raise RuntimeError("No tuning candidates were evaluated.")

    selected = ranked[0]
    (output_root / "selected_candidate.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[SELECTED] {selected['label']} max_price={selected['max_price']} "
        f"beta={selected['incentive_sens']} candidate_ok={selected['candidate_ok']}",
        flush=True,
    )

    if args.skip_validation:
        print("[DONE] Skipped final validation.", flush=True)
        return

    validation_dir = rel_or_abs(root, args.validation_output_dir)
    validation_label = f"FINAL_{selected['label']}"
    validation = run_candidate(
        args=args,
        root=root,
        label=validation_label,
        max_price=float(selected["max_price"]),
        incentive_sens=float(selected["incentive_sens"]),
        seeds=VALIDATION_SEEDS,
        output_dir=validation_dir,
        folder_suffix=args.validation_folder_suffix,
        phase_tag="FINAL",
    )

    validation["validation_pass"] = (
        validation["all_complete"]
        and validation["ordering_count"] >= 2
        and (validation["drpo_profit_mean"] is not None)
        and (validation["dspo_profit_mean"] is not None)
        and (validation["static_profit_mean"] is not None)
        and validation["drpo_profit_mean"] > validation["dspo_profit_mean"] > validation["static_profit_mean"]
        and validation["quit_ok"]
        and validation["steering_ok"]
        and validation["spo_ok"]
    )
    (validation_dir / "final_selection.json").write_text(json.dumps({
        "selected_tuning_candidate": selected,
        "validation": validation,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    persist_summary(validation_dir / "validation_summary.csv", [validation])

    print(
        f"[DONE] validation_pass={validation['validation_pass']} "
        f"ordering_count={validation['ordering_count']}/3 "
        f"summary={validation_dir / 'validation_summary.csv'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
