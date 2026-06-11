#!/usr/bin/env python
"""Stage one-seed tuning runs for the Yanjiao 400 dispersed instance.

This wrapper deliberately avoids a full Cartesian product. It delegates actual
runs to run_yanjiao_experiments.py and records candidate-level metadata for the
project analyzer.
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


OOH_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = OOH_ROOT.parent
PLANNING_DIR = WORKSPACE_ROOT / ".planning" / "yanjiao400_dispersed_one_seed_tuning"
OUTPUT_ROOT_REL = Path("Experiments") / "analysis" / "yanjiao400_dispersed_one_seed_tuning"
OUTPUT_ROOT = OOH_ROOT / OUTPUT_ROOT_REL
MATRIX_PATH = PLANNING_DIR / "04-TUNING-MATRIX.csv"
STATE_PATH = PLANNING_DIR / "STATE.md"
DECISION_LOG_PATH = PLANNING_DIR / "DECISION-LOG.md"
VERIFICATION_PATH = PLANNING_DIR / "08-VERIFICATION.md"

YANJIAO_PREFIX = "yanjiao_dispersed_{n_passengers}_{seed}"
N_PASSENGERS = 400
DATA_SEED = 0
DATA_SEED_TEST = 1
FOLDER_SUFFIX = "_yanjiao400_one_seed"
RUNNER = "scripts/run_yanjiao_experiments.py"

STAGE_STRATEGIES = {
    "calibrate_no": ["No-pricing"],
    "tune_static": ["No-pricing", "Static-pricing"],
    "probe_params": ["DSPO", "DRPO"],
    "probe_data": ["No-pricing", "Static-pricing"],
    "probe_data_dynamic": ["DSPO", "DRPO"],
    "compare_dynamic": ["No-pricing", "Static-pricing", "DSPO", "DRPO"],
    "confirm": ["No-pricing", "Static-pricing", "DSPO", "DRPO"],
}

STRATEGY_ALGOS = {
    "No-pricing": "Baseline",
    "Static-pricing": "Baseline",
    "DSPO": "DSPO",
    "DRPO": "DRPO",
}

MATRIX_FIELDS = [
    "candidate_id",
    "stage",
    "strategy",
    "seed",
    "status",
    "command",
    "log_path",
    "output_dir",
    "home_util",
    "outside_option_util",
    "incentive_sens",
    "walk_distance_weight",
    "price_home",
    "price_pp",
    "max_price",
    "min_price",
    "yanjiao_prefix",
    "dspo_spo_loss_weight",
    "drpo_spo_loss_weight",
    "episodes",
    "notes",
]


@dataclass
class Candidate:
    candidate_id: str
    stage: str
    home_util: float
    outside_option_util: float
    incentive_sens: float
    walk_distance_weight: float
    price_home: float = 0.0
    price_pp: float = 0.0
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    yanjiao_prefix: str = YANJIAO_PREFIX
    dspo_spo_loss_weight: float = 0.0
    drpo_spo_loss_weight: Optional[float] = None
    source: str = "planned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staged Yanjiao 400 dispersed one-seed tuning wrapper")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stage", required=True, choices=sorted(STAGE_STRATEGIES))
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--allow-cpu", "--allow_cpu", dest="allow_cpu", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--max-candidates", type=int, default=0)
    p.add_argument("--candidate-id", action="append", default=[], help="Run only the named candidate id; repeatable.")
    p.add_argument("--python-executable", default=sys.executable)
    return p.parse_args()


def ensure_dirs() -> None:
    PLANNING_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not MATRIX_PATH.exists():
        write_csv(MATRIX_PATH, [], MATRIX_FIELDS)


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_rate(value: Any) -> Optional[float]:
    v = to_float(value)
    if v is None:
        return None
    return v / 100.0 if abs(v) > 1.0 else v


def normalize_quit_rate(value: Any) -> Optional[float]:
    """Yanjiao runner raw CSV stores quit_rate as the number before '%'."""
    v = to_float(value)
    if v is None:
        return None
    return v / 100.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_decision(message: str) -> None:
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with DECISION_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp}\n\n- {message}\n")


def update_state(stage: str, status: str, command_count: int) -> None:
    content = f"""# State

## Current Phase

Stage `{stage}`.

## Status

{status}

## Last Update

- Updated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Planned command groups: {command_count}
- Formal 30-seed runs: not used
- Final test seeds: not used

## Next Step

Run the next staged command from `03-COMMANDS.md`, or run analyze-only to rebuild reports from completed candidate outputs.
"""
    STATE_PATH.write_text(content, encoding="utf-8")


def update_verification(stage: str, dry_run: bool, command_count: int) -> None:
    content = f"""# Verification

## Guardrails

- Seed fixed to `20` unless explicitly overridden for a non-final one-seed diagnostic.
- 30 seeds not used.
- Final test seeds not used.
- Approved final lock not modified.
- Manuscript files not modified.
- Core DSPO/DRPO algorithm definitions not modified by this wrapper.

## Current Stage

- Stage: `{stage}`
- Dry run: `{dry_run}`
- Command groups planned/executed: `{command_count}`
- Result directory: `ooh_code/{OUTPUT_ROOT_REL.as_posix()}`

## Output Checks

- `04-TUNING-MATRIX.csv`: exists
- `05-RESULTS-RAW.csv`: produced by analyzer after completed runs
- `06-RESULTS-SUMMARY.csv`: produced by analyzer after completed runs
- `07-DECISION-REPORT.md`: produced by analyzer after completed runs
- `08-VERIFICATION.md`: updated
"""
    VERIFICATION_PATH.write_text(content, encoding="utf-8")


def load_raw_results() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for raw_path in OUTPUT_ROOT.glob("*/*/yanjiao_raw.csv"):
        for row in read_csv(raw_path):
            row["_raw_path"] = str(raw_path)
            rows.append(row)
    return rows


def score_a1(row: Dict[str, str]) -> float:
    home = normalize_rate(row.get("home_pickup_rate"))
    quit_rate = normalize_quit_rate(row.get("quit_rate"))
    served = normalize_rate(row.get("served_rate"))
    if home is None:
        return -999.0
    target_penalty = 0.0 if 0.95 <= home <= 0.99 else min(abs(home - 0.95), abs(home - 0.99)) * 10.0
    quit_penalty = (quit_rate or 0.0) * 0.5
    served_bonus = served or 0.0
    return served_bonus - target_penalty - quit_penalty


def select_a1_public_candidates(top_k: int) -> List[Candidate]:
    rows = [
        r for r in load_raw_results()
        if r.get("label") == "No-pricing" and str(r.get("candidate_id", "")).startswith("A1_")
    ]
    if not rows:
        return []
    calibrated = [
        r for r in rows
        if (normalize_rate(r.get("home_pickup_rate")) is not None
            and 0.95 <= normalize_rate(r.get("home_pickup_rate")) <= 0.99)
    ]
    if not calibrated:
        write_a1_closest(rows, top_k=5)
        return []
    calibrated.sort(key=score_a1, reverse=True)
    out: List[Candidate] = []
    seen = set()
    for row in calibrated:
        key = (
            row.get("home_util"),
            row.get("outside_option_util"),
            row.get("incentive_sens"),
            row.get("walk_distance_weight"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(Candidate(
            candidate_id=f"A1_SELECTED_{len(out) + 1:02d}",
            stage="tune_static",
            home_util=float(row.get("home_util", 1.6)),
            outside_option_util=float(row.get("outside_option_util", -1.5)),
            incentive_sens=float(row.get("incentive_sens", -0.25)),
            walk_distance_weight=float(row.get("walk_distance_weight", -0.002)),
            source="A1 results",
        ))
        if len(out) >= top_k:
            break
    return out


def write_a1_closest(rows: List[Dict[str, str]], top_k: int = 5) -> None:
    if not rows:
        return
    closest = sorted(rows, key=score_a1, reverse=True)[:top_k]
    fields = [
        "candidate_id",
        "home_util",
        "outside_option_util",
        "incentive_sens",
        "walk_distance_weight",
        "home_pickup_rate",
        "quit_rate",
        "served_rate",
        "net_profit",
        "log_path",
    ]
    path = PLANNING_DIR / "A1-CLOSEST-TOP5.csv"
    write_csv(path, closest, fields)
    append_decision(
        "Stage A1 did not find any No-pricing home_rate in [0.95,0.99]; "
        f"wrote closest Top 5 to {path} and blocked later stages."
    )


def select_a2_candidates(top_k: int, stage: str) -> List[Candidate]:
    rows = load_raw_results()
    by_candidate: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        cid = row.get("candidate_id", "")
        if not cid.startswith("A2_"):
            continue
        by_candidate.setdefault(cid, {})[row.get("label", "")] = row

    scored = []
    for cid, group in by_candidate.items():
        no = group.get("No-pricing")
        static = group.get("Static-pricing") or group.get("Static")
        if not no or not static:
            continue
        profit_no = to_float(no.get("net_profit"))
        profit_static = to_float(static.get("net_profit"))
        home_no = normalize_rate(no.get("home_pickup_rate"))
        home_static = normalize_rate(static.get("home_pickup_rate"))
        if profit_no is None or profit_static is None or home_no is None or home_static is None:
            continue
        quit_no = normalize_quit_rate(no.get("quit_rate")) or 0.0
        quit_static = normalize_quit_rate(static.get("quit_rate")) or 0.0
        pass_profit = profit_static > profit_no
        pass_home_order = home_no > home_static
        pass_home_range = 0.80 <= home_static <= 0.95
        pass_quit = quit_static <= quit_no + 0.01
        pass_a2 = pass_profit and pass_home_order and pass_home_range and pass_quit
        score = (
            (100000.0 if pass_a2 else 0.0)
            + (1000.0 if pass_profit else 0.0)
            + (500.0 if pass_home_order else 0.0)
            + (300.0 if pass_home_range else 0.0)
            + (100.0 if pass_quit else 0.0)
            + (profit_static - profit_no)
        )
        scored.append((score, static))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Candidate] = []
    for _, row in scored[:top_k]:
        out.append(Candidate(
            candidate_id=f"{'A3' if stage == 'compare_dynamic' else 'B'}_C{len(out) + 1:02d}",
            stage=stage,
            home_util=float(row.get("home_util", 1.6)),
            outside_option_util=float(row.get("outside_option_util", -1.5)),
            incentive_sens=float(row.get("incentive_sens", -0.25)),
            walk_distance_weight=float(row.get("walk_distance_weight", -0.002)),
            price_home=float(row.get("price_home", 2.0)),
            price_pp=float(row.get("price_pp", -5.0)),
            source="A2 results",
        ))
    return out


def select_a3_candidates(top_k: int) -> List[Candidate]:
    rows = load_raw_results()
    by_candidate: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        cid = row.get("candidate_id", "")
        if not cid.startswith("A3_"):
            continue
        label = "Static-pricing" if row.get("label") == "Static" else row.get("label", "")
        by_candidate.setdefault(cid, {})[label] = row

    scored = []
    for cid, group in by_candidate.items():
        required = ["No-pricing", "Static-pricing", "DSPO", "DRPO"]
        if any(label not in group for label in required):
            continue
        no, static, dspo, drpo = [group[label] for label in required]
        profit_no = to_float(no.get("net_profit"))
        profit_static = to_float(static.get("net_profit"))
        profit_dspo = to_float(dspo.get("net_profit"))
        profit_drpo = to_float(drpo.get("net_profit"))
        home_no = normalize_rate(no.get("home_pickup_rate"))
        home_static = normalize_rate(static.get("home_pickup_rate"))
        home_dspo = normalize_rate(dspo.get("home_pickup_rate"))
        home_drpo = normalize_rate(drpo.get("home_pickup_rate"))
        quit_dspo = normalize_rate(dspo.get("quit_rate"))
        quit_drpo = normalize_rate(drpo.get("quit_rate"))
        values = [profit_no, profit_static, profit_dspo, profit_drpo, home_no, home_static, home_dspo, home_drpo, quit_dspo, quit_drpo]
        if any(v is None for v in values):
            continue
        pass_profit = profit_drpo > profit_dspo > profit_static > profit_no
        pass_home = home_no > home_static > home_dspo > home_drpo
        pass_quit = quit_drpo <= quit_dspo + 0.03
        score = (profit_drpo - profit_dspo) + (100.0 if pass_profit else 0.0) + (50.0 if pass_home else 0.0) + (30.0 if pass_quit else 0.0)
        scored.append((score, drpo))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: List[Candidate] = []
    for _, row in scored[:top_k]:
        out.append(Candidate(
            candidate_id=f"B_C{len(out) + 1:02d}",
            stage="confirm",
            home_util=float(row.get("home_util", 1.6)),
            outside_option_util=float(row.get("outside_option_util", -1.5)),
            incentive_sens=float(row.get("incentive_sens", -0.25)),
            walk_distance_weight=float(row.get("walk_distance_weight", -0.002)),
            price_home=float(row.get("price_home", 2.0)),
            price_pp=float(row.get("price_pp", -5.0)),
            max_price=to_float(row.get("max_price")),
            min_price=to_float(row.get("min_price")),
            drpo_spo_loss_weight=to_float(row.get("drpo_spo_loss_weight")) or 0.05,
            source="A3 results",
        ))
    return out


def generate_candidates(stage: str, top_k: int, dry_run: bool) -> List[Candidate]:
    if stage == "calibrate_no":
        existing = load_raw_results()
        a1_done = [r for r in existing if r.get("label") == "No-pricing" and str(r.get("candidate_id", "")).startswith("A1_H")]
        if a1_done:
            calibrated = [
                r for r in a1_done
                if (normalize_rate(r.get("home_pickup_rate")) is not None
                    and 0.95 <= normalize_rate(r.get("home_pickup_rate")) <= 0.99)
            ]
            if not calibrated:
                candidates = []
                for hu in [1.0, 1.2, 1.4, 1.6]:
                    for walk in [-0.001, -0.0005, 0.0]:
                        candidates.append(Candidate(
                            candidate_id=f"A1_L{len(candidates) + 1:03d}",
                            stage=stage,
                            home_util=hu,
                            outside_option_util=-1.5,
                            incentive_sens=-0.25,
                            walk_distance_weight=walk,
                            source="A1 lower-home-util expansion",
                        ))
                return candidates
            top_home = sorted(a1_done, key=score_a1, reverse=True)[:max(1, min(2, top_k))]
            home_utils = []
            for row in top_home:
                hu = to_float(row.get("home_util"))
                if hu is not None and hu not in home_utils:
                    home_utils.append(hu)
            candidates = []
            for hu in home_utils:
                for outside in [-2.0, -1.5, -1.0]:
                    for walk in [-0.001, -0.002, -0.003]:
                        candidates.append(Candidate(
                            candidate_id=f"A1_R{len(candidates) + 1:02d}",
                            stage=stage,
                            home_util=hu,
                            outside_option_util=outside,
                            incentive_sens=-0.25,
                            walk_distance_weight=walk,
                            source="A1 refinement",
                        ))
            return candidates
        return [
            Candidate(f"A1_H{i + 1:02d}", stage, hu, -1.5, -0.25, -0.002, source="A1 home_util screen")
            for i, hu in enumerate([1.6, 2.0, 2.4, 2.8])
        ]

    if stage == "tune_static":
        public = select_a1_public_candidates(top_k)
        if not public:
            if not dry_run:
                raise RuntimeError(
                    "Stage A1 has no calibrated public choice candidate with "
                    "No-pricing home_rate in [0.95,0.99]. Stop here and inspect "
                    f"{PLANNING_DIR / 'A1-CLOSEST-TOP5.csv'}."
                )
            public = [Candidate("A1_DRY_01", stage, 2.0, -1.5, -0.25, -0.002, source="dry-run placeholder")]
        candidates = []
        for base in public[:top_k]:
            for price_home in [1, 2, 3, 4, 5, 6, 8]:
                for price_pp in [-1, -2, -3, -4, -5, -6, -8, -10]:
                    candidates.append(Candidate(
                        candidate_id=f"A2_C{len(candidates) + 1:03d}",
                        stage=stage,
                        home_util=base.home_util,
                        outside_option_util=base.outside_option_util,
                        incentive_sens=base.incentive_sens,
                        walk_distance_weight=base.walk_distance_weight,
                        price_home=float(price_home),
                        price_pp=float(price_pp),
                        source=base.candidate_id,
                    ))
        return candidates

    if stage == "probe_params":
        return [
            Candidate(
                candidate_id="P1",
                stage=stage,
                home_util=1.2,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.1,
                min_price=-2.6,
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="parameter probe near A3_M003",
            ),
            Candidate(
                candidate_id="P2",
                stage=stage,
                home_util=1.2,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.2,
                min_price=-2.7,
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="parameter probe near A3_M003",
            ),
            Candidate(
                candidate_id="P3",
                stage=stage,
                home_util=1.2,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.3,
                min_price=-2.8,
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="parameter probe near A3_M003",
            ),
            Candidate(
                candidate_id="P4",
                stage=stage,
                home_util=1.2,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.2,
                min_price=-2.7,
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.20,
                source="parameter probe near A3_M003 with stronger SPO weight",
            ),
        ]

    if stage == "probe_data":
        return [
            Candidate(
                candidate_id=f"D{i + 1}",
                stage=stage,
                home_util=home_util,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                yanjiao_prefix="yanjiao_het_home_mixed_mp_{n_passengers}_{seed}",
                source="data probe using existing het_home_mixed_mp generator variant",
            )
            for i, home_util in enumerate([1.2, 1.4, 1.6, 1.8])
        ]

    if stage == "probe_data_dynamic":
        return [
            Candidate(
                candidate_id="E1",
                stage=stage,
                home_util=1.4,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.2,
                min_price=-2.7,
                yanjiao_prefix="yanjiao_het_home_mixed_mp_{n_passengers}_{seed}",
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="D2 passed data A1/A2; dynamic probe",
            ),
            Candidate(
                candidate_id="E2",
                stage=stage,
                home_util=1.4,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.3,
                min_price=-2.8,
                yanjiao_prefix="yanjiao_het_home_mixed_mp_{n_passengers}_{seed}",
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="D2 passed data A1/A2; slightly stronger dynamic bounds",
            ),
            Candidate(
                candidate_id="E3",
                stage=stage,
                home_util=1.6,
                outside_option_util=-1.5,
                incentive_sens=-0.25,
                walk_distance_weight=-0.001,
                price_home=1.0,
                price_pp=-1.0,
                max_price=2.2,
                min_price=-2.7,
                yanjiao_prefix="yanjiao_het_home_mixed_mp_{n_passengers}_{seed}",
                dspo_spo_loss_weight=0.0,
                drpo_spo_loss_weight=0.10,
                source="D3 passed data A1/A2; dynamic probe",
            ),
        ]

    if stage == "compare_dynamic":
        candidates = select_a2_candidates(top_k, stage)
        if not candidates:
            if not dry_run:
                raise RuntimeError("No A2 results found. Run tune_static before compare_dynamic.")
            candidates = [Candidate("A3_DRY_01", stage, 2.0, -1.5, -0.25, -0.002, 2.0, -5.0, source="dry-run placeholder")]
        out = []
        for base in candidates[:top_k]:
            for max_price, min_price, note in [
                (1.5, -2.0, "moderate dynamic price bounds"),
                (1.0, -1.5, "narrow dynamic price bounds"),
                (2.0, -2.5, "medium dynamic price bounds"),
            ]:
                out.append(Candidate(
                    candidate_id=f"A3_M{len([c for c in out if c.candidate_id.startswith('A3_M')]) + 1:03d}",
                    stage=stage,
                    home_util=base.home_util,
                    outside_option_util=base.outside_option_util,
                    incentive_sens=base.incentive_sens,
                    walk_distance_weight=base.walk_distance_weight,
                    price_home=base.price_home,
                    price_pp=base.price_pp,
                    max_price=max_price,
                    min_price=min_price,
                    dspo_spo_loss_weight=0.0,
                    drpo_spo_loss_weight=0.10,
                    source=f"{base.candidate_id}; {note}",
                ))
            for spo in [0.02, 0.05, 0.10, 0.20]:
                out.append(Candidate(
                    candidate_id=f"A3_C{len([c for c in out if c.candidate_id.startswith('A3_C')]) + 1:03d}",
                    stage=stage,
                    home_util=base.home_util,
                    outside_option_util=base.outside_option_util,
                    incentive_sens=base.incentive_sens,
                    walk_distance_weight=base.walk_distance_weight,
                    price_home=base.price_home,
                    price_pp=base.price_pp,
                    dspo_spo_loss_weight=0.0,
                    drpo_spo_loss_weight=spo,
                    source=base.candidate_id,
                ))
        return out

    if stage == "confirm":
        candidates = select_a3_candidates(top_k)
        if not candidates:
            if not dry_run:
                raise RuntimeError("No A3 results found. Run compare_dynamic before confirm.")
            candidates = [Candidate("B_DRY_01", stage, 2.0, -1.5, -0.25, -0.002, 2.0, -5.0, drpo_spo_loss_weight=0.05, source="dry-run placeholder")]
        return candidates[:top_k]

    raise RuntimeError(f"Unsupported stage: {stage}")


def limit_candidates(candidates: List[Candidate], max_candidates: int) -> List[Candidate]:
    if max_candidates and max_candidates > 0:
        return candidates[:max_candidates]
    return candidates


def filter_candidates(candidates: List[Candidate], candidate_ids: List[str]) -> List[Candidate]:
    if not candidate_ids:
        return candidates
    wanted = set(candidate_ids)
    filtered = [c for c in candidates if c.candidate_id in wanted]
    missing = sorted(wanted - {c.candidate_id for c in filtered})
    if missing:
        available = ", ".join(c.candidate_id for c in candidates)
        raise RuntimeError(f"Unknown candidate id(s): {', '.join(missing)}. Available: {available}")
    return filtered


def candidate_output_rel(candidate: Candidate) -> Path:
    return OUTPUT_ROOT_REL / candidate.stage / candidate.candidate_id


def candidate_output_abs(candidate: Candidate) -> Path:
    return OOH_ROOT / candidate_output_rel(candidate)


def run_prefix(candidate: Candidate) -> str:
    return f"YJ400OS_{candidate.candidate_id}"


def strategy_log_path(candidate: Candidate, strategy: str, seed: int) -> Path:
    algo = STRATEGY_ALGOS[strategy]
    rid = f"{run_prefix(candidate)}_{N_PASSENGERS}_{strategy}_seed{seed}"
    return OOH_ROOT / "Experiments" / "Parcelpoint_py" / "pricing" / algo / f"{rid}{FOLDER_SUFFIX}" / str(seed) / "Logs" / "logfile.log"


def candidate_data_prefix(candidate: Candidate, seed: int) -> str:
    return candidate.yanjiao_prefix.format(n_passengers=N_PASSENGERS, passengers=N_PASSENGERS, seed=seed)


def candidate_data_exists(candidate: Candidate) -> bool:
    for seed in [DATA_SEED, DATA_SEED_TEST]:
        prefix = candidate_data_prefix(candidate, seed)
        if not (OOH_ROOT / "Environments" / "OOH" / "Beijing_Yanjiao" / f"{prefix}_coords.txt").exists():
            return False
        if not (OOH_ROOT / "Environments" / "OOH" / "Beijing_Yanjiao" / f"{prefix}_adjacency10.npy").exists():
            return False
    return True


def ensure_candidate_data(candidate: Candidate, args: argparse.Namespace) -> None:
    if candidate.yanjiao_prefix == YANJIAO_PREFIX or candidate_data_exists(candidate):
        return
    if candidate.yanjiao_prefix != "yanjiao_het_home_mixed_mp_{n_passengers}_{seed}":
        raise RuntimeError(
            f"Missing data for {candidate.yanjiao_prefix}; only het_home_mixed_mp can be generated automatically."
        )
    cmd = [
        args.python_executable,
        "scripts/generate_yanjiao_instance.py",
        "--passengers", str(N_PASSENGERS),
        "--mp", "100",
        "--seeds", str(DATA_SEED), str(DATA_SEED_TEST),
        "--variant", "het_home_mixed_mp",
    ]
    print("[DATA] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=OOH_ROOT, check=True)


def same_base_params(row: Dict[str, Any], candidate: Candidate, include_dynamic_bounds: bool = False) -> bool:
    checks = [
        ("yanjiao_prefix", candidate.yanjiao_prefix),
        ("home_util", candidate.home_util),
        ("outside_option_util", candidate.outside_option_util),
        ("incentive_sens", candidate.incentive_sens),
        ("walk_distance_weight", candidate.walk_distance_weight),
        ("price_home", candidate.price_home),
        ("price_pp", candidate.price_pp),
        ("dspo_spo_loss_weight", candidate.dspo_spo_loss_weight),
    ]
    if include_dynamic_bounds:
        checks.extend([
            ("max_price", candidate.max_price),
            ("min_price", candidate.min_price),
        ])
    for key, expected in checks:
        if key == "yanjiao_prefix":
            if (row.get(key) or YANJIAO_PREFIX) != expected:
                return False
            continue
        if expected is None:
            if row.get(key) not in (None, ""):
                return False
            continue
        value = to_float(row.get(key))
        if value is None or abs(value - expected) > 1e-9:
            return False
    return True


def reusable_compare_strategies(candidate: Candidate) -> set:
    if candidate.stage != "compare_dynamic":
        return set()
    compare_dir = OOH_ROOT / OUTPUT_ROOT_REL / "compare_dynamic"
    if not compare_dir.exists():
        return set()
    found = set()
    for raw_path in compare_dir.glob("*/yanjiao_raw.csv"):
        for row in read_csv(raw_path):
            strategy = row.get("strategy") or row.get("label")
            if strategy in {"No-pricing", "Static-pricing"} and same_base_params(row, candidate, include_dynamic_bounds=False):
                found.add(strategy)
            if strategy == "DSPO" and same_base_params(row, candidate, include_dynamic_bounds=True):
                found.add(strategy)
    return found


def candidate_strategies(candidate: Candidate) -> List[str]:
    if candidate.stage == "compare_dynamic":
        reusable = reusable_compare_strategies(candidate)
        if candidate.candidate_id.startswith("A3_M"):
            return [strategy for strategy in ["DSPO", "DRPO"] if strategy not in reusable]
        if {"No-pricing", "Static-pricing", "DSPO"}.issubset(reusable):
            return ["DRPO"]
    return STAGE_STRATEGIES[candidate.stage]


def build_command(candidate: Candidate, args: argparse.Namespace) -> List[str]:
    strategies = candidate_strategies(candidate)
    cmd = [
        args.python_executable,
        RUNNER,
        "--phase", "main",
        "--seeds", str(args.seed),
        "--episodes", str(args.episodes),
        "--eval_episodes", str(args.eval_episodes),
        "--gpu", str(args.gpu),
        "--strategies", *strategies,
        "--output_dir", candidate_output_rel(candidate).as_posix(),
        "--allow_existing_output_dir",
        "--persist_every_n", "1",
        "--folder_suffix", FOLDER_SUFFIX,
        "--run_prefix", run_prefix(candidate),
        "--n_passengers_override", str(N_PASSENGERS),
        "--yanjiao_prefix", candidate.yanjiao_prefix,
        "--home_util_override", str(candidate.home_util),
        "--outside_option_util_override", str(candidate.outside_option_util),
        "--incentive_sens_override", str(candidate.incentive_sens),
        "--walk_distance_weight_override", str(candidate.walk_distance_weight),
        "--static_price_home", str(candidate.price_home),
        "--static_price_pp", str(candidate.price_pp),
        "--dspo_spo_loss_weight", str(candidate.dspo_spo_loss_weight),
    ]
    if candidate.max_price is not None:
        cmd.extend(["--max_price_override", str(candidate.max_price)])
    if candidate.min_price is not None:
        cmd.extend(["--min_price_override", str(candidate.min_price)])
    if candidate.drpo_spo_loss_weight is not None:
        cmd.extend(["--drpo_spo_loss_weight", str(candidate.drpo_spo_loss_weight)])
    if args.allow_cpu:
        cmd.append("--allow_cpu")
    if args.dry_run:
        cmd.append("--dry_run")
    if args.skip_existing:
        cmd.append("--skip_existing")
    else:
        cmd.append("--no_skip_existing")
    return cmd


def matrix_rows_for(candidate: Candidate, args: argparse.Namespace, command: List[str], status: str) -> List[Dict[str, Any]]:
    rows = []
    for strategy in candidate_strategies(candidate):
        row = asdict(candidate)
        row.update({
            "strategy": strategy,
            "seed": args.seed,
            "status": status,
            "command": " ".join(command),
            "log_path": str(strategy_log_path(candidate, strategy, args.seed)),
            "output_dir": str(candidate_output_abs(candidate)),
            "episodes": args.episodes,
            "notes": candidate.source,
        })
        rows.append(row)
    return rows


def upsert_matrix(rows: List[Dict[str, Any]]) -> None:
    existing = read_csv(MATRIX_PATH)
    keyed = {
        (r.get("candidate_id"), r.get("stage"), r.get("strategy"), str(r.get("seed"))): r
        for r in existing
    }
    for row in rows:
        key = (row.get("candidate_id"), row.get("stage"), row.get("strategy"), str(row.get("seed")))
        keyed[key] = row
    ordered = list(keyed.values())
    ordered.sort(key=lambda r: (r.get("stage", ""), r.get("candidate_id", ""), r.get("strategy", "")))
    write_csv(MATRIX_PATH, ordered, MATRIX_FIELDS)


def annotate_raw(candidate: Candidate, raw_path: Path) -> None:
    rows = read_csv(raw_path)
    if not rows:
        return
    enriched = []
    for row in rows:
        row.update({
            "candidate_id": candidate.candidate_id,
            "stage": candidate.stage,
            "home_util": candidate.home_util,
            "outside_option_util": candidate.outside_option_util,
            "incentive_sens": candidate.incentive_sens,
            "walk_distance_weight": candidate.walk_distance_weight,
            "price_home": candidate.price_home,
            "price_pp": candidate.price_pp,
            "max_price": "" if candidate.max_price is None else candidate.max_price,
            "min_price": "" if candidate.min_price is None else candidate.min_price,
            "yanjiao_prefix": candidate.yanjiao_prefix,
            "dspo_spo_loss_weight": candidate.dspo_spo_loss_weight,
            "drpo_spo_loss_weight": "" if candidate.drpo_spo_loss_weight is None else candidate.drpo_spo_loss_weight,
        })
        enriched.append(row)
    fieldnames = list(enriched[0].keys())
    for row in enriched:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    write_csv(raw_path, enriched, fieldnames)


def run_analyzer(args: argparse.Namespace) -> None:
    cmd = [
        args.python_executable,
        "scripts/analyze_yanjiao400_dispersed_tuning.py",
        "--output_dir", OUTPUT_ROOT_REL.as_posix(),
        "--planning_dir", (Path("..") / PLANNING_DIR.relative_to(WORKSPACE_ROOT)).as_posix(),
    ]
    print("[ANALYZE] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=OOH_ROOT, check=True)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.stage == "confirm" and args.episodes != 200:
        print("[INFO] Confirmation stage requires max_episodes=200; overriding --episodes to 200.", flush=True)
        args.episodes = 200

    if args.analyze_only:
        run_analyzer(args)
        return

    candidates = generate_candidates(args.stage, args.top_k, args.dry_run)
    candidates = filter_candidates(candidates, args.candidate_id)
    candidates = limit_candidates(candidates, args.max_candidates)
    if not candidates:
        raise RuntimeError(f"No candidates generated for stage {args.stage}.")

    print(f"[INFO] stage={args.stage} candidates={len(candidates)} seed={args.seed} episodes={args.episodes}", flush=True)
    planned_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        command = build_command(candidate, args)
        planned_rows.extend(matrix_rows_for(candidate, args, command, "planned" if args.dry_run else "queued"))
    upsert_matrix(planned_rows)

    if args.dry_run:
        for candidate in candidates:
            command = build_command(candidate, args)
            print(f"\n[{candidate.candidate_id}]")
            print(" ".join(command), flush=True)
        append_decision(f"Dry-run generated {len(candidates)} candidate command groups for {args.stage}.")
        update_state(args.stage, "Dry-run completed; no formal experiment was run.", len(candidates))
        update_verification(args.stage, True, len(candidates))
        print(f"\n[DRY-RUN] Matrix: {MATRIX_PATH}")
        return

    completed = []
    for idx, candidate in enumerate(candidates, 1):
        ensure_candidate_data(candidate, args)
        command = build_command(candidate, args)
        out_dir = candidate_output_abs(candidate)
        out_dir.mkdir(parents=True, exist_ok=True)
        command_file = out_dir / "wrapper_command.txt"
        command_file.write_text(" ".join(command) + "\n", encoding="utf-8")
        meta = asdict(candidate)
        meta.update({
            "seed": args.seed,
            "episodes": args.episodes,
            "strategies": candidate_strategies(candidate),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        (out_dir / "candidate_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n[{idx}/{len(candidates)}] {candidate.candidate_id}", flush=True)
        print("[RUN] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=OOH_ROOT, check=True)
        raw_path = out_dir / "yanjiao_raw.csv"
        annotate_raw(candidate, raw_path)
        upsert_matrix(matrix_rows_for(candidate, args, command, "completed"))
        completed.append(candidate.candidate_id)

    append_decision(f"Completed stage {args.stage}: {', '.join(completed)}.")
    update_state(args.stage, f"Completed `{args.stage}` for {len(completed)} candidate command groups.", len(completed))
    update_verification(args.stage, False, len(completed))
    run_analyzer(args)


if __name__ == "__main__":
    main()
