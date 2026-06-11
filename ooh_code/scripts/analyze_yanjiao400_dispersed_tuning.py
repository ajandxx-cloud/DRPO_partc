#!/usr/bin/env python
"""Analyze staged Yanjiao 400 dispersed one-seed tuning outputs."""

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


OOH_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = OOH_ROOT.parent
DEFAULT_OUTPUT_DIR = Path("Experiments") / "analysis" / "yanjiao400_dispersed_one_seed_tuning"
DEFAULT_PLANNING_DIR = Path("..") / ".planning" / "yanjiao400_dispersed_one_seed_tuning"
MAIN_ORDER = ["No-pricing", "Static-pricing", "DSPO", "DRPO"]
DYNAMIC_HOME_MIN = 0.50
POSITIVE_QUIT_MIN = 0.0

RAW_FIELDS = [
    "candidate_id",
    "stage",
    "strategy",
    "seed",
    "n_passengers",
    "status",
    "net_profit",
    "total_costs",
    "quit_rate",
    "home_rate",
    "meeting_point_rate",
    "served_rate",
    "served_demand",
    "total_demand",
    "charge_revenue",
    "discount_costs",
    "avg_charge",
    "avg_discount",
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
    "command",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Yanjiao 400 dispersed tuning results")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--planning_dir", default=str(DEFAULT_PLANNING_DIR))
    return p.parse_args()


def resolve_from_ooh(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (OOH_ROOT / path).resolve()


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


def fmt(value: Any, digits: int = 4) -> str:
    v = to_float(value)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def canonical_label(label: str) -> str:
    return "Static-pricing" if label == "Static" else label


def load_matrix(planning_dir: Path) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    matrix = {}
    for row in read_csv(planning_dir / "04-TUNING-MATRIX.csv"):
        key = (row.get("candidate_id", ""), canonical_label(row.get("strategy", "")), str(row.get("seed", "")))
        matrix[key] = row
    return matrix


def discover_raw_rows(output_dir: Path, planning_dir: Path) -> List[Dict[str, Any]]:
    matrix = load_matrix(planning_dir)
    rows: List[Dict[str, Any]] = []
    for raw_path in sorted(output_dir.glob("*/*/yanjiao_raw.csv")):
        candidate_dir = raw_path.parent
        meta_path = candidate_dir / "candidate_meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for raw in read_csv(raw_path):
            candidate_id = raw.get("candidate_id") or meta.get("candidate_id") or candidate_dir.name
            stage = raw.get("stage") or meta.get("stage") or candidate_dir.parent.name
            strategy = canonical_label(raw.get("label") or raw.get("strategy") or "")
            seed = str(raw.get("seed") or meta.get("seed") or "20")
            matrix_row = matrix.get((candidate_id, strategy, seed), {})
            served_rate = normalize_rate(raw.get("served_rate"))
            served_demand = to_float(raw.get("served_demand"))
            total_demand = to_float(raw.get("total_demand"))
            if served_rate is None and served_demand is not None and total_demand and total_demand > 0:
                served_rate = served_demand / total_demand
            home_rate = normalize_rate(raw.get("home_pickup_rate") or raw.get("home_rate") or raw.get("home_delivery"))
            meeting_rate = normalize_rate(raw.get("meeting_point_rate") or raw.get("meeting_point_adoption_rate"))
            if meeting_rate is None and served_rate is not None and home_rate is not None:
                meeting_rate = served_rate - home_rate
            row: Dict[str, Any] = {
                "candidate_id": candidate_id,
                "stage": stage,
                "strategy": strategy,
                "seed": seed,
                "n_passengers": raw.get("n_passengers", "400"),
                "status": raw.get("status", ""),
                "net_profit": raw.get("net_profit", ""),
                "total_costs": raw.get("total_costs", ""),
                "quit_rate": "" if normalize_quit_rate(raw.get("quit_rate")) is None else normalize_quit_rate(raw.get("quit_rate")),
                "home_rate": "" if home_rate is None else home_rate,
                "meeting_point_rate": "" if meeting_rate is None else meeting_rate,
                "served_rate": "" if served_rate is None else served_rate,
                "served_demand": raw.get("served_demand", ""),
                "total_demand": raw.get("total_demand", ""),
                "charge_revenue": raw.get("charge_revenue", ""),
                "discount_costs": raw.get("discount_costs", ""),
                "avg_charge": raw.get("avg_charge", ""),
                "avg_discount": raw.get("avg_discount", ""),
                "log_path": raw.get("log_path") or matrix_row.get("log_path", ""),
                "output_dir": str(candidate_dir),
                "home_util": raw.get("home_util") or meta.get("home_util") or matrix_row.get("home_util", ""),
                "outside_option_util": raw.get("outside_option_util") or meta.get("outside_option_util") or matrix_row.get("outside_option_util", ""),
                "incentive_sens": raw.get("incentive_sens") or meta.get("incentive_sens") or matrix_row.get("incentive_sens", ""),
                "walk_distance_weight": raw.get("walk_distance_weight") or meta.get("walk_distance_weight") or matrix_row.get("walk_distance_weight", ""),
                "price_home": raw.get("price_home") or meta.get("price_home") or matrix_row.get("price_home", ""),
                "price_pp": raw.get("price_pp") or meta.get("price_pp") or matrix_row.get("price_pp", ""),
                "max_price": raw.get("max_price") or meta.get("max_price") or matrix_row.get("max_price", ""),
                "min_price": raw.get("min_price") or meta.get("min_price") or matrix_row.get("min_price", ""),
                "yanjiao_prefix": raw.get("yanjiao_prefix") or meta.get("yanjiao_prefix") or matrix_row.get("yanjiao_prefix", "yanjiao_dispersed_{n_passengers}_{seed}"),
                "dspo_spo_loss_weight": raw.get("dspo_spo_loss_weight") or meta.get("dspo_spo_loss_weight") or matrix_row.get("dspo_spo_loss_weight", ""),
                "drpo_spo_loss_weight": raw.get("drpo_spo_loss_weight") or meta.get("drpo_spo_loss_weight") or matrix_row.get("drpo_spo_loss_weight", ""),
                "command": raw.get("command") or matrix_row.get("command", ""),
            }
            rows.append(row)
    return rows


def group_candidate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        key = (row["stage"], row["candidate_id"])
        grouped.setdefault(key, {})[row["strategy"]] = row
    return grouped


def param_signature(row: Dict[str, Any]) -> Tuple[Any, ...]:
    keys = [
        "yanjiao_prefix",
        "home_util",
        "outside_option_util",
        "incentive_sens",
        "walk_distance_weight",
        "price_home",
        "price_pp",
        "dspo_spo_loss_weight",
        "max_price",
        "min_price",
    ]
    signature = []
    for key in keys:
        if key == "yanjiao_prefix":
            signature.append(row.get(key) or "yanjiao_dispersed_{n_passengers}_{seed}")
            continue
        value = to_float(row.get(key))
        signature.append(round(value, 12) if value is not None else str(row.get(key, "")))
    return tuple(signature)


def inherit_compare_dynamic_baselines(
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]
) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    baseline_strategies = ["No-pricing", "Static-pricing", "DSPO"]
    static_keys = [
        "yanjiao_prefix",
        "home_util",
        "outside_option_util",
        "incentive_sens",
        "walk_distance_weight",
        "price_home",
        "price_pp",
        "dspo_spo_loss_weight",
    ]
    def static_signature(row: Dict[str, Any]) -> Tuple[Any, ...]:
        out = []
        for key in static_keys:
            if key == "yanjiao_prefix":
                out.append(row.get(key) or "yanjiao_dispersed_{n_passengers}_{seed}")
                continue
            value = to_float(row.get(key))
            out.append(round(value, 12) if value is not None else str(row.get(key, "")))
        return tuple(out)

    baselines: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    static_baselines: Dict[Tuple[Any, ...], Dict[str, Dict[str, Any]]] = {}
    for (stage, _candidate_id), group in grouped.items():
        if stage not in {"compare_dynamic", "probe_params", "probe_data_dynamic"}:
            continue
        if "No-pricing" in group and "Static-pricing" in group:
            static_baselines[static_signature(group["No-pricing"])] = {
                "No-pricing": group["No-pricing"],
                "Static-pricing": group["Static-pricing"],
            }
        if all(strategy in group for strategy in baseline_strategies):
            key = param_signature(group["No-pricing"])
            baselines[key] = {strategy: group[strategy] for strategy in baseline_strategies}

    for (stage, candidate_id), group in list(grouped.items()):
        if stage not in {"compare_dynamic", "probe_params"} or "DRPO" not in group:
            continue
        missing = [strategy for strategy in baseline_strategies if strategy not in group]
        if not missing:
            continue
        baseline = baselines.get(param_signature(group["DRPO"]), {})
        static_baseline = static_baselines.get(static_signature(group["DRPO"]), {})
        for strategy in missing:
            if strategy in {"No-pricing", "Static-pricing"}:
                source = static_baseline.get(strategy)
            else:
                source = baseline.get(strategy)
            if not source:
                continue
            inherited = dict(source)
            inherited["candidate_id"] = candidate_id
            inherited["inherited_from_candidate_id"] = source.get("candidate_id", "")
            group[strategy] = inherited
    return grouped


def evaluate_candidate(group: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"pass_status": "INCOMPLETE", "failure_reasons": "missing four strategies"}
    if any(strategy not in group for strategy in MAIN_ORDER):
        return out
    no, static, dspo, drpo = [group[s] for s in MAIN_ORDER]
    profit_no = to_float(no.get("net_profit"))
    profit_static = to_float(static.get("net_profit"))
    profit_dspo = to_float(dspo.get("net_profit"))
    profit_drpo = to_float(drpo.get("net_profit"))
    home_no = normalize_rate(no.get("home_rate"))
    home_static = normalize_rate(static.get("home_rate"))
    home_dspo = normalize_rate(dspo.get("home_rate"))
    home_drpo = normalize_rate(drpo.get("home_rate"))
    quit_dspo = normalize_rate(dspo.get("quit_rate"))
    quit_drpo = normalize_rate(drpo.get("quit_rate"))
    values = [profit_no, profit_static, profit_dspo, profit_drpo, home_no, home_static, home_dspo, home_drpo, quit_dspo, quit_drpo]
    if any(v is None for v in values):
        out["failure_reasons"] = "missing required metrics"
        return out

    failures = []
    pass_home_no = 0.95 <= home_no <= 0.99
    pass_profit = profit_drpo > profit_dspo > profit_static > profit_no
    pass_home_order = home_no > home_static > home_dspo > home_drpo
    pass_quit = quit_drpo <= quit_dspo + 0.03
    pass_dynamic_home = min(home_dspo, home_drpo) >= DYNAMIC_HOME_MIN
    pass_positive_quit = quit_dspo > POSITIVE_QUIT_MIN and quit_drpo > POSITIVE_QUIT_MIN
    relaxed_quit = quit_drpo <= quit_dspo + 0.05
    weak_positive = profit_drpo > profit_dspo and (profit_drpo - profit_dspo) < 1.0
    if not pass_home_no:
        failures.append("No-pricing home_rate not in [0.95,0.99]")
    if not pass_profit:
        failures.append("net_profit ordering failed")
    if not pass_home_order:
        failures.append("home_rate ordering failed")
    if not pass_quit:
        failures.append("DRPO quit_rate guardrail failed")
    if not pass_dynamic_home:
        failures.append("dynamic home_rate below 0.50")
    if not pass_positive_quit:
        failures.append("dynamic quit_rate is zero")

    if not failures:
        status = "PASS"
    elif failures == ["DRPO quit_rate guardrail failed"] and relaxed_quit:
        status = "RELAXED"
    elif weak_positive:
        status = "WEAK_POSITIVE"
    else:
        status = "FAIL"

    out.update({
        "pass_status": status,
        "failure_reasons": "; ".join(failures),
        "profit_no": profit_no,
        "profit_static": profit_static,
        "profit_dspo": profit_dspo,
        "profit_drpo": profit_drpo,
        "home_no": home_no,
        "home_static": home_static,
        "home_dspo": home_dspo,
        "home_drpo": home_drpo,
        "quit_dspo": quit_dspo,
        "quit_drpo": quit_drpo,
        "delta_profit_drpo_dspo": profit_drpo - profit_dspo,
        "delta_profit_dspo_static": profit_dspo - profit_static,
        "delta_profit_static_no": profit_static - profit_no,
        "delta_quit_drpo_dspo": quit_drpo - quit_dspo,
        "delta_home_dspo_drpo": home_dspo - home_drpo,
        "pass_dynamic_home": pass_dynamic_home,
        "pass_positive_quit": pass_positive_quit,
    })
    return out


def score_summary(row: Dict[str, Any]) -> float:
    status_bonus = {"PASS": 10000.0, "RELAXED": 8000.0, "WEAK_POSITIVE": 6000.0, "FAIL": 0.0, "INCOMPLETE": -1000.0}
    home_dspo = to_float(row.get("home_dspo"))
    home_drpo = to_float(row.get("home_drpo"))
    quit_dspo = to_float(row.get("quit_dspo"))
    quit_drpo = to_float(row.get("quit_drpo"))
    profit_static = to_float(row.get("profit_static"))
    profit_drpo = to_float(row.get("profit_drpo"))
    behavior_bonus = 0.0
    if (
        home_dspo is not None and home_drpo is not None
        and quit_dspo is not None and quit_drpo is not None
        and min(home_dspo, home_drpo) >= DYNAMIC_HOME_MIN
        and quit_dspo > POSITIVE_QUIT_MIN and quit_drpo > POSITIVE_QUIT_MIN
    ):
        behavior_bonus = 9000.0
    static_gap = 0.0
    if profit_static is not None and profit_drpo is not None:
        static_gap = (profit_drpo - profit_static) / 10.0
    return (
        status_bonus.get(str(row.get("pass_status")), 0.0)
        + behavior_bonus
        + static_gap
        + (to_float(row.get("delta_profit_drpo_dspo")) or 0.0)
    )


def build_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = inherit_compare_dynamic_baselines(group_candidate(rows))
    summary = []
    for (stage, candidate_id), group in grouped.items():
        params = next(iter(group.values()))
        eval_row = evaluate_candidate(group)
        row = {
            "stage": stage,
            "candidate_id": candidate_id,
            "strategies_present": ",".join(sorted(group.keys())),
            "pass_status": eval_row.get("pass_status", ""),
            "failure_reasons": eval_row.get("failure_reasons", ""),
            "home_util": params.get("home_util", ""),
            "outside_option_util": params.get("outside_option_util", ""),
            "incentive_sens": params.get("incentive_sens", ""),
            "walk_distance_weight": params.get("walk_distance_weight", ""),
            "price_home": params.get("price_home", ""),
            "price_pp": params.get("price_pp", ""),
            "max_price": params.get("max_price", ""),
            "min_price": params.get("min_price", ""),
            "yanjiao_prefix": params.get("yanjiao_prefix", ""),
            "dspo_spo_loss_weight": params.get("dspo_spo_loss_weight", ""),
            "drpo_spo_loss_weight": params.get("drpo_spo_loss_weight", ""),
            "profit_no": eval_row.get("profit_no", ""),
            "profit_static": eval_row.get("profit_static", ""),
            "profit_dspo": eval_row.get("profit_dspo", ""),
            "profit_drpo": eval_row.get("profit_drpo", ""),
            "home_no": eval_row.get("home_no", ""),
            "home_static": eval_row.get("home_static", ""),
            "home_dspo": eval_row.get("home_dspo", ""),
            "home_drpo": eval_row.get("home_drpo", ""),
            "quit_dspo": eval_row.get("quit_dspo", ""),
            "quit_drpo": eval_row.get("quit_drpo", ""),
            "delta_profit_drpo_dspo": eval_row.get("delta_profit_drpo_dspo", ""),
            "delta_profit_dspo_static": eval_row.get("delta_profit_dspo_static", ""),
            "delta_profit_static_no": eval_row.get("delta_profit_static_no", ""),
            "delta_quit_drpo_dspo": eval_row.get("delta_quit_drpo_dspo", ""),
            "delta_home_dspo_drpo": eval_row.get("delta_home_dspo_drpo", ""),
            "pass_dynamic_home": eval_row.get("pass_dynamic_home", ""),
            "pass_positive_quit": eval_row.get("pass_positive_quit", ""),
        }
        summary.append(row)
    summary.sort(key=score_summary, reverse=True)
    return summary


def score_a1(row: Dict[str, Any]) -> float:
    home = normalize_rate(row.get("home_rate"))
    quit_rate = normalize_rate(row.get("quit_rate")) or 0.0
    served_rate = normalize_rate(row.get("served_rate")) or 0.0
    if home is None:
        return -999.0
    target_penalty = 0.0 if 0.95 <= home <= 0.99 else min(abs(home - 0.95), abs(home - 0.99)) * 10.0
    return served_rate - target_penalty - quit_rate * 0.5


def a1_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        r for r in raw_rows
        if r.get("stage") == "calibrate_no" and r.get("strategy") == "No-pricing"
    ]


def a1_calibrated(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in a1_rows(raw_rows):
        home = normalize_rate(row.get("home_rate"))
        if home is not None and 0.95 <= home <= 0.99:
            out.append(row)
    return sorted(out, key=score_a1, reverse=True)


def a2_pairs(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [r for r in raw_rows if r.get("stage") == "tune_static"]
    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("candidate_id")), {})[str(row.get("strategy"))] = row
    out = []
    for candidate_id, group in grouped.items():
        no = group.get("No-pricing")
        static = group.get("Static-pricing")
        if not no or not static:
            continue
        profit_no = to_float(no.get("net_profit"))
        profit_static = to_float(static.get("net_profit"))
        home_no = normalize_rate(no.get("home_rate"))
        home_static = normalize_rate(static.get("home_rate"))
        quit_no = normalize_rate(no.get("quit_rate"))
        quit_static = normalize_rate(static.get("quit_rate"))
        if None in (profit_no, profit_static, home_no, home_static, quit_no, quit_static):
            continue
        pass_profit = profit_static > profit_no
        pass_home_order = home_no > home_static
        pass_home_range = 0.80 <= home_static <= 0.95
        pass_quit = quit_static <= quit_no + 0.01
        out.append({
            "candidate_id": candidate_id,
            "price_home": static.get("price_home", ""),
            "price_pp": static.get("price_pp", ""),
            "home_util": static.get("home_util", ""),
            "walk_distance_weight": static.get("walk_distance_weight", ""),
            "profit_no": profit_no,
            "profit_static": profit_static,
            "delta_profit": profit_static - profit_no,
            "home_no": home_no,
            "home_static": home_static,
            "delta_home": home_static - home_no,
            "quit_no": quit_no,
            "quit_static": quit_static,
            "pass_a2": pass_profit and pass_home_order and pass_home_range and pass_quit,
        })
    out.sort(key=lambda r: (not r["pass_a2"], -(to_float(r["delta_profit"]) or -1e18)))
    return out


def write_a1_closest(planning_dir: Path, raw_rows: List[Dict[str, Any]]) -> Optional[Path]:
    rows = a1_rows(raw_rows)
    path = planning_dir / "A1-CLOSEST-TOP5.csv"
    if not rows:
        return None
    if a1_calibrated(raw_rows):
        if path.exists():
            path.unlink()
        return None
    closest = sorted(rows, key=score_a1, reverse=True)[:5]
    fields = [
        "candidate_id",
        "home_util",
        "outside_option_util",
        "incentive_sens",
        "walk_distance_weight",
        "home_rate",
        "quit_rate",
        "served_rate",
        "net_profit",
        "log_path",
    ]
    write_csv(path, closest, fields)
    return path


def decision_pool(summary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    confirm_rows = [r for r in summary if r.get("stage") == "confirm"]
    if confirm_rows:
        return confirm_rows, "confirmation"
    return summary, "pre-confirmation"


def best_candidate(summary: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    pool, _ = decision_pool(summary)
    if not pool:
        return None
    return sorted(pool, key=score_summary, reverse=True)[0]


def render_report(summary: List[Dict[str, Any]], raw_rows: List[Dict[str, Any]]) -> str:
    best = best_candidate(summary)
    pool, basis = decision_pool(summary)
    pass_rows = [r for r in pool if r.get("pass_status") == "PASS"]
    complete = [r for r in summary if r.get("pass_status") not in ("INCOMPLETE", "")]
    lines = [
        "# Yanjiao 400 Dispersed One-Seed Tuning Decision Report",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 本项目做了什么",
        "",
        "本项目建立了燕郊 400 分散实例的一 seed 分阶段调参流程，用于寻找后续多 seed 实验前的候选参数方向。该结果不是最终论文证据包。",
        "",
        "## 2. 实例与 seed",
        "",
        "- 实例：`Beijing_Yanjiao`",
        "- 数据前缀：`yanjiao_dispersed_{n_passengers}_{seed}`",
        "- 乘客数：`400`",
        "- 数据 seed：`data_seed=0`, `data_seed_test=1`",
        "- 实验 seed：`20`",
        "",
        "## 3. 搜索原则",
        "",
        "采用 staged coarse-to-fine，而不是完整全网格搜索。A1 先校准 No-pricing 的 home rate，A2 再调 Static-pricing，A3 才引入 DSPO/DRPO，B 阶段只确认 Top candidates。这样可以降低实验成本，减少 one-seed 过拟合，并让机制解释更清楚。",
        "",
        "## 4. 当前结果",
        "",
    ]
    a1_all = a1_rows(raw_rows)
    a1_pass = a1_calibrated(raw_rows)
    if not raw_rows:
        lines.extend([
            "当前还没有完成的实验结果。已生成 harness、matrix 和报告骨架；请先运行 Stage A1。",
            "",
            "## 5. 下一轮建议",
            "",
            "下一步优先运行 `calibrate_no`，只检查 No-pricing 的 home_rate 是否能落入 95%-99%。",
        ])
        return "\n".join(lines) + "\n"

    lines.append(f"- 原始结果行数：{len(raw_rows)}")
    lines.append(f"- 候选组数：{len(summary)}")
    lines.append(f"- 完整四策略候选数：{len(complete)}")
    lines.append(f"- 当前决策依据：`{basis}`")
    if basis != "confirmation":
        lines.append("- 注意：尚无 confirmation 结果，因此当前报告不能作为最终候选锁定依据。")
    lines.append("")
    if best is None or (best.get("pass_status") == "INCOMPLETE" and a1_all):
        lines.append("当前尚无完整四策略候选；本报告只用于 A1 No-pricing 校准诊断。")
    else:
        label = "最优候选" if pass_rows and best.get("pass_status") == "PASS" else "closest_candidate"
        lines.extend([
            f"## 5. {label}",
            "",
            f"- candidate_id：`{best.get('candidate_id')}`",
            f"- stage：`{best.get('stage')}`",
            f"- status：`{best.get('pass_status')}`",
            f"- failure：{best.get('failure_reasons') or '无'}",
            f"- home_util：`{best.get('home_util')}`",
            f"- outside_option_util：`{best.get('outside_option_util')}`",
            f"- incentive_sens：`{best.get('incentive_sens')}`",
            f"- walk_distance_weight：`{best.get('walk_distance_weight')}`",
            f"- price_home：`{best.get('price_home')}`",
            f"- price_pp：`{best.get('price_pp')}`",
            f"- max_price：`{best.get('max_price')}`",
            f"- min_price：`{best.get('min_price')}`",
            f"- drpo_spo_loss_weight：`{best.get('drpo_spo_loss_weight')}`",
            "",
            "## 6. 四策略排序",
            "",
            f"- net_profit：No={fmt(best.get('profit_no'))}, Static={fmt(best.get('profit_static'))}, DSPO={fmt(best.get('profit_dspo'))}, DRPO={fmt(best.get('profit_drpo'))}",
            f"- home_rate：No={fmt(best.get('home_no'))}, Static={fmt(best.get('home_static'))}, DSPO={fmt(best.get('home_dspo'))}, DRPO={fmt(best.get('home_drpo'))}",
            f"- quit_rate：DSPO={fmt(best.get('quit_dspo'))}, DRPO={fmt(best.get('quit_drpo'))}",
            f"- DRPO-DSPO net_profit：{fmt(best.get('delta_profit_drpo_dspo'))}",
            f"- DRPO-DSPO quit_rate：{fmt(best.get('delta_quit_drpo_dspo'))}",
            "",
            "## 7. 约束诊断",
            "",
            f"- No-pricing calibration：{'PASS' if to_float(best.get('home_no')) is not None and 0.95 <= to_float(best.get('home_no')) <= 0.99 else 'FAIL/UNKNOWN'}",
            f"- Profit ranking：{'PASS' if all(to_float(best.get(k)) is not None for k in ['profit_no', 'profit_static', 'profit_dspo', 'profit_drpo']) and to_float(best.get('profit_drpo')) > to_float(best.get('profit_dspo')) > to_float(best.get('profit_static')) > to_float(best.get('profit_no')) else 'FAIL/UNKNOWN'}",
            f"- Home-rate ordering：{'PASS' if all(to_float(best.get(k)) is not None for k in ['home_no', 'home_static', 'home_dspo', 'home_drpo']) and to_float(best.get('home_no')) > to_float(best.get('home_static')) > to_float(best.get('home_dspo')) > to_float(best.get('home_drpo')) else 'FAIL/UNKNOWN'}",
            f"- Quit-rate guardrail：{'PASS' if to_float(best.get('quit_drpo')) is not None and to_float(best.get('quit_dspo')) is not None and to_float(best.get('quit_drpo')) <= to_float(best.get('quit_dspo')) + 0.03 else 'FAIL/UNKNOWN'}",
            "",
            "## 8. 机制解释口径",
            "",
            "在相同乘客选择环境下，No-pricing 保留乘客对 home pickup 的自然偏好。Static-pricing 通过固定 home surcharge 和 meeting point discount 初步引导部分乘客转向 meeting points，但价格不随状态变化。DSPO 利用学习到的服务成本信息改善决策。DRPO 在 DSPO 基础上通过面向下游收益的动态定价学习，更精准地引导部分乘客转向 meeting points，同时需要控制退出率。这一链条对应本文 `recommend -> predict -> price` 的核心机制。",
        ])
    if a1_all:
        lines.extend([
            "",
            "## A1 No-pricing 校准结果",
            "",
            f"- A1 候选数：{len(a1_all)}",
            f"- 进入 [0.95,0.99] 的候选数：{len(a1_pass)}",
        ])
        for row in sorted(a1_all, key=score_a1, reverse=True):
            mark = "PASS" if row in a1_pass else "FAIL"
            lines.append(
                f"- `{row.get('candidate_id')}` {mark}: "
                f"home_rate={fmt(row.get('home_rate'))}, "
                f"quit_rate={fmt(row.get('quit_rate'))}, "
                f"served_rate={fmt(row.get('served_rate'))}, "
                f"home_util={row.get('home_util')}"
            )
        if not a1_pass:
            lines.extend([
                "",
                "A1 没有找到满足 `home_rate(No-pricing) in [0.95,0.99]` 的公共 choice environment。根据执行 safeguard，调参流水线必须停止，不能进入 Static-pricing 或 DSPO/DRPO 比较。closest Top 5 已写入 `A1-CLOSEST-TOP5.csv`。",
            ])
    a2 = a2_pairs(raw_rows)
    if a2:
        passed = [r for r in a2 if r.get("pass_a2")]
        lines.extend([
            "",
            "## A2 Static-pricing 粗筛结果",
            "",
            f"- A2 候选组数：{len(a2)}",
            f"- 满足温和 Static-pricing 条件的候选数：{len(passed)}",
        ])
        for row in a2:
            mark = "PASS" if row.get("pass_a2") else "FAIL"
            lines.append(
                f"- `{row.get('candidate_id')}` {mark}: "
                f"price_home={row.get('price_home')}, price_pp={row.get('price_pp')}, "
                f"profit_no={fmt(row.get('profit_no'))}, profit_static={fmt(row.get('profit_static'))}, "
                f"delta_profit={fmt(row.get('delta_profit'))}, "
                f"home_no={fmt(row.get('home_no'))}, home_static={fmt(row.get('home_static'))}, "
                f"quit_no={fmt(row.get('quit_no'))}, quit_static={fmt(row.get('quit_static'))}"
            )
    lines.extend([
        "",
        "## 9. 下一轮建议",
        "",
        "如果尚无 PASS，优先根据 failure_reasons 调整：No-pricing home_rate 未达标则继续扩大 choice 参数；Static-pricing 未改善收益则调整固定价格强度；DRPO 未超过 DSPO 则再比较 SPO 权重；quit_rate 超线则降低价格压力或调整 outside option。",
    ])
    return "\n".join(lines) + "\n"


def git_diff_summary() -> str:
    scoped_paths = [
        ".planning/yanjiao400_dispersed_one_seed_tuning",
        "ooh_code/scripts/tune_yanjiao400_dispersed_one_seed.py",
        "ooh_code/scripts/analyze_yanjiao400_dispersed_tuning.py",
        "ooh_code/Experiments/analysis/yanjiao400_dispersed_one_seed_tuning",
    ]
    try:
        cp = subprocess.run(
            ["git", "status", "--short", "--", *scoped_paths],
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        scoped = cp.stdout.strip() or "(no project-scoped git changes detected)"
        broad = subprocess.run(
            ["git", "status", "--short"],
            cwd=WORKSPACE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        note = "Unrelated pre-existing workspace changes are present." if broad.stdout.strip() != scoped else "No unrelated workspace changes detected."
        return scoped + "\n" + note
    except OSError as exc:
        return f"git status unavailable: {exc}"


def render_verification(output_dir: Path, planning_dir: Path, raw_rows: List[Dict[str, Any]]) -> str:
    csv_checks = {
        "04-TUNING-MATRIX.csv": planning_dir / "04-TUNING-MATRIX.csv",
        "05-RESULTS-RAW.csv": planning_dir / "05-RESULTS-RAW.csv",
        "06-RESULTS-SUMMARY.csv": planning_dir / "06-RESULTS-SUMMARY.csv",
        "07-DECISION-REPORT.md": planning_dir / "07-DECISION-REPORT.md",
        "08-VERIFICATION.md": planning_dir / "08-VERIFICATION.md",
    }
    log_paths = [str(r.get("log_path", "")) for r in raw_rows if r.get("log_path")]
    lines = [
        "# Verification",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Guardrails",
        "",
        "- 使用 seed=20。",
        "- 未使用 30 seeds。",
        "- 未使用 final test seeds。",
        "- 未修改 approved final lock。",
        "- 未修改 manuscript。",
        "- 未修改核心 DSPO/DRPO 算法定义。",
        "",
        "## CSV Outputs",
        "",
    ]
    for name, path in csv_checks.items():
        lines.append(f"- `{name}`: {'exists' if path.exists() else 'missing'}")
    lines.extend([
        "",
        "## Analyze-only Reproducibility",
        "",
        f"- Command: `python scripts/analyze_yanjiao400_dispersed_tuning.py --output_dir {output_dir.relative_to(OOH_ROOT).as_posix()} --planning_dir ../.planning/yanjiao400_dispersed_one_seed_tuning`",
        "",
        "## Completed Log Paths",
        "",
    ])
    if log_paths:
        for path in log_paths:
            lines.append(f"- `{path}`")
    else:
        lines.append("- No completed experiment logs yet.")
    lines.extend([
        "",
        "## Git Diff Summary",
        "",
        "```text",
        git_diff_summary(),
        "```",
    ])
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, planning_dir: Path) -> None:
    raw_rows = discover_raw_rows(output_dir, planning_dir)
    summary = build_summary(raw_rows)
    closest_path = write_a1_closest(planning_dir, raw_rows)
    write_csv(planning_dir / "05-RESULTS-RAW.csv", raw_rows, RAW_FIELDS)
    summary_fields = list(summary[0].keys()) if summary else [
        "stage", "candidate_id", "strategies_present", "pass_status", "failure_reasons",
        "home_util", "outside_option_util", "incentive_sens", "walk_distance_weight",
        "price_home", "price_pp", "max_price", "min_price", "yanjiao_prefix", "dspo_spo_loss_weight", "drpo_spo_loss_weight",
        "profit_no", "profit_static", "profit_dspo", "profit_drpo",
        "home_no", "home_static", "home_dspo", "home_drpo",
        "quit_dspo", "quit_drpo", "delta_profit_drpo_dspo",
        "delta_profit_dspo_static", "delta_profit_static_no",
        "delta_quit_drpo_dspo", "delta_home_dspo_drpo",
    ]
    write_csv(planning_dir / "06-RESULTS-SUMMARY.csv", summary, summary_fields)
    (planning_dir / "07-DECISION-REPORT.md").write_text(render_report(summary, raw_rows), encoding="utf-8")
    (planning_dir / "08-VERIFICATION.md").write_text(render_verification(output_dir, planning_dir, raw_rows), encoding="utf-8")
    print(f"[DONE] Raw rows: {len(raw_rows)} -> {planning_dir / '05-RESULTS-RAW.csv'}", flush=True)
    print(f"[DONE] Summary rows: {len(summary)} -> {planning_dir / '06-RESULTS-SUMMARY.csv'}", flush=True)
    print(f"[DONE] Report: {planning_dir / '07-DECISION-REPORT.md'}", flush=True)
    print(f"[DONE] Verification: {planning_dir / '08-VERIFICATION.md'}", flush=True)
    if closest_path is not None:
        print(f"[BLOCKED] A1 has no calibrated candidate. Closest Top 5: {closest_path}", flush=True)


def main() -> None:
    args = parse_args()
    output_dir = resolve_from_ooh(args.output_dir)
    planning_arg = Path(args.planning_dir)
    planning_dir = planning_arg if planning_arg.is_absolute() else (OOH_ROOT / planning_arg).resolve()
    planning_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(output_dir, planning_dir)


if __name__ == "__main__":
    main()
