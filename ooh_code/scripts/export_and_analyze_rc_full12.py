#!/usr/bin/env python
import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REQUIRED_FILES = [
    "stage1_raw.csv",
    "stage2_raw.csv",
    "stage1_summary_enhanced.csv",
    "stage2_summary_enhanced.csv",
    "stage2_guardrail_ranking.csv",
    "final_recommendations.csv",
    "validation_report.txt",
]

BUSINESS_FACTORS = {
    "outside_option_util",
    "incentive_sens",
    "home_util",
    "k",
    "revenue",
    "fuel_cost",
    "home_failure",
}

ALGO_FACTORS = {
    "learning_rate",
    "batch_size",
    "spo_warmup_episodes",
    "spo_rampup_episodes",
    "spo_loss_weight",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wait for RC full12 OAT completion, export all CSV rows, and generate management insights.")
    p.add_argument("--output_dir", required=True, help="OAT output directory.")
    p.add_argument("--wait", action="store_true", help="Wait until required files are complete.")
    p.add_argument("--poll_sec", type=int, default=60, help="Polling interval when --wait is set.")
    p.add_argument("--timeout_min", type=int, default=0, help="Timeout in minutes (0 means no timeout).")
    p.add_argument("--export_name", default="", help="Output filename for merged all-results CSV.")
    p.add_argument("--insights_name", default="management_insights.csv", help="Output filename for insights CSV.")
    p.add_argument("--summary_name", default="management_summary.txt", help="Output filename for short text summary.")
    return p.parse_args()


def to_float(x: object) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_bool(x: object) -> bool:
    return str(x).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def validation_complete(output_dir: Path) -> bool:
    vpath = output_dir / "validation_report.txt"
    if not vpath.exists():
        return False
    txt = vpath.read_text(encoding="utf-8", errors="ignore")
    return ("missing_stage1_runs=0" in txt) and ("missing_stage2_runs=0" in txt)


def is_complete(output_dir: Path) -> bool:
    for f in REQUIRED_FILES:
        if not (output_dir / f).exists():
            return False
    return validation_complete(output_dir)


def wait_for_completion(output_dir: Path, poll_sec: int, timeout_min: int) -> None:
    start = time.time()
    timeout_sec = 0 if timeout_min <= 0 else timeout_min * 60
    while True:
        if is_complete(output_dir):
            return
        if timeout_sec > 0 and (time.time() - start) > timeout_sec:
            raise TimeoutError(f"Timeout waiting for completion: {output_dir}")
        time.sleep(max(5, poll_sec))


def merge_all_csvs(output_dir: Path, export_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if export_name.strip():
        out_name = export_name
    else:
        out_name = f"all_results_merged_{ts}.csv"
    out_path = output_dir / out_name

    csv_files = sorted(
        [p for p in output_dir.glob("*.csv") if p.name not in {out_name, "management_insights.csv"}],
        key=lambda p: p.name,
    )
    rows: List[Dict[str, object]] = []
    all_fields = {"source_csv", "source_row"}
    for csv_path in csv_files:
        data = read_csv(csv_path)
        for i, r in enumerate(data, 1):
            row: Dict[str, object] = {"source_csv": csv_path.name, "source_row": i}
            row.update(r)
            rows.append(row)
            all_fields.update(r.keys())

    ordered_fields = ["source_csv", "source_row"] + sorted(f for f in all_fields if f not in {"source_csv", "source_row"})
    write_csv(out_path, rows, ordered_fields)
    return out_path


def classify_factor(factor: str) -> str:
    if factor in BUSINESS_FACTORS:
        return "business"
    if factor in ALGO_FACTORS:
        return "algorithm"
    return "other"


def top_k(rows: List[Dict[str, str]], predicate, k: int = 3) -> List[Dict[str, str]]:
    filtered = [r for r in rows if predicate(r)]
    filtered.sort(key=lambda r: (to_float(r.get("primary_gain_for_ranking")) or -1e30), reverse=True)
    return filtered[:k]


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def build_management_insights(output_dir: Path, insights_name: str, summary_name: str) -> Tuple[Path, Path]:
    ranking = read_csv(output_dir / "stage2_guardrail_ranking.csv")
    recs = read_csv(output_dir / "final_recommendations.csv")
    sens = read_csv(output_dir / "sensitivity_scores.csv") if (output_dir / "sensitivity_scores.csv").exists() else []

    rec_map = {r.get("factor", ""): r for r in recs}
    sens_map = {r.get("factor", ""): r for r in sens}

    best_guardrail = top_k(
        ranking,
        lambda r: parse_bool(r.get("guardrail_pass", "false")) and (to_float(r.get("primary_gain_for_ranking")) or 0.0) > 0.0,
        3,
    )
    risky_high_gain = top_k(
        ranking,
        lambda r: (not parse_bool(r.get("guardrail_pass", "false"))) and (to_float(r.get("primary_gain_for_ranking")) or 0.0) > 0.0,
        3,
    )
    stable_defaults = [r for r in recs if str(r.get("recommendation_type", "")).startswith("default")]

    business_gains: List[float] = []
    algo_gains: List[float] = []
    for r in ranking:
        if not parse_bool(r.get("guardrail_pass", "false")):
            continue
        g = to_float(r.get("primary_gain_for_ranking"))
        if g is None or math.isnan(g):
            continue
        group = classify_factor(str(r.get("factor", "")))
        if group == "business":
            business_gains.append(g)
        elif group == "algorithm":
            algo_gains.append(g)

    business_avg = mean(business_gains)
    algo_avg = mean(algo_gains)

    insight_rows: List[Dict[str, object]] = []
    idx = 1
    for r in best_guardrail:
        factor = str(r.get("factor", ""))
        rec = rec_map.get(factor, {})
        insight_rows.append(
            {
                "insight_id": idx,
                "category": "top_guardrail_gain",
                "factor": factor,
                "factor_group": classify_factor(factor),
                "observation": "Candidate improves primary metric and passes both guardrails.",
                "primary_gain_for_ranking": r.get("primary_gain_for_ranking", ""),
                "quit_rate_delta": r.get("quit_rate_delta_candidate_minus_default", ""),
                "served_rate_delta": r.get("served_rate_delta_candidate_minus_default", ""),
                "recommended_value": rec.get("recommended_value", r.get("candidate_value", "")),
                "recommendation_type": rec.get("recommendation_type", ""),
                "risk_flag": rec.get("risk_flag", ""),
                "management_action": "Prioritize controlled rollout for this knob with weekly service KPI checks.",
            }
        )
        idx += 1

    for r in risky_high_gain:
        factor = str(r.get("factor", ""))
        insight_rows.append(
            {
                "insight_id": idx,
                "category": "risky_profit_tradeoff",
                "factor": factor,
                "factor_group": classify_factor(factor),
                "observation": "Candidate improves profit but violates at least one service guardrail.",
                "primary_gain_for_ranking": r.get("primary_gain_for_ranking", ""),
                "quit_rate_delta": r.get("quit_rate_delta_candidate_minus_default", ""),
                "served_rate_delta": r.get("served_rate_delta_candidate_minus_default", ""),
                "recommended_value": r.get("candidate_value", ""),
                "recommendation_type": "fallback_risky",
                "risk_flag": "RED",
                "management_action": "Treat as stress-test scenario; do not deploy without mitigation budget.",
            }
        )
        idx += 1

    for r in stable_defaults[:3]:
        factor = str(r.get("factor", ""))
        srow = sens_map.get(factor, {})
        insight_rows.append(
            {
                "insight_id": idx,
                "category": "robust_default",
                "factor": factor,
                "factor_group": classify_factor(factor),
                "observation": "Default remains preferred after Stage2 validation.",
                "primary_gain_for_ranking": r.get("primary_delta_candidate_minus_default", ""),
                "quit_rate_delta": "",
                "served_rate_delta": "",
                "recommended_value": r.get("recommended_value", ""),
                "recommendation_type": r.get("recommendation_type", ""),
                "risk_flag": r.get("risk_flag", ""),
                "management_action": "Keep current setting; focus management effort on higher-leverage parameters.",
                "sensitivity_abs_local_slope": srow.get("abs_local_slope", ""),
            }
        )
        idx += 1

    if business_avg is not None or algo_avg is not None:
        insight_rows.append(
            {
                "insight_id": idx,
                "category": "portfolio_allocation",
                "factor": "ALL",
                "factor_group": "summary",
                "observation": "Average guardrail-safe gain by knob family.",
                "primary_gain_for_ranking": "",
                "quit_rate_delta": "",
                "served_rate_delta": "",
                "recommended_value": "",
                "recommendation_type": "",
                "risk_flag": "",
                "management_action": "Allocate tuning effort toward the higher-return family first.",
                "business_avg_gain_guardrail_safe": "" if business_avg is None else f"{business_avg:.6f}",
                "algorithm_avg_gain_guardrail_safe": "" if algo_avg is None else f"{algo_avg:.6f}",
            }
        )

    insight_fields = [
        "insight_id",
        "category",
        "factor",
        "factor_group",
        "observation",
        "primary_gain_for_ranking",
        "quit_rate_delta",
        "served_rate_delta",
        "recommended_value",
        "recommendation_type",
        "risk_flag",
        "management_action",
        "sensitivity_abs_local_slope",
        "business_avg_gain_guardrail_safe",
        "algorithm_avg_gain_guardrail_safe",
    ]
    insights_path = output_dir / (insights_name.strip() or "management_insights.csv")
    write_csv(insights_path, insight_rows, insight_fields)

    summary_lines = [
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Output dir: {output_dir}",
        f"Top guardrail-safe opportunities: {len(best_guardrail)}",
        f"Risky high-gain candidates: {len(risky_high_gain)}",
        f"Default-preferred factors: {len(stable_defaults)}",
        f"Business avg gain (guardrail-safe): {'' if business_avg is None else round(business_avg, 6)}",
        f"Algorithm avg gain (guardrail-safe): {'' if algo_avg is None else round(algo_avg, 6)}",
    ]
    summary_path = output_dir / (summary_name.strip() or "management_summary.txt")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    return insights_path, summary_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    if args.wait:
        wait_for_completion(output_dir, poll_sec=args.poll_sec, timeout_min=args.timeout_min)

    if not is_complete(output_dir):
        missing = [f for f in REQUIRED_FILES if not (output_dir / f).exists()]
        raise RuntimeError(
            "Run not complete or validation failed. "
            f"Missing files={missing}, validation_complete={validation_complete(output_dir)}"
        )

    export_path = merge_all_csvs(output_dir, args.export_name)
    insights_path, summary_path = build_management_insights(output_dir, args.insights_name, args.summary_name)

    print(f"[DONE] merged_csv={export_path}")
    print(f"[DONE] insights_csv={insights_path}")
    print(f"[DONE] summary_txt={summary_path}")


if __name__ == "__main__":
    main()
