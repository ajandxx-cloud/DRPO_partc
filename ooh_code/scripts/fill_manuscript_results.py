#!/usr/bin/env python3
# Fill manuscript.tex placeholders with actual experiment results.
# After running run_ablation_spo.py and run_joint_sensitivity.py,
# run this script to replace all ABL* and JTS* placeholders in manuscript.tex.
# Usage:
#   python scripts/fill_manuscript_results.py           # fill and overwrite
#   python scripts/fill_manuscript_results.py --dry_run # print replacements only
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANUSCRIPT = ROOT / "paper" / "trc_latex" / "manuscript.tex"
ABL_DIR = ROOT / "Experiments" / "analysis" / "ablation_spo"
JTS_DIR = ROOT / "Experiments" / "analysis" / "joint_sensitivity"


# ── Ablation placeholders ─────────────────────────────────────────────────────
# \ABLDRPOPROFIT{}  mean net profit of DRPO
# \ABLABL{}         mean net profit of DSPO-ablation
# \ABLGAINPCT{}     % gain (DRPO - ablation) / ablation * 100
# \ABLCI{}          95% CI half-width of profit diff
# \ABLWINS{}        number of seeds DRPO wins (out of 30)
# \ABLHOMEABL{}     home-pickup % for ablation (mean ± std)
# \ABLQUITABL{}     quit rate % for ablation (mean ± std)
# \ABLCOSTABL{}     total costs for ablation (mean ± std)
# \ABLPROFITABL{}   net profit for ablation (mean ± std)
# \ABLHOMEDRPO{}    home-pickup % for DRPO (mean ± std)
# \ABLQUITDRPO{}    quit rate % for DRPO (mean ± std)
# \ABLCOSTDRPO{}    total costs for DRPO (mean ± std)
# \ABLPROFITDRPO{}  net profit for DRPO (mean ± std)

# ── Joint sensitivity placeholders ───────────────────────────────────────────
# \JTSWINSCELLS{}   number of cells (out of 27) where DRPO majority-wins
# \JTSWINSPAIRS{}   total paired wins (out of 81)
# \JTSBODY{}        full table body rows


def fmt_mean_std(mean, std, decimals=2):
    return f"${mean:.{decimals}f}\\pm{std:.{decimals}f}$"


def fmt_mean_std_pct(mean, std, decimals=2):
    """For percentages already in 0-100 range."""
    return f"${mean:.{decimals}f}\\pm{std:.{decimals}f}$"


def load_ablation():
    """Load ablation and DRPO results, return replacement dict."""
    abl_csv = ABL_DIR / "dspo_ablation_results.csv"
    pa_json = ABL_DIR / "paired_analysis.json"

    if not abl_csv.exists():
        print(f"  [WARN] Ablation results not found: {abl_csv}")
        return {}

    # Load ablation rows
    with open(abl_csv) as f:
        abl_rows = list(csv.DictReader(f))
    abl_rows = [{k: float(v) if _is_num(v) else v for k, v in r.items()} for r in abl_rows]

    # Load DRPO results from main experiment
    from scripts.run_ablation_spo import load_drpo_results, SEEDS
    drpo_rows = load_drpo_results(SEEDS)

    def stats(rows, key):
        vals = [r[key] for r in rows if key in r]
        if not vals:
            return float("nan"), float("nan")
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    abl_profit_m, abl_profit_s = stats(abl_rows, "net_profit")
    abl_cost_m, abl_cost_s = stats(abl_rows, "total_costs")
    abl_quit_m, abl_quit_s = stats(abl_rows, "quit_rate")
    abl_home_m, abl_home_s = stats(abl_rows, "home_delivery")

    drpo_profit_m, drpo_profit_s = stats(drpo_rows, "net_profit")
    drpo_cost_m, drpo_cost_s = stats(drpo_rows, "total_costs")
    drpo_quit_m, drpo_quit_s = stats(drpo_rows, "quit_rate")
    drpo_home_m, drpo_home_s = stats(drpo_rows, "home_delivery")

    # Paired analysis
    if pa_json.exists():
        with open(pa_json) as f:
            pa = json.load(f)
        wins = pa["drpo_wins_profit"]
        gain_pct = pa["profit_diff_pct"]
        ci = pa["profit_diff_ci"]
    else:
        # Compute inline
        abl_by_seed = {r["seed"]: r for r in abl_rows}
        drpo_by_seed = {r["seed"]: r for r in drpo_rows}
        common = sorted(set(abl_by_seed) & set(drpo_by_seed))
        diffs = [drpo_by_seed[s]["net_profit"] - abl_by_seed[s]["net_profit"] for s in common]
        wins = sum(1 for d in diffs if d > 0)
        n = len(diffs)
        mean_diff = sum(diffs) / n if n else float("nan")
        std_diff = statistics.stdev(diffs) if n > 1 else 0.0
        t = 2.045  # ~30 df
        ci = t * std_diff / math.sqrt(n) if n else float("nan")
        gain_pct = mean_diff / abl_profit_m * 100 if abl_profit_m else float("nan")

    return {
        r"\ABLDRPOPROFIT{}": f"{drpo_profit_m:.2f}",
        r"\ABLABL{}": f"{abl_profit_m:.2f}",
        r"\ABLGAINPCT{}": f"{gain_pct:.2f}",
        r"\ABLCI{}": f"{ci:.2f}",
        r"\ABLWINS{}": str(wins),
        r"\ABLHOMEABL{}": fmt_mean_std_pct(abl_home_m * 100, abl_home_s * 100),
        r"\ABLQUITABL{}": fmt_mean_std(abl_quit_m, abl_quit_s),
        r"\ABLCOSTABL{}": fmt_mean_std(abl_cost_m, abl_cost_s),
        r"\ABLPROFITABL{}": fmt_mean_std(abl_profit_m, abl_profit_s),
        r"\ABLHOMEDRPO{}": fmt_mean_std_pct(drpo_home_m * 100, drpo_home_s * 100),
        r"\ABLQUITDRPO{}": fmt_mean_std(drpo_quit_m, drpo_quit_s),
        r"\ABLCOSTDRPO{}": fmt_mean_std(drpo_cost_m, drpo_cost_s),
        r"\ABLPROFITDRPO{}": fmt_mean_std(drpo_profit_m, drpo_profit_s),
    }


def load_joint_sensitivity():
    """Load joint sensitivity results, return replacement dict."""
    summary_csv = JTS_DIR / "grid_summary.csv"
    if not summary_csv.exists():
        print(f"  [WARN] Joint sensitivity results not found: {summary_csv}")
        return {}

    with open(summary_csv) as f:
        rows = list(csv.DictReader(f))
    rows = [{k: float(v) if _is_num(v) else v for k, v in r.items()} for r in rows]

    total_paired = sum(r["n_paired"] for r in rows if r["n_paired"] > 0)
    total_wins = sum(r["drpo_wins_profit"] for r in rows if r["n_paired"] > 0)
    cells_majority = sum(1 for r in rows
                         if r["n_paired"] > 0 and r["drpo_wins_profit"] > r["n_paired"] / 2)

    # Build LaTeX table body
    body_lines = []
    for r in rows:
        wr = r.get("win_rate", "?")
        pct = r.get("profit_diff_pct", float("nan"))
        pct_str = f"{pct:+.1f}\\%" if not math.isnan(pct) else "?"
        body_lines.append(
            f"{r['incentive_sens']:+.2f} & {r['outside_option_util']:+.1f} & "
            f"{r['home_util']:.1f} & {wr} & {pct_str} \\\\"
        )
    body = "\n".join(body_lines)

    return {
        r"\JTSWINSCELLS{}": str(cells_majority),
        r"\JTSWINSPAIRS{}": str(int(total_wins)),
        r"\JTSBODY{}": body,
    }


def _is_num(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"\nFilling manuscript placeholders in:\n  {MANUSCRIPT}\n")

    replacements = {}
    replacements.update(load_ablation())
    replacements.update(load_joint_sensitivity())

    if not replacements:
        print("No results found. Run the experiment scripts first.")
        return

    text = MANUSCRIPT.read_text(encoding="utf-8")
    changed = 0
    for placeholder, value in replacements.items():
        if placeholder in text:
            print(f"  {placeholder}  →  {value}")
            if not args.dry_run:
                text = text.replace(placeholder, value)
            changed += 1
        else:
            print(f"  [SKIP] {placeholder} not found in manuscript")

    if not args.dry_run and changed > 0:
        MANUSCRIPT.write_text(text, encoding="utf-8")
        print(f"\nWrote {changed} replacements to {MANUSCRIPT}")
    elif args.dry_run:
        print(f"\n[dry_run] Would replace {changed} placeholders.")


if __name__ == "__main__":
    main()
