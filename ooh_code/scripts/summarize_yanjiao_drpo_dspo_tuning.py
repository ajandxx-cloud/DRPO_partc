#!/usr/bin/env python
"""Print compact status for Yanjiao DRPO/DSPO tuning outputs."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="Experiments/analysis/yanjiao_drpo_dspo_tuning_run2")
    p.add_argument("--validation_dir", default="Experiments/analysis/yanjiao_final_maxprice5_3seed_run2")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    output_dir = root / args.output_dir
    validation_dir = root / args.validation_dir

    print(f"Tuning dir: {output_dir}")
    tuning_summary = read_csv(output_dir / "tuning_summary.csv")
    if tuning_summary:
        print("\nTuning candidates:")
        for row in tuning_summary:
            print(
                f"{row['label']}: ok={row['candidate_ok']} "
                f"order={row['ordering_count']} "
                f"DRPO={row['drpo_profit_mean']} DSPO={row['dspo_profit_mean']} "
                f"Static={row['static_profit_mean']} "
                f"quit=({row['dspo_quit_mean']},{row['drpo_quit_mean']}) "
                f"home=({row['dspo_home_mean']},{row['drpo_home_mean']})"
            )
    else:
        print("No tuning_summary.csv yet.")
        for child in sorted(output_dir.glob("*")):
            if child.is_dir():
                raw = read_csv(child / "yanjiao_raw.csv")
                labels = {}
                for row in raw:
                    labels[row.get("label", "?")] = labels.get(row.get("label", "?"), 0) + 1
                print(f"{child.name}: raw_rows={len(raw)} labels={labels}")

    selected = output_dir / "selected_candidate.json"
    if selected.exists():
        print("\nSelected:")
        print(json.dumps(json.loads(selected.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))

    validation_summary = read_csv(validation_dir / "validation_summary.csv")
    if validation_summary:
        print("\nValidation:")
        for row in validation_summary:
            print(
                f"{row['label']}: pass={row.get('validation_pass', '')} "
                f"order={row['ordering_count']} "
                f"DRPO={row['drpo_profit_mean']} DSPO={row['dspo_profit_mean']} Static={row['static_profit_mean']}"
            )
    elif validation_dir.exists():
        print(f"\nValidation dir exists but no validation_summary.csv yet: {validation_dir}")


if __name__ == "__main__":
    main()
