#!/usr/bin/env python3
"""Run a minimal smoke check on the NYC_TLC pilot instance."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_ALGOS = ["Baseline", "DSPO", "DRPO"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-run NYC_TLC with baseline and learning policies.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--data-seed-test", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--algos", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for algo in args.algos:
        cmd = [
            args.python,
            "run.py",
            "--instance",
            "NYC_TLC",
            "--algo_name",
            algo,
            "--max_episodes",
            str(args.episodes),
            "--initial_phase_epochs",
            "1",
            "--save_count",
            str(max(1, args.episodes)),
            "--seed",
            str(args.seed),
            "--data_seed",
            str(args.data_seed),
            "--data_seed_test",
            str(args.data_seed_test),
            "--k",
            str(args.k),
            "--n_vehicles",
            "8",
            "--veh_capacity",
            "12",
            "--max_steps_r",
            "8",
            "--max_steps_p",
            "0.5",
            "--log_output",
            "term_file",
            "--experiment",
            "nyc_tlc_smoke_",
            "--folder_suffix",
            algo,
        ]
        print("[smoke] " + " ".join(cmd))
        completed = subprocess.run(cmd, cwd=repo_root, check=False)
        if completed.returncode != 0:
            print(f"[failed] {algo} exited with {completed.returncode}")
            return completed.returncode
    print("[ok] NYC_TLC smoke run completed for: " + ", ".join(args.algos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
