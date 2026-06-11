"""Run Phase 1 tuning round 2 configs G-I sequentially.

Round 1 showed:
- outside_util=-1.0: home_pickup 85-95%, quit <5% (too high home pickup)
- outside_util=0.0:  home_pickup 49%, quit 43% (too high quit)

Round 2 targets the sweet spot with:
- Negative home_util to make home less attractive (with outside=-1.0)
- Intermediate outside_util between -1.0 and 0.0
"""
import subprocess
import sys
import time

CONFIGS = [
    # (label, home_util, incentive_sens, outside_option_util)
    ("G", -0.5, -0.25, -1.0),   # Negative home_util, moderate outside
    ("H", -1.0, -0.25, -1.0),   # Very negative home_util, moderate outside
    ("I",  0.0, -0.50, -0.5),   # Higher sens + intermediate outside
]

BASE_CMD = [
    sys.executable, "scripts/run_yanjiao_experiments.py",
    "--phase", "main",
    "--strategies", "DSPO",
    "--seeds", "40",
    "--episodes", "40",
    "--gpu", "0",
    "--yanjiao_prefix", "yanjiao_dispersed_{n_passengers}_{seed}",
]

def main():
    t0 = time.time()
    for label, home_util, incentive_sens, outside_util in CONFIGS:
        print(f"\n{'='*60}")
        print(f"[TUNE-{label}] home_util={home_util}, incentive_sens={incentive_sens}, outside={outside_util}")
        print(f"{'='*60}")

        cmd = BASE_CMD + [
            "--home_util_override", str(home_util),
            "--incentive_sens_override", str(incentive_sens),
            "--outside_option_util_override", str(outside_util),
            "--run_prefix", f"YJ_DISP_TUNE_{label}",
            "--folder_suffix", f"_disp_tune_{label}",
            "--output_dir", f"Experiments/analysis/tune_{label}",
        ]

        t1 = time.time()
        result = subprocess.run(cmd, cwd=".")
        elapsed = time.time() - t1
        print(f"[TUNE-{label}] done in {elapsed/60:.1f} min, exit={result.returncode}")

        if result.returncode != 0:
            print(f"[TUNE-{label}] FAILED, stopping.")
            sys.exit(1)

    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"[DONE] All configs in {total/3600:.1f} hours")


if __name__ == "__main__":
    main()
