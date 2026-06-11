"""Run Phase 1 tuning configs A-E sequentially (one GPU, no contention)."""
import subprocess
import sys
import time

CONFIGS = [
    # (label, home_util, incentive_sens, outside_option_util)
    ("A", 0.4, -0.25, -1.0),
    ("B", 0.0, -0.25, -1.0),
    ("C", 0.4, -0.50, -1.0),
    ("D", 0.0, -0.50, -1.0),
    ("E", 0.4, -0.25,  0.0),
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
