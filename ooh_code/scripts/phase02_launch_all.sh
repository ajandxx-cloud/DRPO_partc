#!/bin/bash
# Phase 2: Behavioral Calibration - Sequential Launch
# 4 configs × 40 runs × ~20 min = ~52 hours total
set -e

cd "$(dirname "$0")/.."

echo "=== Phase 2 Behavioral Calibration: Sequential Launch ==="
echo "Start time: $(date)"
echo ""

COMMON_ARGS="--seed_split tuning --phase main --strategies No-pricing Static DSPO DRPO --episodes 150 --eval_episodes 20 --drop_params final_yanjiao_mode allow_derived_choice_utility"

echo "=== Config 1 (Baseline): walk=-0.0015, outside=-0.75, incentive=-0.25 ==="
python scripts/run_yanjiao_experiments.py \
  $COMMON_ARGS \
  --walk_distance_weight_override -0.0015 \
  --outside_option_util_override -0.75 \
  --run_prefix P02C1 \
  --output_dir Experiments/analysis/phase02_config1 2>&1
echo "Config 1 complete at $(date)"

echo "=== Config 2 (Walk variant): walk=-0.003, outside=-0.75, incentive=-0.25 ==="
python scripts/run_yanjiao_experiments.py \
  $COMMON_ARGS \
  --walk_distance_weight_override -0.003 \
  --outside_option_util_override -0.75 \
  --run_prefix P02C2 \
  --output_dir Experiments/analysis/phase02_config2 2>&1
echo "Config 2 complete at $(date)"

echo "=== Config 3 (Outside variant): walk=-0.0015, outside=-0.5, incentive=-0.25 ==="
python scripts/run_yanjiao_experiments.py \
  $COMMON_ARGS \
  --walk_distance_weight_override -0.0015 \
  --outside_option_util_override -0.5 \
  --run_prefix P02C3 \
  --output_dir Experiments/analysis/phase02_config3 2>&1
echo "Config 3 complete at $(date)"

echo "=== Config 4 (Incentive variant): walk=-0.0015, outside=-0.75, incentive=-0.35 ==="
python scripts/run_yanjiao_experiments.py \
  $COMMON_ARGS \
  --walk_distance_weight_override -0.0015 \
  --outside_option_util_override -0.75 \
  --incentive_sens_override -0.35 \
  --run_prefix P02C4 \
  --output_dir Experiments/analysis/phase02_config4 2>&1
echo "Config 4 complete at $(date)"

echo ""
echo "=== ALL CONFIGS COMPLETE at $(date) ==="
