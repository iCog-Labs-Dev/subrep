#!/usr/bin/env bash
# Run from the repository; override SAFETY_PYTHON if your simulator env differs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trainer_python="${TRAINER_PYTHON:-.venv/bin/python}"
safety_python="${SAFETY_PYTHON:-/home/tarik/miniconda3/envs/subrep-safety/bin/python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
support_dir="data/safety_margin_support_review_${run_stamp}"
benchmark_dir="data/safety_margin_comparison_${run_stamp}"
# Freeze these before inspecting evaluation outcomes. Use unused seeds for final evaluation.
context_seed="${CONTEXT_SEED:-310000}"
"$trainer_python" -c 'import torch, numpy'
"$safety_python" -c 'import torch, safety_gymnasium'
"$trainer_python" -m generator.train_mdn_margin_support \
  --data-dir data/safety_mdn3_current_20261028_173547/train \
  --scales-file data/safety_mdn3_current_20261028_173547/scales.json \
  --baseline-candidate-id zero_action \
  --payoff-scale 0.6275834153279493 \
  --base-checkpoint models/safety_mdn3_current_20261028_193049/auxiliary.pth \
  --output-dir "$support_dir" \
  --eligibility feasible --epsilon 0.2 --regularization 0.1 --epochs 500 --seed 42
cat > "$support_dir/benchmark_contract.json" <<'JSON'
{"environment":"SafetyPointGoal1-v0","observation_schema":"native_flat_v1","objectives":["safety","task","control_efficiency"]}
JSON
"$safety_python" -m demo.run_contextual_safety_benchmark \
  --output "$benchmark_dir" --contexts 300 --development-contexts 2 \
  --training-seeds 42 --seed "$context_seed" --max-steps 200 --epsilon 0.2 \
  --ppo-template models/safety_ppo_monitored_seed42_v1.pt \
  --lagrangian-template models/safety_ppo_lagrangian_episodecost_seed42_v3.pt \
  --scales-file data/safety_mdn3_current_20261028_173547/scales.json \
  --mdn-checkpoint "$support_dir/support.pth" \
  --mdn-contract "$support_dir/benchmark_contract.json"
"$trainer_python" -m utils.compare_contextual_support \
  --benchmark-dir "$benchmark_dir" --fixed-caps 0.8 0.8 0.8
printf '\nComparison report: %s/support_comparison.md\n' "$benchmark_dir"
