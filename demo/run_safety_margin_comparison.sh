#!/usr/bin/env bash
# Three-objective support comparison. Activate the simulator environment first.
# Required inputs must be from training/development data, not benchmark test data.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${DATA_DIR:?Set DATA_DIR to the training candidate-set directory}"
: "${SCALES_FILE:?Set SCALES_FILE to frozen three-objective development scales}"
: "${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the compatible MDN auxiliary checkpoint}"
: "${PPO_CHECKPOINT:?Set PPO_CHECKPOINT to the PPO policy}"
: "${LAGRANGIAN_CHECKPOINT:?Set LAGRANGIAN_CHECKPOINT to the Lagrangian policy}"
trainer_python="${TRAINER_PYTHON:-python}"
safety_python="${SAFETY_PYTHON:-python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
support_dir="${SUPPORT_OUTPUT:-data/safety_margin_support_${run_stamp}}"
benchmark_dir="${BENCHMARK_OUTPUT:-data/safety_margin_comparison_${run_stamp}}"
context_seed="${CONTEXT_SEED:-310000}"
"$trainer_python" -c 'import torch, numpy'
"$safety_python" -c 'import torch, safety_gymnasium'
# The task payoff and task motive use the same frozen scale.
payoff_scale="$("$trainer_python" - "$SCALES_FILE" <<'PY'
import json, math, sys
with open(sys.argv[1]) as handle:
    scales = json.load(handle)
if scales['objective_names'] != ['Safety', 'Task', 'ControlEfficiency']:
    raise ValueError('This comparison script requires three-objective scales')
value = float(scales['divisors'][1])
if not math.isfinite(value) or value <= 0:
    raise ValueError('Invalid task scale')
print(value)
PY
)"
"$trainer_python" -m generator.train_mdn_margin_support \
  --data-dir "$DATA_DIR" --scales-file "$SCALES_FILE" \
  --baseline-candidate-id zero_action --payoff-scale "$payoff_scale" \
  --base-checkpoint "$BASE_CHECKPOINT" --output-dir "$support_dir" \
  --eligibility feasible --epsilon 0.2 --regularization 0.1 \
  --epochs "${EPOCHS:-500}" --seed 42
cat > "$support_dir/benchmark_contract.json" <<'JSON'
{"environment":"SafetyPointGoal1-v0","observation_schema":"native_flat_v1","objectives":["safety","task","control_efficiency"]}
JSON
"$safety_python" -m demo.run_contextual_safety_benchmark \
  --output "$benchmark_dir" --contexts "${CONTEXTS:-300}" --development-contexts 2 \
  --training-seeds 42 --seed "$context_seed" --max-steps 200 --epsilon 0.2 \
  --ppo-template "$PPO_CHECKPOINT" --lagrangian-template "$LAGRANGIAN_CHECKPOINT" \
  --scales-file "$SCALES_FILE" --mdn-checkpoint "$support_dir/support.pth" \
  --mdn-contract "$support_dir/benchmark_contract.json"
"$trainer_python" -m utils.compare_contextual_support \
  --benchmark-dir "$benchmark_dir" --fixed-caps 0.8 0.8 0.8
printf '\nComparison report: %s/support_comparison.md\n' "$benchmark_dir"
