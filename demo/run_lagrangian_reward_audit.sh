#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Same initialization, training length and development seeds for all arms.
python_bin="${SAFETY_PYTHON:-python}"
run_stamp="$(date +%Y%m%d_%H%M%S)"
for arm in no_action_penalties effort_only effort_and_smoothness; do
  effort=0
  smoothness=0
  if [[ "$arm" != no_action_penalties ]]; then effort=0.01; fi
  if [[ "$arm" == effort_and_smoothness ]]; then smoothness=0.1; fi
  "$python_bin" -m pilot.train_safety_gymnasium_ppo \
    --output "models/lagrangian_audit_${run_stamp}_${arm}.pt" \
    --seed 42 --total-updates "${AUDIT_UPDATES:-100}" \
    --use-lagrangian --cost-limit 1 --lambda-lr 0.005 \
    --effort-penalty "$effort" --smoothness-penalty "$smoothness" \
    --max-episode-steps 200 --eval-every 25 --eval-episodes 100 \
    --eval-seed 420000 --device cpu
done
printf 'Completed controlled runs: models/lagrangian_audit_%s_*_run/metrics.jsonl\n' "$run_stamp"
