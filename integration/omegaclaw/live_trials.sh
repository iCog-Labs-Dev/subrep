#!/usr/bin/env bash
# Repeated live OmegaClaw explanation trials, each with a fresh Omega.
#
# Every trial starts Omega on a new, empty memory volume. Omega replays its saved
# history into every model call, so trials that share a volume are not
# independent: an earlier answer can be copied into a later one.
#
# Run from the SubRep root with Docker running:
#   export ASI_API_KEY=...
#   bash integration/omegaclaw/live_trials.sh
# An OpenAI-compatible endpoint instead (for example a local Ollama):
#   PROVIDER=OpenAIAPI OPENAIAPI_URL=http://host.docker.internal:11434/v1/ MODEL=<model> \
#   bash integration/omegaclaw/live_trials.sh
# Optional settings: TRIALS (default 5), TIMEOUT seconds per request (300),
# IMAGE (singularitynet/omega:latest), PULL=1, SCENARIOS="name name",
# OUT (output folder), PY (python), PORT (8765), and PROXY_TEMPLATE, a file that
# replaces the image's nginx template (for example with a longer model timeout).
#
# Output: one report per trial, audit.jsonl, Omega logs, comparison.md, and
# review.csv for a person to fill in and merge back with --reviewed-csv.
set -euo pipefail
shopt -s nullglob
# Git Bash on Windows rewrites arguments that look like Unix paths
# ("--path /agent", "name:/PeTTa/...") before Windows programs see them.
export MSYS_NO_PATHCONV=1

PROVIDER="${PROVIDER:-ASICloud}"
IMAGE="${IMAGE:-singularitynet/omega:latest}"
TRIALS="${TRIALS:-5}"
TIMEOUT="${TIMEOUT:-300}"
PORT="${PORT:-8765}"
PY="${PY:-python}"
OUT="${OUT:-data/omegaclaw/live_$(date +%Y%m%d_%H%M)}"
read -r -a SCENARIO_LIST <<< "${SCENARIOS:-clear_preferred_skill changed_task_priorities excluded_skill no_admissible_options risk_budget_exclusion runtime_inadmissible mixed_exclusions abstention_mixed_reasons}"

provider_args=("provider=$PROVIDER")
case "$PROVIDER" in
  ASICloud)
    : "${ASI_API_KEY:?set ASI_API_KEY first}"
    MODEL="${MODEL:-minimax/minimax-m3}"
    key_args=(-e ASI_API_KEY)
    ;;
  OpenAIAPI)
    : "${OPENAIAPI_URL:?set OPENAIAPI_URL, for example http://host.docker.internal:11434/v1/}"
    : "${MODEL:?set MODEL}"
    export OPENAIAPI_API_KEY="${OPENAIAPI_API_KEY:-endpoint-without-a-key}"
    key_args=(-e OPENAIAPI_API_KEY)
    provider_args+=("openaiapi_url=$OPENAIAPI_URL")
    ;;
  *)
    echo "PROVIDER must be ASICloud or OpenAIAPI" >&2
    exit 2
    ;;
esac

mount_args=()
if [ -n "${PROXY_TEMPLATE:-}" ]; then
  template_path="$(cd "$(dirname "$PROXY_TEMPLATE")" && (pwd -W 2>/dev/null || pwd))/$(basename "$PROXY_TEMPLATE")"
  mount_args=(--mount "type=bind,source=$template_path,target=/opt/nginx/nginx.conf.template,readonly")
fi

SUBREP_OMEGA_TOKEN="$("$PY" -c 'import secrets; print(secrets.token_hex(24))')"
export SUBREP_OMEGA_TOKEN
CONTAINER=omega-subrep-trial
VOLUME=omega-subrep-trial-memory

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  docker volume rm -f "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [ "${PULL:-0}" = "1" ]; then
  docker pull "$IMAGE" >/dev/null
fi
DIGEST="$(docker image inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
mkdir -p "$OUT"
echo "Image: $DIGEST"
echo "Provider: $PROVIDER; model: $MODEL; trials: $TRIALS; timeout: ${TIMEOUT}s"
echo "Proxy template: ${PROXY_TEMPLATE:-image default}"
echo "Scenarios: ${SCENARIO_LIST[*]}"
echo "Output: $OUT"

for trial in $(seq 1 "$TRIALS"); do
  for scenario in "${SCENARIO_LIST[@]}"; do
    cleanup
    name="${scenario}_trial${trial}"
    echo "--- $(date +%H:%M:%S) trial $trial / $TRIALS: $scenario"
    "$PY" -m integration.omegaclaw.demo --backend live \
      --host 0.0.0.0 --port "$PORT" --path /agent --timeout "$TIMEOUT" \
      --scenario "$scenario" --first-trial "$trial" \
      --run-label "$MODEL" --provider "$PROVIDER" --model "$MODEL" --omega-image "$DIGEST" \
      --audit-path "$OUT/audit.jsonl" \
      --report-path "$OUT/$name.json" >"$OUT/$name.log" 2>&1 &
    demo_pid=$!
    sleep 3
    # -it: the image's nginx proxy writes to /dev/stderr, which needs a terminal.
    docker run -d -it --name "$CONTAINER" \
      --security-opt no-new-privileges:true --init \
      --add-host=host.docker.internal:host-gateway \
      --tmpfs /tmp:size=64m,mode=1777 \
      --tmpfs /var/tmp:size=64m,mode=1777 \
      --tmpfs /run:size=16m,mode=755 \
      --volume "$VOLUME:/PeTTa/repos/OmegaClaw-Core/memory" \
      "${mount_args[@]}" \
      "${key_args[@]}" -e IMPORT_KB_ON_START=0 \
      "$IMAGE" \
      commchannel=websocket embeddingprovider=Local "${provider_args[@]}" \
      "WS_URL=ws://host.docker.internal:$PORT/agent" \
      "WS_TOKEN=$SUBREP_OMEGA_TOKEN" "model=$MODEL" >/dev/null
    if wait "$demo_pid"; then
      echo "    valid response"
    else
      echo "    invalid response or backend error (kept in the report)"
    fi
    # Keep Omega's own log for diagnosis, with the per-run token masked.
    docker logs "$CONTAINER" 2>&1 | sed "s/$SUBREP_OMEGA_TOKEN/***/g" >"$OUT/$name.omega.log" || true
    cleanup  # stop Omega at once, so it makes no further model calls
  done
done

reports=()
for path in "$OUT"/*_trial*.json; do
  reports+=(--report "$path")
done
if [ "${#reports[@]}" -eq 0 ]; then
  echo "No trial wrote a report; see the .log files in $OUT." >&2
  exit 1
fi
"$PY" -m integration.omegaclaw.compare_runs "${reports[@]}" \
  --output "$OUT/comparison.md" --review-csv "$OUT/review.csv"
echo "Next: fill reviewer_factually_correct (yes or no) and reviewer_notes in"
echo "$OUT/review.csv, save it as $OUT/reviewed.csv, then rebuild the report:"
echo "  $PY -m integration.omegaclaw.compare_runs ${reports[*]} --output $OUT/comparison.md --reviewed-csv $OUT/reviewed.csv"
