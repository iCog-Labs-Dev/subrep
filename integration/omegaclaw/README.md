# SubRep to Omega Explanation Integration

## What This Integrates

SubRep and Omega have different responsibilities:

- **SubRep** executes candidate policies, compares them with a baseline, certifies
  them with CDS/PDS, stores certificates, and decides which skills are admissible.
- **Omega** is a stateful neural-symbolic agent. The sibling checkout currently
  identifies itself as `Omega`, although the task and some upstream artifacts use
  the OmegaClaw name.
- **This package** sends an immutable, authoritative SubRep decision snapshot to
  Omega. Omega answers with structured claims about the snapshot, never with prose.
  SubRep verifies every claim and writes the explanation itself, from its own
  decision and the verified claims. It never asks Omega to select, certify, admit,
  exclude, or execute a skill.

The integration belongs in SubRep because SubRep owns the safety boundary and must
validate the external response before anything downstream can use it. The Omega
repository remains an independently runnable dependency.

## Safety Boundary

```text
SubRep SkillLibrary
  -> query_admissible()                    owned by SubRep
  -> apply local risk budget               owned by SubRep
  -> apply explicit exclusions             owned by SubRep
  -> compute final selection/abstention    owned by SubRep
  -> build versioned decision snapshot
  -> request claims from Omega
  -> verify the decision echo and every claim against the snapshot   owned by SubRep
  -> write the explanation from verified facts                       owned by SubRep
  -> save request, response, and explanation
  -> return decision plus explanation      no skill execution
```

Omega has no selection authority. SubRep deterministically computes the largest
`final_selection_score`, using lexical skill ID order for exact ties, before the
request leaves the process. Omega must echo `subrep_decision.selected_skill_id` and
`subrep_decision.abstain` exactly. The outcome always carries SubRep's decision,
whether or not an explanation was accepted.

## Package Structure

| Path | Responsibility |
|---|---|
| `contracts.py` | Versioned request, response, claim, risk, evidence, and outcome contracts |
| `service.py` | Calls SubRep admissibility and applies local risk/exclusion filters |
| `adapter.py` | Builds the prompt, retries, verifies the response, and records every attempt |
| `validation.py` | Parses a response and runs every format and factual check on it |
| `facts.py` | Verifies each claim against the request and writes the explanation from verified facts |
| `websocket_gateway.py` | Private server for Omega's existing WebSocket client channel |
| `audit.py` | Append-only JSONL storage of complete inputs and outcomes |
| `scenarios.py` | Eight predefined acceptance scenarios using synthetic evidence |
| `synthetic_backend.py` | Deterministic non-LLM backend used only for tests and review |
| `demo.py` | Reproducible synthetic and live demo entry point |
| `live_trials.sh` | Repeated live trials, each with a fresh Omega memory |
| `compare_runs.py` | Compares run reports and writes the review sheet for a person |
| `tests/` | Contract, claim, validation, adapter, scenario, demo, and WebSocket flow tests |

## Data Contract

Each request contains:

- a unique request ID and timestamp,
- task context,
- named normalized objective weights,
- a local risk budget,
- the authoritative `subrep_decision`, including the selected and runner-up scores,
- admitted skill IDs and final scores for auditability,
- explanation evidence containing weights and objective deltas,
- audit-only certificate evidence that cannot affect ranking,
- explicit exclusions and SubRep-owned reason codes,
- an evidence label, either `OBSERVED` or `SYNTHETIC`.

The prompt lists `CLAIMABLE_FACTS`: every path a claim may cite, with its claim type
and value. Numbers are shown to twelve significant digits, so float noise such as
`0.8200000000000001` reaches the model as `0.82`. Omega must return one JSON object:

```json
{
  "schema_version": "subrep.omegaclaw.explanation.response.v2",
  "request_id": "the supplied request ID",
  "mode": "EXPLANATION_ONLY",
  "selected_skill_id": "the exact SubRep decision skill ID or null",
  "abstain": false,
  "selection_basis": "SUBREP_DECISION",
  "claims": [
    {"type": "objective_weight", "evidence_ref": "explanation_evidence.objective_weights.safety", "value": 0.8},
    {"type": "objective_delta", "evidence_ref": "explanation_evidence.skills.skill_safety.objective_deltas.efficiency", "value": -0.1}
  ],
  "advisory_concerns": []
}
```

Each claim states one fact: its `type`, the `evidence_ref` path of the fact in the
request, and its `value`. The claim types, and the paths each can cite, are:

| Type | Path |
|---|---|
| `score` | `admitted_skills.<skill>.final_selection_score`, `subrep_decision.selected_score`, `subrep_decision.runner_up_score` |
| `score_gap` | `subrep_decision.score_gap` |
| `objective_weight` | `explanation_evidence.objective_weights.<objective>` |
| `objective_delta` | `explanation_evidence.skills.<skill>.objective_deltas.<objective>` |
| `delta_r` | `explanation_evidence.skills.<skill>.delta_r` |
| `certificate` | `audit_only_certificate_evidence.skills.<skill>.<gate_type, epsilon, certificate_admission_margin, weight_region_type>` |
| `risk_budget` | `risk_budget.<max_certificate_epsilon, minimum_admission_margin>` |
| `exclusion_stage` | `exclusions.<index>.stage` |

A selection needs at least one `objective_weight` or `objective_delta` claim, the facts
that show why the selected skill scores highest; an abstention needs no claims. List items use dotted
indexes (`exclusions.0.stage`); the bracket form `exclusions[0].stage` is accepted and
stored dotted. A response may hold at most 40 claims (`MAX_CLAIMS`).

Optional advisory concerns are non-binding and hold only a `code` and its
`evidence_refs`; SubRep writes their sentence. Each code needs specific evidence:

| Code | Accepted when it cites |
|---|---|
| `SMALL_SCORE_GAP` | `subrep_decision.score_gap`, and the gap is at most 10% of the selected score |
| `NEGATIVE_OBJECTIVE_TRADEOFF` | a negative objective delta of the selected skill |
| `HIGH_PRIORITY_OBJECTIVE_DECLINE` | the weight and negative delta of the selected skill on a highest-weight objective |
| `PDS_EPSILON_USED` | the selected skill's `gate_type` and `epsilon`, when it is PDS with epsilon above 0 |
| `LOW_CERTIFICATE_MARGIN` | the selected skill's `certificate_admission_margin`, at most 10% above `risk_budget.minimum_admission_margin` |

The two 10% thresholds are `SMALL_SCORE_GAP_FRACTION` and
`LOW_CERTIFICATE_MARGIN_FRACTION` in `contracts.py`. An abstention takes no advisory
concerns.

## Validation

Every response is checked before it is accepted, and each problem is recorded as a
finding with a category:

- **Format findings** concern the response itself: valid JSON, the exact schema, the
  request ID, and claims that cite a path that does not exist (`UNKNOWN_EVIDENCE_REF`),
  that holds no claimable fact (`UNCLAIMABLE_EVIDENCE_REF`), or whose type does not
  match the path (`CLAIM_TYPE_MISMATCH`), and a selection without a weight or delta
  claim (`MISSING_CLAIMS`).
- **Factual findings** are verified errors: a claimed value that differs from the
  request value at its path (`CLAIM_VALUE_MISMATCH`, sign included; numbers must be
  equal up to float rounding, so a small value such as an epsilon of 5e-7 cannot pass
  as 0), a decision echo that differs from SubRep's decision (`DECISION_MISMATCH`),
  and an advisory concern that its cited evidence does not support
  (`UNSUPPORTED_ADVISORY_CONCERN`).

When every check passes, SubRep writes the explanation. It states the decision and the
scoring rule, the runner-up or that there is none, the verified claims grouped by
fact, every exclusion by its own stage with its stage description and recorded reason,
each advisory concern, and the evidence label. Omega writes no text that reaches the
reader, so a wrong fact cannot be reworded past the checks, and nothing SubRep must
state can be left out.

A rejected response gets one correction prompt that quotes its findings. If checking a
response fails unexpectedly, the error is recorded as `INTERNAL_VALIDATION_ERROR` on
that attempt, so SubRep's decision still reaches the caller and the audit keeps it. A transport
error is retried once with the same prompt; a timeout is not. A reply whose
`request_id` matches no request sent so far is treated as this request's answer, so a
mistyped ID is rejected at once instead of waiting out the timeout; a late reply to an
earlier request is ignored. All attempts remain in the audit record, and a person
reviews live runs with the review sheet described under
[Repeated Live Trials](#repeated-live-trials).

## Install

From the SubRep root:

```powershell
python -m pip install -r requirements.txt
```

The integration adds `websockets` for compatibility with Omega's `WSChannel`.

## Reproducible Synthetic Demo

This command runs all eight predefined scenarios without an LLM and writes the
complete audit trail. Every input is labeled `SYNTHETIC`.

```powershell
python -m integration.omegaclaw.demo --backend synthetic `
  --audit-path data/omegaclaw/synthetic_demo.jsonl `
  --report-path data/omegaclaw/synthetic_demo_report.json
```

Expected decisions:

| Scenario | What it tests | Expected result |
|---|---|---|
| `clear_preferred_skill` | Safety is dominant | `skill_safety` |
| `changed_task_priorities` | Efficiency becomes dominant | `skill_efficiency` |
| `excluded_skill` | The preferred skill is excluded by the operator | `skill_safety` |
| `no_admissible_options` | The only candidate failed certification | Abstain |
| `risk_budget_exclusion` | A PDS skill exceeds the zero-epsilon risk budget | `skill_efficiency` |
| `runtime_inadmissible` | A skill is not runtime admissible for the current weights | `skill_efficiency` |
| `mixed_exclusions` | One request excludes a skill at every stage | `skill_safety` |
| `abstention_mixed_reasons` | Every option is excluded, for different reasons | Abstain |

## Live Demo With Omega

Omega has no inbound REST API. Its existing programmatic interface is a WebSocket
**client**, so this package hosts the corresponding private server.

1. Build the sibling Omega checkout once, from `iCog/Omega`:

```bash
docker build -t omega:local .
```

2. Start the SubRep live demo from the SubRep root. Use a dedicated short-lived
   token:

```powershell
$env:SUBREP_OMEGA_TOKEN = "replace-with-a-dedicated-random-token"
python -m integration.omegaclaw.demo --backend live `
  --host 0.0.0.0 --port 8765 --path /agent `
  --provider ASICloud --model minimax/minimax-m3 `
  --omega-image singularitynet/omega:latest `
  --audit-path data/omegaclaw/live_demo.jsonl `
  --report-path data/omegaclaw/live_demo_report.json
```

   A live run must name the provider, model, and Omega image, so every report shows
   what was tested; the demo refuses to start without them. Give the image digest
   (`docker image inspect --format '{{index .RepoDigests 0}}' <image>`) for an exact
   record.

3. In Bash or WSL, start Omega with its WebSocket channel and the same token.
   The verified workspace run used ASICloud:

```bash
export WS_URL='ws://host.docker.internal:8765/agent'
export WS_TOKEN='replace-with-a-dedicated-random-token'
export ASI_API_KEY='your-provider-key'
./scripts/omega start -d omega:local -p ASICloud -t websocket
```

No explanation text is copied into Telegram or IRC. The demo sends each
request over the channel and validates the returned JSON automatically.

If Omega runs directly on the host rather than in Docker, use
`ws://127.0.0.1:8765/agent`. Keep the endpoint on a private interface. The gateway
does not terminate TLS itself; place it behind a TLS reverse proxy that exposes
`wss://` if traffic leaves the host or a trusted container network.

The successful verification used the published OmegaClaw image directly because
the local image build was blocked by slow package downloads. When using
`singularitynet/omega:latest` directly, let the image use its default security
policy path and do not pass numeric loop overrides. The published image is rooted
at `/PeTTa/repos/OmegaClaw-Core`, while the current local checkout launcher targets
`/PeTTa/repos/Omega`.

Verified published-image alternative:

```bash
docker run -d -it --name omega-subrep-live \
  --security-opt no-new-privileges:true --init \
  --add-host=host.docker.internal:host-gateway \
  --tmpfs /tmp:size=64m,mode=1777 \
  --tmpfs /var/tmp:size=64m,mode=1777 \
  --tmpfs /run:size=16m,mode=755 \
  --volume omega-subrep-live-memory:/PeTTa/repos/OmegaClaw-Core/memory \
  -e ASI_API_KEY -e IMPORT_KB_ON_START=0 \
  singularitynet/omega:latest \
  commchannel=websocket provider=ASICloud embeddingprovider=Local \
  WS_URL=ws://host.docker.internal:8765/agent \
  WS_TOKEN="$WS_TOKEN" model=minimax/minimax-m3
```

After the demo finishes, stop the continuous Omega process to avoid unnecessary
provider calls:

```bash
docker rm -f omega omega-subrep-live 2>/dev/null || true
```

## Repeated Live Trials

`live_trials.sh` runs every scenario against the published Omega image, starting
each trial on a new, empty memory volume. Omega replays its saved history into every
model call, so trials that share a memory are not independent. Run it from the
SubRep root with Docker running:

```bash
export ASI_API_KEY='your-provider-key'
TRIALS=5 bash integration/omegaclaw/live_trials.sh
```

An OpenAI-compatible endpoint works too:

```bash
export OPENAIAPI_API_KEY='your-provider-key'
PROVIDER=OpenAIAPI OPENAIAPI_URL='https://your-endpoint/v1/' MODEL=minimax/minimax-m3 \
  TRIALS=1 bash integration/omegaclaw/live_trials.sh
```

Other settings are `TIMEOUT` (seconds per request), `IMAGE`, `SCENARIOS`, `OUT`, and
`PY`; the script header lists them all. The output folder holds one report per trial,
the audit trail, Omega's logs with the token masked, `comparison.md`, and
`review.csv`. To record a person's review, fill `reviewer_factually_correct` (yes or
no) and `reviewer_notes` in `review.csv`, save it as `reviewed.csv`, and rebuild the
comparison:

```bash
python -m integration.omegaclaw.compare_runs --report <trial report> [--report ...] \
  --output <folder>/comparison.md --reviewed-csv <folder>/reviewed.csv
```

## Tests

```powershell
python -m pytest integration/omegaclaw/tests -v
```

The WebSocket flow test starts the real SubRep gateway and a simulated Omega
protocol client. It verifies a complete transport and validation round trip but
is still labeled synthetic; only `--backend live` exercises the actual Omega
agent and configured model provider.

## Operational Notes

- One explanation request is in flight at a time because Omega's current
  channel does not natively correlate a response with an inbound sequence.
- Request IDs provide application-level correlation.
- Startup, progress, and unrelated Omega messages are not accepted as valid
  explanations.
- Audit files contain task context and model output. Protect them as decision
  records and do not include secrets in task context.
- The prompt tells Omega not to execute tools, but a natural-language prompt is
  not a sandbox. Run Omega isolated, do not mount the SubRep repository into its
  container, and keep SubRep's final stage explanation-only.
- Omega logs may retain prompts and outputs. Apply the same data handling policy
  to Omega logs as to the SubRep audit trail.
- The current Omega configuration logger may also record resolved WebSocket
  configuration values. Use a dedicated short-lived token and protect Omega logs.
