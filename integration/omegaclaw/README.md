# SubRep to Omega Explanation Integration

## What This Integrates

SubRep and Omega have different responsibilities:

- **SubRep** executes candidate policies, compares them with a baseline, certifies
  them with CDS/PDS, stores certificates, and decides which skills are admissible.
- **Omega** is a stateful neural-symbolic agent. The sibling checkout currently
  identifies itself as `Omega`, although the task and some upstream artifacts use
  the OmegaClaw name.
- **This package** sends an immutable, authoritative SubRep decision snapshot to
  Omega and accepts only an explanation that echoes that decision. It never asks
  Omega to select, certify, admit, exclude, or execute a skill.

The integration belongs in SubRep because SubRep owns the safety boundary and must
validate the external explanation before anything downstream can use it. The
Omega repository remains an independently runnable dependency.

## Safety Boundary

```text
SubRep SkillLibrary
  -> query_admissible()                 owned by SubRep
  -> apply local risk budget            owned by SubRep
  -> apply explicit exclusions          owned by SubRep
  -> compute final selection/abstention  owned by SubRep
  -> build versioned decision snapshot
  -> request explanation from Omega
  -> validate exact decision echo         owned by SubRep
  -> save request and response
  -> return decision plus explanation     no skill execution
```

Omega has no selection authority. SubRep deterministically computes the largest
`final_selection_score`, using lexical skill ID order for exact ties, before the
request leaves the process. Omega must echo `subrep_decision.selected_skill_id` and
`subrep_decision.abstain` exactly. Unknown evidence references, changed decisions,
unsupported modes, malformed JSON, and mismatched request IDs are rejected, and so
are explanations whose claims contradict the request (see [Validation](#validation)).

## Package Structure

| Path | Responsibility |
|---|---|
| `contracts.py` | Versioned request, response, risk, evidence, and outcome contracts |
| `service.py` | Calls SubRep admissibility and applies local risk/exclusion filters |
| `adapter.py` | Builds the prompt, parses output, validates it, and records failures |
| `validation.py` | Format checks on a response, and the factual checks on it and its advisory concerns |
| `claims.py` | Factual checks: compares what an explanation claims with the request |
| `websocket_gateway.py` | Private server for Omega's existing WebSocket client channel |
| `audit.py` | Append-only JSONL storage of complete inputs and outcomes |
| `scenarios.py` | Eight predefined acceptance scenarios using synthetic evidence |
| `synthetic_backend.py` | Deterministic non-LLM backend used only for tests and review |
| `demo.py` | Reproducible synthetic and live demo entry point |
| `live_trials.sh` | Repeated live trials, each with a fresh Omega memory |
| `compare_runs.py` | Compares run reports and writes the review sheet for a person |
| `tests/` | Contract, claims, validation, adapter, scenario, demo, and WebSocket flow tests |

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

Omega must return one JSON object containing:

```json
{
  "schema_version": "subrep.omegaclaw.explanation.response.v1",
  "request_id": "the supplied request ID",
  "mode": "EXPLANATION_ONLY",
  "selected_skill_id": "the exact SubRep decision skill ID or null",
  "abstain": false,
  "selection_basis": "SUBREP_DECISION",
  "explanation": "a short explanation using only supplied evidence",
  "evidence_refs": ["subrep_decision.selected_score"],
  "advisory_concerns": []
}
```

For a selection, `evidence_refs` must include `subrep_decision.selected_score` and,
when present, `subrep_decision.runner_up_score`. For abstention, it must include
`subrep_decision.reason_code`. Every reference must resolve to a path in the exact
request snapshot: a value, or a section that holds values, such as `objective_order`.
List items use dotted indexes (`exclusions.0.stage`); the bracket form
`exclusions[0].stage` is accepted and stored dotted. Optional advisory concerns are
non-binding, restricted to the codes defined in `contracts.py`, and must cite
evidence relevant to that specific concern. The explanation is at most 1,500
characters (`MAX_EXPLANATION_CHARS`). One invalid response may be retried; all
attempts remain in the audit record.

## Validation

Every response is checked before it is accepted, and each problem is recorded as a
finding with a category:

- **Format findings** concern the response itself: valid JSON, the exact schema, the
  request ID, the length limit, and citations that exist in the request.
- **Factual findings** come from comparing what the response says with the request.
  The explanation and every advisory-concern message are checked. The checks flag:
  - a decision echo that differs from SubRep's decision;
  - a skill called selected or runner-up that SubRep did not select or rank second,
    including a name that is not in the request;
  - an excluded skill described by the wrong stage, such as an operator exclusion
    called a certification failure;
  - scores, weights, or objective deltas given to the wrong skill or objective, or
    with the wrong sign ("decreased by 0.1" states a delta of -0.1);
  - numbers that are not in the request;
  - claims that a skill was executed, run, or deployed;
  - synthetic evidence that is presented as real, or not disclosed.
- An advisory concern must also cite the evidence that supports its code.

A rejected response gets one correction prompt that quotes its findings. The factual
checks match known kinds of wrong statements, one clause at a time, and ignore
negated or hypothetical ones ("not a certification failure", "would have selected").
A paraphrase they do not recognise passes, so a response without findings has no
errors detected by the checks; it is not proven accurate. A person reviews live
responses with the review sheet described under
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
