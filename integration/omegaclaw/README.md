# SubRep to Omega Recommendation Integration

## What This Integrates

SubRep and Omega have different responsibilities:

- **SubRep** executes candidate policies, compares them with a baseline, certifies
  them with CDS/PDS, stores certificates, and decides which skills are admissible.
- **Omega** is a stateful neural-symbolic agent. The sibling checkout currently
  identifies itself as `Omega`, although the task and some upstream artifacts use
  the OmegaClaw name.
- **This package** sends an immutable SubRep decision snapshot to Omega and accepts
  only a recommendation or abstention. It never certifies, admits, or executes a
  skill.

The integration belongs in SubRep because SubRep owns the safety boundary and must
validate the external recommendation before anything downstream can use it. The
Omega repository remains an independently runnable dependency.

## Safety Boundary

```text
SubRep SkillLibrary
  -> query_admissible()                 owned by SubRep
  -> apply local risk budget            owned by SubRep
  -> apply explicit exclusions          owned by SubRep
  -> build versioned evidence snapshot
  -> request recommendation from Omega
  -> parse and validate response         owned by SubRep
  -> save request and response
  -> return recommendation only          no skill execution
```

Omega cannot add a skill to the admitted set. A non-abstaining response is valid
only when `selected_skill_id` appears in the exact `admitted_skills` snapshot sent
with that request. Unknown IDs, excluded IDs, malformed JSON, mismatched request
IDs, and backend failures are recorded as explicit outcomes.

## Package Structure

| Path | Responsibility |
|---|---|
| `contracts.py` | Versioned request, response, risk, evidence, and outcome contracts |
| `service.py` | Calls SubRep admissibility and applies local risk/exclusion filters |
| `adapter.py` | Builds the prompt, parses output, validates it, and records failures |
| `websocket_gateway.py` | Private server for Omega's existing WebSocket client channel |
| `audit.py` | Append-only JSONL storage of complete inputs and outcomes |
| `scenarios.py` | Four predefined acceptance scenarios using synthetic evidence |
| `synthetic_backend.py` | Deterministic non-LLM backend used only for tests and review |
| `demo.py` | Reproducible synthetic and live demo entry point |
| `tests/` | Contract, adapter, scenario, and WebSocket flow tests |
| `FINDINGS.md` | Short qualitative findings report |

## Data Contract

Each request contains:

- a unique request ID and timestamp,
- task context,
- named normalized objective weights,
- a local risk budget,
- locally admitted skills,
- each admitted skill's SubRep score and certificate evidence,
- explicit exclusions and SubRep-owned reason codes,
- an evidence label, either `OBSERVED` or `SYNTHETIC`.

Omega must return one JSON object containing:

```json
{
  "schema_version": "subrep.omegaclaw.recommendation.response.v1",
  "request_id": "the supplied request ID",
  "selected_skill_id": "an admitted skill ID or null",
  "abstain": false,
  "explanation": "a short explanation using only supplied evidence",
  "cited_skill_ids": ["IDs referenced by the explanation"]
}
```

When `abstain` is `true`, `selected_skill_id` must be `null`. When selecting, the
selected ID must be admitted and cited. Citations to unknown skills are invalid.

## Install

From the SubRep root:

```powershell
python -m pip install -r requirements.txt
```

The integration adds `websockets` for compatibility with Omega's `WSChannel`.

## Reproducible Synthetic Demo

This command runs all four predefined scenarios without an LLM and writes the
complete audit trail. Every input is labeled `SYNTHETIC`.

```powershell
python -m integration.omegaclaw.demo --backend synthetic `
  --audit-path data/omegaclaw/synthetic_demo.jsonl `
  --report-path data/omegaclaw/synthetic_demo_report.json
```

Expected recommendations:

| Scenario | Expected result |
|---|---|
| Clear preferred skill | `skill_safety` |
| Changed task priorities | `skill_efficiency` |
| Excluded preferred skill | `skill_safety` |
| No admissible options | Abstain |

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
  --audit-path data/omegaclaw/live_demo.jsonl
```

3. In Bash or WSL, start Omega with its WebSocket channel and the same token.
   The verified workspace run used ASICloud:

```bash
export WS_URL='ws://host.docker.internal:8765/agent'
export WS_TOKEN='replace-with-a-dedicated-random-token'
export ASI_API_KEY='your-provider-key'
./scripts/omega start -d omega:local -p ASICloud -t websocket
```

No recommendation text is copied into Telegram or IRC. The demo sends each
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

## Tests

```powershell
python -m pytest integration/omegaclaw/tests -v
```

The WebSocket flow test starts the real SubRep gateway and a simulated Omega
protocol client. It verifies a complete transport and validation round trip but
is still labeled synthetic; only `--backend live` exercises the actual Omega
agent and configured model provider.

## Operational Notes

- One recommendation request is in flight at a time because Omega's current
  channel does not natively correlate a response with an inbound sequence.
- Request IDs provide application-level correlation.
- Startup, progress, and unrelated Omega messages are not accepted as valid
  recommendations.
- Audit files contain task context and model output. Protect them as decision
  records and do not include secrets in task context.
- The prompt tells Omega not to execute tools, but a natural-language prompt is
  not a sandbox. Run Omega isolated, do not mount the SubRep repository into its
  container, and keep SubRep's final stage recommendation-only.
- Omega logs may retain prompts and outputs. Apply the same data handling policy
  to Omega logs as to the SubRep audit trail.
- The current Omega configuration logger may also record resolved WebSocket
  configuration values. Use a dedicated short-lived token and protect Omega logs.
