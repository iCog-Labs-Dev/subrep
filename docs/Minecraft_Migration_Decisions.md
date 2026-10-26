# SubRep Minecraft Migration — Architecture Decision Records

**Status:** Draft for review · **Date:** 2026-10-19 · **Scope:** Minecraft migration only
**Roadmap:** [Minecraft Environment and SubRep Prototype Roadmap](https://docs.google.com/document/d/1emAfORlDPkJTZsKA48gmML7VCIvw0VPKoGDaZgZjkSM/edit?usp=sharing) (verified 2026-10-19 — matches the phase plan used for this document)

## Why this exists

The full roadmap gates all real-server work on a complete Phase 0 (RFC, ownership,
threat model, SLOs, release tiers). Phases 1–4 are already built without it, so
enforcing the full gate now would be paperwork for its own sake. This lite version
keeps only the decisions that **constrain implementation choices** — the ones that
are expensive to retrofit once a server, sidecar, and real certificates exist.
Everything else is explicitly deferred (see §5).


---

## ADR-001 — Canonical motive order

**Context:** Every motive vector, certificate, library entry, and report must agree
on coordinate order, or 6D artifacts become silently incompatible.

**Decision:** The one canonical order, everywhere, is
`[Safety, Reputation, DeadlineSlack, InventoryValue, Sustainability, Infrastructure]`.
Larger value = better outcome on every coordinate. The production estimator is the
discounted sum `n_hat(x, o) = E[sum gamma^t · phi(x_t)]` over bounded, normalized
`phi`. Never replace it with an average or transition-delta signal in production
artifacts (research variants use a separate schema/namespace).

**Consequence:** `schemas/minecraft_objective_schema.py` is the single source of
truth. Any new 6D producer must validate vectors against it.

## ADR-002 — Scalar payoff stays separate from motives

**Context:** The paper's backed-up value is `r_hat + w^T n_hat`. If payoff is
computed as a sum of motives, the two terms collapse and the gates become circular.

**Decision:** Task payoff is defined by contract completion (delivery before
sunrise), versioned under its own payoff schema, and accumulated independently of
the motive vector. `info["task_payoff"]` is the only runtime payoff channel.

**Consequence:** `village_sim/baseline.py` already enforces this; the real
Minecraft adapter (Phase 6) must do the same. No payoff-from-motives shortcuts,
even in demos.

## ADR-003 — FULL_SIMPLEX first, no 6D contextual geometry yet

**Context:** The existing MDN weight-geometry machinery was designed around 2
objectives. Its 6D code path is a shape-only Softplus fallback with no validated
semantics.

**Decision:** All six-dimensional certification uses FULL_SIMPLEX until Phase 12.
The 2D `MDN_WX` support parameterization must not be used for 6D, and 6D CVaR
claims are out of scope until the explicit-constraint representation
(`W_x = {w >= 0, sum w = 1, Uw <= h}` with an LP solver) lands.

**Consequence:** Phase 4's shapes-only 6D MDN smoke tests are the correct scope.
No training or selection code may treat 6D support values as meaningful geometry.

## ADR-004 — Sidecar security model

**Context:** Phase 5 introduces a Node.js sidecar that accepts action commands.
An unauthenticated command endpoint is the single biggest new risk in the system.

**Decision:**
- Single-host deployments bind all local services to `127.0.0.1`. No
  unauthenticated HTTP servers, ever.
- Separate service identities with scoped, expiring credentials for: reset
  administration, action submission, action status, telemetry, emergency stop.
  No permanent shared tokens.
- Typed enum actions behind a strict allowlist. Coordinates, payload sizes, and
  scenario boundaries are validated; rate limits, timeouts, cancellation, and an
  emergency stop are mandatory.
- Forbidden: freeform server/chat/slash commands, shell commands, NBT scripts,
  model-generated commands, `shell=True` anywhere in the path.
- Snapshot restore must reject path traversal and symlinks, verify file hashes,
  restore atomically under an exclusive lock, keep base snapshots immutable, and
  reject partial restores.

**Consequence:** These are build-time constraints on Phase 5, not a review
checklist to run after it's built.

## ADR-005 — Determinism = snapshot restore + digest, not seeds

**Context:** Seed control alone cannot reproduce a Minecraft world (entity state,
raid RNG, tick ordering).

**Decision:** Every reset restores an immutable snapshot and computes a state
digest that is verified before any rollout proceeds. Digest mismatch tolerance is
0% — a mismatch fails the run closed, it is never retried silently.

**Consequence:** Phase 5's exit gate (100/100 staging resets with matching
digests) is the acceptance test for this ADR.

## ADR-006 — Artifact identity and domain separation

**Context:** Synthetic proxy artifacts and real Minecraft artifacts share the
same 6D shape. Mixing them would silently corrupt certification.

**Decision:** Every dataset, certificate, library entry, report, and checkpoint
carries `domain_id`, environment ID, schema versions, motive names/order,
producer version, seeds, and checksums. The synthetic domain
(`village_synthetic`) and the real domain (`minecraft_village_defense_trade`)
are never mixed at runtime. Schema-less legacy artifacts are loadable for
migration but quarantined from selection and admission (the
`schemas_compatible` vs `schemas_loadable` distinction already in the codebase
is the enforcement mechanism — keep it).

**Consequence:** Phase 6+ producers must populate the full identity triple;
reviewers reject any producer that doesn't.

## ADR-007 — Feature flags

**Context:** Minecraft work must not break the working LunarLander and
Safety-Gymnasium paths, and learned components must degrade safely.

**Decision:** Eight flags, all default OFF, overridable via
`SUBREP_FLAG_<NAME>` env vars:
`minecraft_env_enabled`, `real_server_enabled`, `paired_rollout_enabled`,
`minecraft_certification_enabled`, `skill_generator_enabled`, `mdn_ranking_enabled`,
`rl_options_enabled`, `contextual_geometry_enabled`.
When a learned-component flag is off, the system falls back to selection over
certified scripted options — never to uncertified actions.

**Consequence:** Implement the flag mechanism now (prompt 1 in
`OPENCODE_PROMPTS.md`). Wiring flags into runtime paths happens per-phase as
each component lands.

## ADR-008 — Paired evaluation discipline

**Context:** Certification is only meaningful if candidate and baseline faced the
same world.

**Decision:** Candidate and baseline rollouts always start from the identical
restored snapshot, verified by matching initial digests. Failed, timed-out,
cancelled, and digest-mismatched runs stay visible in the dataset and in the
denominator — they are never silently dropped. No option is certified from a
single lucky trajectory, and predicted (MDN) values never bypass measured
admission gates.

**Consequence:** Phase 4's evidence persistence must be extended to rejected
candidates (prompt 4). Phase 7's rollout collector is built around this rule
from day one.

---

## 5. Explicitly deferred (revisit pre-Phase 8)

- Ownership matrix and escalation contacts
- SLIs/SLOs and release tiers
- Full rollback drill and promotion policy
- Nightly determinism suite (1,000 resets / 5 snapshots)
- Eight-hour soak and fault-injection suites

These earn their keep once a real server is running with multiple contributors.
Deferring them does not block Phase 5.

