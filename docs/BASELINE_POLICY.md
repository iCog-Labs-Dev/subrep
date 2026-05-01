# Baseline Policy and Improvement Computation

SubRep certification compares a skill against a fixed baseline, not against raw absolute values. The paper defines improvements as

$$
\Delta r(x, o) = \hat{r}(x, o) - \hat{r}(x, o_{base})
$$
$$
\Delta n(x, o) = \hat{n}(x, o) - \hat{n}(x, o_{base})
$$

This matters because the CDS and PDS gates operate on $\Delta r$ and $\Delta n$. A skill only passes **if it improves enough over the baseline to satisfy the gate condition**.

## Why a baseline is needed

Raw payoff and motive values are not directly certifiable. The certification gates needs to know whether a skill is better than the reference behavior under the same environment and discounting setup. The baseline gives a stable anchor for that comparison.

## Choosing a baseline

The simplest baseline is an idle or do-nothing policy. That is the default in this workspace because it is deterministic and easy to test.

Other possible choices:

- Idle policy: best for a conservative, stable reference.
- Random policy: sometimes useful for exploration, but less stable.
- Conservative hand-coded policy: useful when idle is too weak or unrealistic.

For certification, the important property is consistency. **The same baseline should be used for all skills being compared in a given experiment**.

## How improvements are computed

The baseline runner (`idle_policy.py`) collects discounted rollout summaries for several episodes and stores the mean payoff and mean motive vector. The improvement calculator (`improvement_calculator.py`) then computes:

- $\Delta r = r_{skill} - r_{base}$
- $\Delta n = n_{skill} - n_{base}$

Those values are then passed directly into CDS or PDS gates.

## Validation rules

Before admission, improvements should be checked for:

- finite scalar payoff improvement
- finite 1D motive vector
- matching motive dimensionality between skill and baseline

This prevents malformed statistics from reaching the certification gates.

## Practical note

In this repo, the baseline policy is deterministic and the baseline episodes are seeded so repeated runs with the same seed reproduce the same statistics. That makes certification tests easier to trust and debug.