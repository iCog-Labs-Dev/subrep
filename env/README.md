# Environment and Skill Execution

`env/` adapts benchmark environments to SubRep's motive-vector interface and
provides the rollout executor used by data collection and certification.

## MO-LunarLander Wrapper

`lunar_lander_wrapper.py` maps the raw 4-objective MO-Gymnasium reward into two
SubRep motives:

```text
Safety = terminal result + dense shaping
Fuel   = -(main engine cost + side engine cost)
```

Contract:

- observation shape: `(8,)`
- reward shape: `(2,)`
- reward order: `[Safety, Fuel]`

## Skill Executor

`skill_executor.py` runs any policy callable with signature `policy(obs) -> action`
or `policy(obs) -> (action, behavior_probability)`.

It returns:

- discounted scalar payoff,
- discounted motive returns,
- terminal flag,
- run metadata in `last_run_info`.

The executor also supports loading the trained PPO pilot with
`SkillExecutor.from_pilot_checkpoint()`.

## Optional Safety-Gymnasium Wrapper

`safety_gymnasium_wrapper.py` adapts Safety-Gymnasium benchmark environments to
SubRep motives. The default `objective_mode="2d"` interface is:

```text
Safety = -cost
Task   = reward
```

For stronger benchmarks after the M-objective support-geometry update, use:

```text
objective_mode="3d": [Safety, Task, ControlEfficiency]
objective_mode="4d": [Safety, Task, ControlEfficiency, ActionSmoothness]
```

Control efficiency is `-control_scale * ||action||`, and action smoothness is
`-smoothness_scale * ||action_t - action_{t-1}||`, so larger values are better
for every motive. Both scales default to `0.01` so the extra objectives do not
dominate reward/cost by raw magnitude.

This keeps safety costs compatible with the rest of the SubRep code, where
larger motive values are better. The first recommended pilot environment is
`SafetyPointGoal1-v0` because it exposes both task reward and safety cost while
remaining simpler than robot agents such as Car, Ant, or Doggo.

## Tests

```bash
python -m pytest tests/test_env.py tests/test_executor.py tests/test_safety_gymnasium_adapter.py -v
```
