# Environment and Skill Execution

`env/` adapts MO-LunarLander to the 2-objective SubRep implementation and provides
the rollout executor used by data collection and certification.

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
- discounted 2D motive returns,
- terminal flag,
- run metadata in `last_run_info`.

The executor also supports loading the trained PPO pilot with
`SkillExecutor.from_pilot_checkpoint()`.

## Tests

```bash
python -m pytest tests/test_env.py tests/test_executor.py -v
```
