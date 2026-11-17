# Epsilon scale on MO-LunarLander

These scripts measure why MetaMo's PDS budget ε could not change a LunarLander
admission, and which scale fixes that. They produced the unit
`LUNAR_LANDER_EPSILON_UNIT = 300.9` in `env/lunar_lander_metamo.py`, and the
choice of the guarded fixed unit in `bridge/epsilon_scale.py`. See "ε in
environment units" in `bridge/README.md` for the decision.

Run them from the repository root with the project's environment. They need
the tracked PPO checkpoint `models/pilot_ppo.pt`; `live.py` also needs a MetaMo
checkout. Outputs go to `runs/epsilon_scale/` (untracked).

| Script | What it does | Seeds | Time |
|---|---|---|---|
| `collect.py` | Scores the task, borderline and trade-off candidates against the noop baseline; one JSON row per seed and candidate | 1–40 | about 4 min |
| `analyse.py` | Measures U on seeds 1–20, then compares the ε rules on the held-out seeds 21–40 | reads `collect.py` output | seconds |
| `priorities.py` | Whether changing the Safety/Fuel weights changes the pick under the guarded rule | 21–40 | seconds |
| `live.py` | The real governor and controller with the scale on, beside the raw-ε verdicts | 21–36 | a few minutes |

```bash
python experiments/epsilon_scale/collect.py
python experiments/epsilon_scale/analyse.py
python experiments/epsilon_scale/priorities.py
python experiments/epsilon_scale/live.py
```

Measured with MetaMo `4171b1d`. Expected from `analyse.py`:

- **Scales:** s_r = 50.0, s_n = 250.9, U = 300.9, per-seed CV 0.234.
- **Guarded fixed unit, held-out seeds:**
  - no crash admitted at any ε;
  - 20 of 120 trade-off candidates change admission between ε 0 and 0.5;
  - every `ppo_deterministic` landing admitted at ε 0.1;
  - no monotonicity violations.
- **Raw ε:** no candidate changes admission.
- **Fuel** is the objective that binds in 592 of 800 pairs.

Re-measure U when the candidates, the noop baseline or the reward mapping
change, and update the constant and its comment together.
