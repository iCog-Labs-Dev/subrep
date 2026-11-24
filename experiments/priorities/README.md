# Priorities and the LunarLander pick

These scripts measure how far the Safety/Fuel weights have to move to change
which admitted skill SubRep selects, and how far MetaMo actually moves them.
The findings are in "How far priorities have to move" in
`docs/subrep_metamo_integration.md`.

Run them from the repository root with the project's environment, on Linux:
flight results differ slightly by platform. They need the tracked PPO
checkpoint `models/pilot_ppo.pt`; `reachable_weights.py` also needs a MetaMo
checkout. Outputs go to `runs/priorities/` (untracked). Seeds 1–40 only.

| Script | What it does | Seeds | Time |
|---|---|---|---|
| `switch_weights.py` | Every Safety weight at which the pick changes, on the `ppo` and `tradeoff` candidate sets, with admission held at ε 0.1 (guarded fixed-unit rule) | 1–40, from `experiments/epsilon_scale/collect.py` output | seconds |
| `reachable_weights.py` | The Safety weights the mapping can reach and the demo visits under both risks, and on how many steps MetaMo's movement changed the pick | 1–16 | about 2 min |

```bash
python experiments/epsilon_scale/collect.py
python experiments/priorities/switch_weights.py
python experiments/priorities/reachable_weights.py
```

Measured with MetaMo `4171b1d`. Expected:

- **Switch points,** ε 0.1: `ppo` 12 on 12/40 seeds; `tradeoff` 41 on 31/40
  seeds, 17 of them above Safety 0.9.
- **Reachable Safety weight** at the initial goals: 0.250–0.899.
- **Visited,** seeds 1–16: 0.600–0.722 (`ppo`, motive risk), 0.255–0.632
  (`ppo`, task risk), 0.630–0.747 and 0.261–0.632 (`tradeoff` with the ε
  scale), 0.632–0.914 (`ci`, task risk).
- **Steps where MetaMo's movement changed the pick:** 0, 2, 1, 2 and 0 of 16,
  in the same order.
