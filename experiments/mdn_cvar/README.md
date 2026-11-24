# PDS and CVaR on LunarLander, with the untrained and the trained MDN

`compare_gates.py` evaluates the PDS and CVaR gates apart and together (OR, and
AND for reporting), with the demo's untrained network and the trained
`models/lunar_mdn.pth`, and checks how often MetaMo's weights lie inside the
MDN's support region. The findings are in "Trained MDN and CVaR" in
`docs/subrep_metamo_integration.md`; the relation between MetaMo's weights and
the MDN's region is in "MetaMo weights and the MDN region" in
`bridge/README.md`.

Run it from the repository root with the project's environment, on Linux:
flight results differ slightly by platform. It needs the tracked checkpoints
`models/pilot_ppo.pt` and `models/lunar_mdn.pth` and a MetaMo checkout.
Outputs go to `runs/mdn_cvar/` (untracked). Seeds 1–40 only.

| Part | What it does | Seeds |
|---|---|---|
| Offline | Admitted counts per configuration at (ε, α) ∈ {(0.1, 0.1), (0.0, 0.05), (0.3, 0.3), (0.37, 0.43)}, both MDNs; the PDS rejections CVaR overturns under OR | 1–40, from `experiments/epsilon_scale/collect.py` output |
| α direction | Two landings' CVaR at α ∈ {0.05, 0.1, 0.2, 0.43, 0.5}, on the same samples | 1–40 |
| Live | The demo with the trained MDN, `tradeoff --epsilon-scale on`, every `--gates` value under both risks; on the CVaR-gated runs, what would change on the steps outside the region if PDS decided them | 1–16 |

```bash
python experiments/epsilon_scale/collect.py
python experiments/mdn_cvar/compare_gates.py
```

It takes about 5 minutes. Measured with MetaMo `4171b1d`. Expected:

- **Trained MDN, ε 0.1, α 0.1:** PDS 191, CVaR 299, OR 299, AND 191 of 560
  records; CVaR admits all 132 fuel-wasting landings and 92 of 97 timeouts.
- **Untrained MDN, ε 0.1, α 0.1:** PDS 191, CVaR 186, OR 191, AND 186.
- **α direction:** CVaR non-decreasing in α for every record checked.
- **Live:** the same selections under every `--gates` value; MetaMo's weights
  inside the MDN's region on 9/16 steps (motive risk) and 0/16 (task risk).
