# Context-specific certification

Exact-start evidence and same-start re-execution only. Deterministic policies may repeat identical outcomes.
MDN checks: unavailable (no checkpoint supplied)

| Seed | Context | Policy | Result |
|---|---|---|---|
| 42 | 90011 | small_random | Rejected |
| 42 | 90011 | random | Rejected |
| 42 | 90011 | axis0_positive | Rejected |
| 42 | 90011 | axis0_negative | Rejected |
| 42 | 90011 | axis1_positive | Rejected |
| 42 | 90011 | axis1_negative | Rejected |
| 42 | 90011 | ppo | Rejected |
| 42 | 90011 | ppo_lagrangian | Rejected |
| 42 | 90012 | small_random | FULL_SIMPLEX/PDS |
| 42 | 90012 | random | Rejected |
| 42 | 90012 | axis0_positive | Rejected |
| 42 | 90012 | axis0_negative | Rejected |
| 42 | 90012 | axis1_positive | Rejected |
| 42 | 90012 | axis1_negative | Rejected |
| 42 | 90012 | ppo | Rejected |
| 42 | 90012 | ppo_lagrangian | Rejected |
| 42 | 90013 | small_random | Rejected |
| 42 | 90013 | random | Rejected |
| 42 | 90013 | axis0_positive | Rejected |
| 42 | 90013 | axis0_negative | Rejected |
| 42 | 90013 | axis1_positive | Rejected |
| 42 | 90013 | axis1_negative | Rejected |
| 42 | 90013 | ppo | Rejected |
| 42 | 90013 | ppo_lagrangian | Rejected |
| 42 | 90014 | small_random | FULL_SIMPLEX/PDS |
| 42 | 90014 | random | Rejected |
| 42 | 90014 | axis0_positive | Rejected |
| 42 | 90014 | axis0_negative | Rejected |
| 42 | 90014 | axis1_positive | Rejected |
| 42 | 90014 | axis1_negative | Rejected |
| 42 | 90014 | ppo | Rejected |
| 42 | 90014 | ppo_lagrangian | Rejected |

## Re-execution outcomes

| Seed | Context | Priority | Policy | Fallback | Safety cost | Task return | Effort | Goals |
|---|---|---|---|---|---|---|---|---|
| 42 | 90011 | balanced | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90011 | safety | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90011 | task | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90011 | effort | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90012 | balanced | small_random | False | -0.0000 | -0.0176 | 1.7620 | 0 |
| 42 | 90012 | safety | small_random | False | -0.0000 | -0.0176 | 1.7620 | 0 |
| 42 | 90012 | task | small_random | False | -0.0000 | -0.0176 | 1.7620 | 0 |
| 42 | 90012 | effort | small_random | False | -0.0000 | -0.0176 | 1.7620 | 0 |
| 42 | 90013 | balanced | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90013 | safety | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90013 | task | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90013 | effort | zero_action | True | -0.0000 | 0.0000 | -0.0000 | 0 |
| 42 | 90014 | balanced | small_random | False | -0.0000 | 0.0385 | 1.8003 | 0 |
| 42 | 90014 | safety | small_random | False | -0.0000 | 0.0385 | 1.8003 | 0 |
| 42 | 90014 | task | small_random | False | -0.0000 | 0.0385 | 1.8003 | 0 |
| 42 | 90014 | effort | small_random | False | -0.0000 | 0.0385 | 1.8003 | 0 |
