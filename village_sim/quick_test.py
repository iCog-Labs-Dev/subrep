from village_sim.baseline import run_policy, idle_policy
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
import numpy as np


def torch_corridor_policy(state):
    # torch_corridor needs 1 fuel; fall back to idle once exhausted.
    return "torch_corridor" if state.fuel >= 1 else "idle"


def main():
    # Expected return scale (Section 3.1: the scalar payoff is the discounted
    # task reward and is accumulated independently of the motive vector):
    #   * Scalar task payoff: exactly one task_reward in {-1, 0, +1} is awarded
    #     per episode, discounted by gamma^t. The per-episode discounted payoff
    #     is bounded in [-1, +1], so delta_r = skill - baseline lies in
    #     [-2, +2] and is typically of order 0.01-1.
    #   * Motive returns: each phi coordinate is in [0, 1] per step, so a
    #     discounted per-coordinate total is at most sum_{t=0}^{99} gamma^t =
    #     (1 - gamma^100) / (1 - gamma) ~= 63.4 at gamma=0.99. Each coordinate
    #     of delta_n therefore lies in roughly [-63.4, +63.4].
    #
    # PDS admission condition:  delta_r + min(delta_n) >= -epsilon.
    # The worst-case deficit |delta_r + min(delta_n)| lives on the discounted
    # motive-return scale (tens), so a budget that can absorb it must also be
    # on that scale (tens), not on the payoff scale (units). There is no
    # calibrated risk-preference or empirical distribution backing the value
    # chosen here: epsilon is a DEMONSTRATION value set just above the observed
    # worst-case deficit so the example exhibits a PDS admission. In practice
    # it would be calibrated from the MDN-distributed weight draws or a target
    # risk level.
    base_payoff, base_motives = run_policy(idle_policy, seed=1)
    skill_payoff, skill_motives = run_policy(torch_corridor_policy, seed=1)

    delta_r = skill_payoff - base_payoff
    delta_n = skill_motives - base_motives

    gate = CDSGate()
    admitted = gate.admit(delta_r, delta_n)
    print("Delta r:", round(delta_r, 4))
    print("Delta n:", np.round(delta_n, 4).tolist())
    print("CDS admitted:", admitted)

    # Worst-case score over the weight simplex is delta_r + min(delta_n).
    # The torch corridor drains Sustainability (fuel) to buy Safety and
    # Infrastructure; the motive deficit far exceeds the small task-payoff
    # gain, so CDS fails. PDS admits only if the deficit is within the budget.
    worst_case = delta_r + float(np.min(delta_n))
    # Break-even budget: PDS admits this skill iff epsilon >= -worst_case.
    epsilon_min = -worst_case

    # Demonstration budget on the discounted-motive-return scale (see above).
    epsilon = 60.0
    gate = PDSGate(epsilon=epsilon)
    admitted = gate.admit(delta_r, delta_n)
    print(f"Worst-case score: {worst_case:.4f}")
    print(f"Break-even epsilon (minimum that would admit): {epsilon_min:.4f}")
    print(f"PDS admitted (demonstration epsilon={epsilon}):", admitted)


if __name__ == "__main__":
    main()