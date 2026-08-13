from village_sim.baseline import run_policy, idle_policy
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
import numpy as np


def torch_corridor_policy(state):
    return "torch_corridor"


def main():
    base_payoff, base_motives = run_policy(idle_policy, seed=1)
    skill_payoff, skill_motives = run_policy(torch_corridor_policy, seed=1)

    delta_r = skill_payoff - base_payoff
    delta_n = skill_motives - base_motives

    gate = CDSGate()
    admitted = gate.admit(delta_r, delta_n)
    print("Delta r:", round(delta_r, 4))
    print("Delta n:", np.round(delta_n, 4).tolist())
    print("CDS admitted:", admitted)

    # Worst-case score over the simplex is delta_r + min(delta_n). The torch
    # corridor trades Sustainability (fuel drain) for Safety/Infrastructure;
    # CDS fails, so use a PDS budget on the discounted motive-return scale.
    worst_case = delta_r + float(np.min(delta_n))
    epsilon = 30.0
    gate = PDSGate(epsilon=epsilon)
    admitted = gate.admit(delta_r, delta_n)
    print(f"PDS admitted (epsilon={epsilon}, worst-case score={worst_case:.4f}):", admitted)


if __name__ == "__main__":
    main()