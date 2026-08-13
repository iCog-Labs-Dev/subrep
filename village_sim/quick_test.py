from village_sim.baseline import run_policy, idle_policy
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate 

def torch_policy(state):
    return "torch_corridor"

base_payoff, base_motives = run_policy(idle_policy, seed=1)
skill_payoff, skill_motives = run_policy(torch_policy, seed=1)

delta_r = skill_payoff - base_payoff
delta_n = skill_motives - base_motives

gate = CDSGate()
admitted = gate.admit(delta_r, delta_n)
print("Delta n:", delta_n)
print("CDS admitted:", admitted)

gate = PDSGate(epsilon=60.0)  # allow up to 60 worth of worst-case downside
admitted = gate.admit(delta_r, delta_n)
print("PDS admitted (epsilon=60):", admitted)