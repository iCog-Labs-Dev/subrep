"""Probe existing MDN trainers on synthetic data; never reads/writes checkpoints."""
import json
import numpy as np
import torch
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer
from generator.mdn_support_trainer import MDNSupportTrainer
from generator.train_mdn import build_records_from_prepared_candidate_outcomes, build_auxiliary_records_from_prepared_candidate_outcomes
from utils.mdn_record_builder import PreparedCandidateOutcome
from utils.weight_set_store import WeightSetStore


def main():
    torch.manual_seed(42)
    context = (0.1, 0.2, 0.3, 0.4)
    outcomes = [PreparedCandidateOutcome(context, 'idle', 0., (0.,0.,0.)),
                PreparedCandidateOutcome(context, 'move', 1., (.1,.5,-.1))]
    baseline = {'baseline_payoff':0., 'baseline_motives':[0.,0.,0.]}
    decisions = build_records_from_prepared_candidate_outcomes(prepared_outcomes=outcomes, baseline_stats=baseline, device='cpu')
    auxiliary = build_auxiliary_records_from_prepared_candidate_outcomes(prepared_outcomes=outcomes, baseline_stats=baseline, skill_id_bucket_count=16)
    model = MotiveDecompositionNetwork(input_dim=4, num_objectives=3, num_skills=16)
    x = torch.tensor(context)
    def probe(name, operation):
        model.zero_grad(set_to_none=True)
        before = {k:v.detach().clone() for k,v in model.support_head.named_parameters()}
        with torch.no_grad():
            values_before = model.forward_inference(x)[1].clone()
        operation()
        with torch.no_grad():
            values_after = model.forward_inference(x)[1]
        print(json.dumps({'phase':name,
            'support_head_has_gradient':any(p.grad is not None for p in model.support_head.parameters()),
            'support_head_parameters_changed':any(not torch.equal(before[k],v) for k,v in model.support_head.named_parameters()),
            'support_output_max_change':float((values_after-values_before).abs().max())}, indent=2))
    policy = MDNTrainer(model, device='cpu')
    probe('existing policy trainer', lambda: policy.training_step(decisions[0]))
    aux = MDNAuxiliaryTrainer(model, device='cpu')
    probe('existing auxiliary trainer', lambda: aux.online_step(auxiliary[1]))
    store = WeightSetStore(num_objectives=3)
    store.observe_certified_weight(np.array(context), np.array([.2,.3,.5]))
    support = MDNSupportTrainer(model, store, device='cpu')
    probe('separate support trainer (synthetic positive control)', support.training_step)
    print('Synthetic diagnostic only. No checkpoints saved. Output drift alone is not evidence of support-head learning.')


if __name__ == '__main__':
    main()
