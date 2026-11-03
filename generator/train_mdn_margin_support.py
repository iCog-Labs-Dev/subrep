"""Optional margin training for the existing capped-simplex head.

Adaptation of draft Section 2.6 to the existing CDS/PDS-epsilon gates.
Not the draft's coverage-PDS algorithm. No synthetic support targets are used.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.mdn_checkpoint_loader import load_mdn_checkpoint


def worst_case(delta, caps):
    """Exact fractional-knapsack minimum, differentiable in caps almost everywhere.

    delta: (..., M); caps broadcastable to delta. At active-set changes PyTorch
    supplies a subgradient; finite-difference checks should avoid these kinks.
    """
    if delta.ndim < 1 or caps.ndim < 1 or delta.shape[-1] == 0 or delta.shape[-1] != caps.shape[-1]:
        raise ValueError('Expected matching nonempty objective dimensions')
    if not torch.isfinite(delta).all() or not torch.isfinite(caps).all():
        raise ValueError('Nonfinite geometry')
    if ((caps < 0) | (caps > 1)).any() or (caps.sum(-1) < 1 - 1e-6).any():
        raise ValueError('Infeasible capped simplex')
    delta, caps = torch.broadcast_tensors(delta, caps)
    order = delta.argsort(dim=-1)
    values = delta.gather(-1, order)
    limits = caps.gather(-1, order)
    remaining = torch.ones_like(values[..., 0])
    score = torch.zeros_like(remaining)
    for i in range(delta.shape[-1]):
        mass = torch.minimum(remaining, limits[..., i])
        score = score + mass * values[..., i]
        remaining = (remaining - mass).clamp_min(0)
    return score


def margin_loss(caps, delta, epsilon, regularization):
    if caps.ndim != 2 or delta.ndim != 3 or caps.shape != (delta.shape[0], 3) or delta.shape[-1] != 3:
        raise ValueError('Safety adapter requires caps (B,3) and outcomes (B,K,3)')
    if not np.isfinite([epsilon, regularization]).all() or min(epsilon, regularization) < 0:
        raise ValueError('Invalid loss settings')
    # Fixed eligibility based only on recorded outcomes, never model predictions.
    # Exclude idle before calling; focus on positive-task policies which can pass
    # at at least one simplex weight. This does not demand every policy pass.
    eligible = (delta[..., 1] > 1e-6) & (delta[..., 1] + delta.max(-1).values + epsilon > 1e-6)
    margins = delta[..., 1] + worst_case(delta, caps[:, None, :]) + epsilon
    errors = torch.relu(-margins) * eligible
    counts = eligible.sum(-1)
    per_context = errors.sum(-1) / counts.clamp_min(1)
    active = counts > 0
    fit = per_context[active].mean() if active.any() else caps.sum() * 0
    # Explicit full-simplex anchor discourages shrinking just to pass.
    # This is an implementation choice, not a paper-prescribed coefficient.
    anchor = (1 - caps).square().mean()
    loss = fit + regularization * anchor
    return loss, {'margin_loss': float(fit.detach()), 'anchor_loss': float(anchor.detach()),
                  'eligible_pairs': int(eligible.sum()),
                  'eligible_pass_rate': float((margins[eligible] >= 0).float().mean()) if eligible.any() else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--scales-file', required=True)
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--epsilon', type=float, default=.2, help='Use 0 for CDS')
    p.add_argument('--regularization', type=float, default=.1)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not np.isfinite([args.epsilon, args.regularization, args.learning_rate]).all() or min(args.epsilon,args.regularization) < 0 or args.learning_rate <= 0 or args.epochs < 1:
        p.error('Invalid training settings')
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    torch.manual_seed(args.seed)
    scales = np.asarray(json.loads(Path(args.scales_file).read_text())['divisors'])
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Three finite positive scales required')
    contexts, deltas, sources = [], [], []
    observation_shape = None
    dataset_contract = None
    seen = set()
    names_reference = None
    for path in sorted(Path(args.data_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            contract = (str(data['env_id']), float(data['gamma']), tuple(str(name) for name in data['objective_names']))
            if contract[0] != 'SafetyPointGoal1-v0' or contract[2] != ('Safety', 'Task', 'ControlEfficiency') or not 0 <= contract[1] <= 1:
                raise ValueError('Expected SafetyPointGoal1 safety/task/effort schema')
            if dataset_contract is not None and contract != dataset_contract:
                raise ValueError('Mixed environments, discounts or objective order')
            dataset_contract = contract
            names = data['candidate_skill_ids'].astype(str)
            if names.ndim != 1 or len(set(names)) != len(names) or len(names) < 2:
                raise ValueError('Expected distinct candidate IDs including a non-idle candidate')
            if names_reference is None:
                names_reference = names
            if not np.array_equal(names, names_reference) or (names == 'zero_action').sum() != 1:
                raise ValueError('Require consistent policy order and one idle baseline')
            if not bool(data['values_are_raw']):
                raise ValueError('Expected raw outcomes')
            x = np.asarray(data['context'], dtype=np.float32)
            if x.ndim != 1 or not len(x) or (observation_shape is not None and x.shape != observation_shape):
                raise ValueError('Expected consistent flat contexts')
            observation_shape = x.shape
            key = tuple(np.round(x, 4))
            if key in seen:
                raise ValueError('Duplicate context; refusing split leakage')
            seen.add(key)
            v = np.asarray(data['candidate_motives'], dtype=np.float32)
            if v.shape != (len(names), 3) or not np.isfinite(v).all() or not np.isfinite(x).all():
                raise ValueError('Invalid outcomes or contexts')
            contexts.append(x)
            deltas.append(((v-v[names == 'zero_action'][0])/scales)[names != 'zero_action'])
            sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(contexts) < 10:
        raise ValueError('At least ten distinct contexts required')
    model = load_mdn_checkpoint(args.base_checkpoint, map_location='cpu')
    if model.num_objectives != 3 or len(contexts[0]) != model.input_dim:
        raise ValueError('Checkpoint dimensions do not match data')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.support_head.parameters():
        parameter.requires_grad_(True)
    model.eval()  # Freeze representation behavior as well as its parameters.
    x = torch.tensor(np.stack(contexts))
    d = torch.tensor(np.stack(deltas), dtype=torch.float32)
    indices = np.random.default_rng(args.seed).permutation(len(x))
    a, b = int(.7*len(x)), int(.85*len(x))
    splits = {'train': indices[:a], 'validation': indices[a:b], 'test': indices[b:]}
    optimizer = torch.optim.Adam(model.support_head.parameters(), lr=args.learning_rate)
    def evaluate(ids):
        caps = model.forward_inference(x[ids])[1]
        return margin_loss(caps, d[ids], args.epsilon, args.regularization)
    with torch.no_grad():
        best, _ = evaluate(splits['validation'])
        best = float(best)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = evaluate(splits['train'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.support_head.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            val, metrics = evaluate(splits['validation'])
        if not torch.isfinite(val):
            raise ValueError('Nonfinite validation loss')
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'validation_loss': float(val)})
        if float(val) < best:
            best, best_epoch = float(val), epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 50 == 0:
            print(history[-1], flush=True)
    model.load_state_dict(best_state)
    with torch.no_grad():
        test_loss, test_metrics = evaluate(splits['test'])
        full_loss, full_metrics = margin_loss(torch.ones(len(splits['test']),3), d[splits['test']], args.epsilon,args.regularization)
    report = {'dataset_contract': dataset_contract, 'settings': vars(args), 'best_epoch': best_epoch, 'test_loss': float(test_loss),
              'test': test_metrics, 'full_simplex_loss': float(full_loss), 'full_simplex': full_metrics,
              'history': history, 'sources': sources, 'splits': {k:v.tolist() for k,v in splits.items()},
              'base_sha256': hashlib.sha256(Path(args.base_checkpoint).read_bytes()).hexdigest(),
              'scales_sha256': hashlib.sha256(Path(args.scales_file).read_bytes()).hexdigest(),
              'scope': 'Head-only development split; base model may have seen these contexts. Fresh end-to-end evaluation required.',
              'method': 'Positive-task potentially feasible non-idle policies; equal context weighting; hinge loss plus full-simplex squared-distance anchor. Adaptation, not exact draft replication.'}
    out.mkdir(parents=True)
    torch.save({'model_state_dict':model.state_dict(), 'support_training':report},out/'support.pth')
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('Saved:',out/'support.pth')


if __name__ == '__main__':
    main()
