"""Retrospective goal/cost diagnostics from saved episodes; no new rollouts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def classify(selection, execution, library, cost_limit):
    """Exclusive failure categories, plus candidate-level evidence for inspection."""
    w = np.asarray(selection['weight'], dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(),1):
        raise ValueError('Invalid priority weights')
    outcomes = execution['outcomes']
    admitted, eligible, ranked = set(), set(), []
    for skill_id, entry in library['skills'].items():
        name = skill_id.split(':')[0]
        if name not in outcomes: raise ValueError('Missing candidate execution')
        admitted.add(name)
        cert = entry['certificate']
        region = cert['weight_region_type']
        caps = cert.get('wx_support_values')
        if region not in ('FULL_SIMPLEX','MDN_WX'): raise ValueError('Unknown certificate region')
        inside = region == 'FULL_SIMPLEX'
        if not inside:
            caps = np.asarray(caps,dtype=float)
            if caps.shape != (3,) or not np.isfinite(caps).all(): raise ValueError('Invalid support caps')
            inside = bool(np.all(w <= caps+1e-9))
        if inside:
            eligible.add(name)
            ranked.append((-(cert['delta_r']+np.dot(w,cert['delta_n'])),skill_id,name))
    expected = min(ranked)[2] if ranked else 'zero_action'
    if selection['policy'] != expected or bool(selection['fallback']) != (not ranked):
        raise ValueError('Selection differs from saved certificate ranking')
    goals, suitable = set(), set()
    for name, result in outcomes.items():
        raw = np.asarray(result['undiscounted_motives'],dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(result['goals']):
            raise ValueError('Invalid executed outcomes')
        if name != 'zero_action' and result['goals'] > 0:
            goals.add(name)
            if -raw[0] <= cost_limit: suitable.add(name)
    selected = outcomes[expected]
    goal = selected['goals'] > 0
    if goal: category = 'goal_reached'
    elif not goals: category = 'no_goal_reaching_candidate'
    elif not suitable: category = 'goal_candidates_exceed_cost_limit'
    elif suitable & eligible: category = 'suitable_eligible_candidate_not_selected'
    elif suitable & admitted: category = 'suitable_candidates_outside_region'
    else: category = 'suitable_candidates_rejected'
    return dict(scenario=selection['scenario'],weight=w.tolist(),policy=expected,
        goal_reached=bool(goal),selected_raw_cost=float(-selected['undiscounted_motives'][0]),
        selected_meets_goal_and_cost=expected in suitable,fallback=bool(selection['fallback']),category=category,
        goal_candidates=sorted(goals),suitable_candidates=sorted(suitable),
        suitable_eligible=sorted(suitable & eligible),suitable_outside_region=sorted((suitable & admitted)-eligible),
        suitable_rejected=sorted(suitable-admitted))


def run(root,cost_limit):
    if not np.isfinite(cost_limit) or cost_limit<0: raise ValueError('Cost limit must be finite and nonnegative')
    root=Path(root);sources={}
    def read(path):
        raw=path.read_bytes();sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    manifest=read(root/'manifest.json');rows=[]
    for seed in manifest['config']['training_seeds']:
        folder=root/f'seed_{seed}';seen=set()
        for evidence in read(folder/'evidence.json'):
            context=evidence['context_seed']
            if context in seen: raise ValueError('Duplicate context')
            seen.add(context);local=folder/str(context)
            execution=read(local/'execution.json');library=read(local/'library.json')
            if execution['context_seed']!=context or not np.array_equal(execution['initial_observation'],evidence['initial_observation']):
                raise ValueError('Execution start mismatch')
            if set(execution['outcomes'])!=set(evidence['outcomes']): raise ValueError('Candidate sets differ')
            for selection in read(local/'selection.json'):
                rows.append(dict(training_seed=seed,context_seed=context,**classify(selection,execution,library,cost_limit)))
    summaries=[]
    for scenario in sorted({r['scenario'] for r in rows}):
        group=[r for r in rows if r['scenario']==scenario]
        summaries.append(dict(scenario=scenario,queries=len(group),goal_reached=sum(r['goal_reached'] for r in group),
            missed_goals=sum(not r['goal_reached'] for r in group),fallback=sum(r['fallback'] for r in group),
            suitable_available=sum(bool(r['suitable_candidates']) for r in group),
            suitable_selected=sum(r['selected_meets_goal_and_cost'] for r in group),
            categories=dict(Counter(r['category'] for r in group))))
    scope=('Retrospective diagnosis using each candidate’s saved re-execution. Suitable means goal reached and observed raw episode cost <= the specified limit. '
           'This is an episode-level diagnostic, not an expected-cost guarantee or an oracle usable at selection time. '
           'Categories describe observed bottlenecks, not proven causal bugs. Selection still maximizes the existing scalarized score, not binary goal success. '
           'Failure categories are exclusive: no goal candidate; no goal/cost-suitable candidate; suitable eligible candidate not selected; suitable admitted candidates outside region; all suitable candidates rejected. '
           'Candidate lists retain overlapping rejection/region details. Four queries reuse each context; they are not independent episodes.')
    out=root/'goal_diagnostics';out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(scope=scope,cost_limit=cost_limit,summary=summaries,rows=rows,source_sha256=sources),indent=2,allow_nan=False)+'\n')
    lines=['# Goal-success diagnostics','',scope,'',f'Observed episode cost threshold: **{cost_limit:g}**.','',
        '| Priority | Queries | Goals reached | Goal + cost suitable available | Suitable selected | Missed goals | Idle fallback |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:lines.append(f"| {s['scenario']} | {s['queries']} | {s['goal_reached']} | {s['suitable_available']} | {s['suitable_selected']} | {s['missed_goals']} | {s['fallback']} |")
    lines+=['','## Breakdown of missed goals','','| Priority | No candidate reaches goal | Goal candidates exceed cost limit | Suitable eligible, not selected | Suitable outside region | Suitable rejected |','|---|---:|---:|---:|---:|---:|']
    keys=['no_goal_reaching_candidate','goal_candidates_exceed_cost_limit','suitable_eligible_candidate_not_selected','suitable_candidates_outside_region','suitable_candidates_rejected']
    for s in summaries:lines.append('| '+s['scenario']+' | '+' | '.join(str(s['categories'].get(k,0)) for k in keys)+' |')
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('Saved:',out/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',required=True);p.add_argument('--cost-limit',type=float,default=1.)
    a=p.parse_args();run(a.benchmark_dir,a.cost_limit)
