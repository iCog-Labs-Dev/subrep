"""Offline paired baseline, gate-ablation, Pareto and priority-reuse analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from utils.analyze_contextual_baselines import score

STRATEGIES = ('full_simplex', 'full_plus_fixed', 'full_plus_mdn')
METRICS = ('score','task_return','safety_cost','effort','action_change','goal_success',
           'goal_budget','zero_cost','successful_reuse','fallback')


def query(evidence, execution, audits, selection, scales, caps, baseline_names, budget):
    """Evidence chooses policies; execution alone supplies outcome measurements."""
    outcomes=evidence['outcomes']; actual=execution['outcomes']
    names=sorted(set(outcomes)-{'zero_action'})
    if set(outcomes)!=set(actual) or not set(baseline_names.values())<=set(names):
        raise ValueError('Missing baseline or mismatched candidate pool')
    weight=np.asarray(selection['weight'])
    admitted={a['policy']:a for a in audits if a['admitted']}
    eligible=[n for n,a in admitted.items() if a['region']=='FULL_SIMPLEX' or np.all(weight<=np.asarray(caps)+1e-9)]
    def best(pool):
        return min(pool,key=lambda n:(-score(outcomes[n],outcomes['zero_action'],scales,weight),n)) if pool else 'zero_action'
    chosen=best(eligible)
    if chosen!=selection['policy'] or bool(selection['fallback'])!=bool(not eligible):
        raise ValueError('Saved and reconstructed selection disagree')
    unrestricted=best(names)
    pools={'subrep':[chosen], 'zero_action':['zero_action'],
           **{k:[v] for k,v in baseline_names.items()}, 'random_candidate':names,
           'random_certified':eligible or ['zero_action'], 'without_certification':[unrestricted]}
    rows=[]
    for method,pool in pools.items():
        measures=[]
        for name in pool:
            o=actual[name]; raw=np.asarray(o['undiscounted_motives']); sc=score(o,actual['zero_action'],scales,weight)
            if raw.shape!=(4,) or not np.isfinite(raw).all() or not np.isfinite(sc):
                raise ValueError('Invalid execution outcome')
            is_certified=method in ('subrep','random_certified')
            fallback=is_certified and not eligible
            measures.append([sc,raw[1],-raw[0],-raw[2],-raw[3],float(o['goals']>0),
                             float(o['goals']>0 and -raw[0]<=budget),float(-raw[0]<=1e-9),
                             float(is_certified and not fallback and sc>=-1e-9),float(fallback)])
        rows.append(dict(method=method,**dict(zip(METRICS,np.mean(measures,axis=0).tolist()))))
    return rows,dict(blocked=float(unrestricted not in eligible),
                     rejected=float(unrestricted not in admitted),
                     outside_region=float(unrestricted in admitted and unrestricted not in eligible))


def pareto_mask(points):
    """All columns are oriented to maximize. Equal points do not dominate."""
    a=np.asarray(points)
    return [not any(np.all(b>=p) and np.any(b>p) for b in a) for p in a]


def run(root, output, ppo, lagrangian, samples=2000, seed=42):
    root=Path(root);out=Path(output)
    if out.exists(): raise FileExistsError('Use a new output directory')
    if samples<100 or seed<0: raise ValueError('Invalid bootstrap settings')
    hashes={}
    def read(path):
        blob=path.read_bytes();hashes[str(path.relative_to(root))]=hashlib.sha256(blob).hexdigest()
        return json.loads(blob)
    manifest=read(root/'manifest.json'); comparison=read(root/'support_comparison.json')
    scales=np.asarray(manifest['scales']['divisors'])
    if scales.shape!=(4,) or not np.isfinite(scales).all() or np.any(scales<=0): raise ValueError('Invalid scales')
    contexts=comparison['contexts']; ids=[c['context_seed'] for c in contexts]
    if len(ids)<2 or len(set(ids))!=len(ids) or ids!=manifest['evaluation_seeds']: raise ValueError('Invalid context set')
    rows=[];ablations=[];filters=[]
    for c in contexts:
        local=root/str(c['context_seed']);e=read(local/'evidence.json');x=read(local/'execution.json')
        if e['context_seed']!=c['context_seed'] or x['context_seed']!=c['context_seed'] or not np.array_equal(e['initial_observation'],x['initial_observation']):
            raise ValueError('Context mismatch')
        for strategy in STRATEGIES:
            audits=c['audits'][strategy]
            if {a['policy'] for a in audits}!=set(e['outcomes'])-{'zero_action'}: raise ValueError('Audit pool mismatch')
            caps=c['caps'] if strategy=='full_plus_mdn' else manifest['fixed_caps']
            for a in audits:
                cost=-x['outcomes'][a['policy']]['undiscounted_motives'][0]
                higher=cost > -x['outcomes']['zero_action']['undiscounted_motives'][0]+1e-9
                filters.append(dict(strategy=strategy,context=c['context_seed'],admitted=a['admitted'],cost=cost,higher=higher))
            for selection in c['selections'][strategy]:
                rs,ab=query(e,x,audits,selection,scales,caps,
                            {'always_ppo':ppo,'always_ppo_lagrangian':lagrangian},manifest['settings']['cost_limit'])
                identity=dict(strategy=strategy,context=c['context_seed'],priority=selection['priority'])
                rows.extend(dict(**identity,**r) for r in rs);ablations.append(dict(**identity,**ab))
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(ids),(samples,len(ids)))
    summaries=[];paired=[];absummary=[];filter_summary=[]
    priorities=['overall']+list(dict.fromkeys(r['priority'] for r in rows))
    for strategy in STRATEGIES:
        f=[r for r in filters if r['strategy']==strategy]
        groupstats=dict(strategy=strategy,total=len(f))
        for status,label in [(True,'admitted'),(False,'rejected')]:
            subset=[r for r in f if r['admitted']==status]
            groupstats[label]=len(subset)
            groupstats[label+'_mean_cost']=float(np.mean([r['cost'] for r in subset])) if subset else None
            groupstats['higher_cost_'+label]=sum(r['higher'] for r in subset)
        filter_summary.append(groupstats)
        for priority in priorities:
            group=[r for r in rows if r['strategy']==strategy and (priority=='overall' or r['priority']==priority)]
            arrays={}
            for method in sorted({r['method'] for r in group}):
                entries=[r for r in group if r['method']==method]
                by_context={c:[] for c in ids}
                for r in entries: by_context[r['context']].append([r[k] for k in METRICS])
                a=np.array([np.mean(by_context[c],axis=0) for c in ids])
                arrays[method]=a
                summaries.append(dict(strategy=strategy,priority=priority,method=method,**dict(zip(METRICS,a.mean(0).tolist()))))
            for method,a in arrays.items():
                if method=='subrep':continue
                delta=arrays['subrep']-a; ci=np.quantile(delta[indices].mean(1),[.025,.975],axis=0)
                paired.append(dict(strategy=strategy,priority=priority,baseline=method,
                    differences=dict(zip(METRICS,delta.mean(0).tolist())),ci95={k:ci[:,i].tolist() for i,k in enumerate(METRICS)}))
            ab=[r for r in ablations if r['strategy']==strategy and (priority=='overall' or r['priority']==priority)]
            absummary.append(dict(strategy=strategy,priority=priority,**{k:float(np.mean([r[k] for r in ab])) for k in ('blocked','rejected','outside_region')}))
    pareto=[]
    for priority in priorities:
        group=[r for r in summaries if r['priority']==priority and (r['method']=='subrep' or r['strategy']=='full_plus_mdn')]
        points=[[r['task_return'],-r['safety_cost'],-r['effort'],-r['action_change']] for r in group]
        pareto.extend(dict(strategy=r['strategy'],method=r['method'],priority=priority,nondominated=flag)
                      for r,flag in zip(group,pareto_mask(points)))
    scope=('Offline paired replay; selection uses evidence, outcomes use saved execution. Raw task, cost, effort and action change are undiscounted. '
           'Score = scaled discounted task improvement + weighted scaled discounted objective improvements (task appears in both terms). '
           'Random selectors are exact uniform expectations over saved candidates, not new simulator runs. '
           'Without certification removes both admission and regional eligibility restrictions; blocked selections are split into rejected and outside-region cases. '
           'Reuse success means a nonfallback certified choice with execution score >= zero-action baseline; it does not mean goal or absolute safety success. '
           'Priority changes occur between reset episodes, not during a trajectory. Pareto membership uses four raw mean objectives and is descriptive, not statistically established dominance. '
           'Intervals resample paired contexts, not training seeds. Deterministic same-start execution is not an independent safety test.')
    report=dict(scope=scope,contexts=len(ids),baseline_policies={'always_ppo':ppo,'always_ppo_lagrangian':lagrangian},
                bootstrap_samples=samples,seed=seed,summary=summaries,paired_comparisons=paired,
                ablation=absummary,safety_filtering=filter_summary,pareto=pareto,rows=rows,source_sha256=hashes)
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# Four-objective end-to-end baseline analysis','',scope,'',f'Contexts: {len(ids)}. No policy retraining or additional simulation.','',
           '## Baselines and reuse','', '| Strategy | Priority | Method | Score | Task | Cost | Effort | Action change | Goal | Goal + budget | Reuse success* |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        reuse=f"{r['successful_reuse']:.1%}" if r['method'] in ('subrep','random_certified') else 'N/A'
        lines.append(f"| {r['strategy']} | {r['priority']} | {r['method']} | {r['score']:.3f} | {r['task_return']:.3f} | {r['safety_cost']:.3f} | {r['effort']:.3f} | {r['action_change']:.3f} | {r['goal_success']:.1%} | {r['goal_budget']:.1%} | {reuse} |")
    lines+=['','## With versus without CDS/PDS and region restrictions','', '| Strategy | Priority | With cost | Without cost | With goal | Without goal | Blocked | Rejected | Outside region |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in absummary:
        lookup={r['method']:r for r in summaries if r['strategy']==a['strategy'] and r['priority']==a['priority']};yes=lookup['subrep'];no=lookup['without_certification']
        lines.append(f"| {a['strategy']} | {a['priority']} | {yes['safety_cost']:.3f} | {no['safety_cost']:.3f} | {yes['goal_success']:.1%} | {no['goal_success']:.1%} | {a['blocked']:.1%} | {a['rejected']:.1%} | {a['outside_region']:.1%} |")
    lines+=['','## Safety filtering (execution costs)','', '| Strategy | Admitted | Rejected | Admitted mean cost | Rejected mean cost | Higher-cost admitted | Higher-cost rejected |','|---|---:|---:|---:|---:|---:|---:|']
    for r in filter_summary: lines.append('| '+' | '.join(str(r[k]) for k in ('strategy','admitted','rejected','admitted_mean_cost','rejected_mean_cost','higher_cost_admitted','higher_cost_rejected'))+' |')
    lines+=['','## Four-objective Pareto membership','', '| Priority | Nondominated strategies/methods |','|---|---|']
    for priority in priorities:
        labels=[r['strategy']+'/'+r['method'] for r in pareto if r['priority']==priority and r['nondominated']]
        lines.append(f"| {priority} | {', '.join(labels)} |")
    lines+=['','## Paired differences (SubRep minus baseline)','', '| Strategy | Priority | Baseline | Metric | Difference | 95% context interval |','|---|---|---|---|---:|---|']
    for r in paired:
        for metric in ('score','goal_success','safety_cost'):
            low,high=r['ci95'][metric];lines.append(f"| {r['strategy']} | {r['priority']} | {r['baseline']} | {metric} | {r['differences'][metric]:.3f} | [{low:.3f}, {high:.3f}] |")
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('Report:',out/'report.md')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--ppo-policy',required=True);p.add_argument('--lagrangian-policy',required=True)
    p.add_argument('--bootstrap-samples',type=int,default=2000);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir,a.ppo_policy,a.lagrangian_policy,a.bootstrap_samples,a.seed)

if __name__=='__main__':main()
