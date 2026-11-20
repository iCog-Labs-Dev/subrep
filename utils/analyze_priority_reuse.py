"""Offline changing-priority queries at fixed starts; no within-episode switching."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.analyze_contextual_baselines import score


def schedules(points):
    if points < 2: raise ValueError('At least two points per leg required')
    anchors = np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.8,.1,.1]])
    gradual = np.concatenate([np.linspace(a,b,points,endpoint=False) for a,b in zip(anchors[:-1],anchors[1:])] + [anchors[-1:]])
    abrupt = np.repeat(anchors,points,axis=0)
    return {'gradual':gradual,'abrupt':abrupt}


def query(record, execution, library, scales, weight):
    weight = np.asarray(weight)
    if weight.shape != (3,) or not np.isfinite(weight).all() or np.any(weight < 0) or not np.isclose(weight.sum(),1):
        raise ValueError('Invalid simplex query')
    candidates = []
    for skill_id, entry in library['skills'].items():
        cert = entry['certificate']
        caps = cert.get('wx_support_values')
        if cert['weight_region_type'] == 'FULL_SIMPLEX' or (caps is not None and np.all(weight <= np.asarray(caps)+1e-9)):
            candidates.append((skill_id,cert))
    # Use stored certificate evidence and the runtime selector's lexical tie break.
    chosen = min(candidates,key=lambda item: (-(item[1]['delta_r']+np.dot(weight,item[1]['delta_n'])),item[0])) if candidates else None
    name = chosen[0].split(':')[0] if chosen else 'zero_action'
    actual = execution['outcomes'][name]
    actual_score = score(actual,execution['outcomes']['zero_action'],scales,weight)
    return dict(policy=name,available=float(bool(candidates)),fallback=float(not candidates),
                successful_reuse=float(chosen is not None and actual_score >= -1e-9),
                strict_improvement=float(chosen is not None and actual_score > 1e-9),
                score=actual_score,goal_success=float(actual['goals']>0),
                raw_cost=-actual['undiscounted_motives'][0],task_return=actual['motives'][1],
                effort=-actual['motives'][2],eligible_count=len(candidates))


def run(root,points=20):
    root = Path(root); sources = {}
    def read(path):
        raw=path.read_bytes();sources[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    manifest=read(root/'manifest.json'); records=[]
    for seed in manifest['config']['training_seeds']:
        folder=root/f'seed_{seed}'; scales=np.asarray(read(folder/'scales.json'))
        if scales.shape!=(3,) or not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Invalid scales')
        seen=set()
        for record in read(folder/'evidence.json'):
            context=record['context_seed']
            if context in seen:raise ValueError('Duplicate context')
            seen.add(context);local=folder/str(context)
            execution=read(local/'execution.json');library=read(local/'library.json')
            if context!=execution['context_seed'] or not np.array_equal(record['initial_observation'],execution['initial_observation']):raise ValueError('Context mismatch')
            # Verify replay matches the original selector at its recorded priorities.
            for selection in read(local/'selection.json'):
                replay=query(record,execution,library,scales,selection['weight'])
                if replay['policy']!=selection['policy'] or bool(replay['fallback'])!=selection['fallback']:raise ValueError('Saved selection mismatch')
            records.append((seed,context,record,execution,library,scales))
    all_results={}; summaries={}; details=[]
    metrics=['available','fallback','successful_reuse','strict_improvement','score','goal_success','raw_cost','task_return','effort','eligible_count','switch_rate']
    for schedule,weights in schedules(points).items():
        curve=[];previous={}
        for step,weight in enumerate(weights):
            rows=[]
            for seed,context,record,execution,library,scales in records:
                r=query(record,execution,library,scales,weight)
                key=(seed,context)
                r['switch_rate']=float(key in previous and previous[key]!=r['policy'])
                previous[key]=r['policy'];rows.append(r)
                details.append(dict(schedule=schedule,step=step,training_seed=seed,context_seed=context,**r))
            curve.append(dict(step=step,weight=weight.tolist(),**{k:float(np.mean([r[k] for r in rows])) for k in metrics}))
        all_results[schedule]=curve
        summaries[schedule]={k:float(np.mean([r[k] for r in curve])) for k in metrics if k!='switch_rate'}
        summaries[schedule].update(min_successful_reuse=min(r['successful_reuse'] for r in curve),max_successful_reuse=max(r['successful_reuse'] for r in curve),mean_switch_rate_after_first=float(np.mean([r['switch_rate'] for r in curve[1:]])))
    out=root/'priority_reuse';out.mkdir(exist_ok=True)
    scope=('Frozen libraries and policies queried along safety → task → efficiency → safety weights. '
           'Offline same-start replay, not sequential environment execution or within-episode adaptation. '
           'Successful reuse = a certified non-idle selection with executed scalarized score >= idle (tolerance 1e-9). '
           'Fallback is not counted as successful reuse. PDS admission alone does not guarantee this criterion. '
           'Curves reuse the same contexts; points are correlated. No new training, certification, or rollouts. '
           'Raw cost is undiscounted; task and effort are discounted. Means over query steps depend on the prescribed schedule. '
           'One policy-training seed; no uncertainty intervals are shown.')
    (out/'summary.json').write_text(json.dumps(dict(scope=scope,contexts=len(records),points_per_leg=points,summary=summaries,curves=all_results,rows=details,source_sha256=sources),indent=2,allow_nan=False)+'\n')
    for schedule,curve in all_results.items():
        x=np.arange(len(curve));fig,axes=plt.subplots(3,2,figsize=(13,11))
        w=np.array([r['weight'] for r in curve])
        for i,label in enumerate(['Safety','Task','Efficiency']):axes[0,0].plot(x,w[:,i],label=label)
        for key,label in [('available','Certified available'),('successful_reuse','Successful reuse'),('fallback','Idle fallback')]:axes[0,1].plot(x,[r[key] for r in curve],label=label)
        for ax,key,title in [(axes[1,0],'goal_success','Goal success'),(axes[1,1],'raw_cost','Mean raw safety cost'),(axes[2,0],'task_return','Mean discounted task return'),(axes[2,1],'effort','Mean discounted control effort')]:
            ax.plot(x,[r[key] for r in curve]);ax.set_title(title)
        axes[0,0].set_title('Prescribed priorities');axes[0,1].set_title('Availability, reuse, and fallback')
        for ax in (axes[0,0],axes[0,1]):ax.legend(fontsize=8);ax.set_ylim(-.03,1.05)
        axes[1,0].set_ylim(-.03,1.05)
        for ax in axes.flat:ax.set_xlabel('Priority query index (not environment time)');ax.grid(alpha=.25)
        fig.suptitle(f'{schedule.title()} priority changes: frozen SubRep library\nSame-start offline replay across {len(records)} contexts')
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('png','pdf'):fig.savefig(out/f'{schedule}_reuse.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    lines=['# Reuse under changing priorities','',scope,'','| Schedule | Availability | Successful reuse | Fallback | Goal success | Raw cost |','|---|---:|---:|---:|---:|---:|']
    for name,r in summaries.items():lines.append(f"| {name} | {r['available']:.1%} | {r['successful_reuse']:.1%} | {r['fallback']:.1%} | {r['goal_success']:.1%} | {r['raw_cost']:.3f} |")
    lines+=['','Successful reuse includes ties with idle; strict improvement and per-step outcomes are in summary.json. This is not an absolute safety certificate.','','![Gradual priorities](gradual_reuse.png)','','![Abrupt priorities](abrupt_reuse.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('Saved:',out/'report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',required=True);p.add_argument('--points-per-leg',type=int,default=20)
    a=p.parse_args()
    if a.points_per_leg<2:p.error('points-per-leg must be >=2')
    run(a.benchmark_dir,a.points_per_leg)
