"""Four-dimensional Pareto projections and frozen-library priority sweeps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.analyze_contextual_baselines import score
from utils.analyze_four_objective_baselines import STRATEGIES, pareto_mask


def choose(outcomes, audits, scales, caps, weight):
    admitted=[a['policy'] for a in audits if a['admitted'] and
              (a['region']=='FULL_SIMPLEX' or np.all(weight<=np.asarray(caps)+1e-9))]
    if not admitted:return 'zero_action',True
    return min(admitted,key=lambda n:(-score(outcomes[n],outcomes['zero_action'],scales,weight),n)),False


def weight_path(points):
    if points<2:raise ValueError('At least two points per leg required')
    corners=np.full((4,4),.2/3);np.fill_diagonal(corners,.8)
    return np.concatenate([np.array([(1-t)*corners[i]+t*corners[i+1] for t in np.linspace(0,1,points,endpoint=False)]) for i in range(3)]+[corners[-1:]])


def run(root,out,points=20):
    root=Path(root);out=Path(out)
    if out.exists():raise FileExistsError('Use a new output directory')
    hashes={}
    def read(path):
        raw=path.read_bytes();hashes[str(path.relative_to(root))]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    manifest=read(root/'manifest.json');comparison=read(root/'support_comparison.json')
    analysis=read(root/'end_to_end_analysis/analysis.json')
    # Refuse stale analysis generated from different source files.
    for name,digest in analysis['source_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Analysis sources have changed')
    summaries=[r for r in analysis['summary'] if r['priority']=='overall' and
               (r['method']=='subrep' or r['strategy']=='full_plus_mdn')]
    vectors=np.array([[r['task_return'],-r['safety_cost'],-r['effort'],-r['action_change']] for r in summaries])
    flags=pareto_mask(vectors)
    labels=[r['strategy'] if r['method']=='subrep' else r['method'] for r in summaries]
    colors=plt.cm.tab10(np.linspace(0,1,len(labels)))
    fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
    dimensions=[('task_return','Task return ↑'),('safety_cost','Safety cost ↓'),('effort','Control effort ↓'),('action_change','Action change ↓')]
    for ax,(i,j) in zip(axes.flat,[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
        for k,r in enumerate(summaries):
            ax.scatter(r[dimensions[i][0]],r[dimensions[j][0]],color=colors[k],marker='*' if flags[k] else 'o',s=120 if flags[k] else 45,label=labels[k])
        ax.set(xlabel=dimensions[i][1],ylabel=dimensions[j][1]);ax.grid(alpha=.2)
    handles,legends=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Four-objective trade-offs: pairwise views of observed means\nStars = nondominated across all FOUR objectives; descriptive, not confidence-based')
    scales=np.asarray(manifest['scales']['divisors']);weights=weight_path(points);records=[]
    for c in comparison['contexts']:
        local=root/str(c['context_seed']);e=read(local/'evidence.json');x=read(local/'execution.json')
        if e['context_seed']!=x['context_seed'] or not np.array_equal(e['initial_observation'],x['initial_observation']):raise ValueError('Mismatched execution')
        for method in STRATEGIES:
            caps=c['caps'] if method=='full_plus_mdn' else manifest['fixed_caps']
            for index,w in enumerate(weights):
                name,fallback=choose(e['outcomes'],c['audits'][method],scales,caps,w)
                actual=x['outcomes'][name];sc=score(actual,x['outcomes']['zero_action'],scales,w)
                records.append(dict(context=c['context_seed'],method=method,index=index,policy=name,
                    successful_reuse=float(not fallback and sc>=-1e-9),fallback=float(fallback),
                    goal_success=float(actual['goals']>0),cost=-actual['undiscounted_motives'][0],score=sc))
    curve=[]
    for method in STRATEGIES:
        for i,w in enumerate(weights):
            rs=[r for r in records if r['method']==method and r['index']==i]
            curve.append(dict(method=method,index=i,weight=w.tolist(),**{k:float(np.mean([r[k] for r in rs])) for k in ('successful_reuse','fallback','goal_success','cost','score')}))
    sweep,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(metric,title) in zip(axes.flat,[('successful_reuse','Certified reuse scoring at least as well as idle'),('goal_success','Goal success'),('cost','Mean safety cost'),('fallback','Idle fallback')]):
        for method in STRATEGIES:
            rs=[r for r in curve if r['method']==method];ax.plot([r['index'] for r in rs],[r[metric] for r in rs],label=method)
        ax.set_title(title);ax.set_xticks([0,points,2*points,3*points],['Safety','Task','Effort','Smoothness']);ax.grid(alpha=.2)
        if metric!='cost':ax.set_ylim(-.03,1.03);ax.set_ylabel('Fraction of contexts')
    axes.flat[0].legend(fontsize=8)
    sweep.suptitle('Frozen-library reuse as requested priorities change\nOffline weight sweep on the same 300 contexts; no retraining or new simulation'.replace('300',str(len(comparison['contexts']))))
    out.mkdir(parents=True)
    for figure,name in [(fig,'pareto_four_objective_pairwise'),(sweep,'priority_shift_reuse')]:
        for extension in ('png','pdf'):figure.savefig(out/f'{name}.{extension}',dpi=220,bbox_inches='tight')
        plt.close(figure)
    (out/'plot_data.json').write_text(json.dumps(dict(source_sha256=hashes,points_per_leg=points,pareto=[dict(label=l,nondominated=f) for l,f in zip(labels,flags)],curve=curve,rows=records),indent=2,allow_nan=False)+'\n')
    (out/'README.md').write_text('''# Additional benchmark figures

## Pareto analysis — place after Baseline Comparison

![Four-objective Pareto projections](pareto_four_objective_pairwise.png)

Pairwise projections of four-objective mean outcomes. Stars indicate nondominance across all four objectives, not a separate frontier in each panel. Means average the original five priority profiles; membership is descriptive and does not establish statistical dominance.

## Reuse under changing priorities — place before Certification Ablation

![Priority sweep](priority_shift_reuse.png)

Priorities interpolate from safety to task to effort to smoothness, each endpoint emphasizing one objective with weight 0.8. The certified library and learned region remain frozen within each context. Successful reuse requires a nonfallback certified selection whose execution score is at least that of idle. Goal success and raw safety cost are shown separately. This is offline reuse across reset episodes, not a mid-trajectory experiment or new independent evaluation. The path is a reporting choice made after the original evaluation; do not present it as a predeclared final test. No confidence bands are estimated for this exploratory curve.
''')
    print('Figures and captions:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--points-per-leg',type=int,default=20)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir,a.points_per_leg)

if __name__=='__main__':main()
