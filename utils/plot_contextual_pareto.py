"""Descriptive three-objective Pareto analysis of aggregate selector outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'subrep':'SubRep','always_ppo':'PPO','always_ppo_lagrangian':'PPO-Lagrangian',
          'random_candidate':'Random candidate','random_certified':'Random certified',
          'without_certification':'Without certification','zero_action':'Idle'}


def dominators(values):
    """All objectives minimized; ties do not dominate. Exact observed-mean ordering."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Expected a finite matrix')
    return [[j for j in range(len(values)) if np.all(values[j] <= values[i]) and np.any(values[j] < values[i])]
            for i in range(len(values))]


def run(root):
    root = Path(root)
    source = root/'baseline_comparison.json'
    data = json.loads(source.read_text())
    out = root/'pareto_analysis'
    out.mkdir(exist_ok=True)
    scenarios = sorted({r['scenario'] for r in data['summary']})
    methods = sorted({r['method'] for r in data['summary']})
    colors = dict(zip(methods,plt.get_cmap('tab10').colors))
    analysis = []
    fig3 = plt.figure(figsize=(15,12))
    fig2, axes = plt.subplots(len(scenarios),3,figsize=(17,4*len(scenarios)),squeeze=False)
    for index, scenario in enumerate(scenarios):
        rows = sorted([r for r in data['summary'] if r['scenario']==scenario],key=lambda r:r['method'])
        values = [[r['safety_cost'],-r['task_return'],r['effort']] for r in rows]
        dom = dominators(values)
        ax3 = fig3.add_subplot(2,2,index+1,projection='3d')
        for i,r in enumerate(rows):
            method = r['method']; efficient = not dom[i]
            analysis.append(dict(scenario=scenario,method=method,pareto_efficient=efficient,
                dominated_by=[rows[j]['method'] for j in dom[i]],raw_safety_cost=r['safety_cost'],
                discounted_task_return=r['task_return'],discounted_effort=r['effort']))
            style = dict(color=colors[method],marker='*' if efficient else 'o',s=150 if efficient else 65,
                         edgecolors='black',linewidths=.5)
            ax3.scatter(r['safety_cost'],r['effort'],r['task_return'],**style)
            ax3.text(r['safety_cost'],r['effort'],r['task_return'],LABELS[method],fontsize=7)
            for k,(x,y) in enumerate([('safety_cost','task_return'),('effort','task_return'),('safety_cost','effort')]):
                ax = axes[index,k]
                ax.scatter(r[x],r[y],**style)
                ax.annotate(LABELS[method],(r[x],r[y]),xytext=(5,5+(i%2)*6),textcoords='offset points',fontsize=7)
        ax3.set(xlabel='Raw safety cost ↓',ylabel='Discounted effort ↓',zlabel='Discounted task return ↑',title=scenario.title())
        for k,(x,y) in enumerate([('Raw safety cost ↓','Discounted task return ↑'),('Discounted effort ↓','Discounted task return ↑'),('Raw safety cost ↓','Discounted effort ↓')]):
            axes[index,k].set(xlabel=x,ylabel=y,title=scenario.title())
            axes[index,k].grid(alpha=.2)
    fig3.suptitle('Three-objective trade-offs: observed selector means\nStars = nondominated in all three objectives; circles = dominated',fontsize=15)
    fig2.suptitle('Pairwise views of the three-objective comparison\nStars indicate 3D Pareto status, not a separate 2D frontier',fontsize=15)
    for fig,name in [(fig3,'tradeoffs_3d'),(fig2,'tradeoffs_pairwise')]:
        fig.tight_layout(rect=(0,0,1,.95))
        for extension in ('png','pdf'):
            fig.savefig(out/f'{name}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    scope = ('Pareto analysis of observed method means within each priority, not individual trajectories or a statistically certified frontier. '
             'Minimize undiscounted safety cost and discounted effort; maximize discounted task return. '
             'Dominance requires no worse on all three objectives and strictly better on at least one. '
             'Idle is included and may be nondominated because it minimizes cost and effort without completing the task. '
             'These axes use physical outcomes, not the normalized certification score. '
             'No connecting surface is drawn: mixtures and unobserved policies were not evaluated. '
             'Uncertainty can change Pareto membership; one policy-training seed was evaluated.')
    report = dict(scope=scope,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),results=analysis)
    (out/'pareto_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# Three-objective trade-offs and Pareto analysis','',scope,'',
             '| Priority | Method | Raw cost ↓ | Task return ↑ | Effort ↓ | Pareto-efficient | Dominated by |',
             '|---|---|---:|---:|---:|---|---|']
    for r in analysis:
        lines.append(f"| {r['scenario']} | {LABELS[r['method']]} | {r['raw_safety_cost']:.3f} | {r['discounted_task_return']:.3f} | {r['discounted_effort']:.3f} | {'Yes' if r['pareto_efficient'] else 'No'} | {', '.join(LABELS[m] for m in r['dominated_by']) or '—'} |")
    lines += ['', '![Three-objective plots](tradeoffs_3d.png)','', '![Pairwise projections](tradeoffs_pairwise.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('Saved:',out/'report.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir',required=True)
    run(parser.parse_args().benchmark_dir)
