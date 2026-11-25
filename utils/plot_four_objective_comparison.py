"""Four publication-oriented figures from saved support comparison evidence.

No simulation or retraining. Uncertainty resamples whole contexts, retaining all
paired methods and priority queries. It does not capture training-seed variation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('full_simplex', 'full_plus_fixed', 'full_plus_mdn')
LABELS = ('Full simplex', '+ Fixed region', '+ Learned region')
PRIORITIES = ('balanced', 'safety', 'task', 'effort', 'smoothness')
METRICS = ('goal_success', 'safety_cost', 'joint_success', 'task_return', 'effort', 'action_change')
COLORS = ('#496980', '#d28b2c', '#268576')


def extract(contexts, cost_limit):
    if len(contexts) < 2 or len({c['context_seed'] for c in contexts}) != len(contexts):
        raise ValueError('Require at least two distinct contexts')
    values = np.zeros((len(contexts), 3, 5, 6))
    admissions = np.zeros((3, 3), dtype=int)  # Global, additional regional, rejected.
    selected = np.zeros((3, 3), dtype=int)  # Global, regional, fallback.
    regional = np.zeros((len(contexts), 5))
    inside = np.zeros_like(regional)
    volume = []
    reference_names = None
    for ci, c in enumerate(contexts):
        caps = np.asarray(c['caps'], dtype=float)
        if caps.shape != (4,) or not np.isfinite(caps).all() or np.any(caps < 0) or np.any(caps > 1) or caps.sum() < 1-1e-6:
            raise ValueError('Invalid learned caps')
        v = float(c['coverage']['full_plus_mdn'])
        if not np.isfinite(v) or not 0 <= v <= 1: raise ValueError('Invalid region volume')
        volume.append(v)
        for mi, method in enumerate(METHODS):
            audit = c['audits'][method]
            names = [a['policy'] for a in audit]
            if len(set(names)) != len(names) or not names: raise ValueError('Duplicate/empty policy set')
            if reference_names is None: reference_names = set(names)
            if set(names) != reference_names: raise ValueError('Policy pool changed')
            admitted = {a['policy']:a['region'] for a in audit if a['admitted']}
            for a in audit:
                admissions[mi, 2 if not a['admitted'] else 0 if a['region']=='FULL_SIMPLEX' else 1] += 1
            selections = c['selections'][method]
            if len(selections) != 5 or {s['priority'] for s in selections} != set(PRIORITIES):
                raise ValueError('Require the same five unique priority profiles')
            by_label = {s['priority']:s for s in selections}
            for pi, label in enumerate(PRIORITIES):
                s = by_label[label]
                weight = np.asarray(s['weight'], dtype=float)
                expected = np.full(4,.25) if pi==0 else np.array([.8 if i==pi-1 else .2/3 for i in range(4)])
                if weight.shape!=(4,) or not np.allclose(weight,expected): raise ValueError('Priority weights changed')
                goal = s['goals'] > 0
                joint = not s['fallback'] and goal and s['cost'] <= cost_limit
                values[ci,mi,pi] = [goal,s['cost'],joint,s['task_return'],s['effort'],s['action_change']]
                if s['fallback']:
                    selected[mi,2] += 1
                else:
                    if s['policy'] not in admitted: raise ValueError('Selected uncertified policy')
                    region = admitted[s['policy']]
                    selected[mi,0 if region=='FULL_SIMPLEX' else 1] += 1
                    if method=='full_plus_mdn' and region=='MDN_WX':
                        if not np.all(weight <= caps+1e-9): raise ValueError('Regional selection outside caps')
                        regional[ci,pi] = 1
                if method=='full_plus_mdn': inside[ci,pi] = np.all(weight <= caps+1e-9)
    if not np.isfinite(values).all(): raise ValueError('Nonfinite outcomes')
    return dict(values=values, admissions=admissions, selected=selected,
                regional=regional, inside=inside, volume=np.asarray(volume))


def bootstrap(values, indices):
    """Input axis zero is context; a common index matrix pairs all comparisons."""
    draws = values[indices].mean(axis=1)
    bounds = np.quantile(draws,[.025,.975],axis=0)
    return values.mean(axis=0), bounds[0], bounds[1]


def bars_with_ci(ax, means, low, high, ylabel, percent=False, reference=None):
    factor=100 if percent else 1
    means,low,high = (np.asarray(v)*factor for v in (means,low,high))
    x=np.arange(3)
    ax.bar(x,means,color=COLORS,width=.62)
    ax.vlines(x,low,high,color='#263238',linewidth=1.2)
    ax.hlines(low,x-.07,x+.07,color='#263238',linewidth=1.2)
    ax.hlines(high,x-.07,x+.07,color='#263238',linewidth=1.2)
    span=max(float(high.max()-min(0,low.min())),.1)
    for i,value in enumerate(means):
        ax.text(i,max(value,high[i])+.035*span,f'{value:.1f}%' if percent else f'{value:.2f}',ha='center',fontsize=9)
    ax.set_xticks(x,LABELS,fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_ylim(min(0,float(low.min())-.05*span),max(float(high.max())+.18*span,reference or 0))
    if reference is not None:
        ax.axhline(reference,color='#b84040',linestyle='--',linewidth=1,label=f'Diagnostic budget = {reference:g}')
        ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)


def run(root, output=None, samples=2000, seed=42):
    root=Path(root);source=root/'support_comparison.json'
    data=json.loads(source.read_text());manifest=json.loads((root/'manifest.json').read_text())
    cost_limit=float(manifest['settings']['cost_limit'])
    if not np.isfinite(cost_limit) or cost_limit<0 or samples<100 or seed<0: raise ValueError('Invalid plotting settings')
    records=data['contexts']; arrays=extract(records,cost_limit)
    out=Path(output) if output else root/'figures'
    if out.exists(): raise FileExistsError(f'Use a new figures directory: {out}')
    out.mkdir(parents=True)
    n=len(records);indices=np.random.default_rng(seed).integers(0,n,size=(samples,n))
    overall=bootstrap(arrays['values'].mean(axis=2),indices)
    per_priority=bootstrap(arrays['values'][:,:,:,:2],indices)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
    def save(fig,name):
        fig.savefig(out/f'{name}.png',bbox_inches='tight',facecolor='white')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,counts,title,denom in zip(axes,[arrays['admissions'],arrays['selected']],
        ['Certification decisions','Actual selection outcomes'],['policy–context pairs','priority queries']):
        bottom=np.zeros(3)
        labels=['Full-simplex','Additional regional','Rejected'] if denom.startswith('policy') else ['Full-simplex skill','Regional skill','Idle fallback']
        for k,(label,color) in enumerate(zip(labels,['#496980','#268576','#d4d9dc'])):
            ax.bar(np.arange(3),counts[:,k],bottom=bottom,color=color,label=label,width=.62)
            for i,value in enumerate(counts[:,k]):
                if value: ax.text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white' if k<2 else '#263238',fontsize=10)
            bottom+=counts[:,k]
        ax.set_xticks(np.arange(3),LABELS);ax.set_ylabel(f'Number of {denom}')
        ax.set_title(title);ax.set_ylim(0,max(bottom)*1.08)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1,frameon=False)
    fig.suptitle(f'Certification does not automatically imply reuse ({n} contexts)')
    save(fig,'figure1_certification_and_reuse')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    descriptions=[('Goal success (%)',True,None),('Mean safety cost ↓',False,cost_limit),
                  ('Admitted + goal + episode budget (%)',True,None),('Mean task return ↑',False,None),
                  ('Mean control effort ↓',False,None),('Mean action change ↓',False,None)]
    for j,(ax,(label,percent,ref)) in enumerate(zip(axes.flat,descriptions)):
        bars_with_ci(ax,overall[0][:,j],overall[1][:,j],overall[2][:,j],label,percent,ref)
    fig.suptitle('Task gains and safety–control trade-offs\n95% context-bootstrap intervals; idle fallback included')
    save(fig,'figure2_outcome_tradeoffs')
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
    for j,ax in enumerate(axes):
        factor=100 if j==0 else 1
        for mi,label in enumerate(LABELS):
            x=np.arange(5)+(mi-1)*.24
            mean,low,high=[v[mi,:,j]*factor for v in per_priority]
            ax.bar(x,mean,width=.23,color=COLORS[mi],label=label)
            ax.vlines(x,low,high,color='#263238',linewidth=1)
            ax.hlines(low,x-.035,x+.035,color='#263238',linewidth=1)
            ax.hlines(high,x-.035,x+.035,color='#263238',linewidth=1)
        ax.set_xticks(np.arange(5),[s.title() for s in PRIORITIES],rotation=15)
        ax.set_ylabel('Goal success (%)' if j==0 else 'Mean safety cost ↓')
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        if j==1:ax.axhline(cost_limit,color='#b84040',linestyle='--',label='Diagnostic cost budget')
        ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=2,frameon=False)
    fig.suptitle('Effect of requested priorities\n95% intervals resample paired contexts')
    save(fig,'figure3_priority_dependent_outcomes')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    volume=arrays['volume']*100
    axes[0].hist(volume,bins=np.linspace(0,100,11),color='#268576',edgecolor='white')
    axes[0].axvline(volume.mean(),color='#263238',linestyle='--',label=f'Mean: {volume.mean():.1f}%')
    axes[0].set(xlabel='Learned preference-region volume (%)',ylabel='Number of contexts',xlim=(0,100))
    axes[0].legend();axes[0].set_title('Region size varies by context')
    for i,(key,label,color) in enumerate([('inside','Priority inside learned region','#496980'),('regional','Regional skill actually selected','#268576')]):
        rate=arrays[key].mean(axis=0)*100;x=np.arange(5)+(i-.5)*.35
        axes[1].bar(x,rate,width=.34,label=label,color=color)
        for xx,yy in zip(x,rate):axes[1].text(xx,yy+2,f'{yy:.0f}%',ha='center',fontsize=8)
    axes[1].set_xticks(np.arange(5),[s.title() for s in PRIORITIES],rotation=15)
    axes[1].set(ylabel='Contexts (%)',ylim=(0,110),title='Eligibility is not the same as selection')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.2),frameon=False,fontsize=9)
    fig.suptitle('Learned-region coverage and useful reuse\nPreference coverage is not environmental safety coverage')
    save(fig,'figure4_region_coverage_and_use')
    stats=dict(contexts=n,queries_per_method=n*5,methods=list(METHODS),metrics=list(METRICS),
        admissions=arrays['admissions'].tolist(),selected=arrays['selected'].tolist(),
        outcome_means=overall[0].tolist(),outcome_ci_low=overall[1].tolist(),outcome_ci_high=overall[2].tolist(),
        priority_names=list(PRIORITIES),priority_outcome_means=per_priority[0].tolist(),
        priority_outcome_ci_low=per_priority[1].tolist(),priority_outcome_ci_high=per_priority[2].tolist(),
        priority_inside_rates=arrays['inside'].mean(0).tolist(),regional_selected_rates=arrays['regional'].mean(0).tolist(),
        bootstrap_samples=samples,bootstrap_seed=seed,cost_limit=cost_limit,
        source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        uncertainty='Percentile bootstrap of contexts, paired across methods and priorities; conditional on these fixed trained models. Not training-seed uncertainty.')
    (out/'figure_statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
    captions=[
        ('figure1_certification_and_reuse','Certification and actual reuse','After the Pilot Results table',
         'Certification decisions over policy–context pairs and actual selections over priority queries. Regional admissions are additional to retained full-simplex admissions; being admitted does not ensure selection.'),
        ('figure2_outcome_tradeoffs','Task performance and safety–control trade-offs','Under Safety-Cost Result, after its table',
         'Mean outcomes include idle fallback and average equally over the five priority profiles. Bars show observed means; intervals are 95% context-bootstrap intervals. The dashed cost line is a separate diagnostic budget, not a certification constraint.'),
        ('figure3_priority_dependent_outcomes','Outcomes under changing priorities','After the Baseline Comparison table',
         'Goal success and safety cost for each predefined priority profile. All methods use paired contexts. Intervals resample contexts and do not treat repeated priority queries as independent episodes.'),
        ('figure4_region_coverage_and_use','Learned-region coverage and regional selection','Under Interpretation, before Limitations',
         'Distribution of learned preference-region volumes across contexts, and the frequency with which a priority lies inside its region versus a regional skill being selected. Region volume is a fraction of preference space, not a measure of environmental safety.')]
    lines=['# Figure placement and captions','',f'Source: `{source}`. {n} contexts; {n*5} priority queries per method.',
           'PNG files are suitable for inserting into documents; PDFs are vector versions for publication.',
           'The cost-budget joint metric is per episode; it differs from an average-cost constraint.',
           'Intervals describe context variation for these fixed models, not training-seed uncertainty. These are exact-start empirical reuse results, not safety guarantees.','']
    for filename,title,placement,caption in captions:
        lines += [f'## {title}','',f'**Place:** {placement}.','',f'![{title}]({filename}.png)','',f'**Caption:** {caption}','']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('Figures and captions:',out)
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-dir',required=True)
    p.add_argument('--output-dir')
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run(a.benchmark_dir,a.output_dir,a.bootstrap_samples,a.seed)


if __name__=='__main__':main()
