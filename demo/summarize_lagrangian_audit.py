"""Summarize completed controlled reward-audit runs without running a simulator."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pattern',default='models/lagrangian_audit_*_run/metrics.jsonl')
    a=p.parse_args()
    for path in sorted(Path('.').glob(a.pattern)):
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:continue
        print('\n'+str(path))
        for r in rows:
            if 'evaluation' not in r:continue
            e=r['evaluation'];d=r.get('reward_audit',{})
            print(json.dumps(dict(update=r['update'],goals=e['goal_reached_rate'],cost=e['mean_cost'],
                effort=e.get('mean_effort'),action_change=e.get('mean_action_change'),audit=d),allow_nan=False))
    print('\nCompare all arms at matching updates. Large cost-advantage scale or poor explained variance is diagnostic evidence, not proof of a cause. Short runs cannot establish convergence.')


if __name__=='__main__':main()
