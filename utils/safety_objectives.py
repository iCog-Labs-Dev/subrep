"""Three-objective measurements and frozen, development-only score scaling."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

OBJECTIVE_NAMES = ('Safety', 'Task', 'ControlEfficiency')


def control_effort(action, low, high):
    """Mean squared zero-centered action normalized by each side's bound.

    Requires bounds straddling zero. Reject invalid actions instead of measuring
    a different action from the one sent to the simulator.
    """
    a, lo, hi = (np.asarray(v, dtype=float) for v in (action, low, high))
    if a.shape != lo.shape or a.shape != hi.shape or not a.size:
        raise ValueError('Action and bounds must have matching non-empty shapes')
    if not all(np.all(np.isfinite(v)) for v in (a, lo, hi)):
        raise ValueError('Action and bounds must be finite')
    if np.any(lo >= 0) or np.any(hi <= 0):
        raise ValueError('Action bounds must straddle zero')
    if np.any(a < lo) or np.any(a > hi):
        raise ValueError('Action must be within bounds')
    normalized = a / np.where(a >= 0, hi, -lo)
    return float(np.mean(normalized ** 2))


@dataclass(frozen=True)
class FrozenObjectiveScales:
    """Positive divisors fitted once on development delta returns, never online.

    Task payoff uses the same divisor as the task motive. This preserves the
    existing double contribution: task payoff + weighted task motive.
    """
    divisors: tuple[float, float, float]
    development_source: str

    def __post_init__(self):
        values = tuple(float(x) for x in self.divisors)
        if len(values) != 3 or not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError('Three finite positive divisors are required')
        if not self.development_source.strip():
            raise ValueError('Development source is required')
        object.__setattr__(self, 'divisors', values)

    @classmethod
    def fit_development(cls, delta_returns, *, source):
        values = np.asarray(delta_returns, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError('Expected finite non-empty development returns with shape (N,3)')
        # Mean absolute magnitude per objective; constant-zero objectives use 1.
        scales = np.mean(np.abs(values), axis=0)
        scales = np.where(scales > 1e-12, scales, 1.)
        return cls(tuple(scales), source)

    def normalize(self, delta_payoff, delta_motives):
        dn = np.asarray(delta_motives, dtype=float)
        if dn.shape != (3,) or not np.all(np.isfinite(dn)) or not np.isfinite(delta_payoff):
            raise ValueError('Expected finite payoff and three motive effects')
        return float(delta_payoff) / self.divisors[1], dn / self.divisors

    def score(self, delta_payoff, delta_motives, weights):
        w = np.asarray(weights, dtype=float)
        if w.shape != (3,) or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.isclose(w.sum(), 1., atol=1e-9, rtol=0):
            raise ValueError('Weights must be a three-objective simplex vector')
        dr, dn = self.normalize(delta_payoff, delta_motives)
        return float(dr + w @ dn)

    def save(self, path):
        Path(path).write_text(json.dumps({'schema_version': 1, 'objective_names': OBJECTIVE_NAMES,
            'divisors': self.divisors, 'development_source': self.development_source,
            'estimator': 'mean_absolute_development_delta_return_zero_fallback_1'}, indent=2)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['schema_version'] != 1 or tuple(data['objective_names']) != OBJECTIVE_NAMES:
            raise ValueError('Incompatible objective scaling file')
        return cls(tuple(data['divisors']), data['development_source'])


def rollout_metadata(data, motives):
    """Validate objective schema without fabricating missing measurements."""
    if motives.ndim != 2 or motives.shape[1] not in (2,3) or not np.all(np.isfinite(motives)):
        raise ValueError('Expected finite (candidates, 2 or 3) motives')
    n = motives.shape[1]
    version = int(np.asarray(data['schema_version']).item()) if 'schema_version' in data else 1
    if version not in (1,2) or (n == 3 and version != 2):
        raise ValueError('Three objectives require freshly collected schema-2 rollouts')
    required = ('objective_names','measurement_definitions','objective_scales','scaling_source','values_are_raw','gamma')
    if version == 2 and any(k not in data for k in required):
        raise ValueError('Incomplete rollout metadata')
    names = list(map(str,data['objective_names'])) if version == 2 else list(OBJECTIVE_NAMES[:n])
    scales = np.asarray(data['objective_scales'],dtype=float) if version == 2 else np.ones(n)
    if names != list(OBJECTIVE_NAMES[:n]) or scales.shape != (n,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('Invalid objective names or scales')
    if version == 2 and not bool(np.asarray(data['values_are_raw']).item()):
        raise ValueError('Rollout measurements must be raw')
    definitions = list(map(str,data['measurement_definitions'])) if version == 2 else ['negative environment cost','environment task reward']
    if len(definitions) != n:
        raise ValueError('Measurement definition count mismatch')
    return {'schema_version': version,'objective_names': names, 'objective_scales': scales.tolist(),
            'measurement_definitions': definitions,
            'scaling_source': str(np.asarray(data['scaling_source']).item()) if version == 2 else 'legacy-identity',
            'gamma': float(np.asarray(data['gamma']).item()) if version == 2 else None}
