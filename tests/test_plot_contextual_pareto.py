import pytest
from utils.plot_contextual_pareto import dominators


def test_dominance_tradeoffs_and_ties():
    # Minimized coordinates: cost, negative reward, effort.
    assert dominators([[0,0,0],[1,-2,1],[2,-1,2],[1,-2,1]]) == [[],[],[1,3],[]]
    assert dominators([[1,1,1],[1,1,2]]) == [[],[0]]
    with pytest.raises(ValueError):
        dominators([[float('nan'),0,0]])
