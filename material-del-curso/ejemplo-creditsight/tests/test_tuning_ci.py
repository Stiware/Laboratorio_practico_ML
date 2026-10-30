"""El intervalo de una métrica remuestrea clientes completos y contiene al valor observado."""
import numpy as np

from creditsight.models.train import decide_min_cost, expected_cost
from creditsight.models.tuning import bootstrap_metric_ci


def _cost(y, proba):
    return expected_cost(y, decide_min_cost(proba))


def test_perfect_predictions_have_zero_cost_interval():
    y = np.repeat([0, 1, 2, 1], 8)  # 4 clientes, 8 meses cada uno
    proba = np.eye(3)[y]
    groups = np.repeat(["a", "b", "c", "d"], 8)
    assert bootstrap_metric_ci(y, proba, groups, _cost) == (0.0, 0.0, 0.0)


def test_interval_brackets_the_observed_value_and_is_reproducible():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 8)
    y = np.repeat(rng.integers(0, 3, 40), 8)
    proba = rng.dirichlet([1, 1, 1], size=len(y))
    value, low, high = bootstrap_metric_ci(y, proba, groups, _cost)
    assert low <= value <= high
    assert bootstrap_metric_ci(y, proba, groups, _cost) == (value, low, high)
