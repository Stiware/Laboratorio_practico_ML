"""Pruebas de las métricas: costo asimétrico 3 a 1, MAE y mejor plazo fijo (reports/00_caso.md)."""
import numpy as np

from olistbr.config import COST_RATIOS
from olistbr.models.metrics import best_fixed_deadline, delivery_cost, mae_days, mean_delivery_cost


def test_delivery_cost_penalizes_underestimating_three_times_more():
    """Real 10, predicho 7 (se subestimó 3 días) cuesta 3 x 3 = 9; real 10, predicho 13 (sobrestimó 3) cuesta 3."""
    cost = delivery_cost([10, 10, 10], [7, 13, 10])
    assert list(cost) == [9.0, 3.0, 0.0]


def test_delivery_cost_accepts_custom_costs():
    assert list(delivery_cost([10], [8], under_cost=5, over_cost=2)) == [10.0]
    assert list(delivery_cost([10], [12], under_cost=5, over_cost=2)) == [4.0]


def test_mean_delivery_cost_with_ratio_one_is_mae():
    """Con proporción 1 a 1 el costo medio es exactamente el MAE."""
    real = np.array([5.0, 10.0, 20.0, 3.0])
    predicted = np.array([6.0, 8.0, 20.0, 9.0])
    assert mean_delivery_cost(real, predicted, ratio=1) == mae_days(real, predicted)


def test_mean_delivery_cost_default_ratio_is_three_to_one():
    real, predicted = [10.0, 10.0], [7.0, 13.0]  # costos 9 y 3
    assert mean_delivery_cost(real, predicted) == 6.0
    assert mean_delivery_cost(real, predicted, ratio=9) == (27.0 + 3.0) / 2


def test_mae_days():
    assert mae_days([1, 2, 3], [2, 2, 5]) == 1.0  # errores 1, 0 y 2 -> promedio 1


def test_best_fixed_deadline_is_the_ratio_percentile():
    """Con proporción 3 a 1 el mejor plazo fijo es el percentil 75; con 1 a 1, la mediana."""
    real = np.arange(1, 101, dtype=float)  # 1..100
    assert best_fixed_deadline(real, ratio=1) == np.percentile(real, 50)
    assert best_fixed_deadline(real, ratio=3) == np.percentile(real, 75)


def test_best_fixed_deadline_minimizes_the_mean_cost():
    """Ningún otro plazo fijo del mismo conjunto da un costo medio menor que el percentil elegido."""
    rng = np.random.default_rng(0)
    real = rng.gamma(shape=2.0, scale=5.0, size=2000)
    for ratio in COST_RATIOS:
        best = best_fixed_deadline(real, ratio)
        best_cost = mean_delivery_cost(real, np.full_like(real, best), ratio)
        for other in np.linspace(real.min(), real.max(), 60):
            assert best_cost <= mean_delivery_cost(real, np.full_like(real, other), ratio) + 1e-9
