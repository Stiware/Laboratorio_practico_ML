import numpy as np

from creditsight.models.tuning import bootstrap_cost_difference


def test_bootstrap_difference_is_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 400)
    proba = rng.dirichlet(np.ones(3), 400)
    groups = np.repeat(np.arange(50), 8)
    diff, low, high = bootstrap_cost_difference(y, proba, proba, groups, n_boot=200)
    assert diff == 0 and low == 0 and high == 0


def test_bootstrap_detects_a_clearly_better_model_with_negative_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 800)
    good = np.eye(3)[y] * 0.9 + 0.05            # casi siempre acierta
    bad = np.eye(3)[(y + 1) % 3] * 0.9 + 0.05   # casi siempre se equivoca
    groups = np.repeat(np.arange(100), 8)
    diff, low, high = bootstrap_cost_difference(y, good, bad, groups, n_boot=300)
    assert diff < 0 and high < 0  # el intervalo completo está debajo de cero: A es mejor


def test_every_objective_has_a_direction_and_cost_minimizes():
    from creditsight.models.tuning import OBJETIVOS

    assert OBJETIVOS["cost"] == "minimize" and OBJETIVOS["auc_ovr"] == "maximize"
    assert set(OBJETIVOS.values()) <= {"minimize", "maximize"}


def test_tune_rejects_unknown_metric():
    import pytest

    from creditsight.models.tuning import tune

    with pytest.raises(ValueError, match="no soportada"):
        tune("lightgbm", None, None, None, n_trials=1, metric="inventada")


def test_bootstrap_metric_difference_detects_a_better_ranker():
    from sklearn.metrics import roc_auc_score

    from creditsight.models.tuning import bootstrap_metric_difference

    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, 1200)
    good = np.eye(3)[y] * 0.8 + 0.2 / 3
    bad = rng.dirichlet(np.ones(3), 1200)
    groups = np.repeat(np.arange(150), 8)
    auc = lambda yy, pp: roc_auc_score(yy, pp, multi_class="ovr", average="macro", labels=[0, 1, 2])
    diff, low, high = bootstrap_metric_difference(y, good, bad, groups, auc, n_boot=60)
    assert diff > 0.3 and low > 0  # el intervalo completo está por encima de cero: A ordena mejor
