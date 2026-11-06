"""Pruebas de la optimización de hiperparámetros (etapa 6, `src/olistbr/models/tuning.py`).

Todas usan tablas pequeñas o plazos inventados a mano: ninguna lee `data/processed/`.
"""
import numpy as np
import pandas as pd
import pytest

from olistbr.models.train import candidates
from olistbr.models.tuning import FINALIST_NAMES, starting_candidate


# --- Fase 2: partida de los finalistas ---------------------------------------------------------------


def test_starting_candidate_matches_tournament():
    """La partida de cada finalista es exactamente el candidato del torneo (familia B, sin escalar)."""
    tournament = candidates()
    assert FINALIST_NAMES == ("LightGBM cuantil 0,75", "HistGradientBoosting cuantil 0,75")
    for name in FINALIST_NAMES:
        start = starting_candidate(name)
        assert start.family == "B"
        assert start.scale is False
        assert type(start.model) is type(tournament[name].model)
        assert start.model.get_params() == tournament[name].model.get_params()


def test_starting_candidate_rejects_other_models():
    """Pedir la partida de un modelo que no es finalista falla con un mensaje claro."""
    with pytest.raises(KeyError, match="no es finalista"):
        starting_candidate("Regresión lineal")


# --- Fase 3 (US1): comprobación honesta y reglas de decisión ---------------------------------------

from olistbr.models.metrics import delivery_cost  # noqa: E402
from olistbr.models.tuning import (  # noqa: E402
    bootstrap_cost_difference,
    choose_finalist,
    decide_params,
    fit_and_recommend,
    n_trees_used,
)


def _plans(n_customers=60, per_customer=2, seed=0):
    """Días reales inventados, dos clientes por pedido repetido y un plazo base (el percentil no importa aquí)."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"u{i}" for i in range(n_customers)], per_customer)
    real = rng.uniform(5, 30, len(groups))
    return real, groups


def test_identical_plans_give_zero_interval():
    """Si los dos plazos son iguales, la diferencia y todo el intervalo valen cero."""
    real, groups = _plans()
    plan = real + 2.0
    assert bootstrap_cost_difference(real, plan, plan, groups, n_boot=200) == (0.0, 0.0, 0.0)


def test_constant_difference_collapses_interval():
    """Si A cuesta siempre 3 menos que B en cada pedido, la diferencia es −3 y el intervalo también."""
    real, groups = _plans()
    plan_a = real - 1.0  # subestima 1 día: costo 3
    plan_b = real - 2.0  # subestima 2 días: costo 6
    diff, low, high = bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=200)
    assert diff == pytest.approx(-3.0)
    assert low == pytest.approx(-3.0) and high == pytest.approx(-3.0)


def test_resamples_customers_not_rows():
    """Un cliente con muchos pedidos se sortea como un bloque: el intervalo es más ancho que sorteando pedidos sueltos.

    Caso hecho a mano: un cliente con 50 pedidos donde A es 1 punto peor que B y 50 clientes de un pedido sin diferencia.
    La diferencia total es la media por pedido (50 × 1 / 100 = 0,5), no la media por cliente.
    """
    real = np.full(100, 10.0)
    plan_b = np.full(100, 12.0)  # sobrestima 2 días: costo 2
    plan_a = plan_b.copy()
    plan_a[:50] = 13.0  # el cliente grande: costo 3 (1 punto peor)
    groups = np.array(["grande"] * 50 + [f"u{i}" for i in range(50)])
    by_customer = bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=500)
    by_row = bootstrap_cost_difference(real, plan_a, plan_b, np.arange(100).astype(str), n_boot=500)
    assert by_customer[0] == pytest.approx(0.5) and by_row[0] == pytest.approx(0.5)
    assert (by_customer[2] - by_customer[1]) > (by_row[2] - by_row[1])
    # sortear clientes enteros: o el cliente grande sale (diferencia alta) o no sale (diferencia 0)
    assert by_customer[1] == pytest.approx(0.0)


def test_bootstrap_matches_manual_cost():
    """La diferencia total coincide con el costo 3 a 1 calculado a mano, pedido a pedido."""
    real, groups = _plans(seed=3)
    rng = np.random.default_rng(4)
    plan_a, plan_b = real + rng.normal(0, 3, len(real)), real + rng.normal(1, 3, len(real))
    expected = float((delivery_cost(real, plan_a) - delivery_cost(real, plan_b)).mean())
    diff, low, high = bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=300)
    assert diff == pytest.approx(expected)
    assert low <= diff <= high


def test_bootstrap_is_reproducible():
    """Con la misma semilla el intervalo es el mismo; con otra cambia."""
    real, groups = _plans(seed=5)
    rng = np.random.default_rng(6)
    plan_a, plan_b = real + rng.normal(0, 3, len(real)), real + rng.normal(0, 3, len(real))
    first = bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=300, seed=1)
    assert first == bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=300, seed=1)
    assert first != bootstrap_cost_difference(real, plan_a, plan_b, groups, n_boot=300, seed=2)


def test_bootstrap_rejects_bad_input():
    """Largos distintos, clientes vacíos o cero sorteos fallan con un mensaje claro."""
    real, groups = _plans()
    plan = real + 1.0
    with pytest.raises(ValueError, match="largo"):
        bootstrap_cost_difference(real, plan[:-1], plan, groups)
    with pytest.raises(ValueError, match="largo"):
        bootstrap_cost_difference(real, plan, plan, groups[:-1])
    with_missing = groups.astype(object)
    with_missing[0] = None
    with pytest.raises(ValueError, match="sin cliente"):
        bootstrap_cost_difference(real, plan, plan, with_missing)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_cost_difference(real, plan, plan, groups, n_boot=0)


def test_decide_params_rule():
    """Optimizados solo si TODO el intervalo está por debajo de cero; en cualquier otro caso, partida."""
    assert decide_params(-0.20, -0.05)[0] == "optimizados"
    decision, motive = decide_params(-0.20, 0.10)
    assert decision == "partida" and "cruza" in motive
    decision, motive = decide_params(-0.20, 0.0)  # toca el cero: no está por debajo
    assert decision == "partida" and "cruza" in motive
    decision, motive = decide_params(0.05, 0.20)
    assert decision == "partida" and "peor" in motive
    with pytest.raises(ValueError):
        decide_params(0.2, -0.2)


def test_choose_finalist_rule():
    """Con la diferencia A − B: todo bajo cero gana A, todo sobre cero gana B, si cruza es empate y gana el de desempate."""
    a, b = "LightGBM cuantil 0,75", "HistGradientBoosting cuantil 0,75"
    assert choose_finalist(-0.10, -0.01, a, b, tie_winner=a)[0] == a
    assert choose_finalist(0.01, 0.10, a, b, tie_winner=a)[0] == b
    winner, motive = choose_finalist(-0.05, 0.05, a, b, tie_winner=a)
    assert winner == a and "empate práctico" in motive
    winner, _ = choose_finalist(-0.05, 0.05, a, b, tie_winner=b)
    assert winner == b
    with pytest.raises(ValueError):
        choose_finalist(-0.05, 0.05, a, b, tie_winner="Ridge")
    with pytest.raises(ValueError):
        choose_finalist(0.2, -0.2, a, b, tie_winner=a)


def _train_valid(quantile_frame, n=400, seed=11):
    """Una tabla sintética partida por cliente en entrenamiento y validación."""
    from olistbr.data.split import split_by_group

    return split_by_group(quantile_frame(n, seed), test_size=0.25, seed=7)


def test_fit_and_recommend_does_not_see_valid(quantile_frame):
    """El modelo se ajusta solo con `train`: cambiar los días reales de `valid` no cambia los plazos recomendados."""
    train, valid = _train_valid(quantile_frame)
    candidate = starting_candidate("LightGBM cuantil 0,75")
    real, plan, model = fit_and_recommend(candidate, train, valid)
    assert np.allclose(real, valid["dias_entrega"]) and len(plan) == len(valid)
    assert model.margin == 0.0  # familia B: ya predice el percentil 75
    altered = valid.assign(dias_entrega=valid["dias_entrega"] * 10)
    _, plan_altered, _ = fit_and_recommend(candidate, train, altered)
    assert np.allclose(plan, plan_altered)


def test_n_trees_used_reports_early_stopping(quantile_frame):
    """Con más de 10.000 filas, HistGradientBoosting de partida activa la parada temprana y usa menos árboles que `max_iter`."""
    from sklearn.linear_model import Ridge
    from olistbr.models.train import Candidate

    big = quantile_frame(11000, 2)
    train, valid = big.iloc[:10500], big.iloc[10500:]
    hgb = starting_candidate("HistGradientBoosting cuantil 0,75")
    _, _, model = fit_and_recommend(hgb, train, valid)
    assert 0 < n_trees_used(model) < hgb.model.max_iter

    small_train, small_valid = _train_valid(quantile_frame)
    _, _, lgbm = fit_and_recommend(starting_candidate("LightGBM cuantil 0,75"), small_train, small_valid)
    assert n_trees_used(lgbm) == 300  # LightGBM no tiene parada temprana: usa todos sus árboles

    _, _, ridge = fit_and_recommend(Candidate(Ridge(), "B", True), small_train, small_valid)
    with pytest.raises(TypeError):
        n_trees_used(ridge)


# --- Fase 4 (US2): espacios, constructores y búsqueda ------------------------------------------------

import optuna  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

import olistbr.models.tuning as tuning  # noqa: E402
from olistbr.data.split import make_group_folds  # noqa: E402
from olistbr.models.tuning import (  # noqa: E402
    FINALISTS,
    hgb_quantile_space,
    lgbm_quantile_space,
    make_hgb_quantile,
    make_lgbm_quantile,
    tune,
    tuned_candidate,
)


class _RecordingTrial:
    """Ensayo falso que anota el rango pedido para cada parámetro y devuelve su límite inferior."""

    def __init__(self):
        self.ranges = {}

    def suggest_int(self, name, low, high, log=False):
        self.ranges[name] = (low, high, log)
        return low

    def suggest_float(self, name, low, high, log=False):
        self.ranges[name] = (low, high, log)
        return low


LGBM_RANGES = {
    "n_estimators": (150, 800, False), "learning_rate": (0.01, 0.2, True), "num_leaves": (15, 127, False),
    "min_child_samples": (20, 300, False), "subsample": (0.6, 1.0, False), "colsample_bytree": (0.5, 1.0, False),
    "reg_lambda": (0.001, 10.0, True),
}
HGB_RANGES = {
    "max_iter": (150, 800, False), "learning_rate": (0.01, 0.2, True), "max_leaf_nodes": (15, 127, False),
    "min_samples_leaf": (20, 300, False), "l2_regularization": (0.001, 10.0, True), "max_features": (0.5, 1.0, False),
}


@pytest.mark.parametrize("space, expected", [(lgbm_quantile_space, LGBM_RANGES), (hgb_quantile_space, HGB_RANGES)])
def test_spaces_use_the_approved_ranges(space, expected):
    """Cada espacio pide exactamente los parámetros y rangos de `especificaciones/06-optimizacion.md` §4."""
    trial = _RecordingTrial()
    params = space(trial)
    assert trial.ranges == expected
    assert set(params) == set(expected)


@pytest.mark.parametrize("space, expected", [(lgbm_quantile_space, LGBM_RANGES), (hgb_quantile_space, HGB_RANGES)])
def test_spaces_accept_both_extremes(space, expected):
    """Con un ensayo fijo en los dos extremos de cada rango, el espacio devuelve esos mismos valores."""
    for side in (0, 1):
        values = {name: bounds[side] for name, bounds in expected.items()}
        assert space(optuna.trial.FixedTrial(values)) == values


def test_lgbm_quantile_is_fixed():
    model = make_lgbm_quantile({"n_estimators": 150, "num_leaves": 15})
    assert isinstance(model, LGBMRegressor)
    params = model.get_params()
    assert params["objective"] == "quantile" and params["alpha"] == 0.75 and params["subsample_freq"] == 1
    assert params["n_estimators"] == 150 and params["num_leaves"] == 15 and params["random_state"] == 42


def test_hgb_quantile_is_fixed_and_has_no_early_stopping():
    model = make_hgb_quantile({"max_iter": 150})
    assert isinstance(model, HistGradientBoostingRegressor)
    params = model.get_params()
    assert params["loss"] == "quantile" and params["quantile"] == 0.75
    assert params["early_stopping"] is False and params["max_iter"] == 150


@pytest.mark.parametrize("make, key", [
    (make_lgbm_quantile, "objective"), (make_lgbm_quantile, "alpha"),
    (make_hgb_quantile, "loss"), (make_hgb_quantile, "quantile"), (make_hgb_quantile, "early_stopping"),
])
def test_quantile_cannot_be_overridden(make, key):
    """Un ensayo no puede cambiar el percentil, la pérdida ni la parada temprana: falla con un mensaje claro."""
    with pytest.raises(ValueError, match="fijo"):
        make({key: 0.5})


def test_finalists_and_tuned_candidates():
    assert set(FINALISTS) == set(FINALIST_NAMES)
    lgbm = tuned_candidate("LightGBM cuantil 0,75", {"n_estimators": 150})
    hgb = tuned_candidate("HistGradientBoosting cuantil 0,75", {"max_iter": 150})
    assert lgbm.family == "B" and hgb.family == "B" and not lgbm.scale and not hgb.scale
    assert isinstance(lgbm.model, LGBMRegressor) and isinstance(hgb.model, HistGradientBoostingRegressor)
    with pytest.raises(KeyError, match="no es finalista"):
        tuned_candidate("Ridge", {})


def _search_data(quantile_frame, n=300, seed=31):
    """Tabla sintética pequeña con 3 pliegues por cliente, para búsquedas cortas."""
    df = quantile_frame(n, seed)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    return X, y, make_group_folds(df, n_splits=3)


def test_tune_runs_requested_trials(quantile_frame):
    X, y, folds = _search_data(quantile_frame)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False)
    assert len(study.trials) == 2
    for trial in study.trials:
        assert trial.state == optuna.trial.TrialState.COMPLETE and np.isfinite(trial.value)
        assert trial.value == pytest.approx(trial.user_attrs["costo_media"])
        assert {"costo_desv", "mae_media", "cobertura_media", "fit_seconds_media"} <= set(trial.user_attrs)
        assert set(trial.params) == set(LGBM_RANGES)


def test_tune_is_reproducible(quantile_frame):
    X, y, folds = _search_data(quantile_frame)
    first = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False)
    second = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False)
    assert [t.params for t in first.trials] == [t.params for t in second.trials]
    assert [t.value for t in first.trials] == [t.value for t in second.trials]


def test_tune_resumes_from_storage(quantile_frame, tmp_path):
    """Con el estudio guardado, pedir 2 ensayos después de hacer 1 solo hace el que falta."""
    X, y, folds = _search_data(quantile_frame)
    db = tmp_path / "optuna.db"
    tune("LightGBM cuantil 0,75", X, y, folds, n_trials=1, log=False, storage=db)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False, storage=db)
    assert len(study.trials) == 2
    again = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False, storage=db)
    assert len(again.trials) == 2  # ya estaba completo: no hace ninguno más


def test_tune_counts_failed_trials_in_budget(quantile_frame, monkeypatch):
    """Un ensayo con costo no finito queda como fallido, cuenta dentro del presupuesto y no puede ser el mejor."""
    X, y, folds = _search_data(quantile_frame)
    real_evaluate = tuning.evaluate_candidate_folds
    calls = {"n": 0}

    def evaluate_with_one_failure(*args, **kwargs):
        calls["n"] += 1
        result = real_evaluate(*args, **kwargs)
        if calls["n"] == 1:
            result["costo"] = np.nan
        return result

    monkeypatch.setattr(tuning, "evaluate_candidate_folds", evaluate_with_one_failure)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False)
    states = [t.state for t in study.trials]
    assert len(states) == 2
    assert states[0] == optuna.trial.TrialState.FAIL and states[1] == optuna.trial.TrialState.COMPLETE
    assert study.best_trial.number == 1


def test_tune_rejects_bad_input(quantile_frame):
    X, y, folds = _search_data(quantile_frame)
    with pytest.raises(ValueError, match="n_trials"):
        tune("LightGBM cuantil 0,75", X, y, folds, n_trials=0, log=False)
    with pytest.raises(ValueError, match="customer_unique_id"):
        tune("LightGBM cuantil 0,75", X.drop(columns=["customer_unique_id"]), y, folds, n_trials=1, log=False)
    with pytest.raises(KeyError, match="no es finalista"):
        tune("Ridge", X, y, folds, n_trials=1, log=False)


# --- Fase 5 (US3): registro en MLflow y tabla de ensayos --------------------------------------------

import mlflow  # noqa: E402

from olistbr.models.tuning import trials_table  # noqa: E402
from olistbr.tracking import mlflow_utils as mu  # noqa: E402


def test_tune_logs_one_nested_run_per_trial(quantile_frame, tmp_path):
    """2 ensayos -> 1 ejecución principal y 2 hijas, con parámetros, costo con su desviación y etiquetas; sin datos de clientes."""
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    mu.set_experiment("prueba-optimizacion")
    X, y, folds = _search_data(quantile_frame)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=True)

    runs = mlflow.search_runs(experiment_names=["prueba-optimizacion"], output_format="list")
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 1 and len(children) == 2
    assert parents[0].info.run_name == "optuna-lightgbm-cuantil" and "best_costo" in parents[0].data.metrics
    assert {c.info.run_name for c in children} == {"lightgbm-cuantil-ensayo-00", "lightgbm-cuantil-ensayo-01"}
    customers = set(X["customer_unique_id"])
    for child in children:
        assert child.data.tags["mlflow.parentRunId"] == parents[0].info.run_id
        assert {"etapa", "modelo", "metrica_optimizada"} <= set(child.data.tags)
        assert child.data.tags["etapa"] == "06_optimizacion" and child.data.tags["metrica_optimizada"] == "costo"
        assert set(child.data.params) == set(LGBM_RANGES)
        assert {"costo_media", "costo_desv", "mae_media", "fit_seconds_media"} <= set(child.data.metrics)
        assert not customers & (set(child.data.params.values()) | set(child.data.tags.values()))
    run_ids = {t.user_attrs["mlflow_run_id"] for t in study.trials}
    assert run_ids == {c.info.run_id for c in children}


def test_trials_table_columns(quantile_frame, monkeypatch):
    """Una fila por ensayo (también los fallidos) con las columnas de `data-model.md`."""
    X, y, folds = _search_data(quantile_frame)
    real_evaluate = tuning.evaluate_candidate_folds
    calls = {"n": 0}

    def evaluate_with_one_failure(*args, **kwargs):
        calls["n"] += 1
        result = real_evaluate(*args, **kwargs)
        if calls["n"] == 2:
            result["costo"] = np.inf
        return result

    monkeypatch.setattr(tuning, "evaluate_candidate_folds", evaluate_with_one_failure)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=2, log=False)
    table = trials_table(study, "LightGBM cuantil 0,75")
    expected = (
        ["modelo", "ensayo", "estado"] + [f"param_{p}" for p in LGBM_RANGES]
        + ["costo_media", "costo_desv", "mae_media", "rmse_media", "r2_media", "sesgo_media", "tarde_media", "cobertura_media",
           "fit_seconds_media", "mlflow_run_id"]
    )
    assert list(table.columns) == expected
    assert table["ensayo"].tolist() == [0, 1] and table["estado"].tolist() == ["COMPLETE", "FAIL"]
    assert set(table["modelo"]) == {"LightGBM cuantil 0,75"}
    assert table["mlflow_run_id"].isna().all()  # log=False: no hay ejecuciones


# --- Cambios posteriores a las revisiones (auditor-de-fuga y revisor-de-codigo-ia) -------------------

from olistbr.models.evaluation import evaluate_candidate_holdout  # noqa: E402
from olistbr.models.metrics import mean_delivery_cost  # noqa: E402


def test_starting_params_are_the_tournament_numbers():
    """La partida tiene los números del torneo escritos a mano (no solo «lo mismo que `candidates()`»)."""
    lgbm = starting_candidate("LightGBM cuantil 0,75").model.get_params()
    assert (lgbm["n_estimators"], lgbm["learning_rate"], lgbm["num_leaves"], lgbm["alpha"]) == (300, 0.05, 31, 0.75)
    hgb = starting_candidate("HistGradientBoosting cuantil 0,75").model.get_params()
    assert (hgb["max_iter"], hgb["learning_rate"], hgb["quantile"], hgb["early_stopping"]) == (300, 0.05, 0.75, "auto")


def test_fit_and_recommend_measures_like_the_tournament(quantile_frame):
    """`fit_and_recommend` y `evaluate_candidate_holdout` (torneo) dan el mismo costo con la misma tabla."""
    train, valid = _train_valid(quantile_frame)
    candidate = starting_candidate("LightGBM cuantil 0,75")
    real, plan, _ = fit_and_recommend(candidate, train, valid)
    assert mean_delivery_cost(real, plan) == pytest.approx(evaluate_candidate_holdout("x", candidate, train, valid)["costo"])


def test_promised_deadline_never_reaches_the_model(quantile_frame):
    """El plazo prometido por Olist, el cliente y el pedido no llegan al modelo de la optimización."""
    train, valid = _train_valid(quantile_frame)
    _, _, model = fit_and_recommend(tuned_candidate("LightGBM cuantil 0,75", {"n_estimators": 150}), train, valid)
    used = set(model.pipeline.named_steps["prep"].get_feature_names_out())
    forbidden = ("promised", "order_estimated_delivery_date", "customer_unique_id", "order_id")
    assert not [name for name in used if any(word in name for word in forbidden)]


def test_bootstrap_rejects_missing_values_and_empty_input():
    real, groups = _plans()
    plan = real + 1.0
    with_nan = plan.copy()
    with_nan[3] = np.nan
    with pytest.raises(ValueError, match="no finit"):
        bootstrap_cost_difference(real, with_nan, plan, groups)
    with pytest.raises(ValueError, match="no finit"):
        bootstrap_cost_difference(np.where(np.arange(len(real)) == 0, np.inf, real), plan, plan, groups)
    with pytest.raises(ValueError, match="vací"):
        bootstrap_cost_difference([], [], [], [])


def test_rules_reject_non_finite_intervals():
    """Con un intervalo vacío o infinito, las reglas fallan en vez de decir «cruza el cero»."""
    with pytest.raises(ValueError, match="no finit"):
        decide_params(np.nan, np.nan)
    with pytest.raises(ValueError, match="no finit"):
        choose_finalist(-np.inf, 0.1, "LightGBM cuantil 0,75", "HistGradientBoosting cuantil 0,75", tie_winner="LightGBM cuantil 0,75")


@pytest.mark.parametrize("make, key", [
    (make_lgbm_quantile, "random_state"), (make_lgbm_quantile, "n_jobs"), (make_lgbm_quantile, "subsample_freq"),
    (make_lgbm_quantile, "verbose"), (make_hgb_quantile, "random_state"),
])
def test_other_fixed_settings_cannot_be_overridden(make, key):
    with pytest.raises(ValueError, match="fijo"):
        make({key: 1})


def test_only_non_finite_cost_is_caught(quantile_frame, monkeypatch):
    """Un error real (por ejemplo, de columnas) detiene la búsqueda en vez de quedar escondido como ensayo fallido."""
    X, y, folds = _search_data(quantile_frame)

    def broken(*args, **kwargs):
        raise ValueError("columna inexistente")

    monkeypatch.setattr(tuning, "evaluate_candidate_folds", broken)
    with pytest.raises(ValueError, match="columna inexistente"):
        tune("LightGBM cuantil 0,75", X, y, folds, n_trials=1, log=False)


def test_failed_trial_is_also_logged(quantile_frame, monkeypatch, tmp_path):
    """Un ensayo con costo no finito también queda en MLflow, marcado como fallido (todos los ensayos están registrados)."""
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    mu.set_experiment("prueba-fallido")
    X, y, folds = _search_data(quantile_frame)
    real_evaluate = tuning.evaluate_candidate_folds

    def nan_cost(*args, **kwargs):
        result = real_evaluate(*args, **kwargs)
        result["costo"] = np.nan
        return result

    monkeypatch.setattr(tuning, "evaluate_candidate_folds", nan_cost)
    study = tune("LightGBM cuantil 0,75", X, y, folds, n_trials=1, log=True)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    children = [r for r in mlflow.search_runs(experiment_names=["prueba-fallido"], output_format="list") if "mlflow.parentRunId" in r.data.tags]
    assert len(children) == 1 and children[0].data.tags["estado"] == "FAIL"
