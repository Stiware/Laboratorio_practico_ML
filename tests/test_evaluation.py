"""Pruebas de la medición de ideas: pliegues, mejora/ruido y regla de decisión (especificaciones/04-features.md §2)."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

import olistbr.features.engineering as eng
from olistbr.data.split import forward_split, make_group_folds
from olistbr.models.evaluation import (
    RESULT_COLUMNS,
    compare_to_reference,
    decide_idea,
    evaluate_folds,
    evaluate_holdout,
    kept_groups,
    measure_idea,
    measure_reference,
    reference_row,
    results_table,
)
from olistbr.models.train import build_baseline_pipeline, build_pipeline
from olistbr.models.metrics import mae_days, mean_delivery_cost


def _mean_model():
    """Modelo trivial: predice siempre la media del entrenamiento (fácil de calcular a mano)."""
    return Pipeline([("model", DummyRegressor(strategy="mean"))])


def _data():
    X = pd.DataFrame({"x": np.arange(10, dtype=float)})
    y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    folds = [(np.arange(5, 10), np.arange(0, 5)), (np.arange(0, 5), np.arange(5, 10))]
    return X, y, folds


def test_evaluate_folds_returns_one_row_per_fold():
    X, y, folds = _data()
    result = evaluate_folds(_mean_model(), X, y, folds)
    assert list(result.columns) == ["fold", "mae", "cost", "n_train", "n_valid"]
    assert list(result["fold"]) == [0, 1]
    assert list(result["n_train"]) == [5, 5] and list(result["n_valid"]) == [5, 5]


def test_evaluate_folds_does_not_leak_validation_into_fit():
    """El pipeline se ajusta solo con el entrenamiento de cada pliegue: predice la media de ESE entrenamiento."""
    X, y, folds = _data()
    pipeline = _mean_model()
    result = evaluate_folds(pipeline, X, y, folds)
    for (train_idx, valid_idx), (_, row) in zip(folds, result.iterrows()):
        prediction = np.full(len(valid_idx), y.iloc[train_idx].mean())
        assert row["mae"] == pytest.approx(mae_days(y.iloc[valid_idx], prediction))
        assert row["cost"] == pytest.approx(mean_delivery_cost(y.iloc[valid_idx], prediction))
    with pytest.raises(NotFittedError):  # el pipeline que se pasó queda sin ajustar (se clona en cada pliegue)
        check_is_fitted(pipeline.named_steps["model"])


def test_improvement_and_noise_are_paired_differences():
    """Mejora = media de (referencia - idea) por pliegue; ruido = desviación (ddof=1) de esas diferencias."""
    reference = pd.DataFrame({"mae": [1.0, 2.0, 3.0, 4.0, 5.0], "cost": [2.0, 4.0, 6.0, 8.0, 10.0]})
    variant = pd.DataFrame({"mae": [0.5, 1.2, 2.8, 3.0, 4.5], "cost": [1.5, 3.0, 5.5, 6.0, 9.5]})
    result = compare_to_reference(reference, variant)
    mae_diff = np.array([0.5, 0.8, 0.2, 1.0, 0.5])
    cost_diff = np.array([0.5, 1.0, 0.5, 2.0, 0.5])
    assert result["mae_gain"] == pytest.approx(mae_diff.mean())  # 0,6
    assert result["mae_noise"] == pytest.approx(mae_diff.std(ddof=1))
    assert result["mae_ratio"] == pytest.approx(mae_diff.mean() / mae_diff.std(ddof=1))
    assert result["cost_gain"] == pytest.approx(cost_diff.mean())
    assert result["cost_ratio"] == pytest.approx(cost_diff.mean() / cost_diff.std(ddof=1))
    assert result["mae_evaluable"] and result["cost_evaluable"]


def test_zero_noise_is_not_evaluable():
    """Si la mejora es idéntica en todos los pliegues no hay ruido: el cociente no se calcula (no se divide por cero)."""
    reference = pd.DataFrame({"mae": [1.0, 2.0, 3.0], "cost": [2.0, 4.0, 6.0]})
    variant = pd.DataFrame({"mae": [0.5, 1.5, 2.5], "cost": [1.0, 3.0, 5.0]})
    result = compare_to_reference(reference, variant)
    assert result["mae_gain"] == pytest.approx(0.5) and result["mae_noise"] == pytest.approx(0.0)
    assert np.isnan(result["mae_ratio"]) and not result["mae_evaluable"]


def _cv(ratio=1.5, gain=0.6, cost_gain=0.9):
    return {"mae_gain": gain, "mae_ratio": ratio, "mae_evaluable": True, "cost_gain": cost_gain}


def test_decide_keeps_idea_that_beats_noise():
    decision, reason = decide_idea(_cv(1.5), {"mae_gain": 0.3}, known_at_prediction=True)
    assert decision == "conservar" and reason


def test_decide_discards_when_ratio_below_one():
    decision, reason = decide_idea(_cv(0.8), {"mae_gain": 0.3}, known_at_prediction=True)
    assert decision == "descartar" and "0,80" in reason


def test_decide_discards_when_cost_does_not_improve():
    decision, _ = decide_idea(_cv(1.5, cost_gain=-0.1), {"mae_gain": 0.3}, known_at_prediction=True)
    assert decision == "descartar"


def test_decide_discards_when_forward_flips_sign():
    decision, reason = decide_idea(_cv(1.5), {"mae_gain": -0.2}, known_at_prediction=True)
    assert decision == "descartar" and "adelante" in reason


def test_decide_discards_when_not_known_at_prediction():
    decision, reason = decide_idea(_cv(5.0), {"mae_gain": 1.0}, known_at_prediction=False)
    assert decision == "descartar" and "predecir" in reason


def test_decide_keeps_when_forward_not_evaluable():
    """Sin datos hacia adelante (p. ej. la temporada alta de fin de año) no se descarta; el motivo lo dice."""
    decision, reason = decide_idea(_cv(1.5), None, known_at_prediction=True)
    assert decision == "conservar" and "no evaluable hacia adelante" in reason


def test_decide_discards_when_noise_is_zero():
    cv = {"mae_gain": 0.5, "mae_ratio": float("nan"), "mae_evaluable": False, "cost_gain": 0.5}
    decision, reason = decide_idea(cv, {"mae_gain": 0.3}, known_at_prediction=True)
    assert decision == "descartar" and "ruido" in reason


def test_evaluate_holdout_fits_only_on_train():
    """Se ajusta con el entrenamiento y se evalúa en la validación: predice la media del ENTRENAMIENTO."""
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "dias_entrega": [2.0, 4.0, 6.0, 8.0]})  # media 5
    valid = pd.DataFrame({"x": [4.0, 5.0], "dias_entrega": [20.0, 30.0]})
    result = evaluate_holdout(_mean_model(), train, valid)
    prediction = np.full(2, 5.0)
    assert result["mae"] == pytest.approx(mae_days(valid["dias_entrega"], prediction))  # 20
    assert result["cost"] == pytest.approx(mean_delivery_cost(valid["dias_entrega"], prediction))
    assert result["n_train"] == 4 and result["n_valid"] == 2


# ---------------------------- fila de la tabla de ideas (integración con datos de ejemplo) ----------------------------


def _setup(frame_factory):
    df = eng.add_features(frame_factory(300, seed=5))
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    folds = make_group_folds(df, n_splits=5)
    forward = forward_split(df)
    reference = measure_reference(build_pipeline(), X, y, folds, forward)
    return df, X, y, folds, forward, reference


def test_reference_row_has_all_columns(frame_factory):
    _, _, _, _, _, reference = _setup(frame_factory)
    row = reference_row("Referencia", reference)
    assert list(row) == RESULT_COLUMNS
    assert row["decision"] == "referencia"
    assert row["mae_media"] > 0 and np.isnan(row["mae_mejora_ruido"])


def test_measure_idea_row_for_a_useful_idea(frame_factory):
    """El objetivo de ejemplo depende de la distancia: los logaritmos no deben empeorar el modelo."""
    _, X, y, folds, forward, reference = _setup(frame_factory)
    row = measure_idea("F1 logaritmos", "logs", build_pipeline(["logs"]), X, y, folds, reference, forward)
    assert list(row) == RESULT_COLUMNS
    assert row["idea"] == "F1 logaritmos" and row["grupo"] == "logs"
    assert row["decision"] in {"conservar", "descartar"} and row["motivo"]
    assert np.isfinite(row["mae_mejora"]) and np.isfinite(row["adelante_mejora"])


def test_measure_idea_forward_not_evaluable(frame_factory):
    _, X, y, folds, forward, reference = _setup(frame_factory)
    row = measure_idea(
        "F6 temporada", "peak_season", build_pipeline(["peak_season"]), X, y, folds, reference, forward,
        forward_evaluable=False,
    )
    assert np.isnan(row["adelante_mejora"])
    assert row["decision"] == "descartar" or "no evaluable hacia adelante" in row["motivo"]


def test_measure_idea_reference_only_is_never_kept(frame_factory):
    """El plazo prometido se mide como referencia: la decisión es «referencia», nunca «conservar»."""
    _, X, y, folds, forward, reference = _setup(frame_factory)
    row = measure_idea(
        "F4 plazo prometido", "(referencia)", build_pipeline(include_reference=True), X, y, folds, reference, forward,
        kind="referencia_no_usable",
    )
    assert row["decision"] == "referencia" and "no usable" in row["motivo"]


def test_measure_idea_detects_when_the_forward_validation_lacks_the_phenomenon(frame_factory):
    """Si en la validación hacia adelante la marca vale siempre 0, la prueba se informa como no evaluable."""
    df = eng.add_features(frame_factory(300, seed=5))  # fechas de jun-2017 a abr-2018: sin nov-dic en la validación
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    folds, forward = make_group_folds(df), forward_split(df)
    assert forward[1]["is_year_end_peak"].nunique() == 1  # la validación no contiene la temporada alta
    reference = measure_reference(build_pipeline(), X, y, folds, forward)
    row = measure_idea(
        "F6", "peak_season", build_pipeline(["peak_season"]), X, y, folds, reference, forward,
        forward_check_column="is_year_end_peak",
    )
    assert np.isnan(row["adelante_mejora"]) and np.isnan(row["adelante_mae"])
    assert row["decision"] == "descartar" or "no evaluable hacia adelante" in row["motivo"]
    # una columna que sí varía en la validación se evalúa normalmente
    row2 = measure_idea(
        "F5", "calendar", build_pipeline(["calendar"]), X, y, folds, reference, forward,
        forward_check_column="is_slow_weekday",
    )
    assert np.isfinite(row2["adelante_mejora"]) and np.isfinite(row2["adelante_mae"])


def test_measure_idea_baseline_is_not_decided(frame_factory):
    df = eng.add_features(frame_factory(300, seed=5))
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    folds, forward = make_group_folds(df), forward_split(df)
    reference = measure_reference(build_pipeline(), X, y, folds, forward)
    row = measure_idea("Línea base", "(trivial)", build_baseline_pipeline(), X, y, folds, reference, forward, kind="linea_base")
    assert row["decision"] == "línea base" and "mediana" in row["motivo"]
    assert row["mae_media"] > reference["folds"]["mae"].mean()  # el modelo con variables supera a la regla trivial


def test_kept_groups_and_results_table():
    rows = [
        {**dict.fromkeys(RESULT_COLUMNS, np.nan), "idea": "A", "grupo": "logs", "decision": "conservar"},
        {**dict.fromkeys(RESULT_COLUMNS, np.nan), "idea": "B", "grupo": "history", "decision": "descartar"},
        {**dict.fromkeys(RESULT_COLUMNS, np.nan), "idea": "C", "grupo": "regions", "decision": "conservar"},
    ]
    assert kept_groups(rows) == ["logs", "regions"]
    table = results_table(rows)
    assert list(table.columns) == RESULT_COLUMNS and len(table) == 3
    with pytest.raises(ValueError, match="repetidas"):
        results_table(rows + [rows[0]])


def test_decide_wording_does_not_claim_a_tiny_gain_is_maintained():
    decision, reason = decide_idea(_cv(1.02, gain=0.0004, cost_gain=0.001), {"mae_gain": 0.0009}, known_at_prediction=True)
    assert decision == "conservar" and "no empeora" in reason and "0,0009" in reason
