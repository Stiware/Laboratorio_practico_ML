"""Pruebas de la medición de ideas (etapa 4) y de la evaluación del torneo de modelos (etapa 5)."""
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
import olistbr.models.evaluation as ev
from olistbr.models.train import build_baseline_pipeline, build_pipeline, candidates
from olistbr.models.metrics import mae_days, mean_delivery_cost

# Las tablas pequeñas de estas pruebas tienen pocos clientes de calibración: el aviso es esperado y se prueba en test_models.py.
pytestmark = pytest.mark.filterwarnings("ignore:solo .* clientes de calibración:UserWarning")


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


# ====================== Etapa 5: evaluación de los candidatos del torneo ======================

CANDIDATE_COLUMNS = [
    "modelo", "familia", "pliegue", "mae", "rmse", "r2", "sesgo", "costo", "tarde", "cobertura", "margen",
    "n_ajuste", "n_calibracion", "n_validacion", "fit_seconds",
]


def _tournament_data(quantile_frame, n=500, seed=21):
    df = quantile_frame(n, seed)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    return df, X, y


def test_evaluate_candidate_folds_returns_all_columns(quantile_frame):
    df, X, y = _tournament_data(quantile_frame)
    folds = make_group_folds(df, n_splits=5)
    ridge = candidates()["Ridge"]
    result = ev.evaluate_candidate_folds("Ridge", ridge, X, y, folds)
    assert list(result.columns) == CANDIDATE_COLUMNS
    assert list(result["pliegue"]) == [0, 1, 2, 3, 4] and set(result["modelo"]) == {"Ridge"} and set(result["familia"]) == {"A"}
    assert (result["n_calibracion"] > 0).all()  # la familia A aparta clientes de calibración
    assert (result["n_ajuste"] + result["n_calibracion"]).tolist() == [len(tr) for tr, _ in folds]
    assert (result["fit_seconds"] > 0).all() and np.isfinite(result[["mae", "rmse", "costo", "cobertura", "margen"]].to_numpy()).all()
    # la familia B y las líneas base ajustan con todo el entrenamiento y no llevan margen
    base = ev.evaluate_candidate_folds("mediana", candidates()["Línea base: siempre la mediana"], X, y, folds)
    assert (base["n_calibracion"] == 0).all() and (base["margen"] == 0).all()


def test_evaluate_candidate_folds_is_reproducible_and_does_not_touch_the_candidate(quantile_frame):
    """Dos evaluaciones iguales dan las mismas métricas (el tiempo cambia), y el candidato que se pasa queda sin ajustar."""
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    df, X, y = _tournament_data(quantile_frame)
    folds = make_group_folds(df, n_splits=5)
    candidate = candidates()["LightGBM"]
    first = ev.evaluate_candidate_folds("lgbm", candidate, X, y, folds)
    second = ev.evaluate_candidate_folds("lgbm", candidate, X, y, folds)
    pd.testing.assert_frame_equal(first.drop(columns="fit_seconds"), second.drop(columns="fit_seconds"))
    with pytest.raises(NotFittedError):
        check_is_fitted(candidate.model)  # se clona en cada pliegue


def test_family_b_goes_through_evaluate_candidate_folds_without_margin(quantile_frame):
    df, X, y = _tournament_data(quantile_frame)
    folds = make_group_folds(df, n_splits=5)
    result = ev.evaluate_candidate_folds("hgb-q", candidates()["HistGradientBoosting cuantil 0,75"], X, y, folds)
    assert (result["familia"] == "B").all() and (result["margen"] == 0).all() and (result["n_calibracion"] == 0).all()
    assert (result["n_ajuste"] == [len(tr) for tr, _ in folds]).all()  # ajusta con todo el entrenamiento del pliegue


def test_evaluate_candidate_holdout_fits_only_on_train(quantile_frame):
    """La línea base de la mediana predice la mediana del ENTRENAMIENTO (no la de la validación)."""
    df, X, y = _tournament_data(quantile_frame, n=800)
    train, valid = forward_split(df)
    row = ev.evaluate_candidate_holdout("mediana", candidates()["Línea base: siempre la mediana"], train, valid)
    assert row["pliegue"] == "adelante" and set(CANDIDATE_COLUMNS) <= set(row)
    expected_bias = valid["dias_entrega"].mean() - train["dias_entrega"].median()
    assert row["sesgo"] == pytest.approx(expected_bias)
    assert row["n_validacion"] == len(valid) and row["n_ajuste"] == len(train) and row["margen"] == 0.0


def _by_fold(models, rng):
    rows = []
    for name, family, offset in models:
        for fold in range(5):
            rows.append({"modelo": name, "familia": family, "pliegue": fold, "mae": 5 + offset + fold * 0.01, "rmse": 7.0 + offset,
                         "r2": 0.3, "sesgo": 0.1, "costo": 10 + offset + fold * 0.02, "tarde": 0.25, "cobertura": 0.75,
                         "margen": 1.0, "n_ajuste": 80, "n_calibracion": 20, "n_validacion": 25, "fit_seconds": 1.0 + fold})
    return pd.DataFrame(rows)


def test_tournament_summary_means_and_std():
    by_fold = _by_fold([("M1", "A", 0.0), ("M2", "B", 1.0)], None)
    forward = pd.DataFrame([{"modelo": "M1", "costo": 9.5, "mae": 4.8}, {"modelo": "M2", "costo": 11.2, "mae": 6.1}])
    summary = ev.tournament_summary(by_fold, forward).set_index("modelo")
    assert list(summary.index) == ["M1", "M2"] and summary.loc["M2", "familia"] == "B"
    costs = [10 + k * 0.02 for k in range(5)]
    assert summary.loc["M1", "costo_media"] == pytest.approx(np.mean(costs))
    assert summary.loc["M1", "costo_desv"] == pytest.approx(np.std(costs, ddof=1))
    assert summary.loc["M2", "adelante_costo"] == 11.2 and summary.loc["M1", "adelante_mae"] == 4.8
    assert not any("fit_seconds" in c for c in summary.columns)  # el tiempo no entra al resumen


# ================== Etapa 5: comparación por pares, posiciones y líneas base ==================


def test_compare_models_counts_wins():
    """Diferencias de costo [0,5; 0,6; 0,4; 0,5; 0,7] a favor del mejor: gana 5 de 5 y es consistente."""
    best = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    other = best + np.array([0.5, 0.6, 0.4, 0.5, 0.7])
    result = ev.compare_models(best, other)
    diffs = np.array([0.5, 0.6, 0.4, 0.5, 0.7])
    assert result["diferencia_media"] == pytest.approx(diffs.mean())
    assert result["gana"] == 5 and result["n"] == 5 and result["texto"] == "5 de 5"
    assert result["ruido"] == pytest.approx(diffs.std(ddof=1))
    assert result["ratio"] == pytest.approx(diffs.mean() / diffs.std(ddof=1)) and result["ratio"] > 1
    assert result["evaluable"] and result["consistente"]


def test_compare_models_zero_noise_is_not_evaluable():
    best = np.array([10.0, 11.0, 12.0])
    result = ev.compare_models(best, best + 0.5)  # la diferencia es idéntica en todos los pliegues
    assert result["diferencia_media"] == pytest.approx(0.5) and result["ruido"] == pytest.approx(0.0)
    assert np.isnan(result["ratio"]) and not result["evaluable"] and not result["consistente"]


def test_compare_models_consistent_vs_practical_tie():
    """Gana 3 de 5 y el cociente es menor que 1: es un empate práctico, no una ventaja."""
    best = np.full(5, 10.0)
    result = ev.compare_models(best, best + np.array([0.5, -0.4, 0.3, -0.2, 0.1]))
    assert result["gana"] == 3 and result["texto"] == "3 de 5"
    assert result["ratio"] < 1 and not result["consistente"]
    # ganar en 4 de 5 con cociente mayor que 1 sí es consistente
    ok = ev.compare_models(best, best + np.array([0.9, 1.0, 1.1, 1.0, -0.1]))
    assert ok["gana"] == 4 and ok["ratio"] > 1 and ok["consistente"]
    # ganar en 3 de 5 NO alcanza aunque el cociente sea grande
    three = ev.compare_models(best, best + np.array([1.0, 1.1, 0.9, -0.01, -0.02]))
    assert three["gana"] == 3 and not three["consistente"]


def test_rank_table_orders_by_metric():
    summary = pd.DataFrame({
        "modelo": ["A", "B", "C"],
        "costo_media": [10.0, 9.0, 11.0],
        "r2_media": [0.1, 0.3, 0.2],       # en R² mayor es mejor
        "sesgo_media": [0.5, -0.1, 0.3],   # en el sesgo importa la distancia a cero
    })
    ranks = ev.rank_table(summary, ["costo", "r2", "sesgo"]).set_index("modelo")
    assert ranks["posicion_costo"].tolist() == [2, 1, 3]
    assert ranks["posicion_r2"].tolist() == [3, 1, 2]
    assert ranks["posicion_sesgo"].tolist() == [3, 1, 2]


def _folds_table(costs_by_model):
    rows = [{"modelo": name, "pliegue": fold, "costo": cost} for name, costs in costs_by_model.items() for fold, cost in enumerate(costs)]
    return pd.DataFrame(rows)


def test_beats_baselines_requires_more_than_noise():
    by_fold = _folds_table({
        "bueno": [8.0, 8.1, 7.9, 8.0, 8.2],
        "dudoso": [9.9, 10.1, 9.7, 10.3, 10.0],
        "mediana": [10.0, 10.0, 10.0, 10.0, 10.0],
        "plazo fijo": [9.0, 9.0, 9.0, 9.0, 9.0],
    })
    good = ev.beats_baselines(by_fold, "bueno", ["mediana", "plazo fijo"])
    assert good["mediana"]["supera"] and good["plazo fijo"]["supera"]
    doubtful = ev.beats_baselines(by_fold, "dudoso", ["mediana", "plazo fijo"])
    assert not doubtful["mediana"]["supera"]      # la diferencia (≈ 0) no supera su ruido
    assert not doubtful["plazo fijo"]["supera"]   # es peor que el plazo fijo


def test_practical_ties_lists_indistinguishable_models():
    base = np.array([10.0, 10.2, 9.8, 10.1, 9.9])
    by_fold = _folds_table({
        "mejor": base,
        "casi igual": base + np.array([0.3, -0.3, 0.2, -0.2, 0.0]),
        "claramente peor": base + np.array([2.0, 2.1, 1.9, 2.0, 2.2]),
    })
    assert ev.practical_ties(by_fold, "mejor") == ["casi igual"]


def test_per_thousand():
    assert ev.per_thousand(0.0123) == pytest.approx(12.3)
    assert ev.per_thousand(np.array([0.001, 0.002])).tolist() == pytest.approx([1.0, 2.0])


# ============ Etapa 5: tabla final del torneo y métricas por región (funciones que arman el notebook) ============

BASELINES = {"mediana": "Línea base: siempre la mediana", "plazo_fijo": "Línea base: plazo fijo (percentil 75)"}


def _tournament_inputs():
    costs = {
        "Línea base: siempre la mediana": [14.0, 14.3, 13.8, 14.2, 13.9],
        "Línea base: plazo fijo (percentil 75)": [12.0, 12.3, 11.9, 12.1, 11.9],
        "Modelo bueno": [10.0, 10.1, 9.9, 10.2, 9.8],
        "Modelo parecido": [10.05, 10.0, 10.0, 10.25, 9.8],  # costo medio 10,02: apenas peor que el bueno (10,00)
        "Modelo malo": [13.0, 13.2, 12.9, 13.1, 13.0],
    }
    rows = []
    for name, series in costs.items():
        family = "base" if name.startswith("Línea base") else "A"
        for fold, cost in enumerate(series):
            rows.append({"modelo": name, "familia": family, "pliegue": fold, "mae": 5 + cost / 10, "rmse": 7 + cost / 10, "r2": 0.5 - cost / 100,
                         "sesgo": 0.1, "costo": cost, "tarde": 0.25, "cobertura": 0.75, "margen": 1.0, "n_ajuste": 80,
                         "n_calibracion": 20, "n_validacion": 25, "fit_seconds": 1.0})
    by_fold = pd.DataFrame(rows)
    forward = pd.DataFrame([{"modelo": name, "costo": float(np.mean(series)), "mae": 5 + float(np.mean(series)) / 10} for name, series in costs.items()])
    return by_fold, forward


def test_tournament_table_has_everything_the_report_needs():
    by_fold, forward = _tournament_inputs()
    table = ev.tournament_table(by_fold, forward, BASELINES).set_index("modelo")
    assert table.index[0] == "Modelo bueno"  # ordenada por costo, de menor a mayor
    assert bool(table.loc["Modelo bueno", "mejor"]) and not table["mejor"].drop("Modelo bueno").any()
    assert table.loc["Modelo bueno", "posicion_costo"] == 1 and table.loc["Modelo malo", "posicion_costo"] == 4
    # supera las dos líneas base el modelo bueno; el malo supera a la mediana pero no al plazo fijo
    assert table.loc["Modelo bueno", "supera_mediana"] and table.loc["Modelo bueno", "supera_plazo_fijo"]
    assert table.loc["Modelo malo", "supera_mediana"] and not table.loc["Modelo malo", "supera_plazo_fijo"]
    # el parecido es un empate práctico con el mejor; el malo no
    assert table.loc["Modelo parecido", "empate_con_el_mejor"] and not table.loc["Modelo malo", "empate_con_el_mejor"]
    assert table.loc["Modelo malo", "gana_vs_mejor"] == "5 de 5" and table.loc["Modelo malo", "consistente_vs_mejor"]
    assert "fit_seconds" not in " ".join(table.columns)


def test_unknown_model_raises_instead_of_returning_nan():
    by_fold, _ = _tournament_inputs()
    with pytest.raises(KeyError, match="no hay resultados"):
        ev.beats_baselines(by_fold, "Modelo que no existe", ["Línea base: siempre la mediana"])


def test_tournament_table_best_ignores_baselines():
    """Si una línea base tuviera el menor costo, el «mejor» sigue siendo el mejor modelo (y se ve que no lo supera)."""
    by_fold, forward = _tournament_inputs()
    by_fold.loc[by_fold["modelo"] == "Línea base: plazo fijo (percentil 75)", "costo"] = 5.0
    table = ev.tournament_table(by_fold, forward, BASELINES).set_index("modelo")
    assert bool(table.loc["Modelo bueno", "mejor"])
    assert not table.loc["Modelo bueno", "supera_plazo_fijo"]


def test_regional_metrics_splits_out_of_fold_predictions_by_region(quantile_frame):
    df = quantile_frame(500, seed=33)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    folds = make_group_folds(df, n_splits=5)
    table = ev.regional_metrics("Ridge", candidates()["Ridge"], X, y, folds)
    assert list(table.columns) == ["modelo", "grupo", "n_pedidos", "mae", "tarde"]
    assert set(table["modelo"]) == {"Ridge"} and table["n_pedidos"].sum() == len(X)  # cada pedido se predice una vez, fuera de muestra
    assert set(table["grupo"]) <= set(X["customer_region"].dropna()) | {"(sin dato)"}
    assert table["tarde"].between(0, 1).all() and (table["mae"] >= 0).all()


def test_regional_metrics_are_really_out_of_sample(quantile_frame):
    """Un árbol sin límite memoriza lo que ve: si las predicciones no fueran fuera de muestra, su MAE sería casi 0."""
    from sklearn.tree import DecisionTreeRegressor

    from olistbr.models.train import Candidate

    df = quantile_frame(600, seed=34)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    folds = make_group_folds(df, n_splits=5)
    memorizer = Candidate(DecisionTreeRegressor(random_state=0), "B", False)
    table = ev.regional_metrics("árbol", memorizer, X, y, folds)
    assert (table["mae"] > 1.0).all(), table  # con fuga saldría ≈ 0


def test_regional_metrics_requires_the_region_column(quantile_frame):
    df = quantile_frame(300, seed=35)
    X, y = df.drop(columns=["dias_entrega", "customer_region"]), df["dias_entrega"]
    with pytest.raises(ValueError, match="región"):
        ev.regional_metrics("Ridge", candidates()["Ridge"], X, y, make_group_folds(df, n_splits=5))


def test_coverage_by_month_reports_one_row_per_month(quantile_frame):
    df = quantile_frame(900, seed=36)
    train, valid = forward_split(df)
    table = ev.coverage_by_month("Ridge", candidates()["Línea base: plazo fijo (percentil 75)"], train, valid)
    assert list(table.columns) == ["modelo", "mes", "n_pedidos", "cobertura", "mediana_dias", "p75_dias"]
    assert table["n_pedidos"].sum() == len(valid) and table["mes"].is_monotonic_increasing
    assert table["cobertura"].between(0, 1).all()
    # el plazo fijo es el percentil 75 del entrenamiento: la cobertura de cada mes se puede recalcular a mano
    cut = train["dias_entrega"].quantile(0.75)
    first_month = valid[valid["order_purchase_timestamp"].dt.to_period("M").astype(str) == table.loc[0, "mes"]]
    assert table.loc[0, "cobertura"] == pytest.approx((first_month["dias_entrega"] <= cut).mean())
