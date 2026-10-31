"""Pruebas del pipeline del modelo de referencia: preprocesador + regresión lineal."""
import numpy as np
import pandas as pd

import olistbr.features.engineering as eng
from olistbr.models.train import build_baseline_pipeline, build_pipeline


def test_pipeline_fits_and_predicts(frame_factory):
    df = eng.add_features(frame_factory(150))
    model = build_pipeline(list(eng.GROUPS)).fit(df, df["dias_entrega"])
    predictions = model.predict(df)
    assert predictions.shape == (len(df),) and np.isfinite(predictions).all()
    # el objetivo depende de la distancia: el modelo debe explicar bastante mejor que la media
    error = np.abs(predictions - df["dias_entrega"]).mean()
    baseline = np.abs(df["dias_entrega"].mean() - df["dias_entrega"]).mean()
    assert error < baseline


def test_pipeline_is_fit_only_on_train(frame_factory):
    """Las medianas del imputador salen del entrenamiento; transformar otra tabla no las cambia."""
    train = eng.add_features(frame_factory(100, seed=1))
    other = eng.add_features(frame_factory(100, seed=2))
    other["distance_km"] = other["distance_km"] * 1000  # valores muy distintos
    model = build_pipeline().fit(train, train["dias_entrega"])
    imputer = model.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    before = imputer.statistics_.copy()
    model.predict(other)
    np.testing.assert_array_equal(imputer.statistics_, before)
    distance_position = eng.BASE_NUMERIC.index("distance_km")
    assert before[distance_position] == pd.Series(train["distance_km"]).median()


def test_pipeline_has_named_steps():
    pipeline = build_pipeline()
    assert list(pipeline.named_steps) == ["prep", "model"]


def _used_columns(pipeline):
    """Columnas que el preprocesador realmente usa (se excluye el «remainder», que lista las que descarta)."""
    prep = pipeline.named_steps["prep"]
    used = [c for name, _, cols in prep.transformers_ if name != "remainder" for c in cols]
    dropped = [c for name, _, cols in prep.transformers_ if name == "remainder" for c in cols]
    return used, dropped


def test_final_pipeline_never_sees_the_promised_days(frame_factory):
    """El plazo prometido por Olist no llega al modelo final por ningún camino."""
    df = eng.add_features(frame_factory(150))
    assert "promised_days" in df.columns  # la tabla sí la trae: se calcula para medir la referencia
    pipeline = build_pipeline(list(eng.GROUPS)).fit(df, df["dias_entrega"])
    assert pipeline.named_steps["prep"].remainder == "drop"  # lo que no está en las listas se descarta
    used, dropped = _used_columns(pipeline)
    assert "promised_days" not in used
    assert "promised_days" in dropped  # llega al preprocesador y es descartada


def test_reference_pipeline_is_the_only_one_with_promised_days(frame_factory):
    df = eng.add_features(frame_factory(150))
    with_reference = build_pipeline(include_reference=True).fit(df, df["dias_entrega"])
    used, _ = _used_columns(with_reference)
    assert "promised_days" in used
    without, _ = _used_columns(build_pipeline().fit(df, df["dias_entrega"]))
    assert "promised_days" not in without


def test_baseline_pipeline_predicts_the_training_median(frame_factory):
    df = eng.add_features(frame_factory(100))
    baseline = build_baseline_pipeline().fit(df, df["dias_entrega"])
    np.testing.assert_allclose(baseline.predict(df), np.full(len(df), df["dias_entrega"].median()))
