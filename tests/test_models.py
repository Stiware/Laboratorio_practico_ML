"""Pruebas del pipeline del modelo de referencia: preprocesador + regresión lineal."""
import numpy as np
import pandas as pd

import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

import olistbr.features.engineering as eng
import olistbr.models.train as train_module
from olistbr.config import TOURNAMENT_GROUPS
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


# ======================== Etapa 5: candidatos del torneo y pipeline general ========================

EXPECTED_NAMES = [
    "Línea base: siempre la mediana",
    "Línea base: plazo fijo (percentil 75)",
    "Regresión lineal",
    "Ridge",
    "Bosque aleatorio liviano",
    "LightGBM",
    "XGBoost",
    "LightGBM cuantil 0,75",
    "XGBoost cuantil 0,75",
    "HistGradientBoosting cuantil 0,75",
]


def test_candidates_have_expected_names_and_families():
    cands = train_module.candidates()
    assert list(cands) == EXPECTED_NAMES
    families = [c.family for c in cands.values()]
    assert families.count("base") == 2 and families.count("A") == 5 and families.count("B") == 3
    assert [cands[n].family for n in EXPECTED_NAMES[:2]] == ["base", "base"]
    assert all(cands[n].family == "B" for n in EXPECTED_NAMES[-3:])
    assert cands["Regresión lineal"].scale and cands["Ridge"].scale  # los lineales necesitan variables escaladas
    assert not cands["LightGBM"].scale and not cands["Bosque aleatorio liviano"].scale


def test_candidates_are_fresh_objects_each_time():
    """Cada llamada devuelve modelos nuevos (no se comparten ajustes entre pliegues)."""
    first, second = train_module.candidates(), train_module.candidates()
    assert first["LightGBM"].model is not second["LightGBM"].model


def test_candidate_parameters_match_the_spec():
    cands = train_module.candidates(random_state=42)
    forest = cands["Bosque aleatorio liviano"].model.get_params()
    assert (forest["n_estimators"], forest["max_depth"], forest["min_samples_leaf"]) == (100, 12, 20)
    assert (forest["max_features"], forest["max_samples"], forest["random_state"]) == (0.33, 0.5, 42)
    assert cands["LightGBM cuantil 0,75"].model.get_params()["objective"] == "quantile"
    assert cands["LightGBM cuantil 0,75"].model.get_params()["alpha"] == 0.75
    assert cands["XGBoost cuantil 0,75"].model.get_params()["objective"] == "reg:quantileerror"
    assert cands["XGBoost cuantil 0,75"].model.get_params()["quantile_alpha"] == 0.75
    hist = cands["HistGradientBoosting cuantil 0,75"].model.get_params()
    assert hist["loss"] == "quantile" and hist["quantile"] == 0.75
    assert cands["Línea base: plazo fijo (percentil 75)"].model.get_params()["quantile"] == 0.75
    assert cands["Línea base: siempre la mediana"].model.get_params()["strategy"] == "median"
    assert cands["LightGBM"].model.get_params()["deterministic"] is True  # resultados repetibles


def test_every_candidate_fits_a_tiny_table(quantile_frame):
    df = quantile_frame(150, seed=2)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    for name, candidate in train_module.candidates().items():
        pipeline = build_pipeline(TOURNAMENT_GROUPS, model=candidate.model, scale=candidate.scale).fit(X, y)
        prediction = pipeline.predict(X)
        assert prediction.shape == (len(X),) and np.isfinite(prediction).all(), name


def test_build_pipeline_accepts_any_model(frame_factory):
    df = eng.add_features(frame_factory(120))
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    for model in (Ridge(alpha=1.0), train_module.candidates()["LightGBM"].model):
        pipeline = build_pipeline(["logs"], model=model).fit(X, y)
        assert list(pipeline.named_steps) == ["prep", "model"]
        assert pipeline.named_steps["model"] is not None and np.isfinite(pipeline.predict(X)).all()


def test_build_pipeline_default_is_still_linear_regression(frame_factory):
    """Sin `model`, el pipeline sigue siendo el de la etapa 4: mismas predicciones que prep + LinearRegression."""
    df = eng.add_features(frame_factory(150, seed=4))
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    groups = ["logs", "same_state"]
    new = build_pipeline(groups).fit(X, y)
    manual = Pipeline([("prep", eng.build_preprocessor(groups)), ("model", LinearRegression())]).fit(X, y)
    assert isinstance(new.named_steps["model"], LinearRegression)
    np.testing.assert_allclose(new.predict(X), manual.predict(X))


# ================== Etapa 5: métricas de regresión y modelo de plazo (margen por calibración) ==================


def test_regression_metrics_by_hand():
    """real=[10,20], predicho=[12,18]: MAE 2, RMSE 2, sesgo 0, R² 0,84; costo (2 de sobrestima + 3×2 de subestima)/2 = 4."""
    m = train_module.regression_metrics([10.0, 20.0], [12.0, 18.0])
    assert m["mae"] == pytest.approx(2.0) and m["rmse"] == pytest.approx(2.0)
    assert m["sesgo"] == pytest.approx(0.0) and m["r2"] == pytest.approx(0.84)
    assert m["costo"] == pytest.approx(4.0)
    assert m["tarde"] == 0.5 and m["cobertura"] == 0.5 and m["tarde"] + m["cobertura"] == 1.0


def test_perfect_predictions_give_zero_error():
    m = train_module.regression_metrics([3.0, 8.0, 15.0], [3.0, 8.0, 15.0])
    assert m["mae"] == 0 and m["rmse"] == 0 and m["costo"] == 0 and m["sesgo"] == 0
    assert m["r2"] == pytest.approx(1.0) and m["tarde"] == 0 and m["cobertura"] == 1.0


def test_regression_metrics_uses_recommended_for_cost():
    """El MAE sale de la predicción; el costo, el porcentaje tarde y la cobertura salen del plazo recomendado."""
    m = train_module.regression_metrics([10.0, 20.0], predicted=[10.0, 20.0], recommended=[12.0, 22.0])
    assert m["mae"] == 0.0
    assert m["costo"] == pytest.approx(2.0)  # se sobrestima 2 días en cada pedido (cuesta 1 por día)
    assert m["tarde"] == 0.0 and m["cobertura"] == 1.0


class _Truth(BaseEstimator, RegressorMixin):
    """Modelo de juguete: lee la columna `truth` de X y predice `truth + shift` (shift<0: subestima)."""

    def __init__(self, shift=0.0):
        self.shift = shift

    def fit(self, X, y):
        self.fit_index_ = X.index
        return self

    def predict(self, X):
        return X["truth"].to_numpy() + self.shift


class _OnlyKnowsWhatItSaw(BaseEstimator, RegressorMixin):
    """Acierta exactamente lo que vio en el ajuste y se equivoca por 5 días en lo que no vio (como un árbol que memoriza)."""

    def fit(self, X, y):
        self.seen_ = set(X.index)
        return self

    def predict(self, X):
        seen = np.array([i in self.seen_ for i in X.index])
        return X["truth"].to_numpy() - np.where(seen, 0.0, 5.0)


def _toy(n_customers=400, per_customer=2, seed=0):
    rng = np.random.default_rng(seed)
    customers = np.repeat([f"u{i}" for i in range(n_customers)], per_customer)
    truth = rng.gamma(2.0, 5.0, len(customers))
    X = pd.DataFrame({"truth": truth, "customer_unique_id": customers})
    return X, pd.Series(truth, index=X.index), X["customer_unique_id"]


def _pipe(model):
    return Pipeline([("model", model)])


def test_deadline_model_recommend_adds_margin():
    """Un modelo que siempre subestima 2 días recibe un margen de 2: el plazo recomendado reproduce el valor real."""
    X, y, groups = _toy()
    model = train_module.fit_deadline_model(_pipe(_Truth(shift=-2.0)), "A", X, y, groups)
    np.testing.assert_allclose(model.recommend(X), X["truth"].to_numpy())  # (truth − 2) + 2
    predicted, deadline = model.predict_with_deadline(X)
    np.testing.assert_allclose(predicted, X["truth"].to_numpy() - 2.0)
    np.testing.assert_allclose(deadline - predicted, 2.0)


def test_margin_comes_from_calibration_customers_only():
    X, y, groups = _toy()
    # un modelo perfecto en lo que vio y con 5 días de error en lo que no vio: el margen debe ser 5 (se midió en clientes
    # que NO participaron en el ajuste); si se midiera con los mismos clientes del ajuste saldría 0
    memorizer = train_module.fit_deadline_model(_pipe(_OnlyKnowsWhatItSaw()), "A", X, y, groups)
    assert memorizer.margin == pytest.approx(5.0)
    # un modelo que siempre subestima 2 días da margen 2; uno perfecto, 0
    assert train_module.fit_deadline_model(_pipe(_Truth(shift=-2.0)), "A", X, y, groups).margin == pytest.approx(2.0)
    assert train_module.fit_deadline_model(_pipe(_Truth(shift=0.0)), "A", X, y, groups).margin == pytest.approx(0.0)


def test_no_customer_in_fit_and_calibration():
    X, y, groups = _toy()
    model = train_module.fit_deadline_model(_pipe(_Truth()), "A", X, y, groups)
    seen_by_the_model = model.pipeline.named_steps["model"].fit_index_  # las filas con las que el modelo SE AJUSTÓ de verdad
    assert set(seen_by_the_model) == set(model.fit_index)
    in_fit = set(groups.loc[seen_by_the_model])
    in_calibration = set(groups.loc[model.calibration_index])
    assert in_fit.isdisjoint(in_calibration) and in_fit and in_calibration
    assert model.n_fit + model.n_calibration == len(X)
    assert model.n_calibration == pytest.approx(0.2 * len(X), rel=0.1)  # el 20 % de los clientes


def test_margin_can_be_negative():
    """Un modelo que sobreestima da margen negativo: se informa tal cual, no se corrige a cero."""
    X, y, groups = _toy()
    assert train_module.fit_deadline_model(_pipe(_Truth(shift=+3.0)), "A", X, y, groups).margin == pytest.approx(-3.0)


def test_family_b_has_zero_margin_and_uses_all_rows():
    X, y, groups = _toy()
    for family in ("B", "base"):
        model = train_module.fit_deadline_model(_pipe(_Truth(shift=-2.0)), family, X, y, groups)
        assert model.margin == 0.0 and model.n_fit == len(X) and model.n_calibration == 0
        np.testing.assert_allclose(model.recommend(X), model.predict(X))


def test_margin_gives_about_75_percent_coverage(quantile_frame):
    """Familia A con ruido asimétrico: el plazo (predicción + margen) cubre entre 70 % y 80 % de clientes NUEVOS."""
    df = quantile_frame(4000, seed=11)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    customers = np.array(sorted(set(X["customer_unique_id"])))
    train_customers = set(customers[: len(customers) // 2])
    is_train = X["customer_unique_id"].isin(train_customers).to_numpy()
    pipeline = build_pipeline(TOURNAMENT_GROUPS, model=LinearRegression())
    model = train_module.fit_deadline_model(pipeline, "A", X[is_train], y[is_train], X.loc[is_train, "customer_unique_id"])
    covered = (y[~is_train] <= model.recommend(X[~is_train])).mean()
    # La banda es de ±5 puntos porque el margen se calcula con ≈ 400 pedidos de calibración (error de muestreo ≈ ±2 puntos de
    # cobertura): una banda más estrecha fallaría por azar y no por un error del código. El percentil exacto lo fija
    # `test_margin_is_exactly_the_75th_percentile_of_calibration_errors`.
    assert 0.70 <= covered <= 0.80, covered
    assert model.margin > 0  # el ruido tiene cola a la derecha: hace falta un margen positivo


def test_family_b_quantile_models_cover_about_75_percent(quantile_frame):
    """La configuración cuantílica (objetivo `quantile`, alfa 0,75) con `fit_deadline_model` cubre entre 70 % y 80 % de pedidos nuevos.

    Comprueba que el proyecto arma bien un modelo de la familia B (margen 0, todo el entrenamiento); no prueba la librería en sí.
    """
    from lightgbm import LGBMRegressor

    df = quantile_frame(6000, seed=12)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    customers = np.array(sorted(set(X["customer_unique_id"])))
    is_train = X["customer_unique_id"].isin(set(customers[: len(customers) // 2])).to_numpy()
    model_b = LGBMRegressor(objective="quantile", alpha=0.75, n_estimators=100, learning_rate=0.05, num_leaves=7,
                            min_child_samples=40, random_state=42, deterministic=True, force_row_wise=True, verbose=-1)
    pipeline = build_pipeline(TOURNAMENT_GROUPS, model=model_b, scale=False)
    fitted = train_module.fit_deadline_model(pipeline, "B", X[is_train], y[is_train], X.loc[is_train, "customer_unique_id"])
    covered = (y[~is_train] <= fitted.recommend(X[~is_train])).mean()
    assert 0.70 <= covered <= 0.80, covered


def test_warns_when_there_are_few_calibration_customers():
    X, y, groups = _toy(n_customers=40)  # el 20 % son 8 clientes: muy pocos
    with pytest.warns(UserWarning, match="calibración"):
        train_module.fit_deadline_model(_pipe(_Truth()), "A", X, y, groups)


def test_fit_deadline_model_rejects_unknown_family():
    X, y, groups = _toy()
    with pytest.raises(ValueError, match="familia"):
        train_module.fit_deadline_model(_pipe(_Truth()), "Z", X, y, groups)


def test_fit_deadline_model_does_not_modify_its_inputs():
    X, y, groups = _toy()
    X_before, y_before = X.copy(deep=True), y.copy(deep=True)
    pipeline = _pipe(_Truth(shift=-1.0))
    train_module.fit_deadline_model(pipeline, "A", X, y, groups)
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)
    assert not hasattr(pipeline.named_steps["model"], "fit_index_")  # el pipeline que se pasó queda sin ajustar (se clona)


def test_fit_deadline_model_validates_the_customers():
    X, y, groups = _toy()
    with pytest.raises(ValueError, match="una por fila|un cliente por fila"):
        train_module.fit_deadline_model(_pipe(_Truth()), "A", X, y, groups.iloc[:-5])
    with_missing = groups.astype(object).copy()
    with_missing.iloc[3] = np.nan
    with pytest.raises(ValueError, match="sin cliente"):
        train_module.fit_deadline_model(_pipe(_Truth()), "A", X, y, with_missing)


def test_tournament_pipeline_never_receives_identifiers_or_discarded_ideas(quantile_frame):
    """El cliente, el pedido, el plazo prometido, la marca de fin de año y el historial no llegan al modelo del torneo."""
    df = quantile_frame(300, seed=8)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    assert {"customer_unique_id", "order_id", "promised_days", "is_year_end_peak", "route"} <= set(X.columns)  # están en la tabla
    for name, candidate in train_module.candidates().items():
        pipeline = build_pipeline(TOURNAMENT_GROUPS, model=candidate.model, scale=candidate.scale).fit(X, y)
        used, dropped = _used_columns(pipeline)
        assert not {"customer_unique_id", "order_id", "promised_days", "is_year_end_peak", "route", "dias_entrega"} & set(used), name
        assert "customer_unique_id" in dropped and "promised_days" in dropped


class _KnownNoise(BaseEstimator, RegressorMixin):
    """Modelo de juguete: predice `truth − noise`, así el error (real − predicción) de cada fila es exactamente su `noise`."""

    def fit(self, X, y):
        self.fitted_ = True  # scikit-learn exige algún atributo con guion bajo final en un modelo ajustado
        return self

    def predict(self, X):
        return X["truth"].to_numpy() - X["noise"].to_numpy()


def test_margin_is_exactly_the_75th_percentile_of_calibration_errors():
    """El margen es el percentil 75 (no el 72, el 78 ni la media) de los errores de los clientes de calibración, calculado a mano."""
    rng = np.random.default_rng(5)
    X, y, groups = _toy(n_customers=600, seed=5)
    X["noise"] = rng.gamma(2.0, 2.0, len(X))  # errores asimétricos, conocidos fila por fila
    model = train_module.fit_deadline_model(_pipe(_KnownNoise()), "A", X, y, groups)
    calibration_noise = X.loc[model.calibration_index, "noise"].to_numpy()
    assert model.margin == pytest.approx(np.percentile(calibration_noise, 75))
    assert model.margin != pytest.approx(np.percentile(calibration_noise, 50))
    assert model.margin != pytest.approx(calibration_noise.mean())
