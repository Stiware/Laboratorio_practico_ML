"""Candidatos del torneo, pipeline del modelo y modelo de plazo (etapa 5).

Sigue la forma de `train.py` del ejemplo CreditSight, adaptada de clasificación a **regresión**: en vez de probabilidades
y una regla de decisión por costo esperado, cada modelo produce el **plazo que recomendaría la app**.

Con subestimar 3 veces más caro que sobrestimar, el plazo que minimiza el costo es el **percentil 75** de lo que podría
tardar el pedido (el costo 3 a 1 es 4 veces la pérdida cuantílica de 0,75). Hay dos familias de modelos:
- **A (valor típico + margen):** el modelo predice un valor y se le suma un margen, calculado con clientes apartados.
- **B (percentil 75 directo):** el modelo ya predice el percentil 75; no lleva margen.
Las dos líneas base van aparte (`base`).
"""
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from olistbr.config import CALIBRATION_SIZE, RANDOM_STATE, TARGET_QUANTILE
from olistbr.features.engineering import build_preprocessor
from olistbr.models.metrics import bias, coverage, mae_days, mean_delivery_cost, share_late


class Candidate(NamedTuple):
    """Un candidato del torneo: el modelo con parámetros fijos, su familia y si necesita variables escaladas."""

    model: object
    family: str  # "base", "A" (valor típico + margen) o "B" (percentil 75 directo)
    scale: bool


def candidates(random_state: int = RANDOM_STATE) -> dict:
    """Los 10 candidatos del torneo, con parámetros razonables y SIN optimizar (la optimización es la etapa 6).

    Devuelve modelos nuevos en cada llamada. El bosque aleatorio es liviano (100 árboles, profundidad 12, hoja mínima 20,
    33 % de variables, 50 % de filas por árbol): el de 200 árboles tardaba 254 s por ajuste y este 18 s.
    """
    q = TARGET_QUANTILE
    lgbm = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=random_state, n_jobs=2,
                deterministic=True, force_row_wise=True, verbose=-1)
    xgb = dict(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist", random_state=random_state, n_jobs=2)
    return {
        "Línea base: siempre la mediana": Candidate(DummyRegressor(strategy="median"), "base", False),
        "Línea base: plazo fijo (percentil 75)": Candidate(DummyRegressor(strategy="quantile", quantile=q), "base", False),
        "Regresión lineal": Candidate(LinearRegression(), "A", True),
        "Ridge": Candidate(Ridge(alpha=1.0), "A", True),
        "Bosque aleatorio liviano": Candidate(
            RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=20, max_features=0.33,
                                  max_samples=0.5, n_jobs=2, random_state=random_state),
            "A", False,
        ),
        "LightGBM": Candidate(LGBMRegressor(**lgbm), "A", False),
        "XGBoost": Candidate(XGBRegressor(**xgb), "A", False),
        "LightGBM cuantil 0,75": Candidate(LGBMRegressor(objective="quantile", alpha=q, **lgbm), "B", False),
        "XGBoost cuantil 0,75": Candidate(XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **xgb), "B", False),
        "HistGradientBoosting cuantil 0,75": Candidate(
            HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                          random_state=random_state),
            "B", False,
        ),
    }


def build_pipeline(groups=(), include_reference: bool = False, model=None, scale: bool = True) -> Pipeline:
    """`Pipeline` con los pasos `prep` (preprocesador de los grupos activos) y `model`.

    Sin `model`, el modelo es la regresión lineal de la etapa 4 (mismos resultados que antes). `groups` va primero para
    no romper el código de la etapa 4, que lo pasa en primera posición. `scale=False` para modelos de árboles.
    """
    return Pipeline(
        [
            ("prep", build_preprocessor(groups, include_reference, scale=scale)),
            ("model", LinearRegression() if model is None else model),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Línea base trivial de la etapa 4: predice siempre la mediana de los días de entrega del entrenamiento (sin variables)."""
    return Pipeline([("model", DummyRegressor(strategy="median"))])


def regression_metrics(real, predicted, recommended=None) -> dict:
    """Métricas de una evaluación: de la predicción del modelo y del plazo recomendado.

    - De `predicted` (lo que predice el modelo): `mae`, `rmse`, `r2` y `sesgo`.
    - De `recommended` (el plazo que recomendaría la app; si es `None`, el mismo `predicted`): `costo` 3 a 1,
      `tarde` (proporción con `real > recomendado`) y `cobertura` (su complemento).
    Para los modelos de la familia B, `predicted` ya es el percentil 75: su MAE, RMSE y R² no son comparables con los de la A.
    """
    real = np.asarray(real, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    plan = predicted if recommended is None else np.asarray(recommended, dtype=float)
    return {
        "mae": mae_days(real, predicted),
        "rmse": float(root_mean_squared_error(real, predicted)),
        "r2": float(r2_score(real, predicted)),
        "sesgo": bias(real, predicted),
        "costo": mean_delivery_cost(real, plan),
        "tarde": share_late(real, plan),
        "cobertura": coverage(real, plan),
    }


class DeadlineModel:
    """Un modelo ajustado junto con su margen: predice un valor y recomienda un plazo (`predicción + margen`).

    Guarda también qué filas se usaron para ajustar y cuáles para calibrar el margen (útil para comprobar que no se mezclan).
    """

    def __init__(self, pipeline, margin: float, fit_index, calibration_index):
        self.pipeline = pipeline
        self.margin = float(margin)
        self.fit_index = fit_index
        self.calibration_index = calibration_index

    @property
    def n_fit(self) -> int:
        return len(self.fit_index)

    @property
    def n_calibration(self) -> int:
        return len(self.calibration_index)

    def predict(self, X):
        """Lo que predice el modelo (valor típico en la familia A, percentil 75 en la B)."""
        return self.pipeline.predict(X)

    def recommend(self, X):
        """El plazo que recomendaría la app: la predicción más el margen (el margen es 0 en la familia B y las líneas base)."""
        return self.predict(X) + self.margin

    def predict_with_deadline(self, X):
        """Devuelve `(predicción, plazo recomendado)` prediciendo una sola vez (para no repetir el cálculo del margen)."""
        predicted = self.predict(X)
        return predicted, predicted + self.margin


MIN_CALIBRATION_CUSTOMERS = 50  # con menos clientes apartados el margen sería poco confiable


def fit_deadline_model(
    pipeline,
    family: str,
    X,
    y,
    groups,
    quantile: float = TARGET_QUANTILE,
    calibration_size: float = CALIBRATION_SIZE,
    seed: int = RANDOM_STATE,
) -> DeadlineModel:
    """Ajusta un pipeline y calcula el margen del plazo recomendado, sin usar nunca la validación.

    - Familia `"A"` (valor típico + margen): se aparta el `calibration_size` de los **clientes** (`GroupShuffleSplit`), se ajusta
      con el resto y el margen es el percentil `quantile` de los errores (real − predicción) de los clientes apartados.
      No se reajusta con todo: el margen corresponde al modelo tal como quedó. Ningún cliente está en el ajuste y en la calibración.
    - Familias `"B"` (percentil 75 directo) y `"base"`: se ajusta con todo y el margen es 0.
    `groups` es el cliente de cada fila (misma longitud que `X`). El pipeline que se pasa no se modifica.
    """
    if family not in {"A", "B", "base"}:
        raise ValueError(f"familia desconocida: {family!r} (use 'A', 'B' o 'base')")
    groups_array = np.asarray(groups)
    if len(groups_array) != len(X):
        raise ValueError(f"`groups` tiene {len(groups_array)} entradas y `X` tiene {len(X)} filas: debe haber un cliente por fila")
    if pd.isna(groups_array).any():
        raise ValueError("hay filas sin cliente en `groups`: no se pueden agrupar para la calibración")
    model = clone(pipeline)
    if family != "A":
        model.fit(X, y)
        return DeadlineModel(model, 0.0, X.index, X.index[:0])

    groups = groups_array
    splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=seed)
    fit_idx, calibration_idx = next(splitter.split(X, groups=groups))
    if not set(groups[fit_idx]).isdisjoint(set(groups[calibration_idx])):
        raise AssertionError("hay clientes en el ajuste y en la calibración del margen")
    if len(set(groups[calibration_idx])) < MIN_CALIBRATION_CUSTOMERS:
        warnings.warn(
            f"solo {len(set(groups[calibration_idx]))} clientes de calibración: el margen puede ser poco confiable", stacklevel=2
        )
    model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    residuals = y.iloc[calibration_idx].to_numpy(dtype=float) - model.predict(X.iloc[calibration_idx])
    margin = float(np.percentile(residuals, quantile * 100))
    return DeadlineModel(model, margin, X.index[fit_idx], X.index[calibration_idx])
