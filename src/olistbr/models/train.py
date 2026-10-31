"""Pipeline del modelo de referencia: preprocesador + regresión lineal.

Es el modelo con el que se compara cada idea de variable (etapa 4). El torneo con otros modelos y la
optimización son las etapas 5 y 6. Todo lo que aprende de los datos está dentro del `Pipeline`, así que
se ajusta solo con el entrenamiento de cada pliegue.
"""
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from olistbr.features.engineering import build_preprocessor


def build_pipeline(groups=(), include_reference: bool = False) -> Pipeline:
    """`Pipeline` con los pasos `prep` (preprocesador de los grupos activos) y `model` (regresión lineal)."""
    return Pipeline(
        [
            ("prep", build_preprocessor(groups, include_reference)),
            ("model", LinearRegression()),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """Línea base trivial: predice siempre la mediana de los días de entrega del entrenamiento (sin variables)."""
    return Pipeline([("model", DummyRegressor(strategy="median"))])
