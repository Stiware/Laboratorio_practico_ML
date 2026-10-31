"""Métricas de OnTrack: error en días y costo asimétrico del error (reports/00_caso.md).

Predecir menos días de los reales (el pedido llega más tarde de lo dicho) cuesta más que predecir de más.
El supuesto del grupo es 3 a 1, y se revisa con 1, 2, 3, 5 y 9 a 1 (`COST_RATIOS`). Con proporción 1 a 1
el costo medio es exactamente el MAE. Estas funciones se usan en el caso de negocio, el torneo, la
optimización y la validación, para que todas midan lo mismo.
"""
import numpy as np

from olistbr.config import COST_OVERESTIMATE, COST_UNDERESTIMATE


def delivery_cost(real, predicted, under_cost: float = COST_UNDERESTIMATE, over_cost: float = COST_OVERESTIMATE):
    """Costo relativo de cada predicción: días de error por el costo de su dirección.

    `real - predicted > 0` significa que se subestimó (llega más tarde de lo dicho) y cuesta `under_cost`
    por día; si se sobrestimó, cuesta `over_cost` por día. Devuelve un arreglo con un costo por pedido.
    """
    error = np.asarray(real, dtype=float) - np.asarray(predicted, dtype=float)
    return np.where(error > 0, under_cost * error, -over_cost * error)


def mean_delivery_cost(real, predicted, ratio: float = COST_UNDERESTIMATE / COST_OVERESTIMATE) -> float:
    """Costo medio por pedido cuando subestimar cuesta `ratio` veces más que sobrestimar.

    Con `ratio=1` es el MAE. Por defecto usa la proporción del supuesto del grupo (3 a 1).
    """
    return float(delivery_cost(real, predicted, under_cost=ratio, over_cost=1.0).mean())


def mae_days(real, predicted) -> float:
    """Error absoluto medio, en días (métrica principal del proyecto)."""
    return float(np.abs(np.asarray(real, dtype=float) - np.asarray(predicted, dtype=float)).mean())


def best_fixed_deadline(real, ratio: float = COST_UNDERESTIMATE / COST_OVERESTIMATE) -> float:
    """Mejor plazo fijo (en días) para una proporción de costo: el percentil `ratio / (ratio + 1)`.

    Es la referencia a vencer: prometer siempre el mismo plazo a todos los pedidos.
    """
    return float(np.percentile(np.asarray(real, dtype=float), ratio / (ratio + 1) * 100))
