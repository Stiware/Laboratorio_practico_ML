"""Medición de ideas de variables: pliegues, mejora/ruido y regla de decisión.

Cada idea se compara con la referencia usando el mismo modelo, los mismos pliegues y las mismas métricas
(error absoluto medio en días y costo medio 3 a 1). La regla de decisión escrita en
`especificaciones/04-features.md` §2 vive aquí, en `decide_idea`, para que no dependa de leer cifras a ojo.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone

from olistbr.config import MIN_IMPROVEMENT_RATIO, TARGET
from olistbr.models.metrics import mae_days, mean_delivery_cost

NOISE_FLOOR = 1e-12  # con menos ruido que esto el cociente mejora / ruido no se calcula (se evita dividir por cero)


def _fmt(value: float) -> str:
    """Número con coma decimal y dos decimales (cuatro si es muy pequeño, para que no se lea como cero)."""
    digits = 4 if 0 < abs(value) < 0.01 else 2
    return f"{value:.{digits}f}".replace(".", ",")


def evaluate_folds(pipeline, X: pd.DataFrame, y: pd.Series, folds: list) -> pd.DataFrame:
    """Ajusta y evalúa el pipeline en cada pliegue; devuelve una fila por pliegue.

    El pipeline que se pasa no se modifica (se clona en cada pliegue) y se ajusta solo con el entrenamiento
    de ese pliegue. Columnas: `fold`, `mae` (días), `cost` (costo medio 3 a 1), `n_train`, `n_valid`.
    """
    rows = []
    for number, (train_idx, valid_idx) in enumerate(folds):
        model = clone(pipeline)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = model.predict(X.iloc[valid_idx])
        real = y.iloc[valid_idx]
        rows.append({
            "fold": number,
            "mae": mae_days(real, predicted),
            "cost": mean_delivery_cost(real, predicted),
            "n_train": len(train_idx),
            "n_valid": len(valid_idx),
        })
    return pd.DataFrame(rows)


def evaluate_holdout(pipeline, train: pd.DataFrame, valid: pd.DataFrame, target: str = TARGET) -> dict:
    """Ajusta con `train` y evalúa en `valid` (una sola partición, p. ej. la prueba hacia adelante).

    El pipeline que se pasa no se modifica. Devuelve `{"mae", "cost", "n_train", "n_valid"}`.
    """
    model = clone(pipeline)
    model.fit(train.drop(columns=[target]), train[target])
    predicted = model.predict(valid.drop(columns=[target]))
    return {
        "mae": mae_days(valid[target], predicted),
        "cost": mean_delivery_cost(valid[target], predicted),
        "n_train": len(train),
        "n_valid": len(valid),
    }


def _paired(reference: pd.Series, variant: pd.Series) -> tuple:
    """Mejora (media de referencia - idea), ruido (desviación ddof=1) y cociente; el cociente es NaN si no hay ruido."""
    difference = reference.to_numpy(dtype=float) - variant.to_numpy(dtype=float)
    gain = float(difference.mean())
    noise = float(difference.std(ddof=1)) if len(difference) > 1 else 0.0
    evaluable = noise > NOISE_FLOOR
    return gain, noise, (gain / noise if evaluable else float("nan")), bool(evaluable)


def compare_to_reference(reference: pd.DataFrame, variant: pd.DataFrame) -> dict:
    """Compara una idea con la referencia pliegue por pliegue (diferencias pareadas).

    Una diferencia positiva significa que la idea ayuda. Para el error en días (`mae`) y el costo 3 a 1
    (`cost`) devuelve `*_gain` (mejora media), `*_noise` (desviación de la mejora entre pliegues),
    `*_ratio` (mejora / ruido, NaN si no hay ruido) y `*_evaluable`.
    """
    result = {}
    for metric in ("mae", "cost"):
        gain, noise, ratio, evaluable = _paired(reference[metric], variant[metric])
        result.update({
            f"{metric}_gain": gain,
            f"{metric}_noise": noise,
            f"{metric}_ratio": ratio,
            f"{metric}_evaluable": evaluable,
        })
    return result


def decide_idea(
    cv: dict, forward: dict | None, known_at_prediction: bool, min_ratio: float = MIN_IMPROVEMENT_RATIO
) -> tuple:
    """Aplica la regla de decisión escrita y devuelve `("conservar" | "descartar", motivo)`.

    Se conserva solo si (1) se conoce al momento de predecir, (2) mejora / ruido supera `min_ratio` en el
    error en días y el costo 3 a 1 mejora, y (3) la mejora no cambia de signo hacia adelante.
    `cv` es el resultado de `compare_to_reference`; `forward` es `{"mae_gain": ...}` o `None` si la prueba
    hacia adelante no se puede evaluar (por ejemplo, la temporada alta de fin de año).
    """
    if not known_at_prediction:
        return "descartar", "no se conoce al momento de predecir"
    if not cv.get("mae_evaluable", False):
        return "descartar", "sin ruido entre pliegues: la mejora no es evaluable"
    ratio = cv["mae_ratio"]
    if ratio <= min_ratio:
        return "descartar", f"mejora / ruido = {_fmt(ratio)}, no supera {_fmt(min_ratio)}"
    if cv["cost_gain"] <= 0:
        return "descartar", f"mejora / ruido = {_fmt(ratio)}, pero el costo 3 a 1 no mejora"
    if forward is None:
        return "conservar", f"mejora / ruido = {_fmt(ratio)}; no evaluable hacia adelante"
    if forward["mae_gain"] <= 0:
        return "descartar", (
            f"mejora / ruido = {_fmt(ratio)}, pero empeora hacia adelante ({_fmt(forward['mae_gain'])} días)"
        )
    return "conservar", f"mejora / ruido = {_fmt(ratio)} y no empeora hacia adelante ({_fmt(forward['mae_gain'])} días)"


# --- Una fila de la tabla de ideas medidas (reports/04_features_tabla.csv) ---------------------------------
RESULT_COLUMNS = [
    "idea", "grupo",
    "mae_media", "mae_desv", "mae_mejora", "mae_ruido", "mae_mejora_ruido",
    "costo_media", "costo_mejora", "costo_mejora_ruido",
    "adelante_mejora", "adelante_mae", "decision", "motivo",
]


def measure_reference(pipeline, X: pd.DataFrame, y: pd.Series, folds: list, forward: tuple) -> dict:
    """Mide la referencia (variables base) con los pliegues y con la prueba hacia adelante.

    `forward` es el par `(train, valid)` de `forward_split`. Devuelve `{"folds": tabla por pliegue, "holdout": dict}`.
    """
    return {"folds": evaluate_folds(pipeline, X, y, folds), "holdout": evaluate_holdout(pipeline, *forward)}


def reference_row(name: str, reference: dict) -> dict:
    """Fila de la referencia: sus errores y sin comparación (no hay con qué compararla)."""
    folds = reference["folds"]
    nan = float("nan")
    return {
        "idea": name, "grupo": "(base)",
        "mae_media": float(folds["mae"].mean()), "mae_desv": float(folds["mae"].std(ddof=1)),
        "mae_mejora": nan, "mae_ruido": nan, "mae_mejora_ruido": nan,
        "costo_media": float(folds["cost"].mean()), "costo_mejora": nan, "costo_mejora_ruido": nan,
        "adelante_mejora": nan, "adelante_mae": reference["holdout"]["mae"],
        "decision": "referencia", "motivo": "variables base; punto de comparación",
    }


def measure_idea(
    name: str,
    group: str,
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list,
    reference: dict,
    forward: tuple,
    *,
    known_at_prediction: bool = True,
    forward_evaluable: bool = True,
    forward_check_column: str | None = None,
    kind: str = "idea",
) -> dict:
    """Mide una idea contra la referencia y devuelve una fila de la tabla (columnas de `RESULT_COLUMNS`).

    - `kind="idea"`: aplica la regla de decisión (`decide_idea`).
    - `kind="referencia_no_usable"`: se mide pero nunca se conserva (p. ej. el plazo prometido por Olist).
    - `kind="comprobacion"`: un conjunto de ideas medido junto; se mide, no se decide.
    - `kind="linea_base"`: regla trivial (predecir la mediana) para dar contexto; no se decide.
    - `forward_evaluable=False`: esa prueba hacia adelante se informa como no evaluable.
    - `forward_check_column`: nombre de una columna que la validación hacia adelante debe poder variar para
      que la prueba tenga sentido (p. ej. `is_year_end_peak`: si en la validación vale siempre 0, la marca
      no se puede evaluar hacia adelante). Se decide aquí, en `src/`, y no en el notebook.
    """
    scores = evaluate_folds(pipeline, X, y, folds)
    cv = compare_to_reference(reference["folds"], scores)

    if forward_evaluable and forward_check_column is not None:
        forward_evaluable = bool(forward[1][forward_check_column].nunique() > 1)

    forward_gain = None
    holdout = None
    if forward_evaluable:
        holdout = evaluate_holdout(pipeline, *forward)
        forward_gain = {"mae_gain": reference["holdout"]["mae"] - holdout["mae"]}

    if kind == "referencia_no_usable":
        decision, reason = "referencia", "no usable en producción: la app fija la fecha y no conoce el plazo prometido"
    elif kind == "comprobacion":
        decision, reason = "comprobación", "conjunto de ideas medido junto; no se decide"
    elif kind == "linea_base":
        decision, reason = "línea base", "regla trivial: predecir siempre la mediana del entrenamiento"
    else:
        decision, reason = decide_idea(cv, forward_gain, known_at_prediction)

    return {
        "idea": name, "grupo": group,
        "mae_media": float(scores["mae"].mean()), "mae_desv": float(scores["mae"].std(ddof=1)),
        "mae_mejora": cv["mae_gain"], "mae_ruido": cv["mae_noise"], "mae_mejora_ruido": cv["mae_ratio"],
        "costo_media": float(scores["cost"].mean()), "costo_mejora": cv["cost_gain"],
        "costo_mejora_ruido": cv["cost_ratio"],
        "adelante_mejora": forward_gain["mae_gain"] if forward_gain else float("nan"),
        "adelante_mae": holdout["mae"] if holdout else float("nan"),
        "decision": decision, "motivo": reason,
    }


def kept_groups(rows: list) -> list:
    """Grupos de ideas que la regla de decisión conservó (en el orden en que se midieron)."""
    return [row["grupo"] for row in rows if row["decision"] == "conservar"]


def results_table(rows: list) -> pd.DataFrame:
    """Tabla final de ideas medidas, con las columnas de `RESULT_COLUMNS` y sin filas repetidas."""
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if table["idea"].duplicated().any():
        raise ValueError(f"ideas repetidas en la tabla: {table.loc[table['idea'].duplicated(), 'idea'].tolist()}")
    return table
