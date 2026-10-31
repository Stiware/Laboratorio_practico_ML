"""Medición de ideas de variables: pliegues, mejora/ruido y regla de decisión.

Cada idea se compara con la referencia usando el mismo modelo, los mismos pliegues y las mismas métricas
(error absoluto medio en días y costo medio 3 a 1). La regla de decisión escrita en
`especificaciones/04-features.md` §2 vive aquí, en `decide_idea`, para que no dependa de leer cifras a ojo.
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from olistbr.config import GROUP_COL, MIN_IMPROVEMENT_RATIO, MIN_WIN_SHARE, TARGET, TOURNAMENT_GROUPS
from olistbr.models.metrics import mae_days, mean_delivery_cost, metrics_by_group
from olistbr.models.train import build_pipeline, fit_deadline_model, regression_metrics

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


# --- Etapa 5: evaluación de los candidatos del torneo --------------------------------------------------------------
# (`evaluate_folds` y `evaluate_holdout` de la etapa 4 no cambian: siguen sirviendo para medir ideas de variables.)

CANDIDATE_COLUMNS = [
    "modelo", "familia", "pliegue", "mae", "rmse", "r2", "sesgo", "costo", "tarde", "cobertura", "margen",
    "n_ajuste", "n_calibracion", "n_validacion", "fit_seconds",
]
METRIC_COLUMNS = ["mae", "rmse", "r2", "sesgo", "costo", "tarde", "cobertura", "margen"]


def _evaluate_one(name, candidate, X_train, y_train, X_valid, y_valid, groups_col, fold) -> dict:
    """Ajusta un candidato con un entrenamiento y mide el plazo recomendado en la validación (una fila de resultados)."""
    pipeline = build_pipeline(TOURNAMENT_GROUPS, model=clone(candidate.model), scale=candidate.scale)
    start = time.perf_counter()
    model = fit_deadline_model(pipeline, candidate.family, X_train, y_train, X_train[groups_col])
    seconds = time.perf_counter() - start
    predicted, recommended = model.predict_with_deadline(X_valid)
    metrics = regression_metrics(y_valid, predicted, recommended)
    return {
        "modelo": name, "familia": candidate.family, "pliegue": fold, **metrics, "margen": model.margin,
        "n_ajuste": model.n_fit, "n_calibracion": model.n_calibration, "n_validacion": len(X_valid), "fit_seconds": seconds,
    }


def evaluate_candidate_folds(name, candidate, X: pd.DataFrame, y: pd.Series, folds: list, groups_col: str = GROUP_COL) -> pd.DataFrame:
    """Evalúa un candidato en cada pliegue; devuelve una fila por pliegue con `CANDIDATE_COLUMNS`.

    Cada pliegue ajusta con su entrenamiento (la familia A aparta clientes de calibración dentro de él) y mide en su validación.
    El candidato que se pasa no se modifica (se clona en cada pliegue). `X` debe traer la columna de cliente (`groups_col`),
    que sirve para agrupar la calibración y nunca entra al modelo.
    """
    rows = [
        _evaluate_one(name, candidate, X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx], groups_col, fold)
        for fold, (train_idx, valid_idx) in enumerate(folds)
    ]
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def evaluate_candidate_holdout(name, candidate, train: pd.DataFrame, valid: pd.DataFrame, groups_col: str = GROUP_COL, target: str = TARGET) -> dict:
    """Lo mismo con una sola partición (la prueba hacia adelante): devuelve una fila con `pliegue = "adelante"`."""
    return _evaluate_one(
        name, candidate, train.drop(columns=[target]), train[target], valid.drop(columns=[target]), valid[target], groups_col, "adelante"
    )


def tournament_summary(by_fold: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    """Una fila por modelo: familia, media y desviación (`ddof=1`) entre pliegues de cada métrica y el resultado hacia adelante.

    El tiempo de ajuste no entra: cambia de una ejecución a otra y se guarda aparte.
    """
    grouped = by_fold.groupby("modelo", sort=False)
    summary = grouped.agg(familia=("familia", "first"))
    for metric in METRIC_COLUMNS:
        summary[f"{metric}_media"] = grouped[metric].mean()
        summary[f"{metric}_desv"] = grouped[metric].std(ddof=1)
    holdout = forward.set_index("modelo")
    summary["adelante_costo"] = holdout["costo"]
    summary["adelante_mae"] = holdout["mae"]
    return summary.reset_index()


# --- Etapa 5: comparación por pares, posiciones y líneas base --------------------------------------------------------


def compare_models(best_costs, other_costs) -> dict:
    """Compara el costo por pliegue de dos modelos (guía §2.4); `best_costs` es el del que se espera que gane.

    Por pliegue, `diferencia = costo(otro) − costo(mejor)` (positiva = el mejor gana). Devuelve la diferencia media (`diferencia_media`),
    en cuántos pliegues gana el mejor (`gana`, `n`, `texto` como «4 de 5»), el `ruido` (desviación `ddof=1` de las diferencias) y
    `ratio` = diferencia / ruido (NaN y `evaluable=False` si no hay ruido: nunca se divide por cero).
    `consistente` es verdadero si el mejor gana en al menos `MIN_WIN_SHARE` de los pliegues y diferencia / ruido supera 1;
    en otro caso es un **empate práctico**.
    """
    difference = np.asarray(other_costs, dtype=float) - np.asarray(best_costs, dtype=float)
    n = len(difference)
    wins = int((difference > 0).sum())
    noise = float(difference.std(ddof=1)) if n > 1 else 0.0
    evaluable = noise > NOISE_FLOOR
    ratio = float(difference.mean() / noise) if evaluable else float("nan")
    return {
        "diferencia_media": float(difference.mean()), "gana": wins, "n": n, "texto": f"{wins} de {n}",
        "ruido": noise, "ratio": ratio, "evaluable": bool(evaluable),
        "consistente": bool(evaluable and wins / n >= MIN_WIN_SHARE and ratio > MIN_IMPROVEMENT_RATIO),
    }


def _fold_costs(by_fold: pd.DataFrame, model: str) -> np.ndarray:
    """Costo por pliegue de un modelo, en el orden de los pliegues."""
    rows = by_fold[by_fold["modelo"] == model].sort_values("pliegue")
    if rows.empty:
        raise KeyError(f"no hay resultados del modelo {model!r} en la tabla por pliegue")
    return rows["costo"].to_numpy(dtype=float)


# Dirección de cada métrica al ordenar: "min" menor es mejor, "max" mayor es mejor, "abs" lo más cercano a cero.
RANK_DIRECTION = {"costo": "min", "mae": "min", "rmse": "min", "r2": "max", "sesgo": "abs", "tarde": "min", "margen": "min"}


def rank_table(summary: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Posiciones (1 = mejor) de cada modelo por métrica, a partir de las medias del resumen.

    Devuelve `modelo` y una columna `posicion_<métrica>` por cada métrica pedida. Sirve para ver si el ranking por costo
    coincide con el de exactitud (entre familias distintas no tienen por qué coincidir).
    """
    ranks = pd.DataFrame({"modelo": summary["modelo"].to_numpy()})
    for metric in metrics:
        values = summary[f"{metric}_media"].reset_index(drop=True)
        direction = RANK_DIRECTION[metric]
        key = values.abs() if direction == "abs" else (-values if direction == "max" else values)
        ranks[f"posicion_{metric}"] = key.rank(method="min").astype(int).to_numpy()
    return ranks


def beats_baselines(by_fold: pd.DataFrame, model: str, baselines: list) -> dict:
    """Para cada línea base, si el modelo la supera en costo por más que su ruido.

    Devuelve `{línea_base: {..., "supera": bool}}` con el resultado de `compare_models(modelo, línea_base)`:
    supera si la diferencia media es positiva y diferencia / ruido es mayor que 1.
    """
    model_costs = _fold_costs(by_fold, model)
    result = {}
    for baseline in baselines:
        comparison = compare_models(model_costs, _fold_costs(by_fold, baseline))
        comparison["supera"] = bool(comparison["evaluable"] and comparison["diferencia_media"] > 0 and comparison["ratio"] > MIN_IMPROVEMENT_RATIO)
        result[baseline] = comparison
    return result


def practical_ties(by_fold: pd.DataFrame, best: str) -> list:
    """Modelos que no se distinguen del mejor (el mejor no los supera de forma consistente). Ante un empate gana el más simple y rápido."""
    best_costs = _fold_costs(by_fold, best)
    ties = []
    for model in by_fold["modelo"].unique():
        if model != best and not compare_models(best_costs, _fold_costs(by_fold, model))["consistente"]:
            ties.append(str(model))
    return ties


def per_thousand(difference_per_order):
    """Pasa una diferencia por pedido a la diferencia por cada 1.000 pedidos (sin asignarle ningún valor en dinero)."""
    return np.asarray(difference_per_order, dtype=float)[()] * 1000


def tournament_table(by_fold: pd.DataFrame, forward: pd.DataFrame, baselines: dict, rank_metrics=("costo", "mae", "rmse", "r2", "sesgo")) -> pd.DataFrame:
    """La tabla final del torneo: una fila por modelo, ordenada por costo 3 a 1 (de menor a mayor).

    Trae el resumen (media ± desviación y partición hacia adelante), las posiciones por métrica, si supera a cada línea base
    (`supera_<clave>`), si es el `mejor` (el de menor costo **entre los modelos**, sin contar las líneas base), cómo le va contra
    el mejor (`gana_vs_mejor`, `ratio_vs_mejor`, `consistente_vs_mejor`) y si es un `empate_con_el_mejor`.
    `baselines` es `{clave: nombre}`, por ejemplo `{"mediana": "Línea base: siempre la mediana", ...}`.
    """
    summary = tournament_summary(by_fold, forward)
    table = summary.merge(rank_table(summary, list(rank_metrics)), on="modelo")
    baseline_names = set(baselines.values())
    models = table[~table["modelo"].isin(baseline_names)]
    best = str(models.sort_values("costo_media").iloc[0]["modelo"])
    best_costs = _fold_costs(by_fold, best)
    ties = set(practical_ties(by_fold, best))

    table["mejor"] = table["modelo"] == best
    for key, baseline in baselines.items():
        table[f"supera_{key}"] = [
            False if name in baseline_names else beats_baselines(by_fold, name, [baseline])[baseline]["supera"] for name in table["modelo"]
        ]
    wins, ratios, consistent = [], [], []
    for name in table["modelo"]:
        if name == best:
            wins.append("—"), ratios.append(float("nan")), consistent.append(False)
            continue
        comparison = compare_models(best_costs, _fold_costs(by_fold, name))
        wins.append(comparison["texto"]), ratios.append(comparison["ratio"]), consistent.append(comparison["consistente"])
    table["gana_vs_mejor"], table["ratio_vs_mejor"], table["consistente_vs_mejor"] = wins, ratios, consistent
    table["empate_con_el_mejor"] = table["modelo"].isin(ties)
    return table.sort_values("costo_media").reset_index(drop=True)


def regional_metrics(
    name, candidate, X: pd.DataFrame, y: pd.Series, folds: list, groups_col: str = GROUP_COL, region_col: str = "customer_region"
) -> pd.DataFrame:
    """MAE y porcentaje tarde por región del cliente, con predicciones fuera de muestra (cada pedido se predice una vez).

    Para cada pliegue ajusta con su entrenamiento y predice su validación; junta todo y agrupa por región con `metrics_by_group`.
    Sirve para ver dónde rinde peor un finalista (principio VIII). Columnas: `modelo`, `grupo`, `n_pedidos`, `mae`, `tarde`.
    """
    if region_col not in X.columns:
        raise ValueError(f"`X` no trae la columna de región {region_col!r}: aplicar `add_features` antes")
    real, predicted, recommended, regions = [], [], [], []
    for train_idx, valid_idx in folds:
        pipeline = build_pipeline(TOURNAMENT_GROUPS, model=clone(candidate.model), scale=candidate.scale)
        model = fit_deadline_model(pipeline, candidate.family, X.iloc[train_idx], y.iloc[train_idx], X.iloc[train_idx][groups_col])
        valid = X.iloc[valid_idx]
        prediction, deadline = model.predict_with_deadline(valid)
        real.append(y.iloc[valid_idx].to_numpy()), predicted.append(prediction), recommended.append(deadline)
        regions.append(valid[region_col].to_numpy(dtype=object))
    table = metrics_by_group(np.concatenate(real), np.concatenate(predicted), np.concatenate(recommended), np.concatenate(regions))
    table.insert(0, "modelo", name)
    return table


def coverage_by_month(
    name, candidate, train: pd.DataFrame, valid: pd.DataFrame, groups_col: str = GROUP_COL, target: str = TARGET,
    date_col: str = "order_purchase_timestamp",
) -> pd.DataFrame:
    """Cobertura del plazo recomendado, mes a mes de compra, en una partición (p. ej. la prueba hacia adelante).

    Ajusta con `train` y mide en `valid`. Sirve para ver si la calibración se mantiene en el tiempo: un plazo aprendido del pasado
    puede quedar corto en unos meses y largo en otros cuando cambia el ritmo de entrega. Columnas: `modelo`, `mes`, `n_pedidos`,
    `cobertura` (proporción con `real <= recomendado`), `mediana_dias` y `p75_dias` (de los días reales de ese mes).
    """
    pipeline = build_pipeline(TOURNAMENT_GROUPS, model=clone(candidate.model), scale=candidate.scale)
    X_train = train.drop(columns=[target])
    model = fit_deadline_model(pipeline, candidate.family, X_train, train[target], X_train[groups_col])
    _, recommended = model.predict_with_deadline(valid.drop(columns=[target]))
    frame = pd.DataFrame({
        "mes": valid[date_col].dt.to_period("M").astype(str).to_numpy(),
        "real": valid[target].to_numpy(dtype=float),
        "cubierto": (valid[target].to_numpy(dtype=float) <= recommended).astype(float),
    })
    monthly = frame.groupby("mes").agg(
        n_pedidos=("real", "size"), cobertura=("cubierto", "mean"), mediana_dias=("real", "median"), p75_dias=("real", lambda s: s.quantile(0.75))
    ).reset_index()
    monthly.insert(0, "modelo", name)
    return monthly
