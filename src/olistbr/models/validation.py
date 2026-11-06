"""Validación final del modelo (etapa 7, `especificaciones/07-validacion.md`).

La prueba (`test.parquet`) se abre **una sola vez**: `open_test_once` deja constancia de cuándo y se niega a repetir. Todo lo que se
decide (modelo, parámetros, baseline, segmentos) queda escrito **antes**. Después solo se mide:

- el costo 3 a 1, el % de pedidos tarde y la cobertura del plazo del modelo y de tres reglas simples (baselines), cada uno con un
  intervalo del 95 % que **sortea clientes completos** (los pedidos de un mismo cliente no son independientes);
- los segmentos (región y mes), los días de holgura y de atraso, la sensibilidad del costo y qué variables mueven el plazo (SHAP).

OnTrack es regresión: no hay probabilidades que calibrar. El equivalente es la **cobertura** (¿el plazo cubre ≈ 75 % de los pedidos?).
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from olistbr.config import BOOTSTRAP_ROUNDS, COST_RATIOS, RANDOM_STATE, TARGET
from olistbr.models.metrics import bias, delivery_cost, mae_days, mean_delivery_cost
from olistbr.models.tuning import bootstrap_cost_difference

_BOOTSTRAP_CHUNK = 100  # sorteos por bloque, para no ocupar demasiada memoria con ≈ 19.000 clientes


# --- La prueba, una sola vez ---------------------------------------------------------------------------


def open_test_once(path, mark_path) -> pd.DataFrame:
    """Lee la prueba y deja constancia (fecha y hora) en `mark_path`. Si la constancia ya existe, **se niega** a abrirla otra vez.

    Abrir la prueba más de una vez para «mejorar» el modelo la convierte en un segundo conjunto de entrenamiento (guía §3).
    La constancia se escribe después de leer, para que un error de lectura no deje una apertura falsa.
    """
    mark_path = Path(mark_path)
    if mark_path.exists():
        opened = mark_path.read_text(encoding="utf-8").strip()
        raise RuntimeError(f"la prueba ya se abrió el {opened} (constancia en {mark_path}): no se vuelve a abrir")
    table = pd.read_parquet(path)
    mark_path.parent.mkdir(parents=True, exist_ok=True)
    mark_path.write_text(datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8")
    return table


# --- Intervalos por cliente -------------------------------------------------------------------------------


def bootstrap_mean(values, groups, n_boot: int = BOOTSTRAP_ROUNDS, seed: int = RANDOM_STATE) -> tuple:
    """Media por pedido de `values` con su intervalo del 95 %, **sorteando clientes completos**.

    Suma y número de pedidos por cliente; `n_boot` sorteos de clientes con reemplazo; en cada uno, suma de sumas / suma de pedidos.
    Devuelve `(media, percentil 2,5, percentil 97,5)`. Con un indicador 0/1 (por ejemplo, «llegó tarde») da una proporción.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if len(values) != len(groups):
        raise ValueError(f"los datos tienen distinto largo: valores={len(values)}, clientes={len(groups)}")
    if len(values) == 0:
        raise ValueError("no hay pedidos: las entradas están vacías")
    if not np.isfinite(values).all():
        raise ValueError("hay valores vacíos o infinitos (no finitos)")
    if pd.isna(groups).any():
        raise ValueError("hay pedidos sin cliente: no se pueden sortear clientes completos")
    if n_boot < 1:
        raise ValueError(f"n_boot debe ser al menos 1 (se recibió {n_boot})")
    per_customer = pd.DataFrame({"g": groups, "v": values}).groupby("g", sort=True)["v"].agg(["sum", "count"])
    sums, counts = per_customer["sum"].to_numpy(), per_customer["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_boot, _BOOTSTRAP_CHUNK):
        idx = rng.integers(0, len(sums), size=(min(_BOOTSTRAP_CHUNK, n_boot - start), len(sums)))
        draws.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    draws = np.concatenate(draws)
    return float(sums.sum() / counts.sum()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --- Baselines y tabla de la prueba -----------------------------------------------------------------------

MAIN_BASELINE = "Plazo fijo (percentil 75)"


def baseline_plans(train: pd.DataFrame, test: pd.DataFrame, target: str = TARGET,
                   purchase_col: str = "order_purchase_timestamp", promised_col: str = "order_estimated_delivery_date") -> dict:
    """Plazos de las reglas simples para cada pedido de `test` (Q3 de la etapa 7).

    - **Plazo fijo (percentil 75)** del entrenamiento: la «referencia a vencer» del caso de negocio (baseline principal).
    - **Siempre la mediana** del entrenamiento.
    - **Plazo prometido por Olist** (fecha estimada − fecha de compra, en días): solo referencia; nunca entra al modelo.
    """
    if target not in train.columns:
        raise ValueError(f"al entrenamiento le falta la columna {target!r}")
    for column in (purchase_col, promised_col):
        if column not in test.columns:
            raise ValueError(f"a la tabla de prueba le falta la columna {column!r}")
    if test[[purchase_col, promised_col]].isna().any().any():
        raise ValueError(f"hay fechas vacías en {purchase_col!r} o {promised_col!r}: no se puede calcular el plazo prometido")
    n = len(test)
    promised = (test[promised_col] - test[purchase_col]).dt.total_seconds().to_numpy() / 86400
    return {
        MAIN_BASELINE: np.full(n, float(np.percentile(train[target], 75))),
        "Siempre la mediana": np.full(n, float(np.median(train[target]))),
        "Plazo prometido por Olist": promised,
    }


def test_report(real, plans: dict, groups, main_baseline: str, model_name: str, n_boot: int = BOOTSTRAP_ROUNDS,
                seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Una fila por regla con costo 3 a 1, % tarde y cobertura (cada uno con su intervalo por cliente), MAE y sesgo del plazo.

    Para el modelo agrega la diferencia `costo(modelo) − costo(main_baseline)` con su intervalo y `mejora_con_confianza`
    (todo el intervalo bajo cero). El MAE y el sesgo son del **plazo** (percentil 75), no del valor típico.
    """
    for name in (model_name, main_baseline):
        if name not in plans:
            raise KeyError(f"falta la regla {name!r} en `plans`")
    real = np.asarray(real, dtype=float)
    rows = []
    for name, plan in plans.items():
        plan = np.asarray(plan, dtype=float)
        cost = bootstrap_mean(delivery_cost(real, plan), groups, n_boot, seed)
        late = bootstrap_mean((real > plan).astype(float), groups, n_boot, seed)
        row = {
            "regla": name, "costo": cost[0], "costo_ic_inf": cost[1], "costo_ic_sup": cost[2],
            "tarde": late[0], "tarde_ic_inf": late[1], "tarde_ic_sup": late[2],
            "cobertura": 1 - late[0], "cobertura_ic_inf": 1 - late[2], "cobertura_ic_sup": 1 - late[1],
            "mae": mae_days(real, plan), "sesgo": bias(real, plan),
            "dif_costo": np.nan, "dif_ic_inf": np.nan, "dif_ic_sup": np.nan, "mejora_con_confianza": np.nan,
        }
        if name == model_name:
            diff, low, high = bootstrap_cost_difference(real, plan, plans[main_baseline], groups, n_boot=n_boot, seed=seed)
            row.update({"dif_costo": diff, "dif_ic_inf": low, "dif_ic_sup": high, "mejora_con_confianza": bool(high < 0)})
        rows.append(row)
    return pd.DataFrame(rows)


# --- Holgura, atraso, segmentos y sensibilidad ----------------------------------------------------------


def slack_and_delay(real, plan) -> dict:
    """Días que sobran cuando el pedido llega a tiempo (holgura) y días que faltan cuando llega tarde (atraso), con el % de cada caso.

    Reemplaza la «carga de trabajo» de la guía, que no aplica: aquí el costo de un plazo largo es prometer días de más.
    """
    real, plan = np.asarray(real, dtype=float), np.asarray(plan, dtype=float)
    if len(real) != len(plan) or len(real) == 0:
        raise ValueError("los días reales y los plazos deben tener el mismo largo y no estar vacíos")
    if not (np.isfinite(real).all() and np.isfinite(plan).all()):
        raise ValueError("hay días reales o plazos vacíos o infinitos (no finitos)")
    on_time = real <= plan
    return {
        "pct_a_tiempo": float(on_time.mean()),
        "pct_tarde": float((~on_time).mean()),
        "holgura_media": float((plan - real)[on_time].mean()) if on_time.any() else np.nan,
        "atraso_medio": float((real - plan)[~on_time].mean()) if (~on_time).any() else np.nan,
    }


def segment_table(real, plan, segments: dict) -> pd.DataFrame:
    """Por cada segmento (`{"region": ..., "mes": ...}`) y cada grupo: pedidos, % tarde, cobertura, MAE, holgura y atraso.

    Las etiquetas vacías se agrupan como «sin dato».
    """
    real, plan = np.asarray(real, dtype=float), np.asarray(plan, dtype=float)
    rows = []
    for segment, labels in segments.items():
        labels = np.asarray(labels, dtype=object)
        if len(labels) != len(real):
            raise ValueError(f"el segmento {segment!r} tiene otro largo ({len(labels)}) que los pedidos ({len(real)})")
        labels = np.where(pd.isna(labels), "sin dato", labels).astype(str)
        for group in sorted(pd.unique(labels)):
            mask = labels == group
            sd = slack_and_delay(real[mask], plan[mask])
            rows.append({
                "segmento": segment, "grupo": group, "n_pedidos": int(mask.sum()), "tarde": sd["pct_tarde"],
                "cobertura": sd["pct_a_tiempo"], "mae": mae_days(real[mask], plan[mask]),
                "holgura_media": sd["holgura_media"], "atraso_medio": sd["atraso_medio"],
            })
    return pd.DataFrame(rows, columns=["segmento", "grupo", "n_pedidos", "tarde", "cobertura", "mae", "holgura_media", "atraso_medio"])


def cost_sensitivity(real, plans: dict, ratios=COST_RATIOS) -> pd.DataFrame:
    """Costo medio de cada regla con otras proporciones subestimar : sobrestimar (1, 2, 3, 5 y 9 a 1; con 1 a 1 es el MAE).

    El plazo del modelo es siempre el percentil 75: no se reoptimiza para cada proporción.
    """
    rows = [{"regla": name, **{f"{r} a 1": mean_delivery_cost(real, plan, ratio=r) for r in ratios}} for name, plan in plans.items()]
    return pd.DataFrame(rows)


# --- Interpretación ---------------------------------------------------------------------------------------


def top_shap(pipeline, X: pd.DataFrame, n: int = 10, sample: int = 2000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Las `n` variables que más mueven el plazo según SHAP, con el valor absoluto medio y el valor medio con signo.

    Usa una muestra de `sample` filas de `X`, las transforma con el paso `prep` y explica el paso `model` con `shap.TreeExplainer`
    (salida cruda del modelo: el percentil 75 en días). **Límites:** explica al modelo, no al mundo; las variables correlacionadas
    se reparten el crédito; importancia no es causa.
    """
    import shap

    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as error:  # SHAP lanza su propio error con modelos que no son de árboles
        raise TypeError(f"top_shap solo sirve para modelos de árboles; se recibió {type(model).__name__}") from error
    rows = X.sample(n=min(sample, len(X)), random_state=seed)
    transformed = np.asarray(prep.transform(rows), dtype=float)
    values = np.asarray(explainer.shap_values(transformed))
    names = [name.split("__", 1)[-1] for name in prep.get_feature_names_out()]
    table = pd.DataFrame({"variable": names, "shap_abs_medio": np.abs(values).mean(axis=0), "shap_medio": values.mean(axis=0)})
    return table.sort_values("shap_abs_medio", ascending=False).head(n).reset_index(drop=True)
