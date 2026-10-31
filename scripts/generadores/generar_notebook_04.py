import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
code('''import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 04 · Variables — cada idea es una hipótesis que se mide

Especificación: `especificaciones/04-features.md` · Plan: `specs/002-features-engineering/plan.md`.
Este notebook **solo orquesta**: llama a funciones de `src/olistbr/` y muestra resultados.

**Cómo se decide.** Cada idea se compara con la referencia (las variables base) usando el **mismo modelo**
(regresión lineal), los **mismos 5 pliegues por cliente** y las mismas métricas: error en días (principal) y
costo medio 3 a 1 (control). `mejora / ruido` = mejora media dividida por su desviación entre pliegues.
Una idea se **conserva** solo si se conoce al predecir, `mejora / ruido > 1`, el costo también mejora y la
mejora no cambia de signo en la **prueba hacia adelante** (entrenar hasta 2018-02-28, validar de 2018-03-01 a 2018-08-31).

**Reglas de esta etapa:** solo se usa el conjunto de entrenamiento (la prueba final queda cerrada); el plazo
prometido por Olist se mide una vez como referencia y nunca entra al modelo final (la app fija la fecha).'''),

code('''import pandas as pd

from olistbr.config import (
    FORWARD_TRAIN_END, FORWARD_VALID_END, FORWARD_VALID_START, MIN_IMPROVEMENT_RATIO, N_FOLDS,
    PROCESSED_DIR, RANDOM_STATE, REPORTS_DIR, TARGET,
)
from olistbr.data.split import assert_disjoint_groups, forward_split, make_group_folds
from olistbr.features.engineering import add_features
from olistbr.models.evaluation import kept_groups, measure_idea, measure_reference, reference_row, results_table
from olistbr.models.train import build_baseline_pipeline, build_pipeline

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 90)

# Una fila por idea: nombre -> grupo de variables (src/olistbr/features/engineering.py, GROUPS)
IDEAS = {
    "F1 logaritmos": "logs",
    "F2 mismo estado": "same_state",
    "F3 regiones": "regions",
    "F5 día de la semana": "calendar",
    "F6 temporada alta de fin de año": "peak_season",
    "F7 distancia × mismo estado": "distance_interaction",
    "F8 historial vendedor / categoría / ruta": "history",
    "F9 razones de flete y densidad": "ratios",
    "F10 varios vendedores": "multi_seller",
}
# Columna que debe variar en la validación hacia adelante para que esa prueba tenga sentido (si no, "no evaluable")
FORWARD_CHECK = {"peak_season": "is_year_end_peak"}
print(f"{N_FOLDS} pliegues por cliente · semilla {RANDOM_STATE} · umbral mejora / ruido > {MIN_IMPROVEMENT_RATIO}")'''),

md('''## 1. Datos y variables

Se lee `train.parquet` y se calculan **todas** las variables derivadas una sola vez. `add_features` no aprende de
los datos (cada fila usa sus propios valores y tablas fijas), por eso no filtra información.'''),

code('''train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
df = add_features(train)
X, y = df.drop(columns=[TARGET]), df[TARGET]
print(f"entrenamiento: {len(df):,} pedidos · {df['customer_unique_id'].nunique():,} clientes · {df.shape[1]} columnas con las derivadas")
print(f"fechas de compra: {df['order_purchase_timestamp'].min():%Y-%m-%d} a {df['order_purchase_timestamp'].max():%Y-%m-%d}")'''),

md('''## 2. Pliegues por cliente (una sola vez) y prueba hacia adelante

Los pliegues se crean **una vez** y se reutilizan en todas las comparaciones. Se verifica que ningún cliente
quede en los dos lados de un pliegue y que los 5 pliegues cubran todas las filas.'''),

code('''folds = make_group_folds(df)

assert sorted(i for _, valid_idx in folds for i in valid_idx) == list(range(len(df))), "los pliegues no cubren cada fila una vez"
for train_idx, valid_idx in folds:
    assert_disjoint_groups(df.iloc[train_idx], df.iloc[valid_idx])  # falla si un cliente está en los dos lados

forward = forward_split(df)
fwd_train, fwd_valid = forward
assert fwd_train["order_purchase_timestamp"].max() < pd.Timestamp(FORWARD_TRAIN_END) + pd.Timedelta(days=1)
assert fwd_valid["order_purchase_timestamp"].min() >= pd.Timestamp(FORWARD_VALID_START)
assert fwd_valid["order_purchase_timestamp"].max() < pd.Timestamp(FORWARD_VALID_END) + pd.Timedelta(days=1)

print("tamaño de la validación por pliegue:", [len(v) for _, v in folds])
print(f"hacia adelante: entrenamiento {len(fwd_train):,} pedidos (hasta {FORWARD_TRAIN_END}) · validación {len(fwd_valid):,} pedidos ({FORWARD_VALID_START} a {FORWARD_VALID_END})")
print("la validación hacia adelante contiene la temporada alta de fin de año:", bool(fwd_valid["is_year_end_peak"].nunique() > 1))'''),

md('''## 3. Referencia

Variables base (`price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`,
`missing_coords`, categoría, vendedor y estados) con la regresión lineal.'''),

code('''reference = measure_reference(build_pipeline(), X, y, folds, forward)
reference["folds"].round(3)'''),

md('''## 4. Medir cada idea

Cada grupo de ideas se agrega **solo** a las variables base y se mide con los mismos pliegues. Si la validación hacia
adelante no contiene el fenómeno (la marca de fin de año vale siempre 0 de marzo a agosto de 2018), `measure_idea` informa
esa prueba como **no evaluable** y no la usa para descartar.'''),

code('''rows = [reference_row("Referencia (variables base)", reference)]
for name, group in IDEAS.items():
    rows.append(measure_idea(
        name, group, build_pipeline([group]), X, y, folds, reference, forward,
        forward_check_column=FORWARD_CHECK.get(group),
    ))
len(rows)'''),

md('''## 5. El plazo prometido por Olist: solo referencia

Se mide **una vez** para saber cuánto se pierde sin él. No es usable en producción (la app recomienda la fecha) y no entra
a ningún grupo de variables del modelo final.'''),

code('''rows.append(measure_idea(
    "F4 plazo prometido por Olist (referencia)", "(referencia)", build_pipeline(include_reference=True),
    X, y, folds, reference, forward, kind="referencia_no_usable",
))
rows.append(measure_idea(
    "Línea base trivial (siempre la mediana)", "(trivial)", build_baseline_pipeline(),
    X, y, folds, reference, forward, kind="linea_base",
))
len(rows)'''),

md('''## 6. Conjunto conservado y sensibilidad (comprobaciones)

Se miden juntas las ideas que la regla conservó. No se decide con estas filas: sirven para ver si juntas se estorban
(por ejemplo, ideas muy parecidas como los logaritmos y la distancia × mismo estado). Como la marca de fin de año descansa
en una sola temporada (2017) y no se puede validar hacia adelante, se mide también el conjunto **sin ella**, para saber
cuánto de la mejora total depende de esa marca.'''),

code('''kept = kept_groups(rows)
print("grupos conservados:", kept)
rows.append(measure_idea(
    "Conjunto conservado", " + ".join(kept) if kept else "(ninguno)", build_pipeline(kept), X, y, folds, reference, forward,
    kind="comprobacion",
))
kept_without_peak = [group for group in kept if group != "peak_season"]
rows.append(measure_idea(
    "Conjunto conservado sin la marca de fin de año (sensibilidad)", " + ".join(kept_without_peak) if kept_without_peak else "(ninguno)",
    build_pipeline(kept_without_peak), X, y, folds, reference, forward, kind="comprobacion",
))
table = results_table(rows)
print(len(table), "filas en la tabla")'''),

md('''## 7. Tabla de ideas medidas

`mae_*` es el error absoluto medio en días; `costo_*` el costo medio 3 a 1. `mejora` = referencia − idea (positiva = ayuda).
`adelante_mejora` es la mejora del error en la prueba hacia adelante y `adelante_mae` el error absoluto de esa prueba.'''),

code('''summary = table[["idea", "mae_media", "mae_desv", "mae_mejora", "mae_mejora_ruido", "costo_mejora", "adelante_mejora", "adelante_mae", "decision"]]
summary.round(3)'''),

code('''table[["idea", "decision", "motivo"]]'''),

md('''## 8. Guardar la tabla

`reports/04_features_tabla.csv` es la evidencia; el texto con el motivo de cada decisión está en `reports/04_features.md`.'''),

code('''REPORTS_DIR.mkdir(parents=True, exist_ok=True)
table.round(6).to_csv(REPORTS_DIR / "04_features_tabla.csv", index=False)
print("guardado reports/04_features_tabla.csv con", len(table), "filas")'''),

md('''## 9. Las pruebas automáticas

Desde la raíz del repo:

```bash
pytest tests
```

Cada variable tiene su prueba en `tests/test_features.py`; los pliegues y la prueba hacia adelante en `tests/test_split.py`;
la medición y la regla de decisión en `tests/test_evaluation.py`; el pipeline en `tests/test_models.py`.'''),
]
nbf.write(nb, "notebooks/04_features.ipynb")
print("ok", len(nb.cells))
