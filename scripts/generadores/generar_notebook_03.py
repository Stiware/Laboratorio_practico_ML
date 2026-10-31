import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
code('''import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
# Permite importar el paquete de src/ sin instalarlo
sys.path.insert(0, str(Path.cwd().resolve().parent / "src"))'''),

md('''# 03 · Preprocesamiento — de las decisiones del EDA a código con pruebas

Cada decisión de la tabla del EDA (`especificaciones/02-eda.md`) es una **función pequeña** en
`src/olistbr/data/` con su prueba en `tests/`. Este notebook **solo orquesta**: llama a las funciones,
muestra cuántas filas quedan después de cada regla y guarda los datos. No tiene lógica reutilizable.

Reglas aprobadas: `especificaciones/03-preprocesamiento.md` · Plan: `specs/001-preprocesamiento-datos/plan.md`.

**Qué produce:** `data/processed/train.parquet`, `test.parquet` (cerrado hasta la etapa 7) y `extreme_orders.parquet`.'''),

code('''import pandas as pd

from olistbr.config import (
    GROUP_COL, ID_COL, MAX_DELIVERY_DAYS, MIN_SELLER_ITEMS, PERIOD_END, PERIOD_START,
    PROCESSED_DIR, RANDOM_STATE, RAW_DIR, TARGET, TEST_SIZE,
)
from olistbr.data.cleaning import build_orders_table, load_raw
from olistbr.data.split import assert_disjoint_groups, prepare_train_test
from olistbr.features.geo import zip_centroids

# RAW_DIR solo se lee: NUNCA se modifica. Una fila = un pedido (ID_COL); se parte por la persona (GROUP_COL).
print(f"Periodo {PERIOD_START} a {PERIOD_END} · extremos > {MAX_DELIVERY_DAYS} días · prueba {TEST_SIZE:.0%} · semilla {RANDOM_STATE}")'''),

md('''## 1. Construir la tabla de pedidos (reglas 1 a 7 del EDA)

`build_orders_table` carga los CSV **sin reseñas ni textos libres** (decisión 10), construye `dias_entrega` (1),
filtra el periodo (9), separa los extremos (2), agrega los ítems por pedido (3), limpia productos (7),
calcula la distancia (6) y quita lo que se conoce después de comprar (4). La tabla de abajo es la constancia
de **cuántas filas quedan después de cada regla**.'''),

code('''report = []
orders, extreme = build_orders_table(report=report)
steps = pd.DataFrame(report)
steps["filas_quitadas"] = steps["filas_antes"] - steps["filas_despues"]
steps'''),

md('''## 2. Comprobaciones de la limpieza

Cada decisión se convierte en una afirmación que **falla** si algo no se cumple.'''),

code('''checks = {
    "una fila por pedido (order_id único)": orders[ID_COL].is_unique,
    "dias_entrega entre 0 y 100": orders[TARGET].between(0, MAX_DELIVERY_DAYS).all(),
    "compras dentro del periodo": orders["order_purchase_timestamp"].between(PERIOD_START, f"{PERIOD_END} 23:59:59").all(),
    "los extremos son todos > 100 días": (extreme[TARGET] > MAX_DELIVERY_DAYS).all(),
    "extremos y entrenamiento tienen las mismas columnas": list(extreme.columns) == list(orders.columns),
    "sin columnas de entrega, estado ni reseñas": not (
        {"order_status", "order_delivered_customer_date", "order_delivered_carrier_date"} & set(orders.columns)
        or any(c.startswith("review_") for c in orders.columns)
    ),
    "sin ciudad, código postal ni customer_id": not (
        {"customer_city", "customer_zip_code_prefix", "seller_zip_code_prefix", "seller_city", "customer_id"} & set(orders.columns)
    ),
    "hay distancia o indicador de que falta": (orders["distance_km"].isna() == orders["missing_coords"].astype(bool)).all(),
}
result = pd.Series(checks, name="se cumple")
assert result.all(), result[~result]
result'''),

md('''### Vacíos que quedan para la etapa 4

No se imputan aquí (la mediana depende del entrenamiento y va dentro del `Pipeline`).'''),

code('''pd.DataFrame({
    "vacíos": orders.isna().sum()[lambda s: s > 0],
    "qué significa": {
        "order_approved_at": "sin fecha de aprobación (la causa no se verificó)",
        "weight_g_sum": "algún producto sin peso válido (0 o vacío)",
        "volume_cm3_sum": "algún producto sin dimensiones",
        "distance_km": "falta el punto de algún código postal (missing_coords = 1)",
    },
})'''),

md('''## 3. Cómo quedaron las coordenadas (decisión 6)

Cada código postal toma el promedio de sus puntos buenos; si no tiene, la mediana de su ciudad y estado;
si tampoco, queda sin punto.'''),

code('''centroids = zip_centroids(load_raw(RAW_DIR)["geolocation"])
coverage = centroids["source"].value_counts().rename({
    "zip": "promedio de sus propios puntos buenos",
    "city_state": "mediana de la misma ciudad y estado",
    "none": "sin punto",
}).to_frame("códigos postales")
print(f"Pedidos con missing_coords = 1: {int(orders['missing_coords'].sum()):,} de {len(orders):,}")
coverage'''),

md('''## 4. Partición por cliente y vendedores raros (decisiones 5 y 8)

Una persona puede tener varios pedidos (hasta 15 en la tabla final). **Todos los pedidos de un cliente van al mismo lado.**
Después, la lista de vendedores con al menos 10 ítems se calcula **solo con entrenamiento** y se aplica igual a la prueba.
Ese orden (partir primero, aprender después) está dentro de `prepare_train_test`, en `src/`, con su propia prueba,
para que nadie pueda invertirlo desde el notebook. La prueba queda cerrada: no se usa para decidir nada hasta la etapa 7.'''),

code('''train, test, frequent_sellers = prepare_train_test(
    orders, group_col=GROUP_COL, test_size=TEST_SIZE, seed=RANDOM_STATE, min_seller_items=MIN_SELLER_ITEMS
)

assert_disjoint_groups(train, test)  # falla si algún cliente está en los dos lados
assert len(train) + len(test) == len(orders), "se perdieron filas"
print(f"entrenamiento: {len(train):,} pedidos de {train[GROUP_COL].nunique():,} clientes")
print(f"prueba (cerrada): {len(test):,} pedidos ({len(test) / len(orders):.1%} de las filas)")
print(f"vendedores frecuentes (>= {MIN_SELLER_ITEMS} ítems en entrenamiento): {len(frequent_sellers):,}")
print(f"pedidos de entrenamiento con vendedor 'otros': {(train['seller_id'] == 'otros').mean():.1%}")'''),

md('''**Mi unidad de análisis y por qué parto así:** una fila es **un pedido entregado**. La persona
(`customer_unique_id`) se repite en varias filas (en la tabla final, 2.789 clientes compraron más de una vez, hasta 15 veces),
así que se parte **por cliente** con `GroupShuffleSplit` (20 % de los clientes a prueba, semilla 42). Partir por
fila dejaría al mismo cliente en los dos lados y la métrica saldría mejor de lo que será en producción.
La revisión por fecha (últimos meses) se hace en la etapa 7 sobre este mismo conjunto de prueba.

## 5. Guardar los datos de la etapa

Se guardan en `data/processed/` en Parquet (conserva los tipos y se lee directo en Databricks). No se versionan:
se regeneran con este notebook. **No se vuelve a leer `test`** hasta la validación final.'''),

code('''PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
extreme.to_parquet(PROCESSED_DIR / "extreme_orders.parquet", index=False)
for name in ["train", "test", "extreme_orders"]:
    size_mb = (PROCESSED_DIR / f"{name}.parquet").stat().st_size / 1e6
    print(f"guardado data/processed/{name}.parquet ({size_mb:.1f} MB)")'''),

md('''## 6. Las pruebas automáticas

Desde la raíz del repo:

```bash
pytest tests
```

Hay una prueba por cada regla (`tests/test_cleaning.py` y `tests/test_features.py`), la de la partición
(`tests/test_split.py`, que incluye el contraste «si `split_by_group` partiera por fila, la verificación fallaría» y la prueba de `prepare_train_test`) y la de datos
personales (`test_no_personal_columns` en `tests/test_cleaning.py`).

## Entregable de esta etapa (hito H1)

- [x] La limpieza corre de principio a fin desde el repo, sin pasos manuales.
- [x] La partición respeta la unidad de análisis y está verificada con una prueba.
- [x] `pytest tests` en verde (62 pruebas al 2026-10-31).
- [x] `data/raw/` no cambió (`git status`: está ignorado y sin cambios).
- [x] Unidad de análisis declarada por escrito.'''),
]
nbf.write(nb, "notebooks/03_preprocesamiento.ipynb")
print("ok", len(nb.cells))
