# Implementation Plan: Preprocesamiento de datos de OnTrack (etapa 3)

**Branch**: `main` | **Date**: 2026-10-31 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-preprocesamiento-datos/spec.md`

## Summary

Se convierten los 7 CSV de Olist en dos archivos, `train.parquet` y `test.parquet`, con una fila por pedido entregado, el objetivo
`dias_entrega` y las variables que se conocen al comprar. Se hace con funciones pequeñas de pandas en
`src/olistbr/data/` (una por regla del EDA), cada una con su prueba, y un notebook que solo las llama en el orden
aprobado. La partición es 80/20 por `customer_unique_id` con `GroupShuffleSplit` (semilla 42). Lo que "aprende" de los
datos (la lista de vendedores frecuentes) se calcula solo con el entrenamiento. La imputación y la codificación quedan
para la etapa 4, dentro de un `Pipeline`.

## Technical Context

**Language/Version**: Python 3.12 (según `requirements.txt` y la constitución)

**Primary Dependencies**: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1 (solo `GroupShuffleSplit`) y pyarrow 25.0.1 (Parquet). Todas fijadas en `requirements.txt`; `pyarrow` se agregó el 2026-10-31 (ya estaba instalada en el `.venv`).

**Storage**: archivos. Entrada `data/raw/*.csv` (solo lectura); salida `data/processed/train.parquet`, `test.parquet` y `extreme_orders.parquet`. Se usa Parquet (conserva tipos y fechas, y es el formato natural para Databricks).

**Testing**: pytest 9.1.1 desde la raíz con `pytest tests`; casos hechos a mano con tablas pequeñas, sin leer `data/raw/`.

**Target Platform**: Linux/Windows local del grupo (notebook ejecutado con el `.venv`).

**Project Type**: biblioteca de funciones (`src/olistbr/data/`) más un notebook que orquesta.

**Performance Goals**: la limpieza completa corre en unos minutos en un portátil (el CSV más grande, `geolocation`, tiene 1.000.163 filas). No hay un requisito de velocidad más estricto.

**Constraints**: `data/raw/` solo lectura; el conjunto de prueba no se abre; ningún preprocesamiento que aprenda de los datos antes de partir; textos en español y nombres de código en inglés.

**Scale/Scope**: 99.441 pedidos, 112.650 ítems, 32.951 productos, 3.095 vendedores. Una sola máquina, sin paralelismo.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cómo lo cumple este plan | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | `load_raw()` solo lee; todo se escribe en `data/processed/`; una prueba compara que `data/raw/` no cambia (SC-008) | Sí |
| II. Se parte por la unidad de análisis | `split_by_group` con `GroupShuffleSplit` por `customer_unique_id`; `test_split.py` verifica cero solapamiento y falla si se parte por fila | Sí |
| III. La prueba se abre una vez | `test.parquet` se guarda y no se lee en esta etapa; la lista de vendedores frecuentes se aprende solo con train; la imputación va a la etapa 4 | Sí |
| IV. Decisión con motivo, función y prueba | 10 reglas → 13 funciones → pruebas; `reports/02_preprocesamiento.md` con filas antes y después | Sí |
| V. Nada sensible sale | `load_raw()` no carga reseñas; `data/raw/` y `.env` fuera de Git | Sí |
| VI. El código es nuestro | Funciones cortas y con docstring en español; revisión con los subagentes `auditor-de-fuga`, `escritor-de-pruebas` y `revisor-de-codigo-ia` | Sí |
| VII. No se inventan datos | Las cifras del reporte salen del código; las del EDA son solo referencia para comparar | Sí |
| VIII. Uso responsable | No aplica en esta etapa (no hay modelo ni app); se conserva `customer_state` para poder revisar el desempeño por región en la etapa 7 | Sí |

**Resultado:** sin violaciones. No hay nada en "Complexity Tracking".

**Re-evaluación tras el diseño (Fase 1):** sigue sin violaciones. El diseño mantiene el orden aprobado y no agrega dependencias.

## Project Structure

### Documentation (this feature)

```text
specs/001-preprocesamiento-datos/
├── plan.md              # Este archivo
├── research.md          # Decisiones técnicas y alternativas
├── data-model.md        # Tablas de entrada, intermedias y de salida
├── quickstart.md        # Cómo comprobar que funciona
├── contracts/
│   └── data-functions.md  # Firma y comportamiento de cada función
└── tasks.md             # Lo genera /speckit-tasks (aún no existe)
```

### Source Code (repository root)

```text
src/olistbr/
├── config.py            # rutas (RAW_DIR, PROCESSED_DIR…), columnas, reglas aprobadas y costo del error
├── data/
│   ├── cleaning.py      # load_raw, drop_personal_data, build_target, filter_stable_period,
│   │                    # drop_extreme_targets, clean_products, aggregate_items, drop_leakage_columns,
│   │                    # build_orders_table (encadena las reglas 1 a 7)
│   └── split.py         # split_by_group, assert_disjoint_groups, prepare_train_test
├── features/
│   ├── geo.py           # zip_centroids, haversine_km, add_distance
│   └── categories.py    # frequent_values, group_rare_categories
└── models/
    └── metrics.py       # delivery_cost, mean_delivery_cost, mae_days, best_fixed_deadline

tests/
├── conftest.py          # ya existía; ahora incluye la carpeta de CSV de ejemplo (raw_dir)
├── test_cleaning.py     # carga, reglas del EDA y tabla de pedidos
├── test_features.py     # distancia y categorías raras
├── test_split.py
└── test_metrics.py

notebooks/03_preprocesamiento.ipynb   # solo llama a las funciones y muestra conteos
reports/02_preprocesamiento.md        # decisiones con filas antes/después
data/processed/                       # train.parquet, test.parquet, extreme_orders.parquet (se regeneran; no se versionan)
```

**Structure Decision**: se respeta la estructura obligatoria del repositorio y se sigue el ejemplo del curso
(`material-del-curso/ejemplo-creditsight/src/creditsight/`): un `config.py` central, `data/cleaning.py` con la carga y las
reglas, `data/split.py` con la partición y su verificación, y las variables derivadas en `features/`. *Cambio respecto al
plan inicial (2026-10-31, a pedido del grupo):* el plan original proponía seis módulos dentro de `data/` (`loading`, `cleaning`,
`geo`, `categories`, `split`, `pipeline`); se reorganizaron sin cambiar ninguna regla ni resultado. `models/metrics.py` se agregó
para que el costo del error (3 a 1) no viva dentro del notebook 01. `categories` se aplica con `split.prepare_train_test`.

## Orden de ejecución (aprobado por el grupo)

1. `load_raw` (+ `drop_personal_data`) → 2. `build_target` → 3. `filter_stable_period` → 4. `drop_extreme_targets` →
5. `aggregate_items` (con `clean_products`) → 6. `add_distance` → 7. `drop_leakage_columns` → 8. `split_by_group` →
9. `frequent_values` con train → `group_rare_categories` en train y test → guardar.

Los pasos 1 a 7 los encadena `build_orders_table`; los pasos 8 y 9 se llaman desde el notebook, para que la partición
quede a la vista.

## Riesgos y puntos a confirmar

- **Peso del pedido con productos sin peso:** decidido (el grupo delegó la elección). Si algún ítem no tiene peso válido,
  el peso del pedido queda vacío y la etapa 4 lo imputa con la mediana de train y un indicador. Afecta a 22 de 96.478
  pedidos entregados (medido con los datos reales). Ver `research.md`, D3.
- **Extremos en la prueba:** como los pedidos de más de 100 días se quitan antes de partir, también salen del conjunto de
  prueba. La métrica final no los incluye; se reportan aparte (`extreme_orders.parquet`).
- **Ubicación del cliente:** 39 clientes tienen más de un estado. Se usa la ubicación del pedido (`customer_id`), no la de
  la persona.
- **Nombre del reporte:** `reports/` tiene `00_`, `01_` y `04_`. Se propone `02_preprocesamiento.md`.
