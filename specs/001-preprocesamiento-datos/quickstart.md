# Quickstart: cómo comprobar la etapa 3

Estos pasos se pueden seguir cuando exista el código (después de `/speckit-implement`).

## Requisitos previos

- Estar en la raíz del repositorio con el entorno activado: `source .venv/bin/activate`.
- Los 9 CSV de Olist en `data/raw/`.

## 1. Pruebas

```bash
pytest tests
```

**Resultado esperado:** todo en verde, con pruebas en `test_cleaning.py`, `test_features.py`, `test_split.py`
y `test_metrics.py`.

## 2. Regenerar los datos desde cero

```bash
rm -f data/processed/train.parquet data/processed/test.parquet data/processed/extreme_orders.parquet
jupyter nbconvert --to notebook --execute notebooks/03_preprocesamiento.ipynb --output 03_preprocesamiento.ipynb
ls data/processed
```

**Resultado esperado:** el notebook termina sin errores y aparecen `train.parquet`, `test.parquet` y `extreme_orders.parquet`.

## 3. Comprobaciones rápidas (las hace también el notebook)

- Un solo registro por `order_id`.
- `dias_entrega` entre 0 y 100; compras entre 2017-01-01 y 2018-08-31.
- Ningún `customer_unique_id` en `train` y `test` a la vez; ambos suman las filas de la tabla limpia.
- Prueba cerca del 20 % de las filas (±1 punto) y repetir el notebook da archivos idénticos.
- Sin columnas de reseñas, de entrega ni identificadores como entradas.

## 4. Datos originales intactos

```bash
git status
```

**Resultado esperado:** sin cambios en `data/raw/` (que está en `.gitignore`) ni en `.env`.

## 5. Revisión

- Subagente `auditor-de-fuga` sobre `src/olistbr/data/` y el notebook 03.
- Subagente `revisor-de-codigo-ia` para que cada integrante pueda explicar una función.
- `reports/02_preprocesamiento.md` con las filas antes y después de cada regla.
