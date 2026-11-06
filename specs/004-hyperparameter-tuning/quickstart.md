# Quickstart: comprobar la optimización (etapa 6)

## Requisitos previos

1. Los CSV de Olist en `data/raw/` y `data/processed/train.parquet` generado con el notebook 03:
   ```bash
   .venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/03_preprocesamiento.ipynb --output 03_preprocesamiento.ipynb
   ```
2. `.env` con la conexión a Databricks (opcional: sin ella el seguimiento cae a `mlflow_local.db` y lo avisa).
3. Python 3.12 en el `.venv` (requisito del proyecto).

## 1. Pruebas (no necesitan datos)

```bash
.venv/bin/python -m pytest tests -q
```
Esperado: todas en verde (las 166 de antes más las nuevas de `test_tuning.py` y `test_split.py`).

## 2. Notebook completo

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/06_optimizacion.ipynb --output 06_optimizacion.ipynb
```
Esperado (≈ 20 a 35 minutos, estimado):
- Imprime dónde quedó MLflow (`databricks (...)` o `local`) y el experimento `ontrack-optimizacion`.
- `assert` de reserva y pliegues sin clientes compartidos.
- 20 ensayos por finalista; mejor costo de la búsqueda marcado como **optimista**.
- Por finalista: costo de partida y optimizado en la reserva, diferencia, intervalo del 95 % y decisión de la regla.
- Modelo elegido con su motivo; tablas informativas hacia adelante (por mes) y por región.
- Archivos: `reports/06_ensayos.csv`, `reports/06_best_params.json`.

## 3. Comprobaciones manuales

```bash
grep -c "test.parquet" notebooks/06_optimizacion.ipynb src/olistbr/models/tuning.py   # esperado: 0 en ambos
mlflow ui --backend-store-uri sqlite:///mlflow_local.db                               # solo si quedó en local
```
En MLflow: una ejecución por finalista con 20 ejecuciones hijas, cada una con parámetros, `costo_media`, `costo_desv` y etiquetas.

## 4. Reproducibilidad

Borrar `data/processed/optuna_optimizacion.db` y correr el notebook dos veces: los parámetros elegidos, las diferencias y los intervalos deben coincidir (el tiempo de ajuste no).

## 5. Revisiones

Pasar `auditor-de-fuga` y `revisor-de-codigo-ia` sobre `src/olistbr/models/tuning.py`, `data/split.py` y el notebook 06; resumir en `reports/06_optimizacion.md`.
