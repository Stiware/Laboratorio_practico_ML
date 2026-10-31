# Quickstart: cómo comprobar el torneo

Estos pasos se pueden seguir cuando exista el código (después de `/speckit-implement`).

## Requisitos previos

- Estar en la raíz del repositorio y usar el `.venv` (`.venv/bin/python`).
- `data/processed/train.parquet` generado por el notebook 03 (si no existe, ejecutar el notebook 03 primero).

## 1. Pruebas

```bash
.venv/bin/python -m pytest tests -q
```

**Resultado esperado:** todo en verde (119 pruebas de las etapas anteriores más las nuevas de `test_metrics.py`, `test_models.py` y `test_evaluation.py`).

## 2. Ejecutar el notebook del torneo

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/05_torneo.ipynb --output 05_torneo.ipynb
ls reports/05_torneo_por_pliegue.csv reports/05_torneo_resumen.csv reports/05_torneo_tiempos.csv reports/05_torneo_regiones.csv
```

**Resultado esperado:** el notebook termina sin errores (unos 5 a 6 minutos) y existen los cuatro archivos. `05_torneo_por_pliegue.csv` tiene 10 candidatos × 5 pliegues
más 10 filas de la partición hacia adelante (60 filas).

## 3. Comprobaciones rápidas (las hace también el notebook)

- Todos los candidatos usan los mismos 5 pliegues y la misma partición hacia adelante.
- En cada ajuste de la familia A, ningún cliente está en el ajuste y en la calibración.
- Las dos líneas base están en la tabla y se informa quién las supera.
- La cobertura de los modelos de percentil 75 está cerca de 0,75.
- Dos ejecuciones dan las mismas métricas (el archivo de tiempos puede diferir).

## 4. La prueba final no se abre

```bash
grep -c "test.parquet" notebooks/05_torneo.ipynb src/olistbr/models/train.py src/olistbr/models/evaluation.py src/olistbr/models/metrics.py
```

**Resultado esperado:** `0` en los cuatro archivos.

## 5. Revisión

- Subagente `auditor-de-fuga` sobre `src/olistbr/models/` y el notebook 05.
- Subagente `revisor-de-codigo-ia` para que cada integrante pueda explicar el modelo de plazo y la regla de decisión.
- `reports/05_torneo.md` con las tablas, la lectura de la cobertura, la regresión lineal contra el ganador y 1 o 2 finalistas con su motivo.
