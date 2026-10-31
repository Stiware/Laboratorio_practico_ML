# Quickstart: cómo comprobar la etapa 4

Estos pasos se pueden seguir cuando exista el código (después de `/speckit-implement`).

## Requisitos previos

- Estar en la raíz del repositorio y usar el `.venv` (`.venv/bin/python`).
- `data/processed/train.parquet` generado por el notebook 03 (si no existe: ejecutar el notebook 03 primero).

## 1. Pruebas

```bash
.venv/bin/python -m pytest tests -q
```

**Resultado esperado:** todo en verde (62 pruebas de antes más las nuevas de `test_features.py`, `test_split.py`, `test_models.py` y
`test_evaluation.py`).

## 2. Ejecutar el notebook de la etapa

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/04_features.ipynb --output 04_features.ipynb
ls reports/04_features_tabla.csv
```

**Resultado esperado:** el notebook termina sin errores y existe `reports/04_features_tabla.csv` con 14 filas: la referencia, los 9 grupos de ideas,
el plazo prometido (referencia no usable), la línea base trivial (siempre la mediana), el conjunto conservado y el conjunto
conservado sin la marca de fin de año (sensibilidad).

## 3. Comprobaciones rápidas (las hace también el notebook)

- Los 5 pliegues cubren todas las filas y ningún cliente está en los dos lados de un pliegue.
- La prueba hacia adelante: entrenamiento con compras hasta 2018-02-28, validación 2018-03-01 a 2018-08-31, sin clientes compartidos.
- Ninguna combinación de grupos incluye `promised_days`.
- Dos ejecuciones seguidas dan la misma tabla (misma semilla 42).

## 4. La prueba final no se abre

```bash
grep -c "test.parquet" notebooks/04_features.ipynb src/olistbr/features/engineering.py src/olistbr/models/train.py src/olistbr/models/evaluation.py
```

**Resultado esperado:** `0` en los cuatro archivos.

## 5. Revisión

- Subagente `auditor-de-fuga` sobre `src/olistbr/features/`, `src/olistbr/models/` y el notebook 04.
- Subagente `revisor-de-codigo-ia` para que cada integrante pueda explicar una variable.
- `reports/04_features.md` con una fila por idea, su cifra y su motivo (también las descartadas).
