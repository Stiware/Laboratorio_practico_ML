# Quickstart: comprobar la validación final y el registro (etapas 7 y 8)

## Antes de empezar

1. Etapa 6 cerrada: `reports/06_best_params.json` con la decisión final.
2. `.gitignore` excluye `data/processed/` (`git status` no debe mostrar nada de esa carpeta).
3. `reports/prueba_abierta.txt` **no** existe (la prueba nunca se abrió).

## 1. Pruebas (sin datos)

```bash
.venv/bin/python -m pytest tests -q
```
Todas en verde, incluidas `tests/test_validation.py` y `tests/test_registry.py`.

## 2. Etapa 7 (abre la prueba: se ejecuta UNA vez)

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/07_validacion_interpretacion.ipynb --output 07_validacion_interpretacion.ipynb
```
Esperado: decisiones congeladas impresas con fecha y hora **antes** de la celda que abre la prueba; constancia en `reports/prueba_abierta.txt`;
tabla de la prueba con intervalos y la conclusión de la regla; segmentos, sensibilidad, extremos y SHAP; archivos `07_*`.
**Si el notebook falla después de abrir la prueba**, no se borra la constancia: se corrige el error y se reanuda declarándolo en el reporte.

## 3. Etapa 8

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/08_registro_modelo.ipynb --output 08_registro_modelo.ipynb
```
Esperado: «MLflow: databricks»; versión registrada con alias `champion`; ida y vuelta con `plazos_iguales = True` y el mismo costo que la etapa 7.

## 4. Comprobaciones

```bash
cat reports/prueba_abierta.txt                      # una sola fecha
grep -c "gmail\|databricks.com" notebooks/07_*.ipynb notebooks/08_*.ipynb reports/07_* reports/08_*   # 0 en todos
```
En Databricks: **Catalog → workspace → default → ontrack_plazo_entrega**: una versión con alias `champion`.
