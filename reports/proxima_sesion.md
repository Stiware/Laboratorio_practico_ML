# Punto de reanudación · OnTrack (al 5 nov 2026)

> Nota para retomar el trabajo en otra sesión sin repetir nada. Las decisiones del grupo están en `especificacion-proyecto.md` §7 y, resumidas para la
> exposición, en `reports/bitacora_decisiones.md`.

## Dónde estamos

| Etapa | Estado | Dónde está |
|---|---|---|
| 1 Caso de negocio, 2 EDA, 3 Preprocesamiento | Hecho | `reports/00_caso.md`, `01_eda.md`, `02_preprocesamiento.md` |
| 4 Variables, 5 Torneo | Hecho | `reports/04_features.md`, `05_torneo.md` |
| 6 Optimización | **Hecho, revisado y reproducible** (una segunda ejecución completa dio exactamente lo mismo) | `reports/06_optimizacion.md`, `06_best_params.json` |
| 7 Validación final | **Hecho y revisado. La prueba ya se abrió (2026-11-05 20:44:44): no se vuelve a abrir.** El modelo mejora al plazo fijo con confianza (9,216 contra 11,709; −2,493 [−2,608 ; −2,373]) | `reports/07_validacion.md`, `07_metricas_test.json` |
| 8 Registro | **Hecho y revisado.** `workspace.default.ontrack_plazo_entrega`, versión 1, alias `champion` | `reports/08_registro.md`, `08_registro.json` |
| 9 Inferencia y app | **Borrador de la especificación, pendiente de que el grupo responda Q1 a Q8** | `especificaciones/09-inferencia-app.md` |
| Informe técnico, presentación | Pendiente | — |

Todo el trabajo va en `main`. **Nada de las etapas 6 a 8 tiene commit todavía**: conviene hacerlo pronto (al menos `reports/prueba_abierta.txt`,
`reports/07_decisiones_congeladas.json` y los resultados), en commits separados por etapa, revisando `git status` antes.

## Siguiente paso: etapa 9 (inferencia y app)

1. Aprobar `especificaciones/09-inferencia-app.md` → Spec Kit (especificación, plan, tareas, con aprobación del grupo).
2. Exportar `models/model.joblib` **desde el champion**; comprobar que carga en el entorno de la app (**el modelo se guardó con Python 3.14; el proyecto pide 3.12**).
3. `models/input_schema.json` con rangos posibles (no solo los observados) y adaptar `src/olistbr/inference/predict.py`, hoy hecho para **clasificación**, a regresión:
   días y plazo. El modelo espera las 25 columnas que produce `add_features`.
4. `app/streamlit_app.py`: no consulta Databricks; valida entradas; aviso de uso responsable (el plazo es una estimación; ≈ 1 de cada 4 pedidos llega después;
   vigilar la cobertura cada mes; la ubicación del cliente cambia el plazo).
5. `models/ficha_modelo.md` con las cifras de `reports/07_validacion.md`, sus límites y lo no evaluado.
6. **No volver a leer la prueba**: comprobar el modelo exportado con filas del entrenamiento.

## Cómo retomar

```bash
.venv/bin/python -m pytest tests -q            # 242 pruebas en verde
```

- `data/raw/` (CSV de Kaggle) y `data/processed/` no se versionan. Si faltan: descargar los CSV y ejecutar el notebook 03.
- En esta Mac el kernel `python3` de Jupyter apunta a otro proyecto: ejecutar los notebooks con el `.venv` de este (en VS Code, elegir el kernel `.venv`).
- Los notebooks 03 a 08 se generan con `scripts/generadores/`. **No volver a correr un generador sobre un notebook ya ejecutado** sin guardar antes sus salidas.
- Spec Kit: `pwsh` está instalado (PowerShell 7.6.9 con `brew install powershell`); el script `create-new-feature` crea una rama, así que las carpetas de `specs/` se arman a mano.
- MLflow: conexión de Databricks en `.env` (no se imprime). En los notebooks, `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=true` y el nivel de los mensajes de MLflow en WARNING
  **después** de importarlo, para que no aparezca el correo de la cuenta.
