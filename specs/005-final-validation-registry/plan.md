# Implementation Plan: Validación final y registro del modelo de OnTrack (etapas 7 y 8)

**Branch**: `main` | **Date**: 2026-11-05 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/005-final-validation-registry/spec.md`

## Summary

Etapa 7: se ajusta el modelo congelado (LightGBM cuantil 0,75 optimizado, con las piezas de la etapa 6) con todo el entrenamiento, se escriben las decisiones con fecha y
hora y se abre la prueba **una sola vez** con `open_test_once`, que deja constancia y se niega a repetir. Se mide el costo 3 a 1, el % tarde y la cobertura del modelo y de
tres baselines con intervalos por cliente, la diferencia contra el plazo fijo del percentil 75, los segmentos por región y mes, la holgura y el atraso, la sensibilidad del costo,
el escenario con los 64 extremos y SHAP. Etapa 8: se registra exactamente ese modelo en Unity Catalog (`workspace.default.ontrack_plazo_entrega`) con firma, ejemplo sin
identificadores, parámetros, métricas y etiquetas; alias `champion`; ida y vuelta. La lógica va en `models/validation.py` y `tracking/registry.py` (nuevos), con pruebas;
los notebooks 07 y 08 solo orquestan y se generan por script. Ningún módulo anterior se modifica.

## Technical Context

**Language/Version**: Python 3.12 (requisito); en esta máquina el `.venv` es 3.14 (todas las pruebas pasan en él).

**Primary Dependencies**: pandas, numpy, scikit-learn 1.9.1, LightGBM 4.7.0, SHAP 0.54.1, MLflow 3.17.1, databricks-sdk. Ya están en `requirements.txt`; **no se agrega ninguna**.

**Storage**: entradas `train.parquet`, `test.parquet` (una vez), `extreme_orders.parquet`, `06_best_params.json`. Salidas: `models/final_pipeline.joblib`, `reports/07_*`,
`reports/prueba_abierta.txt`, `reports/08_registro.json`; Unity Catalog y el experimento `ontrack-modelo-final` en Databricks.

**Testing**: pytest con tablas pequeñas, plazos inventados y modelos de juguete; el registro se prueba con MLflow local en una carpeta temporal. Ninguna prueba lee `data/processed/`.

**Target Platform**: equipo local del grupo; seguimiento y registro en Databricks.

**Project Type**: biblioteca (`src/olistbr/`) más dos notebooks que orquestan.

**Performance Goals**: notebook 07 en menos de 5 minutos (un ajuste de ≈ 5 s, 1.000 sorteos por métrica, SHAP sobre 2.000 pedidos); notebook 08 en menos de 3 minutos (subida del modelo).

**Constraints**: la prueba se abre una sola vez y nada cambia después; intervalos por cliente; sin identificadores, correo ni enlace del workspace en archivos o salidas;
`.env` no se abre ni se imprime; textos en español y código en inglés.

**Scale/Scope**: 76.924 pedidos de entrenamiento, 19.215 de prueba, 64 extremos; 4 reglas (modelo y 3 baselines); 1 versión registrada.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principio | Cómo lo cumple este plan | ¿Pasa? |
|---|---|---|
| I. Datos originales no se tocan | Solo lee `data/processed/` | Sí |
| II. Se parte por la unidad de análisis | Todos los intervalos sortean clientes completos | Sí |
| III. La prueba se abre una vez | `open_test_once` con constancia; decisiones congeladas antes; nada cambia después; la lectura de la etapa 8 es solo para la ida y vuelta (decisión del grupo) | Sí |
| IV. Decisión con motivo, función y prueba | Cada función nueva con pruebas; decisiones con fecha y hora en `reports/07_validacion.md` | Sí |
| V. Nada sensible sale | Ejemplo de entrada sin identificadores (prueba); salidas sin correo ni enlace; no se guardan plazos pedido a pedido | Sí |
| VI. El código es nuestro | Funciones cortas con docstring en español; revisión con los subagentes | Sí |
| VII. No se inventan datos | Las cifras salen del notebook; los reportes se completan después de ver los archivos | Sí |
| VIII. Uso responsable | Segmentos por región y mes; atributo sensible declarado; la recalibración como regla de uso | Sí |

**Resultado:** sin violaciones. **Re-evaluación tras el diseño:** sin violaciones. A vigilar: (1) si el notebook 07 falla después de abrir la prueba, no se borra la constancia
(se corrige y se declara); (2) el `.gitignore` debe excluir `data/processed/` antes de abrir la prueba.

## Project Structure

### Documentation (this feature)

```text
specs/005-final-validation-registry/
├── plan.md · research.md · data-model.md · quickstart.md
├── contracts/validation-registry-functions.md
├── checklists/requirements.md
└── tasks.md             # lo genera /speckit-tasks (aún no existe)
```

### Source Code (repository root)

```text
src/olistbr/
├── models/validation.py        # NUEVO: open_test_once, bootstrap_mean, baseline_plans, test_report, slack_and_delay,
│                               #        segment_table, cost_sensitivity, top_shap
└── tracking/registry.py        # NUEVO: input_example, register_final_model, set_champion, load_champion

tests/test_validation.py        # NUEVO
tests/test_registry.py          # NUEVO

scripts/generadores/generar_notebook_07.py   # NUEVO
scripts/generadores/generar_notebook_08.py   # NUEVO
notebooks/07_validacion_interpretacion.ipynb # COMPLETAR la plantilla; solo orquesta
notebooks/08_registro_modelo.ipynb           # COMPLETAR la plantilla; solo orquesta
models/final_pipeline.joblib                 # NUEVO
reports/prueba_abierta.txt, 07_metricas_test.json, 07_segmentos.csv, 07_shap.csv, 08_registro.json   # NUEVOS
reports/07_validacion.md, 08_registro.md     # COMPLETAR las plantillas
```

**Structure Decision**: la validación va en `models/` (usa las métricas y piezas de la etapa 6) y el registro en `tracking/` (junto a `mlflow_utils.py`). No se modifica
ningún módulo existente.

## Orden de ejecución

**Notebook 07:** (1) leer `06_best_params.json` y escribir las decisiones congeladas con fecha y hora; (2) leer `train.parquet`, `add_features`, ajustar y guardar el modelo
final; (3) `open_test_once`; (4) plazos del modelo y de los baselines; `test_report`; (5) `segment_table`, `slack_and_delay`, `cost_sensitivity`, escenario con extremos;
(6) `top_shap`; (7) guardar `07_*`. **Notebook 08:** (1) `configure_mlflow`, `set_experiment("ontrack-modelo-final")`; (2) cargar `final_pipeline.joblib` y `07_metricas_test.json`;
(3) `input_example` y `register_final_model`; (4) `set_champion`; (5) `load_champion` e ida y vuelta; (6) guardar `08_registro.json`.

## Riesgos

- **La prueba se abre una sola vez:** cualquier error de código antes de esa celda se arregla sin consecuencia; después, se declara. Por eso el notebook 07 se prueba antes
  con una **copia temporal que usa una tabla falsa en lugar de la prueba** (como la prueba rápida de la etapa 6).
- **Unity Catalog** puede rechazar el registro por permisos o cuota: respaldo local con el motivo.
- **Tamaño del modelo**: 418 árboles de 70 hojas; unos pocos MB, aceptable en el repositorio.
- **Entorno:** Python 3.14 en esta Mac (el proyecto pide 3.12).

## Complexity Tracking

Sin violaciones que justificar.
