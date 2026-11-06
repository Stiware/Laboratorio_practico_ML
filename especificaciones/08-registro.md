# Especificación · Etapa 8 · Registro del modelo

> Se llena **antes** de pedirle código a Claude. Borrador propuesto por Claude el 5 nov 2026 a partir de la guía
> `material-del-curso/kit/guia-validacion-registro-y-app.md` (sección 5), la plantilla `notebooks/08_registro_modelo.ipynb` y
> `src/olistbr/tracking/mlflow_utils.py`. Depende de la etapa 7: se registra **el modelo que se evaluó en la prueba**, con sus métricas.

**Grupo:** _______________   **Fecha:** 5 nov 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo aceptó todas las propuestas de Q1 a Q6 el 5 nov 2026)_

## 1. Qué es y qué no es esta etapa

- **Tracking** (etapa 6): muchas ejecuciones de prueba → «¿qué probamos?». **Registro** (esta etapa): pocas **versiones del modelo** → «¿qué modelo usamos y en qué versión?».
- El alias **`champion`** es una etiqueta móvil: quien use el modelo pide «el champion» sin importar el número de versión.
- **Fuera de esta etapa** (van en la etapa 9, con la app): exportar `models/model.joblib` desde el champion, escribir `models/input_schema.json` y adaptar
  `src/olistbr/inference/predict.py`, que hoy está hecho para **clasificación** (probabilidades por clase y matriz de costos) y hay que pasarlo a regresión (días y plazo).

## 2. Preguntas y decisiones del grupo (5 nov 2026)

**Decisión:** el grupo aceptó **todas** las propuestas de Claude (columna «Propuesta») el 5 nov 2026.

| # | Pregunta | Por qué importa | Propuesta de Claude |
|---|---|---|---|
| Q1 | **¿Dónde se registra?** | La regla 7 pide Databricks; la conexión ya funciona (token del `.env`) | **Unity Catalog de Databricks**, con el nombre `<UC_CATALOG>.<UC_SCHEMA>.ontrack_plazo_entrega` (por defecto `workspace.default`, según `.env.example`). Si el catálogo o el esquema no existen o no hay permiso, respaldo local (`mlflow_local.db`) y se anota el motivo en `reports/08_registro.md`. **Decisión:** `workspace.default` (valores por defecto). |
| Q2 | **¿En qué experimento queda la ejecución del modelo final?** | Separar «lo que probamos» de «lo que entregamos» | Experimento nuevo **`ontrack-modelo-final`**, con una sola ejecución `modelo-final` |
| Q3 | **¿Qué se registra junto al modelo?** | Que otra persona pueda saber qué es sin preguntar | Parámetros (tipo de modelo, los 7 hiperparámetros, percentil 0,75, grupos de variables) · **métricas de la prueba** de `reports/07_metricas_test.json` (costo 3 a 1 con su intervalo, cobertura, % tarde, MAE) · **firma** (columnas y tipos de entrada) · **ejemplo de entrada** · etiquetas (`etapa=08_registro`, `datos=train.parquet`, fecha del congelamiento) |
| Q4 | **¿Qué filas van como ejemplo de entrada?** | El ejemplo se sube a Databricks; no debe llevar identificadores (regla 2) | **3 pedidos del entrenamiento solo con las columnas que usa el modelo**: sin `customer_unique_id`, sin `order_id` y sin el plazo prometido. Se comprueba con una prueba automática que no haya identificadores |
| Q5 | **¿Cómo se guarda el modelo?** | MLflow rechaza por defecto un `Pipeline` con preprocesamiento | `serialization_format="cloudpickle"`, como dice la guía. **Aviso de seguridad:** un archivo pickle puede ejecutar código al cargarse; solo se cargan modelos del propio grupo |
| Q6 | **¿Cómo se comprueba la ida y vuelta?** | Que lo registrado sea exactamente lo evaluado | Cargar `models:/<nombre>@champion` y comparar sus plazos con los del modelo local en **500 pedidos de la prueba** (`np.allclose`). Además, el costo 3 a 1 en toda la prueba con el modelo recuperado debe coincidir con el de `07_metricas_test.json` |

## 3. Reglas

1. Se registra **exactamente** `models/final_pipeline.joblib` de la etapa 7: no se reentrena nada aquí.
2. No se sube ningún identificador ni el contenido de `.env`; las salidas del notebook no muestran el correo ni el enlace del workspace (`MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT`, como en la etapa 6).
3. La prueba **no se vuelve a abrir para decidir**: solo se lee para la comprobación de ida y vuelta (no cambia nada del modelo). _(Confirmado por el grupo el 5 nov 2026.)_
4. Si el registro en Databricks falla, se usa el respaldo local y se escribe el motivo.

## 4. Código previsto

En `src/olistbr/tracking/registry.py` (nuevo), usando `mlflow_utils.py` sin reescribirlo:

| Función | Qué hace | Pruebas (`tests/test_registry.py`, con MLflow local en una carpeta temporal) |
|---|---|---|
| `input_example(df, n)` | n filas solo con las columnas del modelo, sin identificadores | `test_input_example_has_no_identifiers` |
| `register_final_model(pipeline, params, metrics, example, name)` | ejecución `modelo-final`, `log_model` con firma y ejemplo, registro con `registered_model_name` | con un modelo de juguete: queda una versión |
| `set_champion(name)` | alias `champion` a la última versión | `test_champion_points_to_latest_version` |
| `load_champion(name)` | carga `models:/<name>@champion` | `test_round_trip_predicts_the_same` |

## 5. Entregables

- `notebooks/08_registro_modelo.ipynb` (solo orquesta), generado por `scripts/generadores/generar_notebook_08.py`.
- `reports/08_registro.md` (plantilla existente) y `reports/08_registro.json` (nombre, versión, alias, plataforma, fecha, resultado de la ida y vuelta).
- Línea en `reports/bitacora_decisiones.md`.

## 6. Criterios de cierre

- ☒ Q1 a Q6 respondidas y confirmadas por el grupo (5 nov 2026).
- ☒ Modelo registrado con parámetros, métricas de la prueba, firma, ejemplo sin identificadores y etiquetas.
- ☒ Alias `champion` puesto y comprobación de ida y vuelta en verde.
- ☒ `pytest tests` en verde; revisión con `revisor-de-codigo-ia`.
- ☐ Quien no escribió el código sabe explicar la diferencia entre tracking y registro.
