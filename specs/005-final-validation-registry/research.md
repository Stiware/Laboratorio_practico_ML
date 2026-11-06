# Research: Validación final y registro del modelo de OnTrack (etapas 7 y 8)

Sin dudas abiertas. Verificaciones hechas en el `.venv` el 2026-11-05: SHAP 0.54.1 se importa; `extreme_orders.parquet` tiene 64 filas y las mismas 18 columnas que
`train.parquet`; en Databricks existen el catálogo `workspace` y el esquema `default` (consulta de solo lectura).

## D1. Modelo final: piezas existentes, sin código nuevo de entrenamiento

- **Decisión:** `tuned_candidate("LightGBM cuantil 0,75", parametros_elegidos)` + `fit_and_recommend`/`fit_deadline_model` con **todo** `train.parquet` (familia B: margen 0).
  Se guarda `model.pipeline` con `joblib` en `models/final_pipeline.joblib`. Los parámetros se leen de `reports/06_best_params.json` (no se copian a mano).
- **Motivo:** el modelo final debe construirse exactamente como el que se comprobó en la etapa 6.

## D2. Abrir la prueba una sola vez

- **Decisión:** `open_test_once(path, mark_path)` en `models/validation.py`: si `mark_path` existe → `RuntimeError` con la fecha de la apertura anterior; si no, escribe
  la fecha y hora (ISO, zona local) y devuelve la tabla. El notebook la llama **una vez**, después de guardar el modelo y de escribir las decisiones congeladas.
- **Lectura de la etapa 8:** la ida y vuelta lee la prueba con `pd.read_parquet` directamente (decisión del grupo: se lee para comprobar, no para decidir). No pasa
  por `open_test_once`, porque esa función existe para impedir una **segunda evaluación**, y la comprobación no evalúa nada nuevo; se declara en el reporte.
- **Alternativa descartada:** no dejar constancia: la guía la pide y es la prueba de que la prueba midió generalización.

## D3. Baselines en la prueba

- **Plazo fijo del percentil 75:** `np.percentile(train[TARGET], 75)` (lo mismo que `DummyRegressor("quantile", 0.75)` del torneo), igual para todos los pedidos.
- **Siempre la mediana:** `np.median(train[TARGET])`.
- **Plazo prometido por Olist:** `(order_estimated_delivery_date − order_purchase_timestamp)` en días, pedido a pedido. Solo referencia (Q3 de la etapa 7); no entra al modelo.

## D4. Intervalos por cliente

- **Una media:** `bootstrap_mean(values, groups, n_boot=1000, seed=42)`: suma y conteo por cliente, sorteos de clientes, media por pedido; devuelve `(media, p2,5, p97,5)`.
  Misma forma que `bootstrap_cost_difference` (etapa 6) y que la función de la guía (§3).
- **Una diferencia de costo:** se reutiliza `bootstrap_cost_difference(real, plan_modelo, plan_baseline, clientes)`.
- **% tarde y cobertura:** `bootstrap_mean` sobre el indicador 0/1 de «llegó después del plazo» (su media es la proporción).

## D5. Holgura y atraso

- `slack_and_delay(real, plan)` → `holgura_media` = media de `plan − real` en los pedidos que llegan **a tiempo** (días de sobra) y `atraso_medio` = media de `real − plan`
  en los que llegan **tarde**; más el % de cada grupo. Reemplaza la «carga de trabajo» de la guía, que no aplica (no hay revisión humana).

## D6. Segmentos

- **Región:** `metrics_by_group(real, plan, plan, region)` (existe) + holgura y atraso por región.
- **Mes de compra:** misma tabla agrupando por `order_purchase_timestamp` en meses. Una sola función `segment_table(real, plan, segments)` que arma ambas.

## D7. Sensibilidad del costo

- `cost_sensitivity(real, plans, ratios=COST_RATIOS)` → tabla regla × proporción con `mean_delivery_cost(real, plan, ratio)`. Con 1 a 1 es el MAE (se prueba).
  Ojo de lectura: el plazo del modelo es el percentil 75 para todas las proporciones (no se reoptimiza por proporción); se declara.

## D8. Escenario con los 64 extremos

- Se predicen los 64 pedidos de `extreme_orders.parquet` con el modelo final y se calculan las métricas de prueba ∪ extremos. Algunos de esos clientes pueden estar
  en el entrenamiento; sus pedidos extremos nunca se usaron para ajustar. Es un escenario pesimista, separado del resultado principal y sin intervalo (se declara).

## D9. SHAP

- `top_shap(pipeline, X, n=10, sample=2000, seed=42)`: transforma `X` con el paso `prep`, usa `shap.TreeExplainer` sobre el LightGBM ajustado y agrupa por columna
  transformada (`get_feature_names_out`). Devuelve las n con mayor media del |SHAP| y la media del SHAP con signo. Explica la **salida cruda** del modelo (el percentil 75 en días).
- **Límites declarados:** explica al modelo, no al mundo; las variables correlacionadas se reparten el crédito (por ejemplo, `distance_km` y `log_distance`); importancia ≠ causa.
- **Atributo sensible:** `customer_state` y `customer_region` (ubicación del cliente).

## D10. Registro

- `register_final_model(pipeline, params, metrics, example, name, run_name="modelo-final")`: dentro de `mlflow.start_run`, `log_params`, `log_metrics`, etiquetas y
  `mlflow.sklearn.log_model(pipeline, name="model", input_example=example, signature=infer_signature(example, pipeline.predict(example)),
  registered_model_name=registered_model_name(name), serialization_format="cloudpickle")`. Unity Catalog **exige firma**, por eso se infiere explícitamente.
- `set_champion(full_name)` → `MlflowClient().set_registered_model_alias(full_name, "champion", última_versión)`.
- `load_champion(full_name)` → `mlflow.sklearn.load_model(f"models:/{full_name}@champion")`.
- **Ejemplo de entrada:** `input_example(df, n=3)` toma las columnas de `feature_columns(TOURNAMENT_GROUPS)` (18 numéricas y 7 categóricas) y quita cualquier identificador
  de cliente o pedido y el plazo prometido. Incluye `seller_id`, que es el código anónimo de Olist para el **vendedor** (una empresa, no una persona) y es una variable del modelo.
- **Privacidad de las salidas:** `MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT=true` y el nombre del experimento se muestra sin la ruta (que incluye el correo).

## D11. Notebooks generados por script

- `scripts/generadores/generar_notebook_07.py` y `generar_notebook_08.py`, como en las etapas 3 a 6. El 07 tiene la celda de decisiones congeladas **antes** de la que abre la prueba.
- **Aprendizaje de la etapa 6:** nunca regenerar un notebook ya ejecutado sin guardar antes sus salidas; el generador no se vuelve a correr después de ejecutar el 07.
