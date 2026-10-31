# Research: Torneo de modelos de OnTrack (etapa 5, Fase 1)

No quedaron dudas abiertas en el contexto técnico. Aquí se registran las decisiones de diseño y por qué se tomaron. Las verificaciones se hicieron en el `.venv` el 2026-10-31.

## D1. El plazo óptimo es el percentil 75 (verificado)

- **Hecho:** con subestimar 3 veces más caro que sobrestimar, el costo por pedido es exactamente 4 veces la pérdida cuantílica con τ = 0,75. Se comprobó con
  `mean_delivery_cost` y `mean_pinball_loss` de scikit-learn sobre 5.000 pedidos sintéticos: diferencia 0,0. Además, el plazo fijo que minimiza el costo (13,75) coincide
  con el percentil 75 (13,74).
- **Consecuencia:** el plazo que recomienda la app es el percentil 75 de lo que podría tardar el pedido. En entrenamiento el percentil 75 global es 15,7 días,
  igual al «mejor plazo fijo» del caso de negocio.
- **Se prueba en código:** `test_pinball_loss_times_four_equals_cost`.

## D2. Modelo de plazo (`DeadlineModel`)

- **Decisión:** una clase pequeña que guarda el pipeline ajustado y el margen. `predict(X)` devuelve el valor del modelo; `recommend(X)` devuelve `predict(X) + margin`.
- **Familia A (valor típico + margen):** `fit_deadline_model` aparta el 20 % de los clientes del entrenamiento con `GroupShuffleSplit` (`CALIBRATION_SIZE = 0,2`, semilla 42),
  ajusta con el 80 % restante, predice a los clientes apartados y fija `margin = percentil 75 de (real − predicción)` de ellos. **No se reajusta** con el 100 %.
- **Familia B y líneas base:** ajuste con todo el entrenamiento, `margin = 0`.
- **Alternativas descartadas:**
  - Margen con los residuos del propio entrenamiento: los árboles memorizan y el margen saldría casi cero (optimista).
  - Margen calculado con la validación del pliegue: sería usar la validación para decidir un parámetro (fuga); daría un costo «oráculo».
  - Reajustar con el 100 % tras calcular el margen: mejora algo las predicciones pero el margen ya no corresponde al modelo reajustado; se prefiere la coherencia.
  - Validación cruzada interna para obtener residuos fuera de muestra: 3 veces más ajustes (el bosque ya es el candidato más lento).
- **Costo conocido:** la familia A se ajusta con el 80 % del entrenamiento. Se reporta.

## D3. Qué es comparable entre familias

- **Decisión:** entre familias solo se comparan el **costo 3 a 1**, el porcentaje tarde y la cobertura (todos salen del plazo recomendado). MAE, RMSE, R² y sesgo salen de la
  predicción del modelo: para la familia B esa predicción **es** el percentil 75 (más alta que el valor típico), por lo que su MAE y R² son peores a propósito.
- **Dónde se ve:** la tabla resumen lleva una columna `familia`; el reporte dice que la exactitud se compara dentro de cada familia.

## D4. Candidatos y parámetros (fijos, semilla 42, sin optimizar)

| Candidato | Familia | Parámetros | Escalado |
|---|---|---|---|
| Línea base: mediana | base | `DummyRegressor("median")` | no |
| Línea base: plazo fijo p75 | base | `DummyRegressor("quantile", quantile=0.75)` | no |
| Regresión lineal | A | `LinearRegression()` | sí |
| Ridge | A | `Ridge(alpha=1.0)` | sí |
| Bosque aleatorio liviano | A | `RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=20, max_features=0.33, max_samples=0.5, n_jobs=2)` | no |
| LightGBM | A | `LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=2, deterministic=True, force_row_wise=True, verbose=-1)` | no |
| XGBoost | A | `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist", n_jobs=2)` | no |
| LightGBM cuantil | B | igual que LightGBM con `objective="quantile"`, `alpha=0.75` | no |
| XGBoost cuantil | B | igual que XGBoost con `objective="reg:quantileerror"`, `quantile_alpha=0.75` | no |
| HistGradientBoosting cuantil | B | `HistGradientBoostingRegressor(loss="quantile", quantile=0.75, max_iter=300, learning_rate=0.05)` | no |

- **Motivo de los parámetros:** son los del ejemplo del curso y los medidos en el benchmark de tiempos; el bosque liviano se eligió por tiempo (18,4 s) y no mirando puntajes.
- **Excluido:** `QuantileRegressor` lineal de scikit-learn (16,9 s con 8.000 filas; no escala a 61.571).
- **Línea base de la mediana con margen 0:** representa «prometer la mediana». La línea base del plazo fijo p75 representa «prometer siempre el mismo plazo» (el mejor plazo fijo).

## D5. Métricas

- **Del plazo recomendado:** costo 3 a 1 (`mean_delivery_cost`), `share_late` = proporción de pedidos con `real > recomendado`, `coverage` = `1 − share_late`.
- **De la predicción del modelo:** `mae_days`, RMSE (`root_mean_squared_error`), R² (`r2_score`), sesgo = media de `real − predicción`.
- **Operativa:** `fit_seconds` (tiempo de `fit_deadline_model`).
- **Por región (finalistas):** `metrics_by_group` devuelve, por región del cliente (`customer_region`, 5 regiones), MAE y porcentaje tarde del plazo recomendado y el número de pedidos.

## D6. Comparación por pares y finalistas

- **`compare_models(best, other)`** (guía §2.4): por pliegue, `diferencia = costo(otro) − costo(mejor)`; devuelve diferencia media, pliegues que gana el mejor («k de n»)
  y `diferencia / ruido` con `ddof=1`. Ruido menor a 1e-12 → «no evaluable».
- **Consistente:** gana en al menos el 80 % de los pliegues (`MIN_WIN_SHARE = 0,8`, es decir 4 de 5) **y** `diferencia / ruido > 1`. En otro caso es un **empate práctico**.
- **`beats_baselines`:** un modelo supera una línea base si la diferencia media es positiva y `diferencia / ruido > 1` en costo. Se aplica contra ambas.
- **`practical_ties(best, others)`:** devuelve los modelos que no se distinguen del mejor; ante empate gana el más simple y rápido.
- **Los finalistas** (1 o 2) los redacta el reporte con estas funciones, el tiempo, la interpretabilidad y la lectura de la cobertura; la decisión final de incluir la interpretabilidad es del grupo.

## D7. Prueba hacia adelante

- **Decisión:** cada candidato se ajusta una vez con `forward_split` (entrenar hasta 2018-02-28, validar de 2018-03-01 a 2018-08-31) y se calculan las mismas métricas. Es una sola partición,
  sin desviación; sirve de guardia: si un modelo gana por pliegues y pierde hacia adelante, se señala (lección de F8 en la etapa 4).
- **Calibración en la prueba hacia adelante:** igual que en los pliegues, con el 20 % de los clientes del entrenamiento hacia adelante.

## D8. Diferencia en volumen, sin dinero

- **Decisión:** `per_thousand(diferencia_por_pedido)` multiplica por 1.000. La diferencia de **costo** se lee en «unidades de costo» (1 unidad = un día de sobrestimación) y la
  de **MAE** en días. En entrenamiento hay ≈ 5.300 pedidos por mes.
- **Por qué no dinero:** el costo 3 a 1 es un supuesto en unidades relativas (principio VII); asignarle un valor monetario sería inventarlo.

## D9. Reproducibilidad

- Semilla 42 en `GroupKFold`, `GroupShuffleSplit`, y todos los modelos. LightGBM con `deterministic=True` y `force_row_wise=True`.
- Las métricas deben salir idénticas entre ejecuciones; el **tiempo de ajuste no**, por eso va en `reports/05_torneo_tiempos.csv` y se excluye de la comparación.

## D10. Salidas

- `reports/05_torneo_por_pliegue.csv`: `modelo, familia, pliegue, mae, rmse, r2, sesgo, costo, tarde, cobertura, margen, n_ajuste, n_validacion`.
- `reports/05_torneo_resumen.csv`: media y desviación por modelo, partición hacia adelante, posiciones y comparación con la línea base.
- `reports/05_torneo_tiempos.csv`: tiempos de ajuste por modelo y pliegue.
- `reports/05_torneo_regiones.csv`: por región, para los finalistas.
- `reports/05_torneo.md`: texto con motivos, limitaciones y finalistas; se redacta **después** de ver las cifras (principio VII).
