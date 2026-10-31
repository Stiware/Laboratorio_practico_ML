# Especificación · Etapa 5 · Torneo de modelos (Fase 1: lo que corresponde a `train.py`)

> Borrador propuesto por Claude el 31 oct 2026 a partir de `material-del-curso/ejemplo-creditsight/src/creditsight/models/train.py`,
> la guía `material-del-curso/kit/guia-modelado-y-optimizacion.md` (sección 2) y los prompts de la etapa 5.
> **CreditSight era clasificación; OnTrack es regresión** (predice `dias_entrega`), así que cada pieza del ejemplo se adapta (sección 1).
> Los tiempos de la sección 4 se midieron con `train.parquet` (solo tiempos de ajuste, no puntajes) para no sesgar la decisión.
> La Fase 2 (optimización, lo que corresponde a `tuning.py`) viene después, en `especificaciones/06-optimizacion.md`.

**Grupo:** _______________   **Fecha:** 31 oct 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo aceptó las propuestas de Q1 a Q5 el 31 oct 2026 y delegó Q6 a Claude, que la decidió midiendo tiempos)_

## 1. Del ejemplo (clasificación) a OnTrack (regresión)

| Pieza de `train.py` del ejemplo | Qué hace en CreditSight | Qué haremos en OnTrack |
|---|---|---|
| `encode_target` | convierte `Poor/Standard/Good` en 0/1/2 | **No hace falta**: el objetivo `dias_entrega` ya es un número |
| `expected_cost` + `COST_MATRIX` | costo de equivocarse según una matriz de costos | Ya existe: `mean_delivery_cost` en `models/metrics.py` (subestimar cuesta 3 veces más que sobrestimar) |
| `decide_min_cost(proba)` | elige la clase de menor costo esperado a partir de probabilidades | Su equivalente en regresión es **elegir el plazo que minimiza el costo 3 a 1: el percentil 75** de lo que podría tardar el pedido (sección 3) |
| `candidates()` | modelos del torneo: línea base, regresión logística, bosque, LightGBM, XGBoost | Mismos tipos pero **regresores**, más una familia que predice el percentil 75 directamente (sección 4) |
| `build_pipeline(model, scale, ...)` | preprocesador + modelo dentro de un `Pipeline` | Ya existe `build_pipeline` (etapa 4, fija `LinearRegression`): se **generaliza** para recibir cualquier modelo |
| `metrics_from_proba` | AUC, PR-AUC, F1, recall por clase, log loss | Se reemplaza por métricas de regresión: **MAE, RMSE, R²**, costo 3 a 1, porcentaje que llega tarde y cobertura (sección 5) |
| `evaluate_folds` / `evaluate` | una fila por pliegue con todas las métricas y el tiempo | Ya existe `evaluate_folds` (etapa 4, solo MAE y costo): se **amplía** con las métricas nuevas y el tiempo de ajuste |
| `feature_frame` | quita el objetivo | No hace falta |

**Idea clave que el ejemplo no tenía.** El costo 3 a 1 es exactamente **4 veces la pérdida de la regresión cuantílica con τ = 0,75**
(subestimar cuesta 3, sobrestimar 1: 0,75 / 0,25 = 3). Por eso el plazo que minimiza el costo es el **percentil 75** de la
distribución de días, no la mediana ni el promedio. Esto es lo que ya habíamos anotado en la decisión de la app (fecha recomendada =
predicción + margen del percentil 75 de los errores) y obliga a decidir cómo se obtiene ese plazo (pregunta Q2).

## 2. Preguntas y decisiones del grupo (31 oct 2026)

**Decisiones:** Q1 a Q5 se aceptan **tal como las propuso Claude** (columna «Propuesta»). **Q6 fue delegada a Claude**, que la decidió así:
**bosque aleatorio de 100 árboles, `max_depth` 12, `min_samples_leaf` 20, `max_features` 0,33, `max_samples` 0,5 y `n_jobs` 2.** Motivo (tiempos medidos de un ajuste
en un pliegue, solo tiempo y no puntajes): 254 s con 200 árboles; 55 s con la variante que propuse al inicio (100 árboles, profundidad 14, hoja 10, 50 % de variables;
42,5 s con 4 núcleos); **18,4 s** con esta. Con 18 s el torneo completo cabe en unos minutos y la optimización de la etapa 6 (15 ensayos × 5 pliegues) en unos 25 minutos; con 55 s serían más de
una hora. No se eligió mirando puntajes, sino por tiempo y por menor riesgo de memorizar. Si el bosque llega a ser finalista, la etapa 6 afina esos parámetros.


| # | Pregunta | Por qué importa | Propuesta de Claude |
|---|---|---|---|
| Q1 | **¿Qué métrica decide al ganador del torneo?** La ficha (`especificacion-proyecto.md` §4) dice que la métrica principal es el **MAE**; el costo 3 a 1 es secundario | Un modelo que acierta bien el valor típico (mejor MAE) no es el que minimiza el costo si la app va a prometer un plazo con margen. La guía pide que gane la métrica **que refleja la decisión** | **El costo 3 a 1 decide al ganador**; el MAE sigue siendo la métrica principal de exactitud y se reporta siempre, junto con RMSE y R². Si el grupo acepta, se anota en la sección 7 de `especificacion-proyecto.md` |
| Q2 | **¿Cómo se obtiene el plazo que recomienda la app?** (A) modelos que predicen el valor típico y se les suma un margen; (B) modelos que predicen directamente el percentil 75; (C) las dos familias | El costo 3 a 1 solo se minimiza con el percentil 75. Con (A) se conserva la regresión lineal del proyecto; con (B) se optimiza el costo directamente | **(C), las dos familias**. En (A) el margen sale de un grupo de **clientes de calibración** apartado dentro de cada entrenamiento (nunca de la validación). En (B) solo se usan modelos de boosting; la regresión lineal cuantílica de scikit-learn se descarta porque no escala (16,9 s con 8.000 filas; el entrenamiento tiene 61.571) |
| Q3 | **¿Con qué variables se corre el torneo?** La marca de fin de año (F6) quedó pendiente en la etapa 4 | F6 descansa en una sola temporada (2017). Con ella el conjunto mejora 0,178 días en validación por cliente y 0,373 hacia adelante; sin ella, 0,115 y 0,133 | **Sin F6** en el torneo (7 grupos: logaritmos, mismo estado, regiones, día de la semana, distancia × mismo estado, razones de flete, varios vendedores). Se mide una vez a los finalistas con F6 como sensibilidad. Si el grupo decide conservarla, se corre con F6 |
| Q4 | **¿La regresión lineal es obligatoria como modelo final?** `especificacion-proyecto.md` §3 la llama «candidato principal» | Si un modelo de árboles gana por costo, hay que saber si el grupo lo aceptaría **→ Decisión (31 oct 2026): sí; «al final se busca el mejor modelo»** | **La regresión lineal siempre es candidata y se reporta su diferencia contra el ganador** (en días de error por pedido y por cada 1.000 pedidos; en el entrenamiento hay ≈ 5.300 pedidos por mes). Si el ganador supera su ruido, el grupo decide con esa diferencia a la vista. No se inventa un valor en dinero |
| Q5 | **¿Se usa también la prueba hacia adelante?** (entrenar hasta 2018-02-28, validar de marzo a agosto de 2018) | En la etapa 4 una idea (F8) ganó por pliegues y perdió hacia adelante | **Sí**, para todos los modelos: los 5 pliegues por cliente de la etapa 4 (misma semilla 42) **más** la partición hacia adelante, y se avisa si un modelo gana en una y pierde en la otra |
| Q6 | **¿Se acepta un bosque aleatorio reducido?** El de 200 árboles tarda 254 s por ajuste (≈ 21 minutos solo en los 5 pliegues) | Es el candidato más lento; optimizarlo en la etapa 6 sería inviable | Propuesta inicial: 100 árboles, `max_depth` 14, `min_samples_leaf` 10, `max_features` 0,5 (55 s). **Decisión final (delegada a Claude): una variante más liviana de 18,4 s** (ver arriba) |

## 3. Reglas escritas antes de medir

1. **Datos:** solo `data/processed/train.parquet`. La prueba final **no se abre**.
2. **Pliegues:** los de la etapa 4 (`make_group_folds`, 5 pliegues por `customer_unique_id`, semilla 42) **y** `forward_split`. Todos los modelos usan exactamente los mismos.
3. **Líneas base** (todo modelo serio debe superarlas): **B1** predecir siempre la mediana del entrenamiento y **B2** prometer siempre el percentil 75 del entrenamiento (≈ 15,7 días, el «mejor plazo fijo» del caso de negocio). Un modelo que no las supera por más de su ruido se descarta.
4. **Gana el menor costo 3 a 1** (media de los 5 pliegues), sujeto a Q1.
5. **Comparación por pares** entre los mejores (guía §2.4): diferencia media, en cuántos pliegues gana y `diferencia / ruido`. Una diferencia que gana en 4 de 5 o más y supera 1 vez su ruido es consistente; si no, es **empate práctico**.
6. **Finalistas (1 o 2),** en este orden: (1) consistencia, (2) costo 3 a 1, (3) tiempo de ajuste y simplicidad (ante empate práctico gana el más simple), (4) interpretabilidad, (5) qué dicen las métricas técnicas y la cobertura.
7. **El torneo descarta, no corona:** la optimización puede cambiar el orden y solo la prueba final (etapa 7) dice qué tan bueno es el modelo elegido.
8. **Todo lo que aprende de los datos va dentro del `Pipeline`** y se ajusta solo con el entrenamiento de cada pliegue (principio III).

## 4. Candidatos (parámetros razonables, sin optimizar)

Tiempos: un ajuste en un pliegue (61.571 filas × 101 columnas), equipo con carga alta; son referencias, no promesas.

| # | Candidato | Familia | Tiempo medido (s) | Nota |
|---|---|---|---:|---|
| 1 | Línea base: siempre la mediana | A | — | `DummyRegressor("median")`; ya existe `build_baseline_pipeline` |
| 2 | Línea base: plazo fijo = percentil 75 | B | — | `DummyRegressor("quantile", 0.75)` |
| 3 | Regresión lineal | A (+ margen) | 3,0 | El modelo del proyecto y de la etapa 4 |
| 4 | Ridge (`alpha` = 1) | A (+ margen) | 1,5 | Lineal con regularización; hay muchas columnas codificadas |
| 5 | Bosque aleatorio reducido (Q6): 100 árboles, profundidad 12, hoja 20, 33 % de variables, 50 % de filas por árbol | A (+ margen) | **18,4** (254 con 200 árboles sin reducir) | Decidido por Claude (Q6) |
| 6 | LightGBM (300 árboles, tasa 0,05) | A (+ margen) | 2,8 | |
| 7 | XGBoost (300 árboles, tasa 0,05, profundidad 6) | A (+ margen) | 6,1 | |
| 8 | LightGBM cuantil 0,75 | B | 3,4 | `objective="quantile"`, `alpha=0.75` |
| 9 | XGBoost cuantil 0,75 | B | 11,6 | `objective="reg:quantileerror"`, `quantile_alpha=0.75` |
| 10 | HistGradientBoosting cuantil 0,75 | B | 13,0 | `loss="quantile"`, `quantile=0.75` |
| — | ~~Regresión lineal cuantílica~~ | B | 16,9 con 8.000 filas | **Excluida**: no escala a 61.571 filas |

**Estimación de tiempo total** (todos los modelos, 5 pliegues + prueba hacia adelante = 6 ajustes por modelo): con el bosque reducido, unos **5 a 6 minutos**
(suma de ≈ 60 s por ajuste de todos los modelos). Con el bosque de 200 árboles serían unos 25 minutos más; por eso Q6.

**Familia A (valor típico + margen).** El modelo predice un valor; la app recomienda `predicción + margen`, donde el **margen es el percentil 75 de
los errores** (real − predicción) en un grupo de **clientes de calibración**: dentro de cada entrenamiento se aparta el 20 % de los clientes
(`GroupShuffleSplit`), se ajusta el modelo con el resto y se calcula el margen con los apartados. La validación nunca se usa para el margen.
**Familia B (percentil 75 directo).** El modelo ya predice el plazo recomendado; no lleva margen.

## 5. Métricas (regresión; adaptadas de la guía §2.2)

| Métrica | Familia | Qué dice |
|---|---|---|
| **Costo medio 3 a 1** | negocio (decide, Q1) | Costo por pedido del **plazo recomendado** con subestimar 3 veces más caro |
| **MAE (días)** | exactitud (principal de la ficha) | Error típico del valor central, en días |
| **RMSE y R²** | exactitud | RMSE castiga más los errores grandes; R² es la parte de la variación que explica el modelo |
| **Porcentaje que llega tarde** y **cobertura** | decisión | Porcentaje de pedidos con `real > plazo recomendado` (esperado ≈ 25 % para un percentil 75); la cobertura es su complemento. Es el análogo del «recall por clase»: un modelo puede tener bajo costo con una cobertura rara |
| **Sesgo medio** (real − predicción) | exactitud | Si el modelo subestima o sobrestima en promedio |
| **Tiempo de ajuste (s)** | operación | Cuánto cuesta reentrenar |
| **MAE y porcentaje tarde por región del cliente** (5 regiones) | equidad (principio VIII) | Solo para los finalistas: dónde rinde peor |

**Advertencia de lectura.** Los modelos de la familia B predicen el percentil 75, no el valor típico: su MAE, RMSE y R² **no son comparables**
con los de la familia A (a propósito están más alto). Entre familias se compara **solo el costo 3 a 1**; las métricas de exactitud se comparan dentro de cada
familia. Esto es el equivalente, en regresión, a «el costo puede esconder lo que muestran las técnicas».

## 6. Código previsto (misma forma que `train.py` del ejemplo)

En `src/olistbr/models/train.py` y `models/evaluation.py` (y `models/metrics.py` para las métricas simples):

| Función | Qué hace | Pruebas (`tests/test_models.py`, `test_evaluation.py`) |
|---|---|---|
| `candidates(random_state)` | devuelve los modelos del torneo con su familia y si necesitan variables escaladas | `test_candidates_have_expected_names_and_families`, `test_every_candidate_fits_a_tiny_table` |
| `build_pipeline(model=None, groups=(), include_reference=False)` | se generaliza para recibir cualquier modelo; con `model=None` sigue siendo la regresión lineal de la etapa 4 | las pruebas existentes siguen pasando; `test_build_pipeline_accepts_any_model` |
| `regression_metrics(y_true, y_pred, recommended=None)` | MAE, RMSE, R², sesgo, costo 3 a 1, porcentaje tarde y cobertura | `test_regression_metrics_by_hand`, `test_perfect_predictions_give_zero_error`, `test_pinball_loss_times_four_equals_cost` (la identidad costo = 4 × pérdida cuantílica) |
| `fit_deadline_model(pipeline, X, y, groups, quantile=0.75, calibration_size=0.2, seed)` | familia A: aparta clientes de calibración, ajusta con el resto y calcula el margen | `test_margin_comes_from_calibration_customers_only`, `test_margin_gives_about_75_percent_coverage`, `test_no_customer_in_fit_and_calibration` |
| `evaluate_folds` (ampliada) | una fila por pliegue con todas las métricas y `fit_seconds`; compatible con la etapa 4 | `test_evaluate_folds_returns_all_metrics`, y las pruebas de la etapa 4 siguen pasando |
| `compare_models(best, other)` | diferencia media, pliegues que gana y `diferencia / ruido` | `test_compare_models_counts_wins`, `test_compare_models_zero_noise` |
| `rank_table(results)` | posiciones (1 = mejor) por métrica | `test_rank_table_orders_by_metric` |

## 7. Entregables

- `notebooks/05_torneo.ipynb` (plantilla del curso; **solo orquesta** y reutiliza los pliegues de la etapa 4).
- `reports/05_torneo.md`: tablas de negocio y de exactitud (media ± desviación entre pliegues), la tabla de posiciones, la comparación por pares, la diferencia expresada en días y en pedidos, la lectura de la cobertura, 1 o 2 finalistas con su motivo y por qué el torneo no elige un campeón.
- `reports/05_torneo_por_pliegue.csv` (cada modelo en cada pliegue) y `reports/05_torneo_resumen.csv`.
- Línea nueva en `reports/bitacora_decisiones.md`.

## 8. Criterios de cierre

- ☒ Q1 a Q6 respondidas por escrito y confirmadas por el grupo (Q6 delegada).
- ☒ `notebooks/05_torneo.ipynb` corre de principio a fin y solo orquesta; dos ejecuciones dan las mismas métricas (el tiempo de ajuste va aparte).
- ☒ Todos los modelos usan **los mismos pliegues** y la misma partición hacia adelante; hay dos líneas base y se superan (o se explica por qué no).
- ☒ Se guardó **cada pliegue**, no solo el promedio.
- ☒ Tablas de negocio y de exactitud, cobertura revisada (también mes a mes y por región), comparación por pares del mejor contra los siguientes.
- ☒ 1 o 2 finalistas propuestos y justificados en `reports/05_torneo.md`, con la diferencia de la regresión lineal contra el ganador _(los confirma el grupo)_.
- ☒ `pytest tests` en verde (166 pruebas) y la prueba final **no se abrió** (`grep` de `test.parquet` da 0).
- ☒ Todo lo que aprende de los datos (preprocesador, modelo y margen) está dentro de cada pliegue.
- ☒ Revisión con `auditor-de-fuga` y `revisor-de-codigo-ia` (resumen en `reports/05_torneo.md`).

- ☐ Los finalistas y las preguntas de la sección «Para decidir en grupo» de `reports/05_torneo.md` están confirmados por el grupo.

## 9. Riesgos que conviene conocer

- **Margen por calibración (familia A):** entrena con el 80 % de cada entrenamiento (un poco menos de datos) y el margen depende de los clientes apartados; se reporta su valor por pliegue para ver cuánto varía.
- **Plazo prometido y fechas:** el torneo hereda las limitaciones de la etapa 4 (tendencia entre años, una sola temporada); la partición hacia adelante es una sola y no tiene ruido medido.
- **Tiempo del bosque aleatorio:** ver Q6. El equipo estaba con carga alta durante la medición; los tiempos pueden mejorar.
- **Modelos cuantílicos y cobertura:** un modelo cuantílico mal ajustado puede lograr buen costo con una cobertura lejos del 75 %; por eso se reporta siempre.
- **Ranking por costo y por exactitud pueden no coincidir:** es esperado entre familias (sección 5).
