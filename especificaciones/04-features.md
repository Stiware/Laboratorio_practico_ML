# Especificación · Etapa 4 · Variables (feature engineering)

> Se llena **antes** de pedirle código a Claude. Borrador propuesto por Claude el 31 oct 2026 a partir de `reports/01_eda.md`,
> `reports/02_preprocesamiento.md` y del ejemplo `material-del-curso/ejemplo-creditsight/src/creditsight/features/engineering.py`.
> Guía: `material-del-curso/kit/guia-modelado-y-optimizacion.md`, sección 1 · Prompts: `kit/prompts-por-etapa.md`, sección 4.
> Las cifras de la columna «Pista» son **correlaciones sobre el conjunto de entrenamiento** (nunca la prueba) para ordenar las
> ideas; **no son la decisión**. La decisión se toma midiendo con validación cruzada por cliente (sección 3).

**Grupo:** _______________   **Fecha:** 31 oct 2026   **Estado:** ☐ borrador  ☒ cerrada _(el grupo confirmó Q1 a Q4 y aprobó las ideas de la sección 4.2 el 31 oct 2026)_

## 1. Preguntas y respuestas del grupo (31 oct 2026)

| # | Pregunta | Respuesta |
|---|---|---|
| Q1 | **¿En qué momento predice la app?** | **La app recomienda la fecha prometida: predice los días y no conoce el plazo que Olist ya prometió.** `promised_days` **no es entrada del modelo** (sería circular: se usaría la respuesta de Olist para predecir). Se mide **una sola vez como referencia** para saber cuánto se pierde sin ella; no se usa en el modelo final. *Confirmada por el grupo el 31 oct 2026* (ver nota abajo) |
| Q2 | ¿Se miden variables que aprenden del objetivo (promedio de días por vendedor, categoría o ruta)? | **Sí**, solo con `TargetEncoder` de scikit-learn (valida internamente con validación cruzada), dentro del `Pipeline` |
| Q3 | ¿Se acepta usar el mes de compra, sabiendo que existen temporadas? | **Decidido por Claude (delegado por el grupo): NO como mes suelto; sí una marca de temporada alta de fin de año, con una prueba hacia adelante.** Evidencia en la nota de abajo |
| Q4 | ¿Qué modelo fijo se usa para comparar las ideas? | **`LinearRegression`** con el preprocesador de la sección 5 |

**Nota sobre Q1.** Predecir retrasos como objetivo principal no conviene porque solo el 6,8 % de los pedidos llega después de
la fecha prometida: es una clase muy poco frecuente y desbalanceada (las 6,8 % son unos 6.500 pedidos). Por eso el modelo predice
**días**, y el retraso queda como apartado secundario. Con la app fijando la fecha, el retraso se define respecto a la fecha
**recomendada**: dado un plazo, la app puede mostrar la probabilidad de llegar después, usando la distribución de los errores de
validación. Para fijar el plazo con el costo 3 a 1, la fecha recomendada es la predicción más un margen que corresponde al
percentil 75 de los errores (3 / (3 + 1)). Esto se desarrolla en las etapas 5 a 7; aquí solo define qué variables se pueden usar.

**Nota sobre Q3: qué muestran los datos de entrenamiento** (medianas de días de entrega por mes de compra):

| Mes | 2017 | 2018 |
|---|---:|---:|
| Ene | 10,6 | 11,8 |
| Feb | 10,9 | 14,3 |
| Mar | 10,1 | 13,4 |
| Abr | 12,7 | 9,4 |
| May | 9,7 | 9,1 |
| Jun | 10,8 | 8,0 |
| Jul | 10,1 | 7,5 |
| Ago | 9,7 | 7,0 |
| Sep a Dic | 10,3 · 10,2 · 12,6 · 13,1 | no hay datos |

- El patrón mensual de enero a agosto **casi no se repite** entre años (correlación 0,14): en 2017 es plano (cerca de 10 días) y en
  2018 baja de 13–14 a 7. Lo que parece «temporada» en el promedio de todos los pedidos es sobre todo **tendencia entre años**.
  Meter el mes como 12 categorías haría que el modelo aprenda esa tendencia y la extrapole mal hacia adelante.
- Lo que sí se ve es un **pico de fin de año en 2017**: la semana de Black Friday (24 nov) tiene la mediana más alta (13,7 días) con
  el mayor volumen (2.338 pedidos), y diciembre se mantiene en 13. Pero es **una sola temporada observada**: no se puede comprobar
  que se repita ni validar hacia adelante (los datos de 2018 terminan en agosto).
- El efecto del **día de la semana** sí es estable (correlación 0,72 entre años): lo comprado de jueves a sábado tarda más en 2017 y
  en 2018. Por eso se conserva como idea.
- **Decisión:** se prueba una marca de temporada alta (F6) con calendario fijo (no aprendida de los datos), y el mes suelto queda
  descartado con esta evidencia. La tendencia entre años queda como **limitación** a reportar: el modelo no puede anticipar mejoras
  o empeoramientos de la operación posteriores a agosto de 2018.

## 2. Regla de decisión (escrita antes de medir)

Una idea se **conserva** solo si cumple las tres:

1. **Se conoce al momento de predecir** (ver Q1) y la app puede obtenerla sin pedirle al usuario datos que no tiene.
2. **Mejora / ruido > 1**, donde *mejora* = MAE de la referencia − MAE de la variante (media de los pliegues) y *ruido* = desviación estándar entre pliegues de esa diferencia, con la métrica de control (costo 3 a 1) apuntando en la misma dirección, **y que la mejora no cambie de signo en la prueba hacia adelante** (sección 3). Una idea que solo ayuda en la validación por cliente pero empeora hacia adelante se descarta.
3. **Todo lo que aprende de los datos va dentro del `Pipeline`**, ajustado solo con entrenamiento.

Las ideas descartadas se anotan **con su cifra** en `reports/04_features.md`: descartar es un resultado válido.

## 3. Validación común

- **Datos:** `data/processed/train.parquet` (el `test.parquet` **no se abre**).
- **Pliegues:** `GroupKFold(n_splits=5)` por `customer_unique_id` (regresión: no admite mezcla aleatoria), calculados **una sola vez** y guardados en una lista; `assert` de que ningún cliente cae en los dos lados de un pliegue.
- **Prueba hacia adelante (guardia contra la tendencia):** dentro de `train.parquet`, entrenar con compras hasta 2018-02-28 y validar con 2018-03-01 a 2018-08-31 (con los clientes de la validación fuera del entrenamiento). Imita el uso real (predecir lo que viene). No usa `test.parquet`.
- **Métrica principal:** MAE en días. **Control:** costo medio 3 a 1 (`olistbr.models.metrics.mean_delivery_cost`).
- **Referencia:** variables base de la sección 4.1 con el modelo de Q4.
- **Reporte:** media y desviación entre pliegues por variante, y la columna `mejora / ruido`.

## 4. Variables

### 4.1 Base (ya existen en `train.parquet`)

| Tipo | Columnas |
|---|---|
| Numéricas | `price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`, `distance_km`, `missing_coords` |
| Categóricas | `product_category_name`, `seller_id` (con «otros»), `seller_state`, `customer_state` |
| No entran al modelo | `order_id`, `customer_unique_id`, `main_seller_items`, `order_approved_at` (se mide aparte), `order_purchase_timestamp` y `order_estimated_delivery_date` (solo para derivar variables) |

### 4.2 Ideas nuevas (hipótesis)

Todas se calculan con columnas que ya tenemos. «Aprende» = depende de los datos de entrenamiento.

| ID | Variable | Cómo se calcula | Por qué podría ayudar | ¿Se conoce al predecir? | Aprende | Pista (train) | Prioridad |
|---|---|---|---|---|---|---|---|
| F1 | `log_distance`, `log_freight`, `log_price`, `log_weight`, `log_volume` | `log1p` de cada columna | El modelo es lineal y la relación con los días es creciente pero curva | Sí | No | distancia: Pearson 0,417 → 0,428 con log (Spearman 0,546); flete: 0,18 → 0,27 | Alta |
| F2 | `same_state` | `1` si `customer_state == seller_state` | Dentro del mismo estado el envío no cruza regiones | Sí | No | mediana 6,5 d contra 12,8 d; correlación −0,39 | Alta |
| F3 | `customer_region`, `seller_region` | Región (Norte, Nordeste, Centro-Oeste, Sudeste, Sur) según una tabla fija de los 27 estados | Agrupa 27 estados en 5 y reduce la cantidad de categorías | Sí | No (tabla fija) | por medir (el EDA: SP 7,2 d, AM 25,9 d) | Media |
| F4 | `promised_days` | `order_estimated_delivery_date − order_purchase_timestamp`, en días | Es la estimación de Olist al comprar | **No** (Q1: la app fija la fecha) | No | correlación 0,41 (Spearman 0,52) | **Solo referencia**: se mide una vez para saber cuánto se pierde sin ella, nunca entra al modelo final |
| F5 | `purchase_dow`, `is_slow_weekday` | día de la semana de la compra (0 = lunes); marca jueves a sábado | Lo comprado a fin de semana se despacha después | Sí | No | mediana jueves–sábado ≈ 11 d contra ≈ 9,3 d de lunes a miércoles | Media |
| F6 | `is_year_end_peak` | `1` si la compra cae entre el lunes de la semana de Black Friday y el 31 de diciembre (calendario fijo; en los datos solo existe 2017) | Pico de demanda de fin de año; en 2017 esas semanas tienen la mediana más alta (13,1–13,8 días contra ≈ 10) | Sí | No (calendario) | una sola temporada observada: no se puede validar hacia adelante | Media, con la limitación de Q3 |
| F7 | `distance_x_same_state` | `log_distance × same_state` | La distancia pesa distinto dentro y fuera del estado; un modelo lineal no lo ve solo | Sí | No | sin medir | Media |
| F8 | `seller_mean_days`, `category_mean_days`, `route_mean_days` (ruta = estado vendedor → estado cliente) | promedio de `dias_entrega` por vendedor / categoría / ruta | El desempeño histórico del vendedor y de la ruta es una señal fuerte | Sí (es historia), pero **aprende del objetivo** | **Sí** | sin medir | Media, con Q2 |
| F9 | `freight_ratio`, `density` | `freight_sum / price_sum`; `weight_g_sum / volume_cm3_sum` (división segura) | El flete y la densidad podrían reflejar el tipo de servicio | Sí | No | correlación 0,09 y 0,01: **probable descarte** | Baja |
| F10 | `multi_seller` | `1` si `n_items > main_seller_items` | Pedidos con varios vendedores se consolidan | Sí | No | correlación −0,04 y con signo contrario al esperado: **probable descarte** | Baja |

### 4.3 Descartadas desde ya

| Variable | Motivo |
|---|---|
| `purchase_hour` | Correlación 0,004 con el objetivo en train; no hay hipótesis de negocio |
| `purchase_month` (mes suelto) | El patrón mensual no se repite entre 2017 y 2018 (correlación 0,14 en ene–ago): mezcla temporada con tendencia entre años y se extrapolaría mal (ver Q3) |
| `approval_delay_hours` (`order_approved_at − compra`) | Se conoce **horas después** de comprar (mediana 0,34 h, pero el 10 % peor tarda más de 34 h) y su correlación es 0,09. Si el grupo decide predecir tras la aprobación, se reabre |
| Cualquier `order_delivered_*`, `order_status`, `review_*` | Se conocen después de la decisión (fuga) |
| Identificadores (`order_id`, `customer_unique_id`) | No generalizan |
| Ciudad y código postal | Demasiadas categorías (decisión 8 del EDA) |

## 5. Preprocesamiento (dentro del `Pipeline`, como `build_preprocessor` del ejemplo)

- **Numéricas:** `SimpleImputer(strategy="median", add_indicator=True)` y `StandardScaler`. Medianas de `weight_g_sum`, `volume_cm3_sum` y `distance_km` **solo con entrenamiento**.
- **Categóricas:** imputar con `"Unknown"` y `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)`.
- **Funciones previstas en `src/olistbr/features/engineering.py`**, con la misma forma que el ejemplo:
  - `add_features(df)`: crea F1 a F7, F9 y F10 (sin aprender de los datos). No modifica el original, y división por cero o vacío da vacío (nunca `inf`).
  - `feature_columns(...)`: devuelve las listas de numéricas y categóricas según el grupo de ideas activas (un interruptor por idea, como `use_engineered` en el ejemplo).
  - `build_preprocessor(...)`: el `ColumnTransformer` anterior; F8 se agrega con `TargetEncoder` dentro de este mismo objeto.

## 6. Pruebas que deben existir (`tests/test_features.py`)

- ☐ Una prueba por variable de la sección 4.2 con un caso normal y uno sucio hecho a mano.
- ☐ División por cero y vacíos dan vacío, nunca `inf`.
- ☐ `add_features` no modifica el DataFrame de entrada.
- ☐ El preprocesador maneja una categoría que no vio en entrenamiento y no deja vacíos.
- ☐ F8: una prueba de que el valor de un cliente del pliegue de validación **no** usa su propio objetivo.
- ☐ Los pliegues no comparten clientes (`assert` en el notebook y prueba en `tests/`).

## 7. Criterios de cierre

- ☒ Q1 a Q4 respondidas por escrito y confirmadas por el grupo.
- ☐ `notebooks/04_features.ipynb` corre de principio a fin y solo orquesta (la lógica está en `src/`).
- ☐ Tabla de ideas medidas en `reports/04_features.md`: decisión, **cifra** (media, desviación y `mejora / ruido`) y motivo, también para las descartadas.
- ☐ `pytest tests` en verde.
- ☐ El conjunto de prueba no se abrió.
- ☐ Todo lo que aprende de los datos está dentro del `Pipeline`.
- ☐ Revisión con `auditor-de-fuga` y `revisor-de-codigo-ia`.
