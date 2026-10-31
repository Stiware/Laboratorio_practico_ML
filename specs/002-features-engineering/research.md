# Research: Variables nuevas de OnTrack y su medición

No quedaron dudas abiertas en el contexto técnico. Aquí se registran las decisiones de diseño y por qué se tomaron. Las
verificaciones de scikit-learn se hicieron en el `.venv` el 2026-10-31.

## D1. `add_features` sin estado, calculado una vez fuera de los pliegues

- **Decisión:** `add_features(df)` crea **todas** las columnas derivadas de una vez; ninguna depende de otras filas ni del objetivo.
  `feature_columns(groups)` decide cuáles se usan. Se calcula una sola vez sobre `train.parquet`, antes de los pliegues.
- **Motivo:** como no aprende nada, aplicarla antes de partir no filtra información (es lo que hace el ejemplo del curso).
  Evita recalcular 60 veces lo mismo y deja las variables listas para la app, que las calculará igual.
- **Alternativa descartada:** `FunctionTransformer` dentro del `Pipeline` (más lento y sin ganancia de seguridad para columnas sin
  estado).

## D2. Historial por vendedor, categoría y ruta con `TargetEncoder`

- **Decisión:** columnas `seller_id`, `product_category_name` y `route` (texto «estado del vendedor → estado del cliente», creado en
  `add_features`) pasan por `TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=42))` **dentro del
  `ColumnTransformer`**. El valor de una categoría nueva es la media global del entrenamiento (verificado: 57,55 y 59,50 con media global 59,50
  en una prueba de juguete).
- **Motivo:** el encoder calcula, para cada fila de entrenamiento, el promedio de las otras filas (validación cruzada interna), así que
  una fila no usa su propio objetivo. En validación usa estadísticas del entrenamiento del pliegue.
- **Cuidado documentado:** el separador interno (`KFold`) **no agrupa por cliente**: dos pedidos del mismo cliente pueden quedar en
  pliegues internos distintos y uno informa al otro dentro del entrenamiento. Eso puede hacer que el ajuste dentro de entrenamiento
  parezca mejor, pero **no contamina la validación**, porque el cliente de validación nunca está en el entrenamiento del pliegue
  (los pliegues exteriores sí son por cliente). Se anota en el reporte.
- **Nota de versión:** en scikit-learn 1.9 los argumentos `shuffle` y `random_state` de `TargetEncoder` están obsoletos; se pasa el
  separador en `cv`.
- **Alternativa descartada:** calcular los promedios a mano (más código, más riesgo de fuga).

## D3. Regiones: tabla fija

- **Decisión:** tabla fija estado → región (IBGE): Norte (AC, AP, AM, PA, RO, RR, TO); Nordeste (AL, BA, CE, MA, PB, PE, PI, RN, SE);
  Centro-Oeste (DF, GO, MT, MS); Sudeste (ES, MG, RJ, SP); Sur (PR, RS, SC). Un estado que no esté en la tabla queda vacío.
- **Motivo:** no depende de los datos, así que no filtra; reduce 27 categorías a 5.

## D4. Temporada alta de fin de año

- **Decisión:** `is_year_end_peak = 1` si la fecha de compra está entre el lunes de la semana del cuarto viernes de noviembre
  (Black Friday; en 2017 es 2017-11-24, lunes 2017-11-20) y el 31 de diciembre de ese año.
- **Motivo:** calendario fijo (no aprendido); en los datos de entrenamiento las semanas de noviembre-diciembre de 2017 tienen la
  mediana más alta (13,1 a 13,8 días contra cerca de 10).
- **Limitación:** en los datos solo existe una temporada (2017); no se puede validar hacia adelante.

## D5. Día de la semana

- **Decisión:** `purchase_dow` se trata como **categoría** (lunes a domingo) y `is_slow_weekday` (jueves a sábado) como marca numérica.
- **Motivo:** el día de la semana no es una escala; un modelo lineal con el número 0–6 supondría un efecto proporcional. El patrón
  se repite entre 2017 y 2018 (correlación 0,72).

## D6. Cálculo de `mejora / ruido`

- **Decisión:** para cada idea y cada pliegue: `diferencia_i = error_referencia_i − error_idea_i` (positiva = la idea ayuda).
  `mejora` = media de las diferencias; `ruido` = desviación estándar muestral (`ddof=1`) de las diferencias; `mejora / ruido` = cociente.
  Se calcula para el error en días (principal) y para el costo 3 a 1 (control).
- **Motivo:** son diferencias **pareadas** (mismos pliegues), por eso el ruido mide la variación de la mejora y no la de los datos.
- **Borde:** si `ruido < 1e-12`, el cociente es «no evaluable» (nunca se divide por cero).

## D7. Regla de decisión como función

- **Decisión:** `decide_idea(...)` devuelve `conservar` / `descartar` y un motivo de una línea, según la regla de `04-features.md` §2:
  (1) se conoce al predecir, (2) `mejora / ruido > 1` en el error en días **y** el costo apunta en la misma dirección (mejora media
  positiva), (3) la mejora hacia adelante no cambia de signo. Si no hay ejemplos hacia adelante (temporada alta), la tercera condición
  se marca «no evaluable» y no descarta; el motivo lo dice.
- **Motivo:** la regla escrita queda en código con pruebas, y el reporte no depende de interpretar números a ojo.

## D8. Plazo prometido: solo referencia

- **Decisión:** `promised_days` está en `REFERENCE_ONLY_COLUMNS`. `feature_columns` no lo incluye por ningún grupo; solo se puede
  pedir con el argumento explícito `include_reference=True`, que usa únicamente el notebook para la fila de referencia. Una prueba
  comprueba que ninguna combinación de grupos lo contiene.
- **Motivo:** la app recomienda la fecha (Q1), así que no puede conocer el plazo de Olist; impedirlo en el código evita errores.

## D9. Pliegues y partición hacia adelante

- **Decisión:** `make_group_folds` usa `GroupKFold(n_splits=5, shuffle=True, random_state=42)` (verificado: la versión instalada admite
  `shuffle` y es reproducible) y devuelve la lista de índices. `forward_split(df, train_end, valid_start, valid_end)` devuelve
  entrenamiento (compras ≤ 2018-02-28, sin los clientes que aparecen en la validación) y validación (2018-03-01 a 2018-08-31).
- **Motivo:** la guía del curso decía que `GroupKFold` no admite mezcla; en scikit-learn 1.9.1 sí, y mezclar evita que los pliegues
  dependan del orden de las filas.
- **Cifras esperadas (de la distribución mensual del EDA):** del orden de 45 mil pedidos para entrenar y 31 mil para validar hacia
  adelante; los números reales los dará el notebook.

## D10. Preprocesador (como `build_preprocessor` del ejemplo)

- **Numéricas:** `SimpleImputer(strategy="median", add_indicator=True)` y `StandardScaler`.
- **Categóricas:** `SimpleImputer(strategy="constant", fill_value="Unknown")` y
  `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)`. Con `min_frequency=0.01` los
  vendedores poco frecuentes quedan agrupados, por lo que no se crean 1.165 columnas.
- **Historial:** `TargetEncoder` (D2), con `SimpleImputer` previo de categoría vacía.
- **Modelo:** `LinearRegression` sin parámetros (Q4); el torneo y los ajustes son la etapa 5.

## D11. Salidas

- `reports/04_features_tabla.csv`: una fila por idea con las medias, desviaciones, `mejora / ruido`, resultado hacia adelante y decisión
  (el ejemplo del curso guarda una tabla igual).
- `reports/04_features.md`: el texto con el motivo de cada decisión, las limitaciones y cualquier hallazgo. Se redacta **después** de
  ver las cifras, nunca antes (principio VII).
