# Etapa 3 · Preprocesamiento (limpieza y partición)

> Todas las cifras salen del código (`notebooks/03_preprocesamiento.ipynb`, ejecutado el 2026-10-31 con la semilla 42).
> Las del EDA se citan solo como referencia. Reglas aprobadas: `especificaciones/03-preprocesamiento.md`.
> Código: `src/olistbr/data/` · Pruebas: `tests/` · Plan: `specs/001-preprocesamiento-datos/plan.md`.

## Resumen

Los 9 CSV de Olist se convierten en una tabla de **96.139 pedidos entregados** (una fila por pedido), con `dias_entrega`
como objetivo. Se parte **por cliente**: 76.924 pedidos para entrenamiento y 19.215 (20,0 %) para prueba, que queda cerrada
hasta la etapa 7. Otros **64 pedidos** que tardaron más de 100 días se guardan aparte. Los tres resultados se guardan en
Parquet en `data/processed/` (no se versionan; se regeneran con el notebook, y dos ejecuciones seguidas dan archivos
idénticos).

## Filas antes y después de cada regla

| Paso | Regla (decisión del EDA) | Antes | Después | Quitadas |
|---|---|---:|---:|---:|
| 1 | Solo pedidos `delivered` con fecha de entrega, y se construye `dias_entrega` | 99.441 | 96.470 | 2.971 |
| 9 | Compras entre 2017-01-01 y 2018-08-31 | 96.470 | 96.203 | 267 |
| 2 | Pedidos de más de 100 días se separan | 96.203 | 96.139 | 64 |
| 3 | Una fila por pedido al agregar ítems | 96.139 | 96.139 | 0 |

Las reglas 4, 6, 7, 8 y 10 no quitan filas (quitan o crean columnas, o cambian valores), por eso no aparecen en esta tabla:
tienen su cifra en la tabla de decisiones de abajo y su prueba en `tests/`. La partición (regla 5) se ve en el resumen.

- **Diferencia con el EDA (paso 1):** el EDA esperaba 2.965 pedidos sin fecha de entrega y quedarse con 96.470; el código
  llega a 96.470, y las 2.971 filas quitadas son esos 2.965 más 6 pedidos `canceled` que sí traen fecha de entrega
  (verificado con los datos). Salen porque la decisión 1 exige estado `delivered`. No cambia la cifra final.
- **Paso 2:** los 64 extremos coinciden con los del EDA y todos caen dentro del periodo. Van a `extreme_orders.parquet`
  (máximo 209,6 días).
- **Paso 3:** ningún pedido se pierde al unir con los ítems (los 775 pedidos sin ítems del EDA no estaban entre los
  entregados, como se había previsto).

## Decisiones con su motivo

| # | Decisión | Cifra real | Motivo |
|---|---|---|---|
| 1 | Quedarse con entregados que tengan fecha de entrega | 96.470 pedidos | Sin fecha no hay objetivo que aprender |
| 2 | Excluir más de 100 días y reportarlos aparte | 64 pedidos (0,07 %) | Distorsionan MAE y RMSE; no es lo que la app debe predecir |
| 3 | Una fila por pedido: sumar precio, flete, peso y volumen; contar ítems; vendedor principal | 9.600 pedidos con más de un ítem; 1.272 con más de un vendedor | La unidad de análisis es el pedido |
| 4 | Quitar lo que se conoce después de comprar (`order_delivered_*`, `order_status`, `review_*`) | 0 de esas columnas en la salida | Evitar que el modelo "vea el futuro" |
| 5 | Partir por `customer_unique_id` | 2.789 clientes con más de un pedido (máximo 15) | Un cliente en los dos lados infla la métrica |
| 6 | Distancia vendedor–cliente en km con centroides por código postal | 480 pedidos (0,5 %) sin distancia | Es la variable más fuerte (correlación 0,39 en el EDA) |
| 7 | Productos: categoría vacía → `desconocida`, peso 0 → vacío | 1.350 pedidos con categoría desconocida; 22 con peso vacío | No perder pedidos ni imputar con información de la prueba |
| 8 | Vendedores con menos de 10 ítems en entrenamiento → `otros` | 1.164 vendedores frecuentes; 6,8 % de los pedidos de entrenamiento quedan en `otros` | Demasiadas categorías para pocos datos |
| 9 | Periodo 2017-01 a 2018-08 | 267 pedidos fuera | Los meses casi vacíos no representan la operación |
| 10 | No cargar reseñas ni identificadores como variables | 0 columnas `review_*` | Datos personales potenciales y fuga |

### Decisiones nuevas tomadas al planear (aprobadas por el grupo el 2026-10-31)

- **Peso y volumen vacíos si falta algún ítem:** 22 pedidos con peso vacío y 16 con volumen vacío (0,02 % y 0,02 %). La
  etapa 4 los completa con la mediana del entrenamiento. Sumar solo lo conocido habría subestimado el peso sin avisar.
- **Coordenadas erróneas:** el EDA encontró 42 filas fuera de Brasil, que son 21 códigos postales. Resultado: **16** se
  corrigieron con el promedio de sus otras filas buenas, **1** con la mediana de su ciudad y estado, y **4** quedaron sin
  punto. En total solo 5 de los 19.015 códigos postales usan una vía distinta al promedio propio.
- **Vendedores raros por ítems del vendedor principal**, contados solo con entrenamiento.
- **Parquet** en lugar de CSV, porque el modelo se desplegará en Databricks.

## Vacíos que quedan para la etapa 4 (dentro del `Pipeline`)

| Columna | Vacíos | Qué significa | Tratamiento en la etapa 4 |
|---|---:|---|---|
| `order_approved_at` | 14 | el pedido no se aprobó / no se anotó | se mide con y sin esta columna |
| `weight_g_sum` | 22 | algún producto sin peso válido | mediana del entrenamiento + indicador |
| `volume_cm3_sum` | 16 | algún producto sin dimensiones | mediana del entrenamiento + indicador |
| `distance_km` | 480 | falta el punto de algún código postal (`missing_coords = 1`) | mediana del entrenamiento |

## Objetivo, para tener presente

`dias_entrega` en la tabla final: mediana 10,2; media 12,5; p99 45,4 días; mínimo 0,53 (no hay valores negativos).

## Riesgos y cosas que conviene saber

- **Los extremos también salen de la prueba.** Se quitan antes de partir, así que la métrica final no los incluye; por eso se
  reportan aparte (64 pedidos, `extreme_orders.parquet`).
- **Vendedores que solo aparecen en la prueba** pasan a `otros`, porque la lista de frecuentes se aprende solo con
  entrenamiento. Es lo correcto; no se muestran estadísticas de la prueba para no mirarla antes de la etapa 7.
- **Clientes de la prueba:** 18.607 clientes. La proporción es del 20 % de las filas con una tolerancia de menos de 0,1 puntos.
- **Ubicación del cliente:** se usa la del pedido (`customer_id`), no la de la persona (39 personas tienen más de un estado
  en el EDA).
- **Archivos de datos (resuelto con la guía):** `kit/guia-eda-y-preprocesamiento.md`, sección 4.4, dice que los datos limpios de
  `data/processed/` «se regeneran con el notebook 03: **no se versionan**», y el ejemplo CreditSight también. Se agregó
  `data/processed/*` al `.gitignore` (con excepción de `.gitkeep`), igual que `data/raw/`. Así el conjunto de prueba cerrado no
  puede subirse por accidente.

## Revisiones

### Auditoría de fuga (subagente `auditor-de-fuga`, 2026-10-31)

**Veredicto:** sin fuga grave. La partición es por cliente, la lista de vendedores frecuentes se aprende solo con
entrenamiento, nada ajusta medias ni medianas antes de partir, no se cargan reseñas y la prueba no se usa para decidir.
El auditor solo leyó el código (no ejecutó pruebas ni abrió los Parquet); las pruebas y las comprobaciones del notebook
las ejecutamos nosotros (44 pruebas en verde y notebook sin errores).

| # | Hallazgo del auditor | Gravedad | Qué hacemos |
|---|---|---|---|
| 1 | Los pedidos de más de 100 días también salen de la prueba, así que el MAE de la prueba saldrá algo mejor que el real | Media (métrica optimista, no fuga) | Decisión aprobada. **Etapa 7:** reportar el MAE de la prueba normal y también el MAE incluyendo `extreme_orders.parquet` |
| 2 | `order_estimated_delivery_date` queda como columna de entrada | Baja | El EDA la clasificó como «se conoce al comprar» (la calcula Olist en la compra). Se conserva y se anota aquí; si en la etapa 4 resulta sospechosa por ser demasiado fuerte, se mide con y sin ella, igual que `order_approved_at` |
| 3 | `order_approved_at` sigue en `train` y `test` | Baja | Es lo aprobado (regla 4): se conserva aparte. **Etapa 4:** excluirla de las variables de entrada por defecto y medir con y sin ella |

Otras notas del auditor: `customer_unique_id`, `order_id` y `order_purchase_timestamp` siguen en las tablas guardadas.
Los dos identificadores **no** son variables del modelo; la fecha de compra se decide en la etapa 4. La prueba es aleatoria por
cliente y no por fecha: la revisión por los últimos meses queda para la etapa 7 (decisión aprobada).

### Revisión de código (subagente `revisor-de-codigo-ia`)

**Veredicto:** aceptar con cambios. El código hace lo que dice el plan, las pruebas pasan y no se encontró fuga. El revisor
comprobó con los Parquet que las cifras de este reporte salen de los datos (9.600 pedidos con más de un ítem, 1.272 con más de
un vendedor, 1.350 categoría desconocida, 22 y 16 con peso y volumen vacíos, 480 sin distancia, 2.789 clientes con más de un
pedido, p99 de 45,4 días: todas coinciden). Para hacerlo abrió también `test.parquet` solo para verificar que no hay clientes
repetidos; no se tomó ninguna decisión con ese archivo.

| # | Hallazgo | Qué se hizo |
|---|---|---|
| 1 | Los Parquet no están ignorados por Git | **Corregido:** la guía del curso dice que no se versionan; se ignoran en `.gitignore` (ver «Riesgos») |
| 2 | La tabla de filas antes y después solo cubre 4 reglas | Se aclaró arriba que las demás no quitan filas; sus cifras están en la tabla de decisiones |
| 3 | El orden «partir y luego agrupar vendedores» vivía solo en el notebook | **Corregido:** nueva función `prepare_train_test` en `src/olistbr/data/split.py` con dos pruebas (un vendedor frecuente solo en la prueba no entra a la lista); el notebook solo la llama. Los Parquet salieron idénticos antes y después del cambio |
| 4 | Lógica y una afirmación sin respaldo en el notebook | **Corregido:** el texto de `order_approved_at` ahora dice «la causa no se verificó». Se deja la comprobación de reglas y el recálculo de centroides como están: son solo para mostrar y comprobar, no transforman datos |
| 5 | El contraste «partir por fila» no probaba la función real | **Corregido:** `test_no_overlap_check_fails_if_split_function_splits_by_row` cambia por dentro el separador de `split_by_group` a uno por fila y comprueba que entonces sí hay clientes repetidos |
| 6 | El notebook citaba cifras del EDA (2.997 clientes, 17 veces) | **Corregido:** ahora dice 2.789 y 15, las de la tabla final |
| 7 | Casos borde sin prueba | **Cubiertos:** distancia con dato vacío, puntos opuestos de la Tierra (`np.clip`), filas conservadas en `add_distance`, tablas vacías, fecha de entrega anterior a la compra (sale negativa y el notebook fallaría) y traducción repetida (ahora se deduplica) |
| 8 | Docstring de `features/categories.py` mezclaba cifras del EDA con las del conteo actual | **Corregido:** aclara que el conteo es distinto (solo entrenamiento y ítems del vendedor principal) |
| 9 | El notebook imprimía estadísticas de la prueba | **Corregido:** ya no imprime nada de la prueba salvo el tamaño |
| 10 | Higiene: casillas del notebook, imports de `conftest.py` | **Corregido** (casillas del notebook y especificación 03 marcadas con lo verificado) |

Preguntas para quien no escribió el código (para la puesta en común): (1) ¿por qué `frequent_values` se llama solo con
entrenamiento y qué cambiaría en la métrica si se usara toda la tabla? (2) `order_estimated_delivery_date` correlaciona ~0,4
con el objetivo: ¿por qué no es fuga y cómo se comprobará en la etapa 4 que el modelo no solo copia la fecha prometida?
(3) Si alguien cambiara `split_by_group` para partir por fila, ¿qué prueba falla y por qué?

## Estructura del código (reorganizada siguiendo el ejemplo CreditSight)

A pedido del grupo, el código se ordenó como en `material-del-curso/ejemplo-creditsight/` (criterio de repositorio modular de
la rúbrica: `src/` con la lógica, notebooks que solo orquestan, pruebas por módulo). **No cambió ningún resultado**: los tres
Parquet salen con los mismos hashes y las salidas de los notebooks 01, 02 y 03 son idénticas.

| Antes | Ahora |
|---|---|
| Constantes en `data/__init__.py`; rutas calculadas dentro de cada notebook | `src/olistbr/config.py` (rutas, columnas, reglas aprobadas, costo del error); `data/__init__.py` vacío |
| `data/loading.py`, `data/cleaning.py`, `data/pipeline.py` | `data/cleaning.py` (carga, reglas y `build_orders_table`) |
| `data/geo.py`, `data/categories.py` | `features/geo.py`, `features/categories.py` |
| `data/split.py` con la verificación solo en pruebas y notebook | `data/split.py` con `assert_disjoint_groups`; `split_by_group` falla sola si hubiera clientes repetidos |
| Funciones de costo 3 a 1 definidas dentro del notebook 01 | `models/metrics.py` (`delivery_cost`, `mean_delivery_cost`, `mae_days`, `best_fixed_deadline`) con 7 pruebas |
| 6 archivos de prueba | `test_cleaning.py`, `test_features.py`, `test_split.py`, `test_metrics.py` |

Se dejó fuera de `src/` el análisis exploratorio del notebook 02 (EDA) a propósito: es exploración, sus cifras ya están
validadas por el grupo y repetirlo en funciones no aporta. Solo cambió de dónde toma la ruta de `data/raw/`.
