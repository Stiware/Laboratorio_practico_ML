# Research: Preprocesamiento de datos de OnTrack

No quedaron dudas abiertas en el contexto técnico. Aquí se registran las decisiones de diseño y por qué se tomaron.

## D1. Formato de salida: Parquet

- **Decisión:** `train.parquet`, `test.parquet` y `extreme_orders.parquet` en `data/processed/`.
- **Motivo:** el grupo pidió Parquet porque el modelo se desplegará en Databricks. Conserva los tipos (fechas, enteros,
  vacíos) y se lee directo allí. La especificación aprobada permitía `.csv` o `.parquet`.
- **Dependencia:** `pyarrow==25.0.1` agregado a `requirements.txt`. Ya estaba instalada en el `.venv` (escritura y lectura
  de Parquet comprobadas el 2026-10-31); `pip` no está en el `.venv`, que se gestiona con `uv`.
- **Alternativa descartada:** CSV (pierde los tipos y obliga a releer las fechas).
- **Consecuencia:** el notebook y las pruebas leen y escriben con `to_parquet` / `read_parquet`.

## D2. Partición: `GroupShuffleSplit` en una sola pasada

- **Decisión:** `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` con `groups = customer_unique_id`.
- **Motivo:** es la partición por grupo que nombra la constitución; con una sola pasada el 20 % de **clientes** va a prueba.
  La proporción de filas puede variar un poco (por eso SC-003 permite 1 punto de tolerancia).
- **Alternativas:** `StratifiedGroupKFold` (pensado para validación cruzada y para clasificación; se usará en la etapa 5
  dentro del entrenamiento); partición por fecha (descartada por el grupo: la revisión por fecha se hace en la etapa 7).

## D3. Agregar ítems por pedido

- **Decisión:** `groupby("order_id")`: suma de `price`, `freight_value`, peso y volumen; conteo de ítems; vendedor con más
  ítems (empate: menor `seller_id`).
- **Volumen:** `product_length_cm × product_height_cm × product_width_cm` por ítem, luego suma. Un ítem es una unidad.
- **Peso o dimensiones vacíos:** si algún ítem del pedido no tiene dato, el valor del pedido queda vacío (no se suma
  parcialmente). *Motivo:* sumar solo lo conocido subestimaría el peso sin que nadie lo note. Medido con los datos
  reales: solo 22 de 96.478 pedidos entregados (0,02 %) tienen algún ítem sin peso válido, así que una regla más compleja
  no compensa. La etapa 4 lo imputa con la mediana de train dentro del `Pipeline`, con indicador de vacío.
  **Decidido** (el grupo delegó esta elección).
- **Para contar vendedores raros:** `aggregate_items` también guarda `main_seller_items`, los ítems del vendedor principal
  en ese pedido. Así `frequent_values` puede sumar ítems por vendedor usando solo filas de train, sin volver a leer las
  tablas originales.

## D4. Ubicación y distancia

- **Decisión:** centroide = promedio de `geolocation_lat` y `geolocation_lng` por `geolocation_zip_code_prefix`. El código
  postal del cliente sale de `customers` (por `customer_id`, es decir, de ese pedido) y el del vendedor de `sellers`.
  Distancia haversine en km con numpy, radio terrestre 6371 km.
- **Coordenadas fuera de Brasil (42 filas en 21 códigos postales, medido):** una fila cuenta como errónea si cae fuera del
  recuadro aproximado de Brasil (latitud −34 a 6, longitud −74 a −34). Para cada código postal el punto se obtiene así,
  en este orden:
  1. Promedio de las filas **buenas** del mismo código postal (26 de las 42 filas tienen otra fila buena).
  2. Si no hay ninguna, mediana de los puntos buenos de la **misma ciudad y estado** (`geolocation_city` y
     `geolocation_state`, sin tildes ni mayúsculas para que coincidan). Es la aproximación pedida por el grupo.
  3. Si tampoco hay, el código postal queda sin punto (`missing_coords`). No se usa el estado como último recurso: un
     estado es demasiado grande para que el punto sirva como distancia.
  Se usa la **mediana** de la ciudad y no el promedio, para que un punto raro no la mueva. Es una aproximación (la
  distancia queda con un error de unos pocos km); el reporte indica cuántos códigos postales se corrigieron por cada vía.
  Solo se corrigen los puntos erróneos; los buenos no se tocan.
- **Sin coordenadas:** `distance_km` vacía y `missing_coords = 1`. No se imputa aquí.
- **Alternativa descartada:** `geopy`/`haversine` (dependencia innecesaria para una fórmula de 5 líneas).

## D5. Vendedores raros

- **Decisión:** `frequent_values(train, "seller_id", weight_col="main_seller_items", min_count=10)` devuelve los vendedores
  con 10 o más ítems en train; `group_rare_categories` asigna `"otros"` a los demás, también a los que solo aparecen en test.
- **Motivo:** la lista depende de los datos, así que se aprende con train y se aplica igual a test (constitución, III).

## D6. Columnas que se conservan

- **Se conservan como entradas posibles:** `price_sum`, `freight_sum`, `n_items`, `weight_g_sum`, `volume_cm3_sum`,
  `product_category_name` (del producto principal), `seller_id` agrupado, `seller_state`, `customer_state`,
  `distance_km`, `missing_coords`, `order_estimated_delivery_date` y `order_purchase_timestamp` (se conocen al comprar).
- **Se conservan solo para trabajar, fuera del modelo:** `order_id`, `customer_unique_id` (agrupa la partición) y
  `order_approved_at` (se mide con y sin ella en la etapa 4).
- **Fuera:** `order_delivered_*`, `order_status`, `review_*`, `customer_id`, ciudad, códigos postales.
- **Categoría del pedido:** la del producto del vendedor principal con más ítems; si hay empate, el primer producto por
  `order_item_id`. Es un detalle que el EDA no fijó; es la elección más simple y se anota en el reporte.

## D7. Verificación de no fuga y de datos originales

- La prueba de partición compara conjuntos de `customer_unique_id`; una segunda prueba parte por fila a propósito y debe
  mostrar solapamiento (es el contraste que pide la especificación).
- Para `data/raw/`: una prueba (o paso del notebook) compara tamaño y fecha de modificación antes y después, y `git status`
  confirma que no hay cambios.

## D8. Cifras

- Las cifras del EDA (96.470, 64, 9.803, 478…) son **referencia** para comparar, no valores a forzar. El reporte muestra lo
  que produzca el código y señala cualquier diferencia. Las del EDA se calcularon antes del filtro de periodo, por lo que
  se espera que los conteos finales sean menores.
