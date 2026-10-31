# Modelo de datos: torneo de modelos

## Entrada

`data/processed/train.parquet` más las variables derivadas de `add_features` (etapa 4): 76.924 filas, una por pedido.

| Columna | Uso en el torneo |
|---|---|
| `dias_entrega` | **objetivo** |
| `customer_unique_id` | agrupa los pliegues y la calibración del margen; **nunca** entra al modelo |
| `order_purchase_timestamp` | solo para la prueba hacia adelante y las variables de calendario |
| 7 grupos de variables | `logs`, `same_state`, `regions`, `calendar`, `distance_interaction`, `ratios`, `multi_seller` (`TOURNAMENT_GROUPS`) |
| `customer_region` | para la revisión por región de los finalistas (no entra como variable de agrupación del modelo; ya es variable del grupo `regions`) |
| Fuera del torneo | `peak_season` (F6), `history` (F8), `promised_days` (nunca) |

Variables resultantes: 100 columnas tras el preprocesador (medido con los 7 grupos, sin F6).

## Candidatos

| Campo | Significado |
|---|---|
| `name` | nombre en la tabla (por ejemplo `LightGBM cuantil 0,75`) |
| `model` | estimador de scikit-learn con parámetros fijos |
| `family` | `base`, `A` (valor típico + margen) o `B` (percentil 75 directo) |
| `scale` | si el preprocesador escala las numéricas |

## Resultado por pliegue (`reports/05_torneo_por_pliegue.csv`)

| Columna | Tipo | Qué es |
|---|---|---|
| `modelo`, `familia` | texto | candidato y familia |
| `pliegue` | entero | 0 a 4; la partición hacia adelante se guarda con `pliegue = "adelante"` |
| `mae`, `rmse`, `r2`, `sesgo` | número | de la **predicción** del modelo (para la familia B es el percentil 75) |
| `costo` | número | costo medio 3 a 1 del **plazo recomendado** (decide el torneo) |
| `tarde`, `cobertura` | número | proporción con `real > recomendado` y su complemento |
| `margen` | número | margen sumado (0 en la familia B y las líneas base) |
| `n_ajuste`, `n_calibracion`, `n_validacion` | entero | tamaños (la familia A ajusta con el 80 %) |

## Resumen (`reports/05_torneo_resumen.csv`)

Una fila por modelo: `modelo`, `familia`, `<métrica>_media` y `<métrica>_desv` para `mae`, `rmse`, `r2`, `sesgo`, `costo`, `tarde`, `cobertura`, `margen`;
`adelante_costo`, `adelante_mae`; `posicion_costo`, `posicion_mae`; `supera_mediana`, `supera_plazo_fijo` (verdadero/falso); `empate_con_el_mejor`.

## Tiempos (`reports/05_torneo_tiempos.csv`)

`modelo`, `pliegue`, `fit_seconds`. Cambian de una ejecución a otra y no entran en la comprobación de reproducibilidad.

## Regiones (`reports/05_torneo_regiones.csv`)

Para cada finalista y cada región del cliente: `modelo`, `region`, `n_pedidos`, `mae`, `tarde`.

## Reglas de validez (se prueban)

- Los 5 pliegues cubren todas las filas una vez y no comparten clientes (ya probado en la etapa 4).
- En cada ajuste de la familia A, ningún cliente está en el ajuste y en la calibración.
- `costo` = 4 × pérdida cuantílica de 0,75 de la serie `(real, recomendado)`.
- `tarde + cobertura = 1`.
- `promised_days` y `peak_season` no están en las columnas usadas por el torneo.
- Las columnas de salida del preprocesador no tienen vacíos ni infinitos.
