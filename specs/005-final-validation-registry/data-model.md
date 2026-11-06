# Data Model: Validación final y registro (etapas 7 y 8)

## Entradas

| Entidad | Origen | Filas | Uso |
|---|---|---:|---|
| Entrenamiento | `data/processed/train.parquet` + `add_features` | 76.924 | ajustar el modelo final y los baselines (percentil 75, mediana) |
| Prueba | `data/processed/test.parquet` + `add_features` (una sola vez) | 19.215 | evaluar; segmentar; SHAP (muestra de 2.000) |
| Extremos | `data/processed/extreme_orders.parquet` + `add_features` | 64 | escenario pesimista |
| Parámetros | `reports/06_best_params.json` → `finalistas["LightGBM cuantil 0,75"]["parametros_elegidos"]` | — | construir el modelo final |

## Salidas de la etapa 7

| Archivo | Contenido |
|---|---|
| `models/final_pipeline.joblib` | el pipeline ajustado (preprocesador + LightGBM cuantil 0,75) |
| `reports/prueba_abierta.txt` | fecha y hora de la apertura de la prueba (una línea) |
| `reports/07_metricas_test.json` | decisiones congeladas (con fecha y hora), tabla de la prueba (`test_report`), sensibilidad, escenario con extremos, holgura y atraso, conclusión de la regla |
| `reports/07_segmentos.csv` | `segmento` (`region` o `mes`), `grupo`, `n_pedidos`, `tarde`, `cobertura`, `mae`, `holgura_media`, `atraso_medio` |
| `reports/07_shap.csv` | `variable`, `shap_abs_medio`, `shap_medio` (10 filas) |
| `reports/07_plazos_test.csv` | **no se guarda**: los plazos pedido a pedido podrían identificar pedidos; la ida y vuelta de la etapa 8 vuelve a predecir |

## Salidas de la etapa 8

| Archivo / lugar | Contenido |
|---|---|
| Unity Catalog `workspace.default.ontrack_plazo_entrega` | versión del modelo con firma, ejemplo, parámetros, métricas y etiquetas; alias `champion` |
| Experimento `ontrack-modelo-final` | ejecución `modelo-final` |
| `reports/08_registro.json` | `plataforma`, `nombre`, `version`, `alias`, `run_id`, `fecha`, `ida_y_vuelta` (`plazos_iguales`, `costo_registro`, `costo_etapa_7`, `coincide`) |

## Reglas

- La tabla de prueba se lee **una vez** para evaluar (constancia) y otra vez en la etapa 8 **solo** para la ida y vuelta.
- Ningún archivo guarda identificadores de cliente o pedido; el ejemplo de entrada no los tiene.
- Todas las cifras de `reports/07_validacion.md` y `08_registro.md` salen de estos archivos.
