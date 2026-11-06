# Contrato: inferencia y app (etapa 9)

## `src/olistbr/inference/schema.py`

| Firma | Devuelve | Pruebas |
|---|---|---|
| `FORM_FIELDS` | lista ordenada de los 11 campos: `order_purchase_timestamp`, `seller_state`, `customer_state`, `distance_km`, `product_category_name`, `n_items`, `main_seller_items`, `price_sum`, `freight_sum`, `weight_g_sum`, `volume_cm3_sum` | `test_form_has_the_eleven_fields` |
| `VALID_RANGES` | rangos posibles de FR-004 | `test_valid_ranges_match_the_spec` |
| `build_schema(train, test_results=None, segments=None, path=None) -> dict` | campos, etiquetas, tipos, estados y categorías, rangos posibles y observados, medianas, % tarde total y por región (si se dan), título, uso responsable; guarda JSON si hay `path` | `test_build_schema_round_trip`, `test_schema_has_no_identifiers` |
| `load_schema(path) -> dict` | el esquema guardado | (en la de arriba) |

## `src/olistbr/inference/predict.py`

| Firma | Devuelve | Errores | Pruebas |
|---|---|---|---|
| `InputValidationError(ValueError)` | — | — | — |
| `load_model(path)` | el pipeline | `FileNotFoundError` claro | `test_load_model_missing_file` |
| `validate_orders(orders, schema) -> list[list[str]]` | advertencias por fila | `InputValidationError` con todos los motivos | `test_impossible_values_are_rejected`, `test_required_fields`, `test_unknown_state_rejected`, `test_seller_items_not_above_items`, `test_old_date_rejected`, `test_unusual_values_warn` |
| `order_to_features(orders) -> pd.DataFrame` | las columnas de `add_features` con vendedor «otros», `missing_coords` y categoría «desconocida» | — | `test_order_to_features_fills_seller_and_missing_coords`, `test_features_match_manual_construction` |
| `predict_deadline(orders, model, schema) -> pd.DataFrame` | `plazo_dias`, `fecha_recomendada`, `region_cliente`, `pct_tarde_region`, `advertencias`; `attrs["warnings"]` | `InputValidationError` | `test_predict_normal_order`, `test_predict_with_unknown_values`, `test_date_rounds_up`, `test_several_orders` |

## `app/streamlit_app.py`

Rutas por `ONTRACK_MODEL_PATH` y `ONTRACK_SCHEMA_PATH` (por defecto `models/`). Pruebas (`tests/test_app.py`, `AppTest`): sin archivos muestra el error actual;
con modelo y esquema de juguete muestra el formulario y el aviso; al enviar un pedido normal muestra una fecha; con un dato imposible, un error.

## `scripts/verificar_modelo.py`

`python scripts/verificar_modelo.py MODEL ENTRADA.parquet [ESPERADO.npy]`: carga el modelo, predice; sin `ESPERADO` guarda los plazos; con `ESPERADO` compara y sale con
código 1 si difieren. Imprime la versión de Python.
