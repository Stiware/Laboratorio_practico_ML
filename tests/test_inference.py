"""Pruebas de la inferencia de OnTrack (etapa 9): esquema del formulario, validación y plazo recomendado.

Usan una tabla sintética con las columnas de `train.parquet` (`frame_factory` de `conftest.py`) y un LightGBM de juguete:
no leen `data/processed/` ni `models/model.joblib`. Reemplazan las pruebas de clasificación de la plantilla.
"""
import json

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from olistbr.config import OTHER_LABEL, TOURNAMENT_GROUPS, UNKNOWN_CATEGORY
from olistbr.features.engineering import add_features
from olistbr.inference.predict import (
    InputValidationError,
    load_model,
    order_to_features,
    predict_deadline,
    validate_orders,
)
from olistbr.inference.schema import FORM_FIELDS, VALID_RANGES, build_schema, load_schema
from olistbr.models.train import build_pipeline


@pytest.fixture
def train_raw(frame_factory):
    """Tabla de entrenamiento sintética (columnas de `train.parquet`)."""
    return frame_factory(400, 3)


@pytest.fixture
def model(train_raw):
    """Pipeline del torneo con un LightGBM pequeño, ajustado con la tabla sintética."""
    df = add_features(train_raw)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    return build_pipeline(TOURNAMENT_GROUPS, model=LGBMRegressor(n_estimators=30, verbose=-1, random_state=0), scale=False).fit(X, y)


@pytest.fixture
def schema(train_raw):
    results = {"resultado": [{"regla": "LightGBM cuantil 0,75", "tarde": 0.264}]}
    segments = pd.DataFrame({"segmento": ["region", "region", "mes"], "grupo": ["Sudeste", "Norte", "2018-01"], "tarde": [0.268, 0.297, 0.3]})
    return build_schema(train_raw, test_results=results, segments=segments)


def _order(**changes):
    """Un pedido normal, con valores dentro de lo observado en la tabla sintética."""
    order = {
        "order_purchase_timestamp": "2017-10-02 10:00", "seller_state": "SP", "customer_state": "RJ", "distance_km": 400.0,
        "product_category_name": "toys", "n_items": 1, "main_seller_items": 1, "price_sum": 100.0, "freight_sum": 20.0,
        "weight_g_sum": 800.0, "volume_cm3_sum": 5000.0,
    }
    order.update(changes)
    return pd.DataFrame([order])


# --- Esquema -----------------------------------------------------------------------------------------


def test_form_has_the_eleven_fields():
    assert FORM_FIELDS == [
        "order_purchase_timestamp", "seller_state", "customer_state", "distance_km", "product_category_name", "n_items",
        "main_seller_items", "price_sum", "freight_sum", "weight_g_sum", "volume_cm3_sum",
    ]


def test_valid_ranges_match_the_spec():
    assert VALID_RANGES == {
        "distance_km": (0, 4500), "price_sum": (0, 20000), "freight_sum": (0, 2000), "n_items": (1, 30), "main_seller_items": (1, 30),
        "weight_g_sum": (0, 200000), "volume_cm3_sum": (0, 1500000),
    }


def test_build_schema_round_trip(train_raw, tmp_path):
    path = tmp_path / "input_schema.json"
    results = {"resultado": [{"regla": "LightGBM cuantil 0,75", "tarde": 0.264}]}
    segments = pd.DataFrame({"segmento": ["region"], "grupo": ["Norte"], "tarde": [0.297]})
    schema = build_schema(train_raw, test_results=results, segments=segments, path=path)
    assert load_schema(path) == json.loads(json.dumps(schema))
    assert schema["fields"] == FORM_FIELDS and schema["states"] == ["MG", "RJ", "SP"]
    assert UNKNOWN_CATEGORY not in schema["categories"] and "toys" in schema["categories"]
    assert schema["late_share_total"] == pytest.approx(0.264) and schema["late_share_by_region"] == {"Norte": 0.297}
    assert schema["observed_range"]["distance_km"][0] >= 0 and set(schema["labels"]) == set(FORM_FIELDS)
    assert schema["app_title"] == "OnTrack · Fecha de entrega recomendada" and "no una promesa" in schema["responsible_use"]


def test_schema_has_no_identifiers(schema, train_raw):
    """Ni los nombres ni los VALORES de los identificadores (clientes, pedidos, vendedores) aparecen en el esquema."""
    text = json.dumps(schema)
    for column in ("customer_unique_id", "order_id", "seller_id"):
        assert f'"{column}"' not in text
        for value in train_raw[column].astype(str).unique():
            assert f'"{value}"' not in text


# --- Pedido → variables y plazo ------------------------------------------------------------------------


def test_order_to_features_fills_seller_and_missing_coords():
    orders = pd.concat([_order(), _order(distance_km=np.nan, product_category_name=None)], ignore_index=True)
    features = order_to_features(orders)
    assert (features["seller_id"] == OTHER_LABEL).all()
    assert features["missing_coords"].tolist() == [0, 1]
    assert features.loc[1, "product_category_name"] == UNKNOWN_CATEGORY
    assert features.loc[0, "same_state"] == 0 and features.loc[0, "customer_region"] == "Sudeste"
    assert features.loc[0, "purchase_dow"] == "0"  # 2017-10-02 fue lunes


def test_features_match_manual_construction(model, schema):
    """El plazo de `predict_deadline` es el mismo que da el pipeline con las columnas armadas a mano."""
    manual = pd.DataFrame([{
        "order_purchase_timestamp": pd.Timestamp("2017-10-02 10:00"), "order_estimated_delivery_date": pd.NaT, "seller_state": "SP",
        "customer_state": "RJ", "distance_km": 400.0, "product_category_name": "toys", "n_items": 1, "main_seller_items": 1,
        "price_sum": 100.0, "freight_sum": 20.0, "weight_g_sum": 800.0, "volume_cm3_sum": 5000.0, "seller_id": OTHER_LABEL, "missing_coords": 0,
    }])
    expected = model.predict(add_features(manual))[0]
    assert predict_deadline(_order(), model, schema).loc[0, "plazo_dias"] == pytest.approx(expected)


def test_predict_normal_order(model, schema):
    result = predict_deadline(_order(), model, schema)
    assert list(result.columns) == ["plazo_dias", "fecha_recomendada", "region_cliente", "pct_tarde_region", "advertencias"]
    row = result.iloc[0]
    assert row["plazo_dias"] > 0 and row["region_cliente"] == "Sudeste" and row["pct_tarde_region"] == pytest.approx(0.268)
    assert row["fecha_recomendada"] > pd.Timestamp("2017-10-02").date()


def test_date_rounds_up(model, schema, monkeypatch):
    """Compra a las 10:00 + 9,3 días = día 11 a las 17:12 → la fecha recomendada es el día 12 (nunca antes del plazo)."""
    monkeypatch.setattr(model, "predict", lambda X: np.full(len(X), 9.3))
    assert predict_deadline(_order(), model, schema).loc[0, "fecha_recomendada"] == pd.Timestamp("2017-10-12").date()
    monkeypatch.setattr(model, "predict", lambda X: np.full(len(X), 9.0))
    assert predict_deadline(_order(order_purchase_timestamp="2017-10-02 00:00"), model, schema).loc[0, "fecha_recomendada"] == pd.Timestamp("2017-10-11").date()


def test_several_orders(model, schema):
    orders = pd.concat([_order(), _order(customer_state="SP", distance_km=10.0)], ignore_index=True)
    result = predict_deadline(orders, model, schema)
    assert len(result) == 2 and result["region_cliente"].tolist() == ["Sudeste", "Sudeste"]


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook 09"):
        load_model(tmp_path / "no_existe.joblib")


# --- Validación ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("field, value", [
    ("distance_km", 9000.0), ("distance_km", -1.0), ("price_sum", 25000.0), ("freight_sum", 2500.0), ("n_items", 31),
    ("weight_g_sum", 250000.0), ("volume_cm3_sum", 2_000_000.0), ("main_seller_items", 0),
])
def test_impossible_values_are_rejected(model, schema, field, value):
    """Cada rango posible se aplica a SU campo: el mensaje nombra ese campo y dice «fuera de lo posible»."""
    with pytest.raises(InputValidationError) as error:
        predict_deadline(_order(**{field: value}), model, schema)
    message = str(error.value)
    assert "fuera de lo posible" in message and schema["labels"][field] in message
    assert "vendedor principal no pueden" not in message  # no lo dispara otra regla


def test_zero_items_is_impossible(model, schema):
    with pytest.raises(InputValidationError, match="«Número de ítems» = 0 está fuera de lo posible"):
        predict_deadline(_order(n_items=0, main_seller_items=None), model, schema)


def test_required_fields(model, schema):
    for field in ("order_purchase_timestamp", "seller_state", "customer_state"):
        with pytest.raises(InputValidationError, match="obligatorio"):
            predict_deadline(_order(**{field: None}), model, schema)


def test_unknown_state_rejected(model, schema):
    with pytest.raises(InputValidationError, match="no es un estado"):
        predict_deadline(_order(customer_state="XX"), model, schema)


def test_seller_items_not_above_items(model, schema):
    with pytest.raises(InputValidationError, match="vendedor principal"):
        predict_deadline(_order(n_items=2, main_seller_items=3), model, schema)


def test_old_date_rejected(model, schema):
    with pytest.raises(InputValidationError, match="2016"):
        predict_deadline(_order(order_purchase_timestamp="2015-05-01"), model, schema)


def test_errors_are_reported_together(model, schema):
    with pytest.raises(InputValidationError) as error:
        predict_deadline(_order(distance_km=9000.0, price_sum=-5.0), model, schema)
    assert "distancia" in str(error.value).lower() and "precio" in str(error.value).lower()


def test_unusual_values_warn(model, schema):
    observed_max = schema["observed_range"]["distance_km"][1]
    warnings = validate_orders(_order(distance_km=min(observed_max + 500, 4400), product_category_name="libros_raros",
                                      order_purchase_timestamp="2019-03-01"), schema)[0]
    text = " ".join(warnings)
    assert "poco común" in text and "categoría" in text.lower() and "periodo" in text


def test_predict_with_unknown_values(model, schema):
    result = predict_deadline(_order(distance_km=np.nan, product_category_name=None, weight_g_sum=np.nan), model, schema)
    assert np.isfinite(result.loc[0, "plazo_dias"])
    assert "menos preciso" in result.loc[0, "advertencias"]
    assert result.attrs["warnings"]


# --- Cambios posteriores a la revisión (revisor-de-codigo-ia) -----------------------------------------


def test_items_must_be_whole_numbers(model, schema):
    with pytest.raises(InputValidationError, match="número entero"):
        predict_deadline(_order(n_items=2.5), model, schema)


def test_state_must_be_text(model, schema):
    with pytest.raises(InputValidationError, match="no es un estado"):
        predict_deadline(_order(customer_state=5), model, schema)


def test_date_with_time_zone_is_accepted(model, schema):
    result = predict_deadline(_order(order_purchase_timestamp=pd.Timestamp("2017-10-02 10:00", tz="America/Sao_Paulo")), model, schema)
    assert np.isfinite(result.loc[0, "plazo_dias"])


def test_unknown_items_cannot_be_checked_against_seller_items(model, schema):
    warnings = validate_orders(_order(n_items=np.nan, main_seller_items=2), schema)[0]
    assert any("no se puede comprobar" in w for w in warnings)
