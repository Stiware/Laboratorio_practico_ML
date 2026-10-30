"""El esquema de entrada se construye con los datos de entrenamiento, se guarda y se recarga igual."""
import json

import pandas as pd
import pytest

from creditsight.config import CLASSES, COST_MATRIX
from creditsight.inference.predict import load_schema
from creditsight.inference.schema import build_schema


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [20, 35, 50, 41],
            "Annual_Income": [20_000.5, 40_000.0, 90_000.0, 55_000.0],
            "Occupation": ["Doctor", "Teacher", "Doctor", None],
        }
    )


def test_schema_records_ranges_categories_and_integer_columns():
    schema = build_schema(_train(), ["Age", "Annual_Income"], ["Occupation"], CLASSES, cost_matrix=COST_MATRIX)
    assert schema["observed_range"]["Age"] == [20.0, 50.0]
    assert schema["categories"]["Occupation"] == ["Doctor", "Teacher"]  # sin vacíos, ordenadas
    assert schema["integer_columns"] == ["Age"]  # el ingreso tiene decimales
    assert schema["median"]["Annual_Income"] == pytest.approx(47_500.0)
    assert schema["cost_matrix"] == COST_MATRIX.tolist()


def test_schema_roundtrip_through_file(tmp_path):
    path = tmp_path / "input_schema.json"
    built = build_schema(
        _train(), ["Age"], ["Occupation"], CLASSES, valid_range={"Age": (14, 100)}, examples={"Joven": {"Age": 20}}, path=path
    )
    assert load_schema(path) == json.loads(json.dumps(built))  # lo guardado es lo construido
    assert load_schema(path)["valid_range"]["Age"] == [14.0, 100.0]


def test_schema_rejects_missing_columns_and_bad_cost_matrix():
    with pytest.raises(ValueError, match="Faltan columnas"):
        build_schema(_train(), ["Edad"], [], CLASSES)
    with pytest.raises(ValueError, match="cost_matrix"):
        build_schema(_train(), ["Age"], [], CLASSES, cost_matrix=[[0, 1], [1, 0]])
