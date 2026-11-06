"""Pruebas del registro del modelo (etapa 8, `src/olistbr/tracking/registry.py`).

Usan MLflow local en una carpeta temporal y un modelo de juguete: nada llega a Databricks.
"""
import mlflow
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor

from olistbr.config import TOURNAMENT_GROUPS
from olistbr.models.train import build_pipeline
from olistbr.tracking import mlflow_utils as mu
from olistbr.tracking.registry import input_example, load_champion, register_final_model, set_champion


@pytest.fixture
def local_mlflow(tmp_path):
    """MLflow local en `tmp_path`, con los archivos del modelo también ahí (no en `mlruns/` del repo)."""
    mu.configure_mlflow(force_local=True, local_db=tmp_path / "mlflow.db", env_file=tmp_path / "no_existe.env")
    mlflow.create_experiment("prueba-registro", artifact_location=(tmp_path / "artefactos").as_uri())
    mlflow.set_experiment("prueba-registro")
    return tmp_path


def _toy(quantile_frame):
    df = quantile_frame(300, 9)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    pipe = build_pipeline(TOURNAMENT_GROUPS, model=LGBMRegressor(n_estimators=20, verbose=-1, random_state=0), scale=False).fit(X, y)
    return df, X, pipe


def test_input_example_predicts_like_the_original_rows(quantile_frame):
    """Pasar los enteros a decimales (por ejemplo, el día de la semana) no cambia la predicción."""
    df, X, pipe = _toy(quantile_frame)
    assert np.allclose(pipe.predict(input_example(df, n=20)), pipe.predict(X.head(20)))


def test_input_example_has_no_identifiers(quantile_frame):
    df, _, _ = _toy(quantile_frame)
    df = df.assign(promised_days=10.0)
    example = input_example(df, n=3)
    assert len(example) == 3 and len(example.columns) == 25
    forbidden = {"customer_unique_id", "order_id", "order_estimated_delivery_date", "promised_days", "dias_entrega"}
    assert not forbidden & set(example.columns)
    # sin columnas de enteros: así la firma acepta datos faltantes (la app permite «No sé»)
    assert not [c for c in example.columns if pd.api.types.is_integer_dtype(example[c])]
    with pytest.raises(ValueError, match="distance_km"):
        input_example(df.drop(columns=["distance_km"]))


def test_register_creates_a_version(quantile_frame, local_mlflow):
    df, _, pipe = _toy(quantile_frame)
    info = register_final_model(pipe, {"modelo": "juguete"}, {"costo": 9.1}, input_example(df), "ontrack_prueba", tags={"etapa": "08_registro"})
    assert info["nombre"] == "ontrack_prueba" and info["version"] == 1
    run = mlflow.get_run(info["run_id"])
    assert run.data.params["modelo"] == "juguete" and run.data.metrics["costo"] == 9.1 and run.data.tags["etapa"] == "08_registro"


def test_champion_points_to_latest_version(quantile_frame, local_mlflow):
    df, _, pipe = _toy(quantile_frame)
    for _ in range(2):
        register_final_model(pipe, {}, {}, input_example(df), "ontrack_prueba")
    assert set_champion("ontrack_prueba") == 2
    version = mlflow.MlflowClient().get_model_version_by_alias("ontrack_prueba", "champion")
    assert int(version.version) == 2
    with pytest.raises(ValueError, match="no hay versiones"):
        set_champion("no_existe")


def test_round_trip_predicts_the_same(quantile_frame, local_mlflow):
    df, X, pipe = _toy(quantile_frame)
    register_final_model(pipe, {}, {}, input_example(df), "ontrack_prueba")
    set_champion("ontrack_prueba")
    recovered = load_champion("ontrack_prueba")
    assert np.allclose(recovered.predict(X), pipe.predict(X))


def test_register_saves_signature_and_example(quantile_frame, local_mlflow):
    df, _, pipe = _toy(quantile_frame)
    info = register_final_model(pipe, {}, {}, input_example(df), "ontrack_prueba")
    model_info = mlflow.models.get_model_info(f"models:/ontrack_prueba/{info['version']}")
    assert model_info.signature is not None
    assert len(model_info.signature.inputs.input_names()) == 25


def test_signature_accepts_missing_values(quantile_frame, local_mlflow):
    """Con la firma registrada, el modelo cargado como pyfunc (que sí aplica la firma) acepta un dato faltante («No sé»)."""
    df, X, pipe = _toy(quantile_frame)
    register_final_model(pipe, {}, {}, input_example(df), "ontrack_prueba")
    set_champion("ontrack_prueba")
    pyfunc = mlflow.pyfunc.load_model("models:/ontrack_prueba@champion")
    row = input_example(df, n=1)
    row.loc[0, "weight_g_sum"] = np.nan
    row.loc[0, "n_items"] = np.nan
    assert np.isfinite(np.asarray(pyfunc.predict(row), dtype=float)).all()


def test_set_champion_can_point_to_a_given_version(quantile_frame, local_mlflow):
    df, _, pipe = _toy(quantile_frame)
    for _ in range(2):
        register_final_model(pipe, {}, {}, input_example(df), "ontrack_prueba")
    assert set_champion("ontrack_prueba", version=1) == 1
    assert int(mlflow.MlflowClient().get_model_version_by_alias("ontrack_prueba", "champion").version) == 1
    with pytest.raises(ValueError, match="no existe"):
        set_champion("ontrack_prueba", version=7)
