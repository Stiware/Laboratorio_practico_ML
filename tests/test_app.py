"""Pruebas de la app de Streamlit con `streamlit.testing.AppTest`: la ejecutan sin abrir el navegador.

Usan un modelo y un esquema de juguete en una carpeta temporal (por las variables `ONTRACK_MODEL_PATH` y
`ONTRACK_SCHEMA_PATH`): no necesitan `models/model.joblib` ni los datos del proyecto.

La app integra el diseño del prototipo OnTrack (dashboard, EDA, metodología y simulador). El
simulador es la parte que se prueba aquí porque es la que llama al modelo.
"""
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from streamlit.testing.v1 import AppTest

from olistbr.config import TOURNAMENT_GROUPS
from olistbr.features.engineering import add_features
from olistbr.inference.schema import build_schema
from olistbr.models.train import build_pipeline

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
SIMULATOR = "Simulador OnTrack"


@pytest.fixture
def toy_files(frame_factory, tmp_path, monkeypatch):
    train = frame_factory(400, 5)
    df = add_features(train)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    model = build_pipeline(TOURNAMENT_GROUPS, model=LGBMRegressor(n_estimators=20, verbose=-1, random_state=0), scale=False).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    results = {"resultado": [{"regla": "LightGBM cuantil 0,75", "tarde": 0.264}]}
    segments = pd.DataFrame({"segmento": ["region"], "grupo": ["Sudeste"], "tarde": [0.268]})
    build_schema(train, test_results=results, segments=segments, path=tmp_path / "input_schema.json")
    monkeypatch.setenv("ONTRACK_MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("ONTRACK_SCHEMA_PATH", str(tmp_path / "input_schema.json"))
    return tmp_path


def _open_simulator(at):
    at.segmented_control(key="view").set_value(SIMULATOR).run()
    return at


def _fill(at, **numbers):
    at.date_input(key="order_purchase_timestamp").set_value(date(2017, 10, 2))
    at.selectbox(key="seller_state").set_value("SP")
    at.selectbox(key="customer_state").set_value("RJ")
    for key, value in numbers.items():
        if key == "distance_km":
            at.slider(key=key).set_value(value)
        else:
            at.number_input(key=key).set_value(value)
    return at


def test_app_without_files_shows_the_missing_files_message(tmp_path, monkeypatch):
    monkeypatch.setenv("ONTRACK_MODEL_PATH", str(tmp_path / "no.joblib"))
    monkeypatch.setenv("ONTRACK_SCHEMA_PATH", str(tmp_path / "no.json"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("notebook" in e.value and "09" in e.value for e in at.error)


def test_app_shows_form_and_responsible_use(toy_files):
    at = _open_simulator(AppTest.from_file(APP, default_timeout=60).run())
    assert not at.exception
    assert any("OnTrack" in element.value for element in at.html)
    assert len(at.number_input) == 8 and len(at.selectbox) == 3 and len(at.date_input) == 2
    assert len(at.slider) == 1 and len(at.checkbox) == 7
    assert any("no una promesa" in m.value for m in at.markdown)


def test_app_recommends_a_date_for_a_normal_order(toy_files):
    at = _fill(_open_simulator(AppTest.from_file(APP, default_timeout=60).run()), distance_km=400.0)
    at.button(key="analizar").click().run()
    assert not at.exception and not at.error
    metrics = {m.label: m.value for m in at.metric}
    deadline = datetime.strptime(metrics["Fecha de entrega recomendada"], "%d/%m/%Y").date()
    assert deadline > date(2017, 10, 2)  # después de la compra
    assert metrics["Plazo recomendado"].endswith(" días") and float(metrics["Plazo recomendado"].split()[0].replace(",", ".")) > 0


def test_app_explains_an_impossible_value(toy_files):
    at = _fill(
        _open_simulator(AppTest.from_file(APP, default_timeout=60).run()),
        length_cm=150.0,
        width_cm=150.0,
        height_cm=150.0,
    )
    at.button(key="analizar").click().run()
    assert not at.exception
    assert any("fuera de lo posible" in e.value for e in at.error)
    assert not at.metric


def test_app_accepts_unknown_distance(toy_files):
    at = _open_simulator(AppTest.from_file(APP, default_timeout=60).run())
    at.checkbox(key="unk_distance_km").check()
    at.button(key="analizar").click().run()
    assert not at.exception and not at.error
    assert any("menos preciso" in w.value for w in at.warning)


def test_app_rejects_more_seller_items_than_items(toy_files):
    at = _fill(
        _open_simulator(AppTest.from_file(APP, default_timeout=60).run()),
        n_items=2,
        main_seller_items=3,
    )
    at.button(key="analizar").click().run()
    assert any("vendedor principal" in e.value for e in at.error) and not at.metric


def test_app_renders_the_three_informative_views(toy_files):
    at = AppTest.from_file(APP, default_timeout=60).run()
    for view in ("Datos & EDA", "Metodología & Flujo", "Dashboard General"):
        at.segmented_control(key="view").set_value(view).run()
        assert not at.exception
        assert not at.error
