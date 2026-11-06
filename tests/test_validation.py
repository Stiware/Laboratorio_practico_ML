"""Pruebas de la validación final (etapa 7, `src/olistbr/models/validation.py`).

Usan archivos temporales, plazos inventados y modelos de juguete: ninguna lee `data/processed/`.
"""
import numpy as np
import pandas as pd
import pytest

from olistbr.models.metrics import mae_days, mean_delivery_cost
from olistbr.models.validation import (
    baseline_plans,
    bootstrap_mean,
    cost_sensitivity,
    open_test_once,
    segment_table,
    slack_and_delay,
    test_report,
    top_shap,
)

test_report.__test__ = False  # pytest no debe tratar esta función como una prueba (su nombre empieza por «test»)


# --- US1: la prueba se abre una sola vez ------------------------------------------------------------


def _parquet(tmp_path):
    path = tmp_path / "test.parquet"
    pd.DataFrame({"customer_unique_id": ["u1", "u2"], "dias_entrega": [5.0, 9.0]}).to_parquet(path)
    return path


def test_test_set_opens_only_once(tmp_path):
    path, mark = _parquet(tmp_path), tmp_path / "prueba_abierta.txt"
    first = open_test_once(path, mark)
    assert len(first) == 2 and mark.exists()
    with pytest.raises(RuntimeError, match="ya se abrió"):
        open_test_once(path, mark)


def test_mark_records_date_and_time(tmp_path):
    mark = tmp_path / "prueba_abierta.txt"
    open_test_once(_parquet(tmp_path), mark)
    stamp = pd.Timestamp(mark.read_text(encoding="utf-8").strip())
    assert abs((pd.Timestamp.now(tz=stamp.tz) - stamp).total_seconds()) < 60


# --- US1: intervalo de una media por cliente -----------------------------------------------------------


def test_bootstrap_mean_constant():
    values, groups = np.full(40, 2.5), np.repeat([f"u{i}" for i in range(20)], 2)
    assert bootstrap_mean(values, groups, n_boot=200) == pytest.approx((2.5, 2.5, 2.5))


def test_bootstrap_mean_resamples_customers():
    """Un cliente con 50 pedidos se sortea como un bloque: el intervalo es más ancho que sorteando pedidos."""
    values = np.r_[np.ones(50), np.zeros(50)]
    groups = np.array(["grande"] * 50 + [f"u{i}" for i in range(50)])
    by_customer = bootstrap_mean(values, groups, n_boot=500)
    by_row = bootstrap_mean(values, np.arange(100).astype(str), n_boot=500)
    assert by_customer[0] == pytest.approx(0.5) and by_row[0] == pytest.approx(0.5)
    assert (by_customer[2] - by_customer[1]) > (by_row[2] - by_row[1])


def test_bootstrap_mean_reproducible():
    rng = np.random.default_rng(0)
    values, groups = rng.normal(size=60), np.repeat([f"u{i}" for i in range(30)], 2)
    assert bootstrap_mean(values, groups, n_boot=200, seed=3) == bootstrap_mean(values, groups, n_boot=200, seed=3)


def test_bootstrap_mean_rejects_bad_input():
    groups = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="largo"):
        bootstrap_mean([1.0, 2.0], groups)
    with pytest.raises(ValueError, match="vací"):
        bootstrap_mean([], [])
    with pytest.raises(ValueError, match="no finit"):
        bootstrap_mean([1.0, np.nan, 2.0], groups)
    with pytest.raises(ValueError, match="sin cliente"):
        bootstrap_mean([1.0, 2.0, 3.0], np.array(["a", None, "c"], dtype=object))
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_mean([1.0, 2.0, 3.0], groups, n_boot=0)


# --- US1: baselines y tabla de la prueba -----------------------------------------------------------------


def test_baseline_plans_by_hand():
    train = pd.DataFrame({"dias_entrega": [1.0, 2.0, 3.0, 4.0, 5.0]})
    purchase = pd.to_datetime(["2018-01-01", "2018-01-10"])
    test = pd.DataFrame({
        "order_purchase_timestamp": purchase,
        "order_estimated_delivery_date": purchase + pd.to_timedelta([20, 12.5], unit="D"),
    })
    plans = baseline_plans(train, test)
    assert list(plans) == ["Plazo fijo (percentil 75)", "Siempre la mediana", "Plazo prometido por Olist"]
    assert np.allclose(plans["Plazo fijo (percentil 75)"], 4.0)
    assert np.allclose(plans["Siempre la mediana"], 3.0)
    assert np.allclose(plans["Plazo prometido por Olist"], [20.0, 12.5])
    with pytest.raises(ValueError, match="order_estimated_delivery_date"):
        baseline_plans(train, test.drop(columns=["order_estimated_delivery_date"]))


def _report_data():
    real = np.array([10.0, 12.0, 8.0, 20.0, 15.0, 9.0])
    groups = np.array(["a", "a", "b", "c", "d", "d"])
    plans = {"modelo": real + np.array([1.0, -1.0, 2.0, 0.5, -2.0, 1.0]), "fijo": np.full(6, 14.0)}
    return real, groups, plans


def test_test_report_matches_manual_metrics():
    real, groups, plans = _report_data()
    table = test_report(real, plans, groups, main_baseline="fijo", model_name="modelo", n_boot=200).set_index("regla")
    for name, plan in plans.items():
        row = table.loc[name]
        assert row["costo"] == pytest.approx(mean_delivery_cost(real, plan))
        assert row["tarde"] == pytest.approx(np.mean(real > plan))
        assert row["cobertura"] == pytest.approx(1 - np.mean(real > plan))
        assert row["mae"] == pytest.approx(mae_days(real, plan))
        assert row["sesgo"] == pytest.approx(np.mean(real - plan))
        assert row["costo_ic_inf"] <= row["costo"] <= row["costo_ic_sup"]
    model = table.loc["modelo"]
    assert model["dif_costo"] == pytest.approx(mean_delivery_cost(real, plans["modelo"]) - mean_delivery_cost(real, plans["fijo"]))
    assert np.isnan(table.loc["fijo", "dif_costo"])


def test_test_report_flags_confident_improvement():
    real = np.linspace(5, 30, 40)
    groups = np.repeat([f"u{i}" for i in range(20)], 2)
    better = test_report(real, {"m": real + 0.5, "f": real + 5.0}, groups, "f", "m", n_boot=200).set_index("regla")
    assert bool(better.loc["m", "mejora_con_confianza"]) is True
    same = test_report(real, {"m": real + 1.0, "f": real + 1.0}, groups, "f", "m", n_boot=200).set_index("regla")
    assert bool(same.loc["m", "mejora_con_confianza"]) is False
    with pytest.raises(KeyError):
        test_report(real, {"m": real}, groups, "f", "m")


# --- US2: holgura, atraso, segmentos y sensibilidad ----------------------------------------------------


def test_slack_and_delay_by_hand():
    real = np.array([10.0, 10.0, 10.0, 10.0])
    plan = np.array([12.0, 14.0, 7.0, 10.0])  # sobran 2 y 4 días, falta 3, justo a tiempo
    result = slack_and_delay(real, plan)
    assert result["pct_a_tiempo"] == pytest.approx(0.75) and result["pct_tarde"] == pytest.approx(0.25)
    assert result["holgura_media"] == pytest.approx(2.0)  # (2 + 4 + 0) / 3
    assert result["atraso_medio"] == pytest.approx(3.0)


def test_segment_table_by_hand():
    real = np.array([10.0, 10.0, 10.0, 10.0])
    plan = np.array([12.0, 8.0, 15.0, 9.0])
    table = segment_table(real, plan, {"region": np.array(["N", "N", "S", "S"]), "mes": np.array(["m1", "m2", "m1", "m2"])})
    assert list(table.columns) == ["segmento", "grupo", "n_pedidos", "tarde", "cobertura", "mae", "holgura_media", "atraso_medio"]
    north = table[(table.segmento == "region") & (table.grupo == "N")].iloc[0]
    assert north["n_pedidos"] == 2 and north["tarde"] == pytest.approx(0.5) and north["mae"] == pytest.approx(2.0)
    assert north["holgura_media"] == pytest.approx(2.0) and north["atraso_medio"] == pytest.approx(2.0)
    assert set(table[table.segmento == "mes"]["grupo"]) == {"m1", "m2"}


def test_ratio_one_is_mae():
    real, _, plans = _report_data()
    table = cost_sensitivity(real, plans, ratios=(1, 3)).set_index("regla")
    assert table.loc["modelo", "1 a 1"] == pytest.approx(mae_days(real, plans["modelo"]))
    assert table.loc["fijo", "3 a 1"] == pytest.approx(mean_delivery_cost(real, plans["fijo"]))


# --- US3: SHAP ---------------------------------------------------------------------------------------


def test_top_shap_finds_the_only_useful_feature(quantile_frame):
    """En la tabla sintética el plazo depende solo de la distancia: una variable de distancia sale primera."""
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge

    from olistbr.config import TOURNAMENT_GROUPS
    from olistbr.models.train import build_pipeline

    df = quantile_frame(500, 4)
    X, y = df.drop(columns=["dias_entrega"]), df["dias_entrega"]
    pipe = build_pipeline(TOURNAMENT_GROUPS, model=LGBMRegressor(n_estimators=40, verbose=-1, random_state=0), scale=False).fit(X, y)
    table = top_shap(pipe, X, n=5, sample=200)
    assert list(table.columns) == ["variable", "shap_abs_medio", "shap_medio"] and len(table) == 5
    assert "distance" in table.iloc[0]["variable"]
    assert table["shap_abs_medio"].is_monotonic_decreasing
    ridge = build_pipeline(TOURNAMENT_GROUPS, model=Ridge()).fit(X, y)
    with pytest.raises(TypeError):
        top_shap(ridge, X)


# --- Cambios posteriores a las revisiones (auditor-de-fuga y revisor-de-codigo-ia) -------------------


def test_coverage_interval_mirrors_late_interval():
    """La cobertura es 1 − % tarde, y su intervalo es el espejo del de % tarde (no invertido)."""
    real, groups, plans = _report_data()
    row = test_report(real, plans, groups, "fijo", "modelo", n_boot=300).set_index("regla").loc["modelo"]
    assert row["cobertura_ic_inf"] <= row["cobertura"] <= row["cobertura_ic_sup"]
    assert row["cobertura_ic_inf"] == pytest.approx(1 - row["tarde_ic_sup"])
    assert row["cobertura_ic_sup"] == pytest.approx(1 - row["tarde_ic_inf"])


def test_bootstrap_mean_changes_with_seed():
    rng = np.random.default_rng(1)
    values, groups = rng.normal(size=80), np.repeat([f"u{i}" for i in range(40)], 2)
    assert bootstrap_mean(values, groups, n_boot=200, seed=1) != bootstrap_mean(values, groups, n_boot=200, seed=2)


def test_slack_and_delay_rejects_non_finite():
    with pytest.raises(ValueError, match="no finit"):
        slack_and_delay([10.0, np.nan], [12.0, 9.0])


def test_segment_table_handles_missing_labels_and_lengths():
    real, plan = np.array([10.0, 10.0, 10.0]), np.array([12.0, 8.0, 11.0])
    table = segment_table(real, plan, {"region": np.array(["N", None, "N"], dtype=object)})
    assert set(table["grupo"]) == {"N", "sin dato"}
    with pytest.raises(ValueError, match="largo"):
        segment_table(real, plan, {"region": np.array(["N", "S"])})


def test_baseline_plans_rejects_missing_target_and_dates():
    purchase = pd.to_datetime(["2018-01-01", "2018-01-10"])
    test = pd.DataFrame({"order_purchase_timestamp": purchase, "order_estimated_delivery_date": [purchase[0] + pd.Timedelta(days=5), pd.NaT]})
    with pytest.raises(ValueError, match="dias_entrega"):
        baseline_plans(pd.DataFrame({"otra": [1.0]}), test)
    with pytest.raises(ValueError, match="fechas vacías"):
        baseline_plans(pd.DataFrame({"dias_entrega": [1.0, 2.0]}), test)
