"""Pruebas de las variables derivadas: ubicación y distancia (decisión 6), categorías raras (decisión 8)
y las variables nuevas de la etapa 4 (`engineering.py`)."""
import numpy as np
import pandas as pd
import pytest

import olistbr.features.engineering as eng
from olistbr.features.categories import frequent_values, group_rare_categories
from olistbr.features.geo import add_distance, haversine_km, zip_centroids


def _geolocation():
    rows = [
        # zip 1000: dos puntos buenos -> promedio
        (1000, -23.5, -46.6, "sao paulo", "SP"),
        (1000, -23.7, -46.8, "sao paulo", "SP"),
        # zip 3000: un punto bueno y uno fuera de Brasil -> solo cuenta el bueno
        (3000, -19.9, -43.9, "belo horizonte", "MG"),
        (3000, 40.0, -8.0, "belo horizonte", "MG"),
        # zip 4000: solo tiene un punto erróneo, pero su ciudad (escrita distinto) tiene puntos buenos
        (4000, 28.0, -15.0, "Santa Maria", "RJ"),
        (4100, -22.0, -43.0, "santa mária", "RJ"),
        (4200, -22.2, -43.2, "SANTA MARIA", "RJ"),
        (4300, -23.0, -44.0, "santa maria ", "RJ"),
        # zip 5000: punto erróneo y ninguna ciudad con puntos buenos -> queda sin punto
        (5000, 60.0, 10.0, "ciudad fantasma", "AM"),
    ]
    return pd.DataFrame(rows, columns=[
        "geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng",
        "geolocation_city", "geolocation_state",
    ])


def test_zip_centroids():
    result = zip_centroids(_geolocation()).set_index("zip_code_prefix")

    # un punto por código postal
    assert result.index.is_unique

    # promedio de los puntos buenos del mismo código postal
    assert abs(result.loc[1000, "lat"] - (-23.6)) < 1e-9
    assert abs(result.loc[1000, "lng"] - (-46.7)) < 1e-9
    assert result.loc[1000, "source"] == "zip"

    # el punto fuera de Brasil no cuenta si el código postal tiene otro bueno
    assert result.loc[3000, "lat"] == -19.9
    assert result.loc[3000, "source"] == "zip"

    # sin punto bueno propio: mediana de la misma ciudad y estado (nombres escritos distinto)
    assert abs(result.loc[4000, "lat"] - (-22.2)) < 1e-9  # mediana de -22.0, -22.2, -23.0 (el promedio daría -22.4)
    assert abs(result.loc[4000, "lng"] - (-43.2)) < 1e-9
    assert result.loc[4000, "source"] == "city_state"

    # ninguna vía posible: sin punto
    assert np.isnan(result.loc[5000, "lat"])
    assert result.loc[5000, "source"] == "none"

    # los puntos buenos de otros códigos postales no se tocan
    assert result.loc[4100, "lat"] == -22.0
    assert result.loc[4100, "source"] == "zip"


def test_haversine_km():
    """Referencia a mano: un grado de latitud en el ecuador mide 6371 * pi / 180 = 111,195 km."""
    one_degree = 6371 * np.pi / 180
    assert abs(haversine_km(0, 0, 1, 0) - one_degree) < 0.01
    assert abs(haversine_km(0, 0, 0, 1) - one_degree) < 0.01
    # mismo punto: distancia cero
    assert haversine_km(-23.5, -46.6, -23.5, -46.6) == 0
    # a 60 grados de latitud un grado de longitud mide aproximadamente la mitad
    assert abs(haversine_km(60, 0, 60, 1) - one_degree * 0.5) < 0.1


def test_add_distance():
    df = pd.DataFrame({
        "order_id": ["o1", "o2", "o3"],
        "customer_zip_code_prefix": [100, 100, 999],  # 999 no tiene punto
        "seller_zip_code_prefix": [200, 999, 200],
    })
    centroids = pd.DataFrame({
        "zip_code_prefix": [100, 200],
        "lat": [0.0, 1.0],
        "lng": [0.0, 0.0],
        "source": ["zip", "zip"],
    })
    result = add_distance(df, centroids).set_index("order_id")
    assert abs(result.loc["o1", "distance_km"] - 6371 * np.pi / 180) < 0.01
    assert result.loc["o1", "missing_coords"] == 0
    # falta el punto del vendedor o del cliente: distancia vacía e indicador en 1
    for order in ["o2", "o3"]:
        assert np.isnan(result.loc[order, "distance_km"])
        assert result.loc[order, "missing_coords"] == 1
    # no se agregan columnas auxiliares
    assert "lat" not in result.columns


def test_haversine_km_with_missing_value_is_missing():
    """Si falta algún dato, la distancia queda vacía (no 0 ni un número inventado)."""
    assert np.isnan(haversine_km(np.nan, 0, 1, 0))
    assert np.isnan(haversine_km(0, 0, 1, np.nan))


def test_haversine_km_antipodes_do_not_break_with_rounding():
    """Puntos opuestos de la Tierra: el redondeo no debe dar NaN (media vuelta = pi * radio)."""
    assert abs(haversine_km(0, 0, 0, 180) - np.pi * 6371) < 0.01


def test_add_distance_keeps_row_count():
    """Unir con los centroides no debe multiplicar ni perder pedidos."""
    df = pd.DataFrame({
        "order_id": ["o1", "o2"],
        "customer_zip_code_prefix": [100, 100],
        "seller_zip_code_prefix": [200, 200],
    })
    centroids = pd.DataFrame({
        "zip_code_prefix": [100, 200], "lat": [0.0, 1.0], "lng": [0.0, 0.0], "source": ["zip", "zip"],
    })
    assert len(add_distance(df, centroids)) == len(df)


def test_frequent_values():
    """10 ítems entran; 9 no. Se cuentan sumando `main_seller_items`."""
    train = pd.DataFrame({
        "seller_id": ["s1", "s1", "s2", "s2", "s3"],
        "main_seller_items": [6, 4, 5, 4, 20],
    })
    frequent = frequent_values(train, "seller_id", "main_seller_items", min_count=10)
    assert sorted(frequent) == ["s1", "s3"]  # s1 suma 10, s2 suma 9, s3 suma 20


def test_group_rare_categories():
    """Raro -> 'otros'; un vendedor que solo aparece en la prueba también -> 'otros'."""
    test = pd.DataFrame({"seller_id": ["s1", "s2", "s_nuevo"], "price_sum": [1.0, 2.0, 3.0]})
    result = group_rare_categories(test, "seller_id", frequent=["s1"])
    assert list(result["seller_id"]) == ["s1", "otros", "otros"]
    assert list(test["seller_id"]) == ["s1", "s2", "s_nuevo"]  # la entrada no cambia


def test_frequent_list_is_not_recomputed_from_test():
    """La lista aprendida en train se aplica igual a test aunque en test el vendedor sea frecuente."""
    train = pd.DataFrame({"seller_id": ["s1"], "main_seller_items": [50]})
    test = pd.DataFrame({"seller_id": ["s2"] * 30, "main_seller_items": [1] * 30})  # frecuente solo en test
    frequent = frequent_values(train, "seller_id", "main_seller_items")
    assert set(group_rare_categories(test, "seller_id", frequent)["seller_id"]) == {"otros"}


# =============================== Etapa 4: add_features (variables sin estado) ===============================


def _features(features_frame, changes=None):
    """add_features sobre la tabla de ejemplo, indexada por order_id; `changes` = {(order_id, columna): valor}."""
    frame = features_frame.copy()
    for (order_id, column), value in (changes or {}).items():
        frame.loc[frame["order_id"] == order_id, column] = value
    return eng.add_features(frame).set_index("order_id")


def test_log_features(features_frame):
    """F1: log1p de cada columna; vacío o negativo -> NaN."""
    out = _features(features_frame)
    assert out.loc["o0", "log_distance"] == pytest.approx(np.log1p(500))
    assert out.loc["o0", "log_freight"] == pytest.approx(np.log1p(20))
    assert out.loc["o0", "log_price"] == pytest.approx(np.log1p(100))
    assert out.loc["o0", "log_weight"] == pytest.approx(np.log1p(1000))
    assert out.loc["o0", "log_volume"] == pytest.approx(np.log1p(2000))
    assert out.loc["o1", "log_distance"] == 0.0 and out.loc["o1", "log_price"] == 0.0  # log1p(0) = 0
    assert np.isnan(out.loc["o2", "log_distance"])  # dato vacío
    assert np.isnan(out.loc["o6", "log_distance"]) and np.isnan(out.loc["o6", "log_price"])  # negativos


def test_same_state(features_frame):
    """F2: SP/SP -> 1, SP/RJ -> 0, estado vacío -> NaN."""
    out = _features(features_frame, {("o3", "seller_state"): np.nan})
    assert out.loc["o1", "same_state"] == 1.0
    assert out.loc["o0", "same_state"] == 0.0
    assert np.isnan(out.loc["o3", "same_state"])


def test_regions_table(features_frame):
    """F3: SP -> Sudeste, AM -> Norte, estado fuera de la tabla -> NaN."""
    out = _features(features_frame)
    assert out.loc["o0", "seller_region"] == "Sudeste" and out.loc["o0", "customer_region"] == "Sudeste"
    assert out.loc["o3", "customer_region"] == "Norte"
    assert pd.isna(out.loc["o2", "customer_region"])  # "XX" no existe en la tabla


def test_region_table_covers_27_states():
    assert len(eng.REGION_BY_STATE) == 27
    assert set(eng.REGION_BY_STATE.values()) == {"Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"}
    assert eng.REGION_BY_STATE["SP"] == "Sudeste" and eng.REGION_BY_STATE["AM"] == "Norte"
    assert eng.REGION_BY_STATE["BA"] == "Nordeste" and eng.REGION_BY_STATE["DF"] == "Centro-Oeste"
    assert eng.REGION_BY_STATE["RS"] == "Sul"


def test_calendar_features(features_frame):
    """F5: día de la semana como texto (0 = lunes); marca jueves a sábado; fecha vacía -> NaN."""
    out = _features(features_frame, {("o4", "order_purchase_timestamp"): pd.NaT})
    assert out.loc["o0", "purchase_dow"] == "0" and out.loc["o0", "is_slow_weekday"] == 0.0  # lunes
    assert out.loc["o1", "purchase_dow"] == "3" and out.loc["o1", "is_slow_weekday"] == 1.0  # jueves
    assert out.loc["o6", "is_slow_weekday"] == 1.0  # viernes
    assert out.loc["o7", "purchase_dow"] == "5" and out.loc["o7", "is_slow_weekday"] == 1.0  # sábado
    assert out.loc["o2", "purchase_dow"] == "6" and out.loc["o2", "is_slow_weekday"] == 0.0  # domingo
    assert pd.isna(out.loc["o4", "purchase_dow"]) and np.isnan(out.loc["o4", "is_slow_weekday"])


def test_black_friday_dates():
    """Black Friday = cuarto viernes de noviembre."""
    assert eng.black_friday(2017) == pd.Timestamp("2017-11-24")
    assert eng.black_friday(2018) == pd.Timestamp("2018-11-23")


def test_year_end_peak_window(features_frame):
    """F6: 1 desde el lunes de la semana de Black Friday (2017-11-20) hasta el 31 de diciembre; si no, 0."""
    out = _features(features_frame)
    assert out.loc["o2", "is_year_end_peak"] == 0.0  # 2017-11-19, un día antes
    assert out.loc["o3", "is_year_end_peak"] == 1.0  # 2017-11-20
    assert out.loc["o4", "is_year_end_peak"] == 1.0  # 2017-12-31
    assert out.loc["o5", "is_year_end_peak"] == 0.0  # 2018-01-01
    assert out.loc["o0", "is_year_end_peak"] == 0.0  # marzo


def test_distance_interaction(features_frame):
    out = _features(features_frame)
    assert out.loc["o4", "log_distance_x_same_state"] == pytest.approx(np.log1p(10) * 1.0)  # mismo estado
    assert out.loc["o0", "log_distance_x_same_state"] == 0.0  # distinto estado
    assert np.isnan(out.loc["o2", "log_distance_x_same_state"])  # distancia vacía


def test_route(features_frame):
    out = _features(features_frame, {("o3", "customer_state"): np.nan})
    assert out.loc["o0", "route"] == "SP→RJ"
    assert pd.isna(out.loc["o3", "route"])


def test_ratios_and_density(features_frame):
    out = _features(features_frame)
    assert out.loc["o0", "freight_ratio"] == pytest.approx(0.2)  # 20 / 100
    assert out.loc["o0", "density"] == pytest.approx(0.5)  # 1000 / 2000
    assert np.isnan(out.loc["o1", "freight_ratio"])  # precio 0
    assert np.isnan(out.loc["o1", "density"])  # volumen 0
    assert np.isnan(out.loc["o2", "freight_ratio"]) and np.isnan(out.loc["o2", "density"])  # vacíos


def test_multi_seller(features_frame):
    """F10: 1 si hay más ítems que los del vendedor principal."""
    out = _features(features_frame)
    assert out.loc["o0", "multi_seller"] == 1.0  # 3 ítems, 2 del principal
    assert out.loc["o1", "multi_seller"] == 0.0  # 3 y 3
    assert out.loc["o7", "multi_seller"] == 1.0  # 2 y 1


def test_promised_days(features_frame):
    """F4 (solo referencia): días entre la compra y la fecha prometida."""
    out = _features(features_frame)
    assert out.loc["o0", "promised_days"] == pytest.approx(24.0)
    assert np.isnan(out.loc["o2", "promised_days"])


def test_division_by_zero_or_missing_gives_nan_not_inf(features_frame):
    """Ninguna variable numérica derivada contiene infinito (división por cero o dato ausente -> NaN)."""
    out = eng.add_features(features_frame)
    numeric = out.select_dtypes("number").to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_add_features_does_not_mutate_input(features_frame):
    before = features_frame.copy(deep=True)
    eng.add_features(features_frame)
    pd.testing.assert_frame_equal(features_frame, before)
    assert "log_distance" not in features_frame.columns


# ====================== Etapa 4: grupos de ideas y preprocesador (dentro del Pipeline) ======================

ALL_GROUPS = list(eng.GROUPS)


def test_nine_groups_and_no_reference_column_in_them():
    assert ALL_GROUPS == [
        "logs", "same_state", "regions", "calendar", "peak_season",
        "distance_interaction", "history", "ratios", "multi_seller",
    ]


def test_feature_columns_toggles():
    base = eng.feature_columns()
    assert base.numeric == eng.BASE_NUMERIC and base.categorical == eng.BASE_CATEGORICAL
    assert base.target_encoded == []

    logs = eng.feature_columns(["logs"])
    assert "log_distance" in logs.numeric and "log_distance" not in base.numeric

    regions = eng.feature_columns(["regions"])
    assert "customer_region" in regions.categorical and "seller_region" in regions.categorical

    calendar = eng.feature_columns(["calendar"])
    assert "purchase_dow" in calendar.categorical and "is_slow_weekday" in calendar.numeric

    history = eng.feature_columns(["history"])
    assert history.target_encoded == ["seller_id", "product_category_name", "route"]

    # varios grupos: las columnas repetidas se cuentan una sola vez (log_distance y same_state están en dos)
    several = eng.feature_columns(["logs", "same_state", "distance_interaction"])
    assert len(several.numeric) == len(set(several.numeric))
    assert several.numeric.count("log_distance") == 1 and several.numeric.count("same_state") == 1
    assert "log_distance_x_same_state" in several.numeric


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="grupo"):
        eng.feature_columns(["no_existe"])


def test_promised_days_never_in_any_group_combination():
    """El plazo prometido por Olist solo se puede pedir con include_reference=True (para medirlo una vez)."""
    import itertools

    for size in range(len(ALL_GROUPS) + 1):
        for combo in itertools.combinations(ALL_GROUPS, size):
            cols = eng.feature_columns(combo)
            assert "promised_days" not in cols.numeric + cols.categorical + cols.target_encoded
    reference = eng.feature_columns(include_reference=True)
    assert "promised_days" in reference.numeric


def _training_table(frame_factory, groups, n=120):
    """Tabla de entrenamiento con las variables derivadas, y su objetivo."""
    df = eng.add_features(frame_factory(n))
    return df, df["dias_entrega"]


def test_preprocessor_handles_unseen_category_and_missing(frame_factory):
    """Un vendedor y una ruta nunca vistos, más datos vacíos, no dejan vacíos ni fallan."""
    train, y = _training_table(frame_factory, ALL_GROUPS)
    prep = eng.build_preprocessor(ALL_GROUPS).fit(train, y)
    new = train.iloc[:5].copy()
    new["seller_id"] = "vendedor_nunca_visto"
    new["product_category_name"] = "categoria_nueva"
    new["seller_state"] = "ZZ"
    new.loc[new.index[:2], ["weight_g_sum", "volume_cm3_sum", "distance_km"]] = np.nan
    new = eng.add_features(new[[c for c in new.columns if c in frame_factory(5).columns]])
    assert not np.isnan(prep.transform(new)).any()


def test_preprocessor_output_has_no_missing_or_infinite(frame_factory):
    df = eng.add_features(frame_factory(150, seed=1))
    df.loc[df.index[:10], "price_sum"] = 0.0  # razones con denominador cero -> vacío -> se imputa
    df.loc[df.index[10:20], "distance_km"] = np.nan
    df = eng.add_features(df[frame_factory(5).columns])
    prep = eng.build_preprocessor(ALL_GROUPS)
    matrix = prep.fit_transform(df, df["dias_entrega"])
    assert np.isfinite(matrix).all()


def test_history_group_uses_target_encoder(frame_factory):
    train, y = _training_table(frame_factory, ["history"])
    with_history = eng.build_preprocessor(["history"]).fit(train, y)
    without = eng.build_preprocessor([]).fit(train, y)
    names_with = [name for name, *_ in with_history.transformers_]
    names_without = [name for name, *_ in without.transformers_]
    assert "history" in names_with and "history" not in names_without
    assert with_history.transform(train).shape[1] == without.transform(train).shape[1] + 3  # una columna por variable


def test_history_does_not_use_own_target(frame_factory):
    """Una fila de entrenamiento no usa su propio objetivo en su promedio histórico (validación cruzada interna)."""
    df = eng.add_features(frame_factory(120, seed=3))
    df["seller_id"] = "s1"  # un solo vendedor, así el promedio histórico depende solo de los demás pedidos
    y = pd.Series(10.0, index=df.index)
    y.iloc[0] = 1000.0  # un pedido atípico
    prep = eng.build_preprocessor(["history"]).fit(df, y)
    matrix = prep.transform(df)  # transform (no fit_transform): usa el promedio global del vendedor
    fitted = prep.fit_transform(df, y)  # fit_transform: cada fila usa solo las demás
    seller_column = fitted[:, -3]
    others = np.delete(seller_column, 0)
    assert seller_column[0] == pytest.approx(10.0, abs=1.5)  # sin fuga ≈ 10; con su propio 1000 daría ≈ 20
    assert seller_column[0] < others.max()  # las filas de otros pliegues sí ven el 1000
    assert matrix.shape == fitted.shape


def test_validation_row_gets_the_training_average_not_its_own_target(frame_factory):
    """Una fila de validación recibe el promedio de ENTRENAMIENTO de su vendedor (`transform` no ve su objetivo)."""
    train = eng.add_features(frame_factory(150, seed=7))
    prep = eng.build_preprocessor(["history"]).fit(train, train["dias_entrega"])
    encoder = prep.named_transformers_["history"].named_steps["target"]
    categories = list(encoder.categories_[0])  # la primera columna con historial es seller_id
    expected = encoder.encodings_[0][categories.index("s1")]

    valid = eng.add_features(frame_factory(30, seed=8))
    valid["seller_id"] = "s1"
    valid["dias_entrega"] = 99999.0  # un objetivo absurdo: si se usara, el valor se dispararía
    seller_encoding = prep.transform(valid)[:, -3]
    np.testing.assert_allclose(seller_encoding, expected)


def test_infinite_inputs_give_missing_not_infinite(features_frame):
    """Un infinito de entrada (dato corrupto) tampoco produce infinitos en las variables derivadas."""
    frame = features_frame.copy()
    frame.loc[frame["order_id"] == "o0", ["distance_km", "price_sum"]] = np.inf
    frame.loc[frame["order_id"] == "o3", "volume_cm3_sum"] = np.inf
    out = eng.add_features(frame).set_index("order_id")
    assert np.isnan(out.loc["o0", "log_distance"]) and np.isnan(out.loc["o0", "freight_ratio"])
    assert np.isnan(out.loc["o3", "density"])
    derived = [c for c in out.columns if c not in features_frame.columns]  # solo las columnas nuevas (las de entrada conservan su inf)
    assert not np.isinf(out[derived].select_dtypes("number").to_numpy(dtype=float)).any()
