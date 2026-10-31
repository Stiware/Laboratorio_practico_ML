"""Pruebas de las variables derivadas: ubicación y distancia (decisión 6) y categorías raras (decisión 8)."""
import numpy as np
import pandas as pd

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
