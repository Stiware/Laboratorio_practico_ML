"""Ubicación: centroides por código postal y distancia vendedor-cliente (decisión 6 del EDA).

La distancia es la variable más fuerte del problema (correlación 0,39 con los días de entrega).
La tabla de geolocalización trae varias filas por código postal y algunos puntos erróneos
(fuera de Brasil), que aquí se corrigen sin inventar ubicaciones lejanas.
"""
import unicodedata

import numpy as np
import pandas as pd

from olistbr.config import BRAZIL_LAT_RANGE, BRAZIL_LNG_RANGE

EARTH_RADIUS_KM = 6371.0


def _normalize_city(series: pd.Series) -> pd.Series:
    """Minúsculas, sin tildes y sin espacios sobrantes, para que 'Santa Mária ' = 'santa maria'."""
    mapping = {
        city: unicodedata.normalize("NFKD", city).encode("ascii", "ignore").decode().lower().strip()
        for city in series.dropna().unique()
    }
    return series.map(mapping)


def zip_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Un punto (lat, lng) por código postal, corrigiendo los puntos fuera de Brasil.

    Para cada código postal, en este orden:
    1. Promedio de sus filas buenas (dentro del recuadro de Brasil). `source = "zip"`.
    2. Si no tiene ninguna, mediana de los puntos buenos de la misma ciudad y estado
       (nombres comparados sin tildes ni mayúsculas). `source = "city_state"`.
    3. Si tampoco hay, queda sin punto. `source = "none"`.

    Los puntos buenos nunca se modifican. Devuelve las columnas
    `zip_code_prefix, lat, lng, source`.
    """
    g = geolocation.rename(columns={
        "geolocation_zip_code_prefix": "zip_code_prefix",
        "geolocation_lat": "lat",
        "geolocation_lng": "lng",
        "geolocation_city": "city",
        "geolocation_state": "state",
    })[["zip_code_prefix", "lat", "lng", "city", "state"]].copy()
    g["city_key"] = _normalize_city(g["city"])

    inside = g["lat"].between(*BRAZIL_LAT_RANGE) & g["lng"].between(*BRAZIL_LNG_RANGE)
    good = g[inside]

    by_zip = good.groupby("zip_code_prefix").agg(lat=("lat", "mean"), lng=("lng", "mean")).reset_index()
    by_zip["source"] = "zip"

    by_city = (
        good.dropna(subset=["city_key"])
        .groupby(["city_key", "state"])
        .agg(lat=("lat", "median"), lng=("lng", "median"))
        .reset_index()
    )

    # Códigos postales sin ninguna fila buena: se buscan por ciudad y estado.
    no_good_row = g[~g["zip_code_prefix"].isin(by_zip["zip_code_prefix"])]
    fallback = (
        no_good_row.drop_duplicates("zip_code_prefix")[["zip_code_prefix", "city_key", "state"]]
        .merge(by_city, on=["city_key", "state"], how="left")
    )
    fallback["source"] = np.where(fallback["lat"].notna(), "city_state", "none")

    result = pd.concat(
        [by_zip, fallback[["zip_code_prefix", "lat", "lng", "source"]]], ignore_index=True
    )
    return result


def haversine_km(lat1, lng1, lat2, lng2):
    """Distancia en km sobre la superficie de la Tierra (fórmula de haversine, radio 6371 km).

    Acepta números o arreglos; si algún dato es vacío, el resultado es vacío.
    """
    lat1, lng1, lat2, lng2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lng1, lat2, lng2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # clip: el redondeo no debe dar NaN


def add_distance(df: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    """Agrega `distance_km` (cliente-vendedor) y `missing_coords` (1 si faltó algún punto).

    Usa `customer_zip_code_prefix` y `seller_zip_code_prefix` de `df`. No imputa: la distancia
    vacía se completa en la etapa 4, dentro del `Pipeline`, con datos de entrenamiento.
    """
    points = centroids[["zip_code_prefix", "lat", "lng"]]
    customer = points.rename(columns={
        "zip_code_prefix": "customer_zip_code_prefix", "lat": "customer_lat", "lng": "customer_lng",
    })
    seller = points.rename(columns={
        "zip_code_prefix": "seller_zip_code_prefix", "lat": "seller_lat", "lng": "seller_lng",
    })
    result = df.merge(customer, on="customer_zip_code_prefix", how="left")
    result = result.merge(seller, on="seller_zip_code_prefix", how="left")
    result["distance_km"] = haversine_km(
        result["customer_lat"], result["customer_lng"], result["seller_lat"], result["seller_lng"]
    )
    result["missing_coords"] = result["distance_km"].isna().astype(int)
    return result.drop(columns=["customer_lat", "customer_lng", "seller_lat", "seller_lng"])
