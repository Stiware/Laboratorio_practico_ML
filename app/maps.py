"""Mapa de Google Maps para la app OnTrack.

Dibuja las capitales de los 27 estados de Brasil, coloreadas por el % de pedidos tarde de su
región (dato del esquema del modelo), y la ruta vendedor → cliente.

La llave de Google Maps se lee de `GOOGLE_MAPS_API_KEY` en el archivo `.env` (o del entorno /
`st.secrets` en Streamlit Cloud). Es una llave de navegador: se incrusta en la página del mapa,
así que debe restringirse en Google Cloud por «referente HTTP» y a la API «Maps JavaScript».
Si falta, la app no se rompe: muestra un aviso en lugar del mapa.
"""
from __future__ import annotations

import json
import os
from html import escape
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv

from theme import STATE_INFO

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Capital de cada estado (lat, lng): punto de referencia del mapa, no la ubicación real del vendedor.
STATE_COORDS: dict[str, tuple[float, float]] = {
    "AC": (-9.9747, -67.8100),
    "AL": (-9.6658, -35.7353),
    "AM": (-3.1190, -60.0217),
    "AP": (0.0349, -51.0694),
    "BA": (-12.9714, -38.5014),
    "CE": (-3.7319, -38.5267),
    "DF": (-15.7939, -47.8828),
    "ES": (-20.3155, -40.3128),
    "GO": (-16.6869, -49.2648),
    "MA": (-2.5307, -44.3068),
    "MG": (-19.9167, -43.9345),
    "MS": (-20.4697, -54.6201),
    "MT": (-15.6014, -56.0979),
    "PA": (-1.4558, -48.4902),
    "PB": (-7.1195, -34.8450),
    "PE": (-8.0476, -34.8770),
    "PI": (-5.0892, -42.8019),
    "PR": (-25.4284, -49.2733),
    "RJ": (-22.9068, -43.1729),
    "RN": (-5.7945, -35.2110),
    "RO": (-8.7612, -63.9004),
    "RR": (2.8235, -60.6758),
    "RS": (-30.0346, -51.2177),
    "SC": (-27.5954, -48.5480),
    "SE": (-10.9472, -37.0731),
    "SP": (-23.5505, -46.6333),
    "TO": (-10.1689, -48.3317),
}

# El esquema del modelo escribe la región del sur como «Sul»; el diseño la muestra como «Sur».
_REGION_ALIASES = {"Sur": "Sul"}


def get_api_key() -> str:
    """Devuelve la llave de Google Maps, o una cadena vacía si no está configurada."""
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("GOOGLE_MAPS_API_KEY", "")).strip()
        except Exception:  # sin secrets.toml
            key = ""
    return key


def haversine_km(a: str, b: str) -> float:
    """Distancia en línea recta entre las capitales de dos estados (referencia visual)."""
    (lat1, lng1), (lat2, lng2) = STATE_COORDS[a], STATE_COORDS[b]
    d_lat, d_lng = radians(lat2 - lat1), radians(lng2 - lng1)
    h = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lng / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(h))


def _risk_color(share: float | None) -> str:
    if share is None:
        return "#64748b"
    if share >= 0.28:
        return "#f43f5e"
    if share >= 0.25:
        return "#f59e0b"
    return "#10b981"


def _states_payload(region_late: dict[str, float]) -> list[dict]:
    payload = []
    for code, (lat, lng) in STATE_COORDS.items():
        name, region = STATE_INFO[code]
        share = region_late.get(_REGION_ALIASES.get(region, region))
        payload.append(
            {
                "code": code,
                "name": name,
                "region": _REGION_ALIASES.get(region, region),
                "lat": lat,
                "lng": lng,
                "late": None if share is None else round(float(share) * 100, 1),
                "color": _risk_color(share),
            }
        )
    return payload


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body, #map { height: 100%; margin: 0; background: #0d1219; font-family: Inter, system-ui, sans-serif; }
  #legend { position: absolute; left: 10px; bottom: 10px; z-index: 5; background: rgba(13,18,25,.88);
    border: 1px solid #1e293b; border-radius: 10px; padding: 8px 10px; color: #cbd5e1; font-size: 11px; line-height: 1.7; }
  #legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }
  #badge { position: absolute; left: 10px; top: 10px; z-index: 5; background: rgba(13,18,25,.9); border: 1px solid #10b981;
    border-radius: 10px; padding: 8px 12px; color: #e2e8f0; font-size: 12px; line-height: 1.6; max-width: 70%; display: none; }
  #badge b { color: #6ee7b7; }
  #msg { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; text-align: center;
    color: #cbd5e1; font-size: 13px; padding: 24px; }
  .iw { color: #0f172a; font-size: 12px; line-height: 1.5; }
  .iw b { font-size: 13px; }
</style></head>
<body>
<div id="map"></div>
<div id="legend">
  <div><i style="background:#10b981"></i>Retraso &lt; 25 %</div>
  <div><i style="background:#f59e0b"></i>25 – 28 %</div>
  <div><i style="background:#f43f5e"></i>≥ 28 % de pedidos tarde</div>
</div>
<div id="badge"></div>
<div id="msg"></div>
<script>
const STATES = __STATES__;
const ROUTE = __ROUTE__;
const BADGE = __BADGE__;
const DARK = [
  {elementType: "geometry", stylers: [{color: "#0d1219"}]},
  {elementType: "labels.text.fill", stylers: [{color: "#64748b"}]},
  {elementType: "labels.text.stroke", stylers: [{color: "#0d1219"}]},
  {featureType: "administrative", elementType: "geometry.stroke", stylers: [{color: "#1e293b"}]},
  {featureType: "poi", stylers: [{visibility: "off"}]},
  {featureType: "transit", stylers: [{visibility: "off"}]},
  {featureType: "road", elementType: "geometry", stylers: [{color: "#141b24"}]},
  {featureType: "road", elementType: "labels", stylers: [{visibility: "off"}]},
  {featureType: "water", elementType: "geometry", stylers: [{color: "#0b2a3d"}]}
];
function fail(text) {
  document.getElementById("map").style.display = "none";
  document.getElementById("legend").style.display = "none";
  const m = document.getElementById("msg"); m.style.display = "flex"; m.textContent = text;
}
window.gm_authFailure = () => fail("Google Maps rechazó la llave. Revisa GOOGLE_MAPS_API_KEY, que la API «Maps JavaScript» esté habilitada y las restricciones de referente.");
function initMap() {
  if (BADGE) { const bd = document.getElementById("badge"); bd.style.display = "block";
    BADGE.forEach((l) => { const d = document.createElement("div"); d.innerHTML = l; bd.appendChild(d); }); }
  const map = new google.maps.Map(document.getElementById("map"), {
    center: {lat: -14.2, lng: -51.9}, zoom: 4, styles: DARK, disableDefaultUI: true,
    zoomControl: true, gestureHandling: "cooperative", minZoom: 3, maxZoom: 9,
  });
  const info = new google.maps.InfoWindow();
  const bounds = new google.maps.LatLngBounds();
  const active = ROUTE ? [ROUTE.from, ROUTE.to] : [];
  STATES.forEach((s) => {
    const isActive = active.includes(s.code);
    const marker = new google.maps.Marker({
      map, position: {lat: s.lat, lng: s.lng}, title: s.name + " (" + s.code + ")",
      icon: {path: google.maps.SymbolPath.CIRCLE, scale: isActive ? 10 : 6.5, fillColor: s.color,
             fillOpacity: isActive ? 1 : .85, strokeColor: isActive ? "#ffffff" : "#0d1219", strokeWeight: isActive ? 3 : 1.5},
      zIndex: isActive ? 10 : 1,
    });
    marker.addListener("click", () => {
      const late = s.late === null ? "sin dato" : String(s.late).replace(".", ",") + " % tarde";
      info.setContent('<div class="iw"><b>' + s.name + " · " + s.code + "</b><br>Región " + s.region + "<br>" + late + "</div>");
      info.open(map, marker);
    });
    if (isActive) bounds.extend(marker.getPosition());
  });
  if (ROUTE) {
    const a = STATES.find((s) => s.code === ROUTE.from), b = STATES.find((s) => s.code === ROUTE.to);
    const path = [{lat: a.lat, lng: a.lng}, {lat: b.lat, lng: b.lng}];
    new google.maps.Polyline({map, path, geodesic: true, strokeOpacity: 0,
      icons: [{icon: {path: "M 0,-1 0,1", strokeOpacity: 1, strokeColor: "#10b981", scale: 3}, offset: "0", repeat: "14px"}]});
    new google.maps.Polyline({map, path, geodesic: true, strokeColor: "#10b981", strokeOpacity: .25, strokeWeight: 6});
    if (!bounds.isEmpty() && ROUTE.from !== ROUTE.to) {
      map.fitBounds(bounds, 70);
      // Rutas cortas: no acercar tanto que se pierda el contexto del país.
      google.maps.event.addListenerOnce(map, "idle", () => { if (map.getZoom() > 6) map.setZoom(6); });
    }
    else if (ROUTE.from === ROUTE.to) { map.setCenter({lat: a.lat, lng: a.lng}); map.setZoom(6); }
  }
}
window.initMap = initMap;
</script>
<script async src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap&loading=async&language=es"
        onerror="fail('No se pudo cargar Google Maps. Revisa la conexión a internet.')"></script>
</body></html>"""


def _build_page(
    key: str, region_late: dict[str, float], route: tuple[str, str] | None, badge: list[str] | None = None
) -> str:
    route_json = json.dumps({"from": route[0], "to": route[1]} if route else None)
    return (
        _PAGE.replace("__STATES__", json.dumps(_states_payload(region_late), ensure_ascii=False))
        .replace("__ROUTE__", route_json)
        .replace("__BADGE__", json.dumps(badge or None, ensure_ascii=False))
        .replace("__KEY__", quote(key, safe=""))
    )


def render_map(
    region_late: dict[str, float],
    route: tuple[str, str] | None = None,
    height: int = 380,
    badge: list[str] | None = None,
) -> None:
    """Muestra el mapa de Google Maps.

    `route` = (estado del vendedor, estado del cliente). `badge` = líneas de texto (HTML ya
    escapado) que se superponen al mapa, p. ej. los parámetros y el resultado del modelo.
    """
    key = get_api_key()
    if not key:
        st.html(
            '<div class="ot-map-empty"><strong>Mapa no disponible</strong>'
            "Falta la llave de Google Maps. Pégala en <code>GOOGLE_MAPS_API_KEY</code> del archivo "
            "<code>.env</code> y reinicia la app.</div>"
        )
        return
    if route and (route[0] not in STATE_COORDS or route[1] not in STATE_COORDS):
        route = None
    st.iframe(_build_page(key, region_late, route, badge), height=height)


def route_summary(seller: str, customer: str) -> str:
    """Texto corto con la ruta y su distancia de referencia (línea recta entre capitales)."""
    km = haversine_km(seller, customer)
    km_text = f"{km:,.0f}".replace(",", ".")
    return f"{escape(STATE_INFO[seller][0])} ({seller}) → {escape(STATE_INFO[customer][0])} ({customer}) · ≈ {km_text} km en línea recta"
