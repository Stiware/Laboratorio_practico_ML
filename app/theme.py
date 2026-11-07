"""Piezas visuales compartidas por la app de Streamlit.

Aquí viven los tokens del diseño (estados de Brasil, categorías traducidas, ejemplos del
simulador) y funciones que devuelven HTML. La lógica de inferencia sigue en `src/olistbr`.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent / "assets"
CSS_PATH = ASSETS / "css" / "ontrack.css"

# Estado -> (nombre en español, región de Brasil). Se usan los 27 estados del esquema.
STATE_INFO: dict[str, tuple[str, str]] = {
    "AC": ("Acre", "Norte"),
    "AL": ("Alagoas", "Nordeste"),
    "AM": ("Amazonas", "Norte"),
    "AP": ("Amapá", "Norte"),
    "BA": ("Bahía", "Nordeste"),
    "CE": ("Ceará", "Nordeste"),
    "DF": ("Distrito Federal", "Centro-Oeste"),
    "ES": ("Espírito Santo", "Sudeste"),
    "GO": ("Goiás", "Centro-Oeste"),
    "MA": ("Maranhão", "Nordeste"),
    "MG": ("Minas Gerais", "Sudeste"),
    "MS": ("Mato Grosso del Sur", "Centro-Oeste"),
    "MT": ("Mato Grosso", "Centro-Oeste"),
    "PA": ("Pará", "Norte"),
    "PB": ("Paraíba", "Nordeste"),
    "PE": ("Pernambuco", "Nordeste"),
    "PI": ("Piauí", "Nordeste"),
    "PR": ("Paraná", "Sur"),
    "RJ": ("Río de Janeiro", "Sudeste"),
    "RN": ("Río Grande del Norte", "Nordeste"),
    "RO": ("Rondonia", "Norte"),
    "RR": ("Roraima", "Norte"),
    "RS": ("Río Grande del Sur", "Sur"),
    "SC": ("Santa Catarina", "Sur"),
    "SE": ("Sergipe", "Nordeste"),
    "SP": ("São Paulo", "Sudeste"),
    "TO": ("Tocantins", "Norte"),
}

# Etiqueta del prototipo -> categoría real de Olist usada por el modelo.
CATEGORY_OPTIONS: list[tuple[str, str]] = [
    ("Belleza y Salud", "health_beauty"),
    ("Informática y Accesorios", "computers_accessories"),
    ("Cama, Mesa y Baño", "bed_bath_table"),
    ("Deportes y Ocio", "sports_leisure"),
    ("Muebles y Decoración", "furniture_decor"),
    ("Telefonía y Móviles", "telephony"),
    ("Herramientas y Jardín", "garden_tools"),
    ("Relojes y Regalos", "watches_gifts"),
    ("Automotriz y Repuestos", "auto"),
    ("Juguetes y Bebés", "toys"),
    ("Libros y Papelería", "books_general_interest"),
    ("Otra / no la conozco", None),
]
CATEGORY_LABELS = [label for label, _ in CATEGORY_OPTIONS]
CATEGORY_BY_LABEL = dict(CATEGORY_OPTIONS)

# Ejemplos del simulador, adaptados al caso Olist y al periodo de entrenamiento.
EXAMPLES = {
    "SP Local": {
        "seller_state": "SP",
        "customer_state": "SP",
        "distance_km": 85,
        "weight_g_sum": 650,
        "length_cm": 20,
        "width_cm": 15,
        "height_cm": 10,
        "price_sum": 95.0,
        "freight_sum": 12.8,
        "n_items": 1,
        "main_seller_items": 1,
        "product_category_name": "Belleza y Salud",
        "order_purchase_timestamp": "2018-08-20",
        "order_estimated_delivery_date": "2018-09-05",
    },
    "SP → AM Remoto": {
        "seller_state": "SP",
        "customer_state": "AM",
        "distance_km": 2680,
        "weight_g_sum": 1800,
        "length_cm": 35,
        "width_cm": 25,
        "height_cm": 18,
        "price_sum": 420.0,
        "freight_sum": 78.4,
        "n_items": 1,
        "main_seller_items": 1,
        "product_category_name": "Informática y Accesorios",
        "order_purchase_timestamp": "2018-08-20",
        "order_estimated_delivery_date": "2018-09-20",
    },
    "Pesado": {
        "seller_state": "PR",
        "customer_state": "BA",
        "distance_km": 1950,
        "weight_g_sum": 9500,
        "length_cm": 60,
        "width_cm": 50,
        "height_cm": 40,
        "price_sum": 650.0,
        "freight_sum": 115.0,
        "n_items": 2,
        "main_seller_items": 2,
        "product_category_name": "Muebles y Decoración",
        "order_purchase_timestamp": "2018-08-20",
        "order_estimated_delivery_date": "2018-09-15",
    },
}


def inject_css() -> None:
    """Carga la tipografía y la hoja de estilos del diseño una sola vez por rerun."""
    css = CSS_PATH.read_text(encoding="utf-8")
    st.html(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
        """
    )
    st.html(f"<style>{css}</style>")


def state_label(code: str) -> str:
    """Etiqueta del selectbox: «SP — São Paulo (Sudeste)»."""
    name, region = STATE_INFO.get(code, (code, ""))
    return f"{code} — {name} ({region})" if region else f"{code} — {name}"


def hero(title: str, eyebrow: str, subtitle: str = "", chip: str = "", chip_kind: str = "") -> str:
    """Encabezado de sección con la misma jerarquía del prototipo."""
    chip_html = f'<span class="ot-chip {chip_kind}">{chip}</span>' if chip else ""
    subtitle_html = f'<p class="ot-subtitle">{subtitle}</p>' if subtitle else ""
    return (
        '<div class="ot-section-head">'
        "<div>"
        f'<div class="ot-eyebrow"><span class="ot-eyebrow-dot"></span>{eyebrow}</div>'
        f'<h2 class="ot-title">{title}</h2>'
        f"{subtitle_html}"
        "</div>"
        f"{chip_html}"
        "</div>"
    )


def kpi(label: str, value: str, note: str = "", kind: str = "") -> str:
    """Tarjeta de indicador. `kind` puede ser `accent` o `amber`."""
    value_class = f"ot-kpi-value {kind}".strip()
    note_html = f'<div class="ot-kpi-note">{note}</div>' if note else ""
    return (
        f'<div class="ot-kpi {kind}">'
        f'<div class="ot-kpi-label">{label}</div>'
        f'<div class="{value_class}">{value}</div>'
        f"{note_html}"
        "</div>"
    )


def kpi_grid(items: list[str]) -> str:
    return '<div class="ot-kpis">' + "".join(items) + "</div>"


def bar(label: str, value: str, percent: float, kind: str = "") -> str:
    """Fila de barra horizontal. `percent` va de 0 a 100."""
    width = max(0.0, min(100.0, percent))
    return (
        '<div class="ot-bar-row">'
        f'<div class="ot-bar-label"><span>{label}</span><strong>{value}</strong></div>'
        '<div class="ot-bar-track">'
        f'<div class="ot-bar-fill {kind}" style="width:{width:.1f}%"></div>'
        "</div>"
        "</div>"
    )


def bars(items: list[str]) -> str:
    return '<div class="ot-bars">' + "".join(items) + "</div>"


def list_rows(rows: list[tuple[str, str, str]]) -> str:
    """Lista de pares etiqueta/valor con color opcional (`ot-emerald`, `ot-amber`, `ot-rose`)."""
    body = "".join(
        f'<div class="ot-list-row"><span>{label}</span><strong class="{kind}">{value}</strong></div>'
        for label, value, kind in rows
    )
    return f'<div class="ot-list">{body}</div>'


def footer() -> str:
    return (
        '<div class="ot-footer">'
        "<span><strong>OnTrack Logistics</strong> · Predicción Inteligente de Tiempos de Entrega</span>"
        "<span>Modelo LightGBM cuantil 0,75 · Datos Olist 2016–2018</span>"
        "</div>"
    )


def sparkline(values: list[float]) -> str:
    """Área suave en SVG (como «Delivery volume» del prototipo) a partir de una serie de valores."""
    if len(values) < 2:
        return ""
    w, h, pad = 300.0, 80.0, 6.0
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pts = [
        (pad + i * (w - 2 * pad) / (len(values) - 1), h - pad - (v - lo) / span * (h - 2 * pad))
        for i, v in enumerate(values)
    ]
    path = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        mx = (x0 + x1) / 2
        path += f" C{mx:.1f},{y0:.1f} {mx:.1f},{y1:.1f} {x1:.1f},{y1:.1f}"
    area = f"{path} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#fff" stroke="#10b981" stroke-width="2"/>' for x, y in pts)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none">'
        '<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0" stop-color="#10b981" stop-opacity=".28"/><stop offset="1" stop-color="#10b981" stop-opacity="0"/>'
        "</linearGradient></defs>"
        f'<path d="{area}" fill="url(#sg)"/><path d="{path}" fill="none" stroke="#10b981" stroke-width="2.5"/>{dots}</svg>'
    )
    # Streamlit limpia el SVG en línea; como imagen embebida se muestra sin problema.
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img class="ot-spark" alt="Pedidos por mes" src="data:image/svg+xml;base64,{encoded}">'


def vbars(items: list[tuple[str, float, str]], top: float = 1.0) -> str:
    """Barras verticales (como «Sales Overview»). items = (etiqueta, valor 0..top, texto del valor)."""
    cols = []
    for label, value, text in items:
        height = max(6.0, min(100.0, value / top * 100))
        kind = "on" if value >= 0.85 * top else "mid" if value >= 0.7 * top else ""
        cols.append(
            f'<div class="ot-vbar"><b>{text}</b><i class="{kind}" style="height:{height * 0.8:.0f}%"></i><span>{label}</span></div>'
        )
    return '<div class="ot-vbars">' + "".join(cols) + "</div>"
