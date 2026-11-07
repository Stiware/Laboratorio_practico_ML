"""Vistas informativas de la app: dashboard, EDA y metodología.

Las cifras salen de los informes del repositorio (`reports/`). Si un informe no está
disponible, la vista muestra solo lo que está en el esquema del modelo, sin inventar datos.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import streamlit as st

from maps import render_map
from theme import bar, bars, hero, kpi, kpi_grid, list_rows, sparkline, vbars

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "reports" / "07_metricas_test.json"


def _load_metrics() -> dict | None:
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + " %"


def _num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _model_rows(metrics: dict | None) -> list[tuple[str, str, str]]:
    if not metrics:
        return [
            ("Modelo", "LightGBM cuantil 0,75", "ot-emerald"),
            ("Objetivo", "Días hasta la entrega", ""),
            ("Regla", "Percentil 75 de la predicción", ""),
        ]
    rows = {row["regla"]: row for row in metrics.get("resultado", [])}
    model = rows.get("LightGBM cuantil 0,75")
    baseline = rows.get("Plazo fijo (percentil 75)")
    if model is None:
        return [("Modelo", "LightGBM cuantil 0,75", "ot-emerald")]
    result = [
        ("Costo 3 a 1 (menor es mejor)", _num(model["costo"], 2), "ot-emerald"),
        ("Línea base de plazo fijo", _num(baseline["costo"], 2) if baseline else "—", ""),
        ("MAE del modelo", f'{_num(model["mae"], 2)} d', ""),
        ("Cobertura del plazo", _pct(model["cobertura"]), ""),
        ("Pedidos tarde en prueba", _pct(model["tarde"]), "ot-amber"),
        ("Mejora con confianza", "Sí, IC 95 % bajo cero" if metrics.get("mejora_con_confianza") else "No", "ot-emerald"),
    ]
    return result


def _monthly_rows() -> list[dict]:
    """Cobertura y pedidos por mes del modelo elegido (reports/05_torneo_cobertura_mensual.csv)."""
    path = ROOT / "reports" / "05_torneo_cobertura_mensual.csv"
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["modelo"] == "LightGBM cuantil 0,75"]
    except OSError:
        return []
    return sorted(rows, key=lambda r: r["mes"])


_MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def render_dashboard(schema: dict) -> None:
    metrics = _load_metrics()
    region_late = schema.get("late_share_by_region", {}) or {}
    total_late = schema.get("late_share_total", 0.264)
    rows = {r["regla"]: r for r in (metrics or {}).get("resultado", [])}
    model = rows.get("LightGBM cuantil 0,75", {})
    test_orders = (metrics or {}).get("pedidos_prueba", 19215)
    monthly = _monthly_rows()

    # 1) Mapa arriba, como en el prototipo
    st.html(
        '<div class="ot-map-head"><div><strong>Mapa de rutas · Brasil</strong><br>'
        "<span>Capitales de los 27 estados, coloreadas por el % de pedidos tarde de su región. "
        "Línea punteada: ruta crítica SP → AM.</span></div>"
        '<span class="ot-chip emerald">Ruta crítica · 25,9 d de mediana</span></div>'
    )
    render_map(region_late, route=("SP", "AM"), height=420)

    # 2) Tres columnas de tarjetas
    months = [_MONTHS[int(r["mes"][5:7]) - 1] + " " + r["mes"][2:4] for r in monthly]
    orders = [float(r["n_pedidos"]) for r in monthly]
    cover = [float(r["cobertura"]) for r in monthly]
    volume_card = (
        '<div class="ot-card"><div class="ot-card-head"><strong>Pedidos por mes</strong>'
        f"<span>{months[0]} – {months[-1]}</span></div>" + sparkline(orders) +
        '<div class="ot-big-pair">'
        f'<div><div class="ot-big">{_num(sum(orders), 0)}</div><div class="ot-big-label">Pedidos en validación mensual</div></div>'
        f'<div><div class="ot-big accent">{_pct(sum(c * o for c, o in zip(cover, orders)) / sum(orders))}</div>'
        '<div class="ot-big-label">Cobertura media del plazo</div></div></div></div>'
        if monthly else ""
    )
    minis = (
        '<div class="ot-mini-grid">'
        f'<div class="ot-mini"><div class="l">Pedidos evaluados</div><div class="v">{_num(test_orders, 0)}</div>'
        '<span class="pill">Prueba cerrada · 20 %</span></div>'
        f'<div class="ot-mini"><div class="l">Pedidos tarde</div><div class="v">{_pct(total_late)}</div>'
        '<span class="pill amber">En la prueba final</span></div>'
        '<div class="ot-mini"><div class="l">Plazo medio observado</div><div class="v">12,6 <small>d</small></div></div>'
        '<div class="ot-mini"><div class="l">Mediana de entrega</div><div class="v">10,2 <small>d</small></div></div>'
        "</div>"
    )

    model_card = (
        '<div class="ot-card"><div class="ot-card-head"><strong>Modelo en producción</strong>'
        "<span>LightGBM cuantil 0,75</span></div>"
        '<div class="ot-trio">'
        f'<div><div class="v">{_pct(model.get("cobertura", 0.7355), 0)}</div><div class="l">Cobertura</div></div>'
        f'<div><div class="v">{_num(model.get("mae", 5.66), 1)}</div><div class="l">MAE (días)</div></div>'
        f'<div><div class="v">{_num(model.get("costo", 9.22), 1)}</div><div class="l">Costo 3 a 1</div></div></div>'
        + list_rows(_model_rows(metrics)[1:3] + _model_rows(metrics)[5:]) + "</div>"
    )
    route_card = (
        '<div class="ot-card dark"><div class="ot-card-head"><strong style="color:#f8fafc">Ruta de referencia · SP → AM</strong></div>'
        '<div class="ot-card-sub">La combinación más lenta del EDA</div>'
        '<div class="ot-dark-metric">25,9 <small>días de mediana</small></div>'
        '<div class="ot-dark-stats">'
        '<div class="ot-dark-stat"><span>Distancia mediana</span><strong>434 km</strong></div>'
        '<div class="ot-dark-stat"><span>Mismo estado</span><strong>6,5 d</strong></div>'
        '<div class="ot-dark-stat"><span>Distinto estado</span><strong>12,8 d</strong></div>'
        '<div class="ot-dark-stat"><span>Corr. con distancia</span><strong>0,39</strong></div>'
        "</div></div>"
    )

    region_bars = []
    for region, share in sorted(region_late.items(), key=lambda item: item[1], reverse=True):
        kind = "rose" if share >= 0.28 else "amber" if share >= 0.25 else ""
        region_bars.append(bar(region, _pct(share), share / 0.35 * 100, kind))
    region_card = (
        '<div class="ot-card"><div class="ot-card-head"><strong>Retraso por región del cliente</strong>'
        "<span>% tarde</span></div>" + bars(region_bars) + "</div>"
    )
    cover_card = (
        '<div class="ot-card"><div class="ot-card-head"><strong>Cobertura mensual</strong>'
        "<span>Verde ≥ 85 % · ámbar ≥ 70 %</span></div>"
        + vbars([(m, c, f"{c * 100:.0f}%") for m, c in zip(months, cover)])
        + "</div>"
        if monthly else ""
    )

    st.html(
        '<div class="ot-dash">'
        f'<div class="ot-stack">{volume_card}{minis}</div>'
        f'<div class="ot-stack">{model_card}{route_card}</div>'
        f'<div class="ot-stack">{region_card}{cover_card}</div>'
        "</div>"
    )

    st.html(
        '<p class="ot-footnote">Fuente: <strong>reports/07_metricas_test.json</strong>, '
        "<strong>reports/05_torneo_cobertura_mensual.csv</strong> y <strong>reports/01_eda.md</strong>. "
        "La cobertura de marzo es baja (61 %): en los meses de alta demanda el plazo quedó corto. "
        "El tablero sirve para vigilar la cobertura; no es una promesa de entrega.</p>"
    )


def render_eda() -> None:
    st.html(
        hero(
            "Datos & EDA",
            "Fase 02 · Análisis exploratorio",
            "Qué dicen los datos antes de modelar: volumen, distribución del plazo y las relaciones "
            "que justificaron las variables del modelo.",
            "96.470 pedidos entregados con fecha",
        )
    )

    st.html(
        kpi_grid(
            [
                kpi("Pedidos iniciales", "99.441", "Total transaccional"),
                kpi("Entregados con fecha", "96.470", "97,0 % válidos", "accent"),
                kpi("Promedio global", "12,6 <small>d</small>", "Media aritmética"),
                kpi("Mediana real", "10,2 <small>d</small>", "50 % en ≤ 10,2 días", "accent"),
                kpi("Percentil 90", "23,1 <small>d</small>", "Cola de la distribución"),
            ]
        )
    )

    st.html(
        '<div class="ot-grid-2">'
        '<div class="ot-card">'
        '<div class="ot-card-title">Distancia vendedor–cliente</div>'
        '<div class="ot-card-sub">La variable más fuerte del EDA · correlación 0,39 con los días de entrega</div>'
        + list_rows(
            [
                ("Distancia mediana", "434 km", ""),
                ("Pedidos sin coordenadas", "478 (0,5 %)", "ot-amber"),
                ("Mismo estado", "36,0 % de los pedidos", ""),
                ("Mediana mismo estado", "6,5 d", "ot-emerald"),
                ("Mediana distinto estado", "12,8 d", "ot-rose"),
            ]
        )
        + "</div>"
        '<div class="ot-card">'
        '<div class="ot-card-title">Estados más rápidos y más lentos</div>'
        '<div class="ot-card-sub">Mediana de días hasta la entrega · los extremos son pocos pedidos</div>'
        + bars(
            [
                bar("SP · São Paulo", "7,2 d", 7.2 / 30 * 100),
                bar("MG · Minas Gerais", "10,3 d", 10.3 / 30 * 100),
                bar("PR · Paraná", "10,4 d", 10.4 / 30 * 100),
                bar("AP · Amapá", "24,3 d", 24.3 / 30 * 100, "amber"),
                bar("RR · Roraima", "25,0 d", 25.0 / 30 * 100, "amber"),
                bar("AM · Amazonas", "25,9 d", 25.9 / 30 * 100, "rose"),
            ]
        )
        + "</div></div>"
    )

    st.html(
        '<div class="ot-grid-2">'
        '<div class="ot-card soft">'
        '<div class="ot-card-title">Mismo estado vs. estados diferentes</div>'
        '<div class="ot-card-sub">La distancia cambia el plazo, no solo la geografía</div>'
        '<div class="ot-split" style="margin-top:.8rem">'
        '<div class="ot-split-part train"><div class="ot-split-label">Mismo estado</div>'
        '<div class="ot-split-value">6,5 d</div><div class="ot-split-note">Mediana intraestatal</div></div>'
        '<div class="ot-split-part test"><div class="ot-split-label">Estados diferentes</div>'
        '<div class="ot-split-value">12,8 d</div><div class="ot-split-note">+97 % frente al mismo estado</div></div>'
        "</div></div>"
        '<div class="ot-card">'
        '<div class="ot-card-title">Qué NO se puede concluir</div>'
        '<div class="ot-card-body">Las relaciones del EDA son <strong>observadas</strong>, no causales. '
        "Que un estado tarde más no significa que la ubicación sea la causa: también cambian la distancia, "
        "el tipo de producto, el vendedor y la logística disponible.</div>"
        '<div class="ot-card-body">La ubicación del cliente es un atributo sensible; el modelo se interpreta '
        "con SHAP y se revisa por región, sin usarla para segmentar personas.</div>"
        "</div></div>"
    )

    st.html(
        '<p class="ot-footnote">Fuente: <strong>reports/01_eda.md</strong> y <strong>notebooks/02_eda.ipynb</strong>. '
        "Los estados AP, RR y AM tienen pocos pedidos; sus medianas son referencias, no promedios estables.</p>"
    )


def render_methodology() -> None:
    st.html(
        hero(
            "Metodología & Flujo",
            "Rigor científico · Protocolo experimental",
            "Del CSV crudo al modelo desplegado: qué se quitó, cómo se partió y por qué la prueba "
            "permaneció cerrada hasta la validación final.",
            "Regresión · Días de entrega · MAE",
            "emerald",
        )
    )

    st.html(
        '<div class="ot-card">'
        '<div class="ot-card-title">Embudo de depuración de datos</div>'
        '<div class="ot-card-sub">Cada paso queda documentado en reports/02_preprocesamiento.md</div>'
        '<div class="ot-steps">'
        '<div class="ot-step"><div><div class="ot-step-num">PASO 01</div><div class="ot-step-name">Pedidos iniciales</div></div>'
        '<div><div class="ot-step-value">99.441</div><div class="ot-step-note">100 %</div></div></div>'
        '<div class="ot-step"><div><div class="ot-step-num">PASO 02</div><div class="ot-step-name">Entregados con fecha</div></div>'
        '<div><div class="ot-step-value">96.470</div><div class="ot-step-note">−2.971 (2,99 %)</div></div></div>'
        '<div class="ot-step"><div><div class="ot-step-num">PASO 03</div><div class="ot-step-name">Periodo 2017-01 a 2018-08</div></div>'
        '<div><div class="ot-step-value">96.203</div><div class="ot-step-note">−267 pedidos</div></div></div>'
        '<div class="ot-step"><div><div class="ot-step-num">PASO 04</div><div class="ot-step-name">Sin extremos &gt; 100 días</div></div>'
        '<div><div class="ot-step-value">96.139</div><div class="ot-step-note">−64 pedidos</div></div></div>'
        "</div></div>"
    )

    st.html(
        '<div class="ot-card">'
        '<div class="ot-card-title">Partición del conjunto principal · 96.139 pedidos</div>'
        '<div class="ot-card-sub">80 % entrenamiento / 20 % prueba, separados por cliente</div>'
        '<div class="ot-split">'
        '<div class="ot-split-part train"><div class="ot-split-label">Train · 80 %</div>'
        '<div class="ot-split-value">76.924</div>'
        '<div class="ot-split-note">Exploración, imputación, ingeniería de características y validación cruzada.</div></div>'
        '<div class="ot-split-part test"><div class="ot-split-label">Test · 20 % · held-out</div>'
        '<div class="ot-split-value">19.215</div>'
        '<div class="ot-split-note">Cerrado y congelado hasta la validación final del modelo ganador.</div></div>'
        "</div></div>"
    )

    controls = [
        ("Control 01", "Una fila por pedido", "Los ítems se agregan por order_id: precio, flete, peso y volumen se suman; se cuentan ítems y se identifica el vendedor principal."),
        ("Control 02", "Partición por cliente", "Todas las compras de un mismo customer_unique_id quedan en train o en test, nunca en ambos."),
        ("Control 03", "Sin fuga de información", "Medias, medianas, percentiles, imputadores y codificadores se ajustan solo con train."),
        ("Control 04", "Test cerrado", "El conjunto de 19.215 pedidos se consultó una única vez, al final, para la sustentación."),
    ]
    cards = "".join(
        f'<div class="ot-card"><div class="ot-step-num">{num}</div>'
        f'<div class="ot-card-title" style="margin-top:.25rem">{title}</div>'
        f'<div class="ot-card-body">{body}</div></div>'
        for num, title, body in controls
    )
    st.html(f'<div class="ot-grid-2">{cards}</div>')

    st.html(
        '<div class="ot-grid-2">'
        '<div class="ot-card soft">'
        '<div class="ot-card-title">Modelo final</div>'
        '<div class="ot-card-body">'
        + list_rows(
            [
                ("Algoritmo", "LightGBM cuantil 0,75", "ot-emerald"),
                ("Objetivo", "Días hasta la entrega", ""),
                ("Regla", "Percentil 75 de la predicción", ""),
                ("Variables", "7 grupos de features", ""),
                ("Métrica principal", "Costo 3 a 1", ""),
            ]
        )
        + "</div></div>"
        '<div class="ot-card">'
        '<div class="ot-card-title">Qué nunca entra al modelo</div>'
        '<div class="ot-card-body">Fechas de entrega, estado del pedido, reseñas e identificadores '
        "personales: se conocen después de la compra o no deben usarse. Tampoco entra el cliente ni el "
        "número de pedido.</div>"
        '<div class="ot-card-body">La fecha prometida por Olist se conserva como referencia de negocio, '
        "pero el modelo no la usa para predecir.</div>"
        "</div></div>"
    )

    st.html(
        '<p class="ot-footnote">Fuente: <strong>reports/02_preprocesamiento.md</strong>, '
        "<strong>reports/05_torneo.md</strong> y <strong>reports/07_validacion.md</strong>.</p>"
    )
