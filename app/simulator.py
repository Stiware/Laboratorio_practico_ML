"""Simulador de tiempos de entrega conectado al modelo real.

El formulario pide las mismas variables que `models/input_schema.json`. La predicción se
delega en `olistbr.inference.predict`: aquí solo se dibuja y se explican los resultados.
"""
from __future__ import annotations

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from olistbr.inference.predict import InputValidationError, predict_deadline
from maps import haversine_km, render_map, route_summary
from theme import CATEGORY_BY_LABEL, CATEGORY_LABELS, EXAMPLES, bar, bars, hero, state_label

MODEL_LABEL = "LightGBM cuantil 0,75"
UNKNOWN_KEYS = [
    ("unk_distance_km", "No sé la distancia"),
    ("unk_n_items", "No sé el número de ítems"),
    ("unk_main_seller_items", "No sé los ítems del vendedor principal"),
    ("unk_price_sum", "No sé el valor del pedido"),
    ("unk_freight_sum", "No sé el flete"),
    ("unk_weight_g_sum", "No sé el peso"),
    ("unk_volume_cm3_sum", "No sé las dimensiones"),
]


def _default_date(schema: dict) -> date:
    return pd.Timestamp(schema["observed_period"][1]).date()


def _init_state(schema: dict) -> None:
    """Valores iniciales tomados del entrenamiento, para que el formulario abra con datos razonables."""
    defaults = {
        "seller_state": "SP",
        "customer_state": "RJ",
        "distance_km": max(20, min(3500, int(round(schema["median"]["distance_km"] / 10) * 10))),
        "weight_g_sum": float(schema["median"]["weight_g_sum"]),
        "length_cm": 25.0,
        "width_cm": 20.0,
        "height_cm": 15.0,
        "price_sum": float(schema["median"]["price_sum"]),
        "freight_sum": float(schema["median"]["freight_sum"]),
        "n_items": int(schema["median"]["n_items"]),
        "main_seller_items": int(schema["median"]["main_seller_items"]),
        "product_category_name": "Belleza y Salud",
        "order_purchase_timestamp": _default_date(schema),
        "order_estimated_delivery_date": _default_date(schema) + timedelta(days=10),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key, _ in UNKNOWN_KEYS:
        st.session_state.setdefault(key, False)
    # Al entrar por primera vez se analiza el pedido por defecto, para que la vista no abra vacía.
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
        st.session_state["auto_run"] = True


def _random_order(schema: dict) -> None:
    """Rellena el formulario con un pedido aleatorio dentro de los rangos observados en el entrenamiento."""
    rng = random.Random()
    obs = schema["observed_range"]
    states = schema["states"]
    seller = "SP" if rng.random() < 0.6 else rng.choice(states)
    customer = rng.choice(states)
    line_km = haversine_km(seller, customer)
    n_items = rng.choices([1, 2, 3, 4, 5], weights=[70, 15, 8, 4, 3])[0]
    price = min(obs["price_sum"][1], round(rng.lognormvariate(4.5, 0.9) * n_items, 2))
    weight = min(obs["weight_g_sum"][1], round(rng.lognormvariate(6.7, 1.0) * n_items, -1))
    purchase = pd.Timestamp(rng.choice(pd.date_range(*schema["observed_period"]))).date()
    values = {
        "seller_state": seller,
        "customer_state": customer,
        "distance_km": int(max(20, min(3500, round(line_km * rng.uniform(0.8, 1.15) / 10) * 10))),
        "weight_g_sum": float(max(obs["weight_g_sum"][0], weight)),
        "length_cm": float(rng.randint(10, 60)),
        "width_cm": float(rng.randint(8, 45)),
        "height_cm": float(rng.randint(3, 40)),
        "price_sum": float(max(1.0, price)),
        "freight_sum": float(round(min(obs["freight_sum"][1], 8 + price * rng.uniform(0.05, 0.3) + line_km * 0.01), 2)),
        "n_items": n_items,
        "main_seller_items": rng.randint(1, n_items),
        "product_category_name": rng.choice([c for c in CATEGORY_LABELS if not c.startswith("Otra")]),
        "order_purchase_timestamp": purchase,
        "order_estimated_delivery_date": purchase + timedelta(days=rng.randint(7, 30)),
    }
    st.session_state.update(values)
    for key, _ in UNKNOWN_KEYS:
        st.session_state[key] = False
    st.session_state["prediction_error"] = None
    st.session_state["auto_run"] = True


def _use_map_distance(line_km: float) -> None:
    """Copia la distancia del mapa al control de distancia, dentro de su rango (20–3500 km, paso 10)."""
    st.session_state["distance_km"] = int(max(20, min(3500, round(line_km / 10) * 10)))


def _route_panel(schema: dict, seller: str, customer: str) -> None:
    """Mapa de la ruta elegida, con los parámetros y el resultado del modelo superpuestos."""
    badge = [f"<b>{route_summary(seller, customer)}</b>"]
    last = st.session_state.get("last_prediction")
    if last and last.get("seller_state") == seller and last.get("customer_state") == customer:
        fecha = pd.Timestamp(last["fecha"]).strftime("%d/%m/%Y")
        badge.append(f"Distancia enviada al modelo: <b>{last['distance']}</b>")
        badge.append(f"Plazo recomendado: <b>{last['plazo']:.1f} d</b>".replace(".", ",") + f" · entrega <b>{fecha}</b>")
    else:
        badge.append("Pulsa «Analizar pedido» para ver el plazo en el mapa.")
    render_map(schema.get("late_share_by_region", {}) or {}, route=(seller, customer), height=380, badge=badge)


def _apply_example(name: str) -> None:
    """Copia un ejemplo en el estado de los widgets y limpia el resultado anterior."""
    values = EXAMPLES[name]
    for key, value in values.items():
        if key in ("order_purchase_timestamp", "order_estimated_delivery_date"):
            st.session_state[key] = date.fromisoformat(value)
        else:
            st.session_state[key] = value
    for key, _ in UNKNOWN_KEYS:
        st.session_state[key] = False
    st.session_state["last_prediction"] = None
    st.session_state["prediction_error"] = None


def _num(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + " %"


def _attention(region: str, late_share: float | None) -> tuple[str, str, str]:
    """Devuelve (nivel, clase CSS, explicación) a partir del % de retraso de la región."""
    if late_share is None:
        return "Sin dato", "bajo", "No hay porcentaje de retraso para esta región."
    if late_share < 0.25:
        return "Bajo", "bajo", f"{region}: {_pct(late_share)} de pedidos tarde en la prueba."
    if late_share <= 0.28:
        return "Medio", "medio", f"{region}: {_pct(late_share)} de pedidos tarde en la prueba."
    return "Alto", "alto", f"{region}: {_pct(late_share)} de pedidos tarde en la prueba."


def _result_html(prediction: dict) -> str:
    warnings = prediction["warnings"]
    validation = "Aprobada" if not warnings else f"Con {len(warnings)} aviso(s)"
    validation_class = "ok" if not warnings else "warn"
    return (
        '<div class="ot-result-head">'
        '<div class="ot-result-topline">'
        '<span class="ot-result-badge">Análisis del pedido</span>'
        f'<span class="ot-result-time">{prediction["time"]}</span>'
        "</div>"
        '<div class="ot-result-title">'
        '<span class="ot-result-check">✓</span>'
        "<span>Datos recibidos correctamente</span>"
        "</div>"
        '<div class="ot-result-sub">El pedido pasó la validación y fue procesado por el modelo.</div>'
        "</div>"
        '<div class="ot-pipeline">'
        '<div class="ot-pipeline-label">Estado del pipeline logístico</div>'
        '<div class="ot-pipeline-row"><span>Datos del pedido</span><span class="ok">✓ Correctos</span></div>'
        f'<div class="ot-pipeline-row"><span>Validación de entrada</span><span class="{validation_class}">'
        f'{"✓ " if not warnings else "⚠ "}{validation}</span></div>'
        '<div class="ot-pipeline-row"><span>Preprocesamiento</span><span class="ok">✓ Estructurado</span></div>'
        f'<div class="ot-pipeline-row model"><span>Modelo predictivo</span><span class="ok">✓ {MODEL_LABEL}</span></div>'
        "</div>"
    )


def _idle_panel() -> None:
    st.html(
        '<div class="ot-idle">'
        '<div class="ot-idle-title">'
        '<span class="ot-icon-chip">▶</span> Panel de resultados del simulador</div>'
        '<div class="ot-idle-copy">Completa los parámetros logísticos y pulsa '
        '<strong style="color:#059669">“Analizar pedido”</strong>. El modelo responderá con la fecha '
        "de entrega que conviene prometer y el plazo recomendado en días.</div>"
        '<div style="margin-top:.75rem">'
        '<div class="ot-placeholder-row"><span>Días estimados</span><span class="pending">Pendiente</span></div>'
        '<div class="ot-placeholder-row"><span>Fecha estimada</span><span class="pending">Pendiente</span></div>'
        '<div class="ot-placeholder-row"><span>Comparación con fecha prometida</span><span class="pending">Pendiente</span></div>'
        '<div class="ot-placeholder-row"><span>Nivel de atención logística</span><span class="pending">Pendiente</span></div>'
        "</div></div>"
    )


def _show_result(prediction: dict) -> None:
    st.html(_result_html(prediction))

    recommended = pd.Timestamp(prediction["fecha"]).date()
    promised = prediction["promised"]
    delta = (recommended - promised).days

    left, right = st.columns(2)
    with left:
        st.metric("Fecha de entrega recomendada", recommended.strftime("%d/%m/%Y"))
    with right:
        st.metric("Plazo recomendado", f'{prediction["plazo"]:.1f} días'.replace(".", ","))

    if delta > 0:
        comparison = (
            f'<div class="ot-attention alto"><span>⚠ La fecha prometida queda <strong>{delta} día(s)</strong> '
            "antes del plazo recomendado</span><span>Riesgo de incumplir</span></div>"
        )
    elif delta < 0:
        comparison = (
            f'<div class="ot-attention bajo"><span>✓ Margen de <strong>{abs(delta)} día(s)</strong> '
            "frente al plazo recomendado</span><span>Promesa holgada</span></div>"
        )
    else:
        comparison = (
            '<div class="ot-attention medio"><span>La fecha prometida coincide con el plazo recomendado</span>'
            "<span>Sin margen</span></div>"
        )

    level, css, detail = _attention(prediction["region"], prediction["late_region"])
    attention = (
        f'<div class="ot-attention {css}"><span>Nivel de atención logística: <strong>{level}</strong></span>'
        f"<span>{detail}</span></div>"
    )
    st.html(comparison + attention)

    if prediction["warnings"]:
        for message in prediction["warnings"]:
            st.warning(message[0].upper() + message[1:])

    st.html(
        '<p class="ot-form-note">Puedes modificar los campos y volver a pulsar '
        "<strong>“Analizar pedido”</strong>. La fecha se cuenta desde la compra y se redondea al día.</p>"
    )


def render_simulator(model, schema: dict) -> None:
    _init_state(schema)

    st.html(
        hero(
            "Simulador de Tiempos de Entrega",
            "Simulador OnTrack · Terminal logístico",
            "Ingresa los parámetros de ruta, paquete y fechas. El modelo LightGBM cuantil 0,75 devuelve "
            "la fecha que conviene prometer y el plazo en días.",
            "Modelo conectado · respuesta en línea",
            "emerald",
        )
    )

    left, right = st.columns([7, 5], gap="large")

    with left:
        st.html(
            '<div class="ot-sim-head">'
            '<div class="ot-sim-title">'
            '<div class="ot-sim-icon"><span class="ot-sim-glyph">◫</span></div>'
            "<div><strong>Parámetros logísticos del pedido</strong>"
            "<span>Información requerida para el pipeline de inferencia</span></div>"
            "</div></div>"
        )

        st.button(
            "🎲 Simulación aleatoria",
            key="random_order",
            on_click=_random_order,
            args=(schema,),
            help="Genera un pedido al azar dentro de los rangos del entrenamiento y lo analiza.",
        )
        example_cols = st.columns([1.1, 1.15, 1.35, 1.0])
        example_cols[0].markdown('<div class="ot-examples-label">Ejemplos rápidos:</div>', unsafe_allow_html=True)
        for column, name in zip(example_cols[1:], EXAMPLES):
            if column.button(name, key=f"example_{name}", width="stretch"):
                _apply_example(name)
                st.toast(f"Ejemplo aplicado: {name}")

        st.html('<div class="ot-derived"><span><strong>1 · Ruta</strong> (el mapa se actualiza al instante)</span></div>')
        first, second = st.columns(2)
        seller_state = first.selectbox(
            "Estado origen (vendedor) *", schema["states"], format_func=state_label, key="seller_state"
        )
        customer_state = second.selectbox(
            "Estado destino (cliente) *", schema["states"], format_func=state_label, key="customer_state"
        )
        line_km = haversine_km(seller_state, customer_state)
        st.button(
            f"Usar la distancia del mapa (≈ {line_km:,.0f} km)".replace(",", "."),
            key="use_map_distance",
            on_click=_use_map_distance,
            args=(line_km,),
            help="Copia la distancia en línea recta entre las capitales al campo de distancia del modelo.",
        )

        with st.form("pedido"):
            distance = st.slider(
                "Distancia estimada vendedor–cliente (km) *",
                min_value=20,
                max_value=3500,
                step=10,
                key="distance_km",
            )
            same_state = seller_state == customer_state
            historic = "6,5 d mediana" if same_state else "12,8 d mediana"
            classification = "Intraestatal (mismo estado)" if same_state else "Interestatal (cruza frontera)"
            st.html(
                '<div class="ot-derived">'
                f'<span>Distancia: <strong>{distance:,} km</strong></span>'
                f"<span>Clasificación: <strong>{classification}</strong> · Histórico: {historic}</span>"
                "</div>"
            )

            package = st.columns(4)
            weight = package[0].number_input(
                "Peso total (g) *",
                min_value=0.0,
                max_value=200_000.0,
                step=50.0,
                key="weight_g_sum",
            )
            length = package[1].number_input(
                "Largo (cm) *", min_value=1.0, max_value=150.0, step=1.0, key="length_cm"
            )
            width = package[2].number_input(
                "Ancho (cm) *", min_value=1.0, max_value=150.0, step=1.0, key="width_cm"
            )
            height = package[3].number_input(
                "Alto (cm) *", min_value=1.0, max_value=150.0, step=1.0, key="height_cm"
            )
            volume = float(length) * float(width) * float(height)
            net_weight = float(weight) / 1000.0
            volume_text = f"{volume:,.0f}".replace(",", ".")
            weight_text = f"{net_weight:.2f}".replace(".", ",")
            st.html(
                '<div class="ot-derived">'
                f'<span>Volumen: <strong>{volume_text} cm³</strong></span>'
                f'<span>Peso neto: <strong>{weight_text} kg</strong></span>'
                "</div>"
            )

            commerce = st.columns(3)
            category_label = commerce[0].selectbox(
                "Categoría *", CATEGORY_LABELS, key="product_category_name"
            )
            price = commerce[1].number_input(
                "Valor del pedido (R$) *",
                min_value=0.0,
                max_value=20_000.0,
                step=10.0,
                key="price_sum",
            )
            freight = commerce[2].number_input(
                "Flete (R$) *",
                min_value=0.0,
                max_value=2_000.0,
                step=5.0,
                key="freight_sum",
            )

            items = st.columns(2)
            n_items = items[0].number_input(
                "Número de ítems *", min_value=1, max_value=30, step=1, key="n_items"
            )
            main_seller_items = items[1].number_input(
                "Ítems del vendedor principal *",
                min_value=1,
                max_value=30,
                step=1,
                key="main_seller_items",
            )

            dates = st.columns(2)
            purchase = dates[0].date_input(
                "Fecha de compra *",
                min_value=date(2016, 1, 1),
                key="order_purchase_timestamp",
            )
            promised = dates[1].date_input(
                "Fecha prometida inicial",
                min_value=date(2016, 1, 1),
                key="order_estimated_delivery_date",
            )

            with st.expander("Datos que no conozco"):
                st.caption("Marca lo que no sepas: el modelo usará un valor típico y te avisará de que el plazo es menos preciso.")
                unknown = {}
                columns = st.columns(2)
                for index, (key, label) in enumerate(UNKNOWN_KEYS):
                    unknown[key] = columns[index % 2].checkbox(label, key=key)

            submitted = st.form_submit_button(
                "Analizar pedido", type="primary", key="analizar", width="stretch"
            )

        st.html('<p class="ot-form-note">Puedes modificar los campos cuantas veces quieras y volver a analizar.</p>')

        if st.session_state.pop("auto_run", False):
            submitted = True
        if submitted:
            order = {
                "order_purchase_timestamp": pd.Timestamp(purchase),
                "seller_state": seller_state,
                "customer_state": customer_state,
                "product_category_name": CATEGORY_BY_LABEL[category_label],
                "distance_km": np.nan if unknown["unk_distance_km"] else float(distance),
                "n_items": np.nan if unknown["unk_n_items"] else float(n_items),
                "main_seller_items": np.nan if unknown["unk_main_seller_items"] else float(main_seller_items),
                "price_sum": np.nan if unknown["unk_price_sum"] else float(price),
                "freight_sum": np.nan if unknown["unk_freight_sum"] else float(freight),
                "weight_g_sum": np.nan if unknown["unk_weight_g_sum"] else float(weight),
                "volume_cm3_sum": np.nan if unknown["unk_volume_cm3_sum"] else volume,
            }
            try:
                result = predict_deadline(pd.DataFrame([order]), model, schema)
            except InputValidationError as error:
                st.session_state["last_prediction"] = None
                st.session_state["prediction_error"] = str(error)
            else:
                row = result.iloc[0]
                late = row["pct_tarde_region"]
                st.session_state["last_prediction"] = {
                    "fecha": row["fecha_recomendada"],
                    "plazo": float(row["plazo_dias"]),
                    "region": row["region_cliente"],
                    "late_region": None if pd.isna(late) else float(late),
                    "warnings": list(result.attrs.get("warnings", [])),
                    "promised": promised,
                    "distance": "no la sé (valor típico)" if unknown["unk_distance_km"] else f"{int(distance)} km",
                    "seller_state": seller_state,
                    "customer_state": customer_state,
                    "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                }
                st.session_state["prediction_error"] = None

    with right:
        error = st.session_state.pop("prediction_error", None)
        if error:
            st.error(f"Revisa los datos: {error}")
            _idle_panel()
        elif st.session_state.get("last_prediction"):
            _show_result(st.session_state["last_prediction"])
        else:
            _idle_panel()
        _route_panel(schema, st.session_state["seller_state"], st.session_state["customer_state"])

    with st.expander("Uso responsable", expanded=True):
        st.markdown(schema["responsible_use"])

    st.html(
        '<p class="ot-footnote">El simulador no consulta Databricks: usa '
        "<strong>models/model.joblib</strong> y <strong>models/input_schema.json</strong> del repositorio. "
        "La fecha es una estimación para planificar, no una promesa contractual.</p>"
    )
