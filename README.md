# Nombre del proyecto

> Una frase: qué problema de negocio resuelve este proyecto y para quién.

**App desplegada:** _enlace público de Streamlit (se completa en la clase 5)_

## Qué contiene
Estructura obligatoria del curso. Cada carpeta tiene un propósito:

| Carpeta | Contenido |
|---|---|
| `.specify/` y `.claude/` | Entorno de IA: constitución del proyecto, Spec Kit (`/speckit-…`) y subagentes |
| `specs/` | Lo que genera Spec Kit: `spec.md`, `plan.md`, `tasks.md` de cada pieza (aparece al usarlo) |
| `especificaciones/` | Guías que se llenan **antes** de programar cada etapa |
| `data/raw/` | Datos originales, intocables |
| `data/processed/` | Datos limpios generados por el código |
| `notebooks/` | Un notebook por etapa; solo orquestan |
| `src/olistbr/` | Código reutilizable (data, features, models, tracking, inference) |
| `tests/` | Pruebas con `pytest` |
| `models/` | `model.joblib` y `ficha_modelo.md` |
| `app/` | `streamlit_app.py` |
| `reports/` | Una sección por hito e `informe_tecnico.md` |
| `presentacion/` | `ejecutiva.md` (máximo 5 diapositivas) |

## Cómo ejecutarlo
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # en Mac/Linux: .venv/bin/python
cp .env.example .env     # completa PROFILE o DATABRICKS_HOST y DATABRICKS_TOKEN
python scripts/check_databricks.py
pytest
streamlit run app/streamlit_app.py
```

## La app (OnTrack · Simulador de entregas)
```bash
streamlit run app/streamlit_app.py
```
La app integra el diseño del prototipo OnTrack con el modelo del laboratorio:

- **Dashboard General**: desempeño del modelo en la prueba, cobertura y tasa de retraso por región.
- **Datos & EDA**: volumen, distribución del plazo, distancia vendedor–cliente y diferencias regionales.
- **Metodología & Flujo**: embudo de depuración, partición por cliente y controles contra fuga de información.
- **Simulador OnTrack**: formulario de 11 datos del pedido (fecha de compra, estados, distancia, categoría, ítems, precio, flete, peso y volumen; «No sé» en los que no se conozcan) que recomienda la fecha de entrega y el plazo en días.

Usa `models/model.joblib` y `models/input_schema.json` (los genera `notebooks/09_inferencia.ipynb` desde el modelo `champion` registrado en
Databricks) y **no consulta Databricks**. Si un dato es imposible, explica qué corregir; si es poco común, avisa. Límites y cifras: `models/ficha_modelo.md`.

Archivos de la app:

```text
app/
├── streamlit_app.py          # arranque, navegación y carga del modelo
├── simulator.py              # formulario y panel de resultados conectados al modelo
├── views.py                  # Dashboard, EDA y Metodología
├── theme.py                  # estados, categorías, ejemplos y componentes visuales
└── assets/css/ontrack.css    # diseño OnTrack (tarjeta blanca, acentos esmeralda, Inter + JetBrains Mono)
```


## Datos
Dataset: [*Brazilian E-Commerce Public Dataset by Olist*](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): ~100.000 pedidos de comercio electrónico brasileño (2016–2018) en 9 archivos CSV.
**Los CSV no están en el repositorio** (`data/raw/` está en `.gitignore`). Para reproducir el proyecto: descarga el dataset desde Kaggle, descomprímelo y copia los 9 CSV a `data/raw/`. Licencia del dataset: CC BY-NC-SA 4.0 (atribución a Olist, uso no comercial, compartir igual).

## De dónde viene este repositorio
Nació de la rama `feature/estudiantes` del repositorio del curso (ver `LEEME-PRIMERO.md`): una persona del grupo la clonó con
`git clone --branch feature/estudiantes --single-branch …`, la apuntó a este repo con `git remote set-url origin …` y hizo `push`. Los demás integrantes
clonan **este** repositorio. Para arrancar, sigue `material-del-curso/guias/montaje-proyecto.md`.

## Equipo
| Integrante | Rol |
|---|---|
| | |

Especificación del proyecto: [`especificacion-proyecto.md`](especificacion-proyecto.md) · Especificaciones por etapa: [`especificaciones/`](especificaciones/)
