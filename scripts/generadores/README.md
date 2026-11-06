# Generadores de notebooks y reportes

Scripts que usó Claude para escribir los notebooks 03 a 06 y los reportes `reports/04_features.md` y `reports/05_torneo.md`.
Se guardan aquí para poder regenerarlos y para que el grupo pueda revisarlos. **Se ejecutan desde la raíz del repositorio** con el `.venv`.

| Script | Qué hace | Precaución |
|---|---|---|
| `generar_notebook_03.py`, `_04.py`, `_05.py` | Escriben `notebooks/0N_*.ipynb` (solo orquestan funciones de `src/`) | **Sobrescriben el notebook y borran sus salidas.** Después hay que ejecutarlo (`jupyter nbconvert --to notebook --execute …`; el 04 tarda ≈ 8 min y el 05 ≈ 10 min) |
| `generar_notebook_06.py` | Escribe `notebooks/06_optimizacion.ipynb` (solo orquesta `models/tuning.py`) | **Sobrescribe el notebook y borra sus salidas.** Ejecutarlo tarda ≈ 40 a 90 min (20 ensayos por finalista) y registra en MLflow (Databricks si `.env` conecta). El estudio de Optuna se guarda en `data/processed/optuna_optimizacion.db`: si se corta, al reejecutar solo hace los ensayos que faltan; para empezar de cero, borrar ese archivo |
| `generar_reporte_04.py` | Escribe `reports/04_features.md` leyendo `reports/04_features_tabla.csv` | Seguro de repetir: solo lee CSV |
| `generar_reporte_05.py` | Escribe `reports/05_torneo.md` leyendo los `reports/05_torneo_*.csv` | Seguro de repetir. Lee también `train.parquet` y la tabla original de pedidos (solo lectura). **Si una afirmación del texto deja de cumplirse con los datos, falla en vez de escribir algo falso** |
| `salidas_de_notebook_a_texto.py` | Vuelca las salidas de un notebook a texto para compararlas | `python salidas_de_notebook_a_texto.py notebooks/05_torneo.ipynb salida.txt` |

Los textos de interpretación están dentro de los scripts de reporte: si cambian los datos o las decisiones, se editan ahí y se regeneran.
