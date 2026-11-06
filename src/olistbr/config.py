"""Constantes de OnTrack: rutas, columnas, objetivo, reglas aprobadas y costo del error.

Un solo lugar para lo que se repite en varios módulos y notebooks. Si una regla cambia,
se cambia aquí y se anota en `reports/` (las reglas vienen de `especificaciones/03-preprocesamiento.md`).
"""
from pathlib import Path

# src/olistbr/config.py -> la raíz del repositorio está dos niveles arriba de src/
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"  # datos originales: solo se leen, NUNCA se modifican
PROCESSED_DIR = ROOT / "data" / "processed"  # se regenera con el notebook 03; no se versiona
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
GROUP_COL = "customer_unique_id"  # unidad de análisis para particionar: la persona, no el pedido
ID_COL = "order_id"  # una fila = un pedido entregado
TARGET = "dias_entrega"  # fecha de entrega al cliente - fecha de compra, en días

# --- Reglas de preprocesamiento (decisiones del EDA) -------------------------------------------
# Periodo estable de compras (decisión 9): ambos extremos incluidos.
PERIOD_START = "2017-01-01"
PERIOD_END = "2018-08-31"

# Los pedidos que tardaron más de este número de días se excluyen y se reportan aparte (decisión 2).
MAX_DELIVERY_DAYS = 100

# Proporción de clientes que va a la prueba (decisión 5).
TEST_SIZE = 0.2

# Vendedores con menos de este número de ítems en entrenamiento pasan a "otros" (decisión 8).
MIN_SELLER_ITEMS = 10
OTHER_LABEL = "otros"

# Categoría de los productos que no la traen (decisión 7).
UNKNOWN_CATEGORY = "desconocida"

# Recuadro aproximado de Brasil, para detectar coordenadas erróneas (decisión 6).
BRAZIL_LAT_RANGE = (-34.0, 6.0)
BRAZIL_LNG_RANGE = (-74.0, -34.0)

# --- Costo del error (reports/00_caso.md) ------------------------------------------------------
# Costo por día de error, en unidades relativas (importa la proporción, no el valor absoluto).
COST_UNDERESTIMATE = 3.0  # predicho < real: el pedido llega más tarde de lo dicho
COST_OVERESTIMATE = 1.0  # predicho > real: el pedido llega más temprano de lo dicho
# Proporciones con las que se revisa la sensibilidad del supuesto 3 a 1.
COST_RATIOS = (1, 2, 3, 5, 9)

# --- Etapa 4: medición de variables (especificaciones/04-features.md) ----------------------------
N_FOLDS = 5  # pliegues por cliente para comparar cada idea con la referencia

# Prueba hacia adelante dentro de entrenamiento: entrenar hasta febrero de 2018, validar de marzo a agosto.
FORWARD_TRAIN_END = "2018-02-28"
FORWARD_VALID_START = "2018-03-01"
FORWARD_VALID_END = "2018-08-31"

# Las categorías con menos de este porcentaje de filas se agrupan como "poco frecuentes" al codificar.
MIN_CATEGORY_FREQUENCY = 0.01

# Una idea se conserva solo si mejora / ruido supera este valor (regla de decisión escrita).
MIN_IMPROVEMENT_RATIO = 1.0

# --- Etapa 5: torneo de modelos (especificaciones/05-torneo.md) ----------------------------------
# Grupos de variables del torneo: los conservados en la etapa 4 SIN la marca de fin de año (peak_season),
# que descansa en una sola temporada, y sin el historial (history), que se descartó. Nunca el plazo prometido.
TOURNAMENT_GROUPS = (
    "logs", "same_state", "regions", "calendar", "distance_interaction", "ratios", "multi_seller",
)

# Con subestimar 3 veces más caro que sobrestimar, el plazo que minimiza el costo es el percentil 3 / (3 + 1).
TARGET_QUANTILE = COST_UNDERESTIMATE / (COST_UNDERESTIMATE + COST_OVERESTIMATE)  # 0,75

# Proporción de clientes que se aparta dentro de cada entrenamiento para calcular el margen del plazo.
CALIBRATION_SIZE = 0.2

# Un modelo gana "de forma consistente" si gana en al menos esta fracción de los pliegues (4 de 5).
MIN_WIN_SHARE = 0.8

# --- Etapa 6: optimización de hiperparámetros (especificaciones/06-optimizacion.md) --------------
# Proporción de CLIENTES de entrenamiento que la búsqueda nunca ve; con ellos se comprueba si optimizar ayudó (guía §3.4).
RESERVE_SIZE = 0.25
RESERVE_SEED = 7  # semilla de la guía para la reserva; distinta de RANDOM_STATE a propósito

# Presupuesto de ensayos por finalista, fijado antes de empezar (Q3). Un ensayo fallido también cuenta.
TUNING_TRIALS = 20

# Sorteos de clientes para el intervalo del 95 % de la comprobación honesta (regla 5).
BOOTSTRAP_ROUNDS = 1000

# Experimento de MLflow donde queda cada ensayo (Q8).
TUNING_EXPERIMENT = "ontrack-optimizacion"

# Si los dos finalistas empatan en la reserva (el intervalo cruza el cero), gana el más rápido y ganador del torneo (Q7).
TIE_WINNER = "LightGBM cuantil 0,75"
