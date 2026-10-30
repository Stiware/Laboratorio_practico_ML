# Guía de trabajo: variables, torneo y optimización (clase 3 · sábado 31 de octubre)

Esta guía es para **tu dataset**, no para el de CreditSight. El ejemplo del instructor
(`material-del-curso/ejemplo-creditsight/notebooks/04_features.ipynb`, `05_torneo.ipynb` y `06_optimizacion.ipynb`) te muestra
**cómo se razona**, pero sus cifras y sus decisiones no valen para otros datos: cada comparación de tu
proyecto la tienes que medir tú.

**Regla de oro del día:** una mejora que no supera el **ruido de la validación** no es una mejora.
Cada idea, cada modelo y cada configuración se compara con la **misma validación**, y se acepta solo
si la diferencia es mayor que lo que varía entre pliegues.

**Lo que se te pide al cerrar el día** (el hito H2 se cierra el jueves 5 de noviembre, en la clase 4):
- una **tabla de ideas de variables medidas** con su decisión,
- un **torneo con al menos 3 modelos y una línea base**, con 1–2 finalistas justificados,
- una **optimización con presupuesto acotado**, con cada ensayo **registrado** en MLflow y una
  **comprobación honesta** de si mejoró.

---

## 0. Antes de empezar

- Necesitas `data/processed/train.joblib` (lo deja tu notebook 03). **`test.joblib` no se abre hoy:
  se toca una sola vez, en la validación final (clase 4).**
- Define dos cosas al inicio de cada notebook: tu **ID_COL** (la columna que identifica a la entidad
  repetida; si no hay entidades repetidas, déjala en `None`) y tu **TARGET**.
- Si `python scripts/verificar_entorno.py` muestra algún `FALTA`, arréglalo antes de modelar.
- Los esqueletos están en `notebooks/04_features.ipynb`, `05_torneo.ipynb` y `06_optimizacion.ipynb`.

---

## 1. Variables como hipótesis

Crear variables es la parte más "creativa" y la más peligrosa: es fácil añadir columnas que *suenan
bien* y no aportan nada, y cada columna extra hay que calcularla, validarla y pedírsela al usuario de
la app. Por eso **cada idea es una hipótesis que se mide**.

### 1.1 De la idea a la hipótesis

Para cada idea escribe (antes de medirla) tres cosas:

1. **Qué variable** es y cómo se calcula.
2. **Por qué podría ayudar** (una frase).
3. **Si se conoce al momento de predecir.** Si solo se conoce *después* de la decisión que quieres
   apoyar, es fuga: descártala.

### 1.2 El experimento

Mismo modelo (con parámetros fijos), **mismos pliegues** y la misma métrica. Solo cambia el conjunto
de variables.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, cross_validate

X, y = train.drop(columns=[TARGET]), train[TARGET]

# Los pliegues se calculan UNA vez y se reutilizan en todas las comparaciones
if ID_COL is not None:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # clasificación
    # Regresión: GroupKFold(n_splits=5)   (no admite shuffle)
    pliegues = list(cv.split(X, y, groups=X[ID_COL]))
else:
    from sklearn.model_selection import StratifiedKFold      # regresión: KFold
    pliegues = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y))

def evaluar(pipeline, scoring):
    """Devuelve un DataFrame con una fila por pliegue."""
    res = cross_validate(pipeline, X, y, cv=pliegues, scoring=scoring, n_jobs=1)
    return pd.DataFrame({k.replace("test_", ""): v for k, v in res.items() if k.startswith("test_")})
```

> **Una métrica de negocio como `scoring`.** Si equivocarte cuesta distinto en cada dirección, escribe
> una función `costo(y_true, y_pred)` con tu matriz de costos y úsala con
> `make_scorer(costo, greater_is_better=False)`. Scikit-learn devuelve los puntajes con **signo
> negativo** cuando `greater_is_better=False`: multiplícalos por `-1` para leer el costo.

### 1.3 Cómo leer el resultado: mejora / ruido

```python
def mejora_sobre_ruido(referencia, variante):
    """Ambos: valores por pliegue de una métrica donde MENOR es mejor (p. ej. el costo)."""
    mejora = np.mean(referencia) - np.mean(variante)          # positivo: la variante es mejor
    ruido = np.std(variante, ddof=1)                          # cuánto varía entre pliegues
    return mejora, ruido, mejora / ruido if ruido > 0 else np.inf
```

- Una **mejora / ruido menor que ~1** está dentro del ruido: con otra partición podría desaparecer.
- Mira también una **segunda métrica** (por ejemplo la calidad de las probabilidades, `log_loss`): a
  veces la mejora aparece ahí y no en la métrica de decisión.
- Cuenta cuántos pliegues gana la variante, no solo el promedio.

### 1.4 Tu tabla de ideas

| # | Idea (variable) | Variante que se mide | Métrica principal (± ruido) | Mejora / ruido | Decisión |
|---|---|---|---|---|---|
| 0 | — | **Referencia** (variables actuales) | | 0 | — |
| 1 | | | | | ☐ conservar ☐ descartar |
| 2 | | | | | ☐ conservar ☐ descartar |
| 3 | | | | | ☐ conservar ☐ descartar |

**Decide con una regla escrita antes de medir:** "Conservo la idea solo si la mejora / ruido es
mayor que 1 **y** se conoce al momento de predecir". Cualquier excepción se justifica por escrito.

### 1.5 Lo que le pasó al ejemplo (con su dataset, no con el tuyo)

En CreditSight se midieron tres ideas con LightGBM, 5 pliegues por cliente y el costo esperado como
métrica (referencia: costo 0,489 ± 0,007):

| Idea | Resultado | Decisión |
|---|---|---|
| Razones financieras (deuda/ingreso, cuota/sueldo…) | Sin cambio: mejora / ruido ≈ −0,02 | **Descartada** |
| Historial del mes anterior | Mejora pequeña en costo (mejora / ruido ≈ 0,6) y algo en probabilidades | **Descartada**: complica la app y la inferencia |
| `Credit_Mix` (una columna ya existente) | Mejora clara en la calidad de las probabilidades (log loss 0,695 → 0,669) | **Conservada**, con un supuesto escrito: debe conocerse al decidir |

**Dos de tres ideas no sirvieron, y eso es información, no un fracaso.** La mayor mejora del
proyecto vino de **entender y limpiar los datos** (clase 2), no de inventar columnas. Un modelo con
menos variables es más fácil de explicar, de desplegar y de mantener.

### 1.6 Reglas de las variables

- **Todo cálculo que "aprende" algo de los datos** (medias, escalas, categorías) va **dentro del
  pipeline**, ajustado solo con entrenamiento.
- Una variable calculada con **varias filas de la misma entidad** (promedios históricos, acumulados)
  debe usar **solo el pasado**: nunca el mismo periodo ni los siguientes.
- **Las ideas descartadas se conservan** en `reports/04_features.md` con su cifra: documentan lo que
  ya se probó.

---

## 2. Torneo de modelos

El torneo no busca un campeón: busca **uno o dos finalistas** para optimizar, con evidencia.

### 2.1 Las reglas

1. **Mismos datos y mismos pliegues para todos** (los `pliegues` de la sección 1.2), por grupo si
   hay entidades repetidas.
2. **Una línea base** que cualquier modelo serio debe vencer (clase más frecuente, media, regla simple).
3. **Guarda cada pliegue**, no solo el promedio: con los pliegues puedes comparar de a pares.
4. **Misma configuración de variables** para todos (la que decidiste en la sección 1).

### 2.2 Las métricas: de negocio **y** técnicas

Un torneo serio reporta **dos familias de métricas**, siempre las dos:

- **De negocio (la principal):** la que refleja la decisión (por ejemplo, un costo esperado). Es la que decide
  quién gana.
- **Técnicas:** las que dicen *qué tan bien funciona el modelo* sin depender del negocio. Sirven para entender
  el resultado y para ver lo que la métrica de negocio esconde.

**Si tu problema es de clasificación**, reporta como mínimo:

| Métrica | Qué mide | ¿Depende de la decisión? |
|---|---|---|
| **AUC macro (uno contra el resto)** y **AUC de la clase de interés** | Qué tan bien **ordena** los casos: 0,5 es azar, 1 es perfecto | No: usa las probabilidades |
| **PR-AUC de la clase de interés** | Lo mismo, más exigente cuando esa clase es poco frecuente. Su piso es la proporción de esa clase | No |
| **Exactitud balanceada** | Promedio del recall de todas las clases | Sí |
| **Precisión y recall por clase** | Recall: de los que **son** de la clase, ¿cuántos detectó? Precisión: de los que **marcó**, ¿cuántos lo eran? | Sí |
| **F1 macro** | Resumen de precisión y recall con todas las clases por igual | Sí |
| **Log loss** | Calidad de las probabilidades (menor es mejor) | No |

**Si tu problema es de regresión**, además de tu error de negocio reporta **RMSE, MAE y R²** (y, si ayuda, el
error relativo respecto al valor medio). El RMSE castiga más los errores grandes; el MAE es el error típico en
las unidades del objetivo; R² dice qué parte de la variación explica el modelo.

> **Advertencia: el costo puede esconder lo que muestran las técnicas.** Un modelo puede ganar en costo siendo
> muy conservador, por ejemplo con el recall de una clase que casi nunca se predice. Si solo miras la métrica de
> negocio, no lo ves. Mira siempre el **recall de cada clase**.

- **Evita la exactitud simple** con clases desbalanceadas (la clase mayoritaria sola ya da un número alto): usa
  la exactitud balanceada.
- Un modelo puede ganar en la métrica principal y perder en las técnicas. Eso no es un error: **la métrica que
  eliges decide quién gana**, y debes explicar por qué.

**Tu tabla del torneo** (una fila por modelo; media ± desviación entre pliegues):

| Modelo | Métrica de negocio (± ruido) | AUC macro | PR-AUC clase de interés | Exactitud balanceada | Precisión clase de interés | Recall por clase | F1 macro |
|---|---|---|---|---|---|---|---|
| línea base | | | | | | | |
| modelo 1 | | | | | | | |
| modelo 2 | | | | | | | |

### 2.3 Correr el torneo

```python
from sklearn.dummy import DummyClassifier           # regresión: DummyRegressor
from sklearn.linear_model import LogisticRegression  # regresión: Ridge / LinearRegression
from sklearn.ensemble import RandomForestClassifier  # regresión: RandomForestRegressor

candidatos = {
    "línea base": DummyClassifier(strategy="most_frequent"),
    "regresión logística": LogisticRegression(max_iter=1000),
    "bosque aleatorio": RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42),
    # TODO: un modelo de boosting (LightGBM o XGBoost)
}

por_pliegue = {}
for nombre, modelo in candidatos.items():
    pipeline = armar_pipeline(modelo)        # TODO: tu función: preprocesamiento + modelo
    por_pliegue[nombre] = evaluar(pipeline, scoring)   # un DataFrame por modelo (una fila por pliegue)
    print(f"{nombre:25s} listo")
```

Tabla resumen (media ± desviación entre pliegues, ordenada por la métrica principal) y un
`boxplot` por modelo para **ver** la variabilidad.

**Métricas técnicas por pliegue (clasificación).** Para cada modelo y pliegue calcula también las técnicas, a
partir de las probabilidades (`proba`) y de la decisión que usa tu negocio (`decision`):

```python
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)


def metricas_tecnicas(y_real, proba, decision, clase_interes=0):
    """y_real: etiquetas 0..k-1; proba: (n, k); decision: clase elegida (n,)."""
    es_interes = (np.asarray(y_real) == clase_interes).astype(int)
    return {
        "AUC macro": roc_auc_score(y_real, proba, multi_class="ovr", average="macro"),
        "AUC clase de interés": roc_auc_score(es_interes, proba[:, clase_interes]),
        "PR-AUC clase de interés": average_precision_score(es_interes, proba[:, clase_interes]),
        "exactitud balanceada": balanced_accuracy_score(y_real, decision),
        "F1 macro": f1_score(y_real, decision, average="macro"),
        "precisión clase de interés": precision_score(y_real, decision, labels=[clase_interes],
                                                      average="macro", zero_division=0),
        **{f"recall clase {k}": recall_score(y_real, decision, labels=[k], average="macro")
           for k in range(proba.shape[1])},
        "log loss": log_loss(y_real, proba),
    }
```

Si tu problema es **binario**, usa `roc_auc_score(y_real, proba[:, 1])` (sin `multi_class`). En **regresión**
calcula `root_mean_squared_error`, `mean_absolute_error` y `r2_score` de `sklearn.metrics`.

### 2.4 Comparación pareada

Como todos los modelos se evaluaron en **los mismos pliegues**, puedes comparar pliegue a pliegue:
¿gana siempre el mismo?

```python
def comparar(mejor, otro):
    """Ambos: costo por pliegue (menor es mejor). Positivo en 'diferencia': 'mejor' gana."""
    dif = np.asarray(otro) - np.asarray(mejor)
    ruido = dif.std(ddof=1)
    return {
        "diferencia media": dif.mean(),
        "pliegues que gana": f"{(dif > 0).sum()} de {len(dif)}",
        "diferencia / ruido": dif.mean() / ruido if ruido > 0 else np.inf,
    }
```

- Una diferencia que **gana en todos los pliegues** y es varias veces mayor que su ruido es
  consistente.
- Una que gana en 3 de 5 con ruido mayor que la diferencia es un **empate práctico**.

### 2.5 Elegir finalistas

Elige **1 o 2** mirando, en este orden:

1. **Consistencia** (gana en la mayoría de pliegues, diferencia / ruido > 1).
2. **Costo / métrica principal.**
3. **Tiempo de ajuste** y simplicidad de operar (ante un empate, gana el más simple).
4. **Interpretabilidad**, si tu caso la exige por regulación o por confianza del usuario. Si un
   modelo interpretable cuesta algo más, traduce esa diferencia a **dinero o volumen**
   (`costo por decisión × decisiones al mes`) y deja que el negocio decida.
5. **Compara el ranking de negocio con el técnico.** ¿El mejor por costo también es el mejor por AUC? Si dos
   modelos tienen un AUC casi igual pero costos distintos, la diferencia viene de **cómo deciden**, no de qué
   tan bien ordenan. Revisa también el recall de **cada** clase: una clase que casi nunca se predice es un
   riesgo de negocio aunque el costo se vea bien.

**Por qué el torneo no elige un campeón:** la optimización puede cambiar el orden, y solo la prueba
final (clase 4) dice qué tan bueno es el modelo que elegiste. El torneo descarta, no corona.

### 2.6 Lo que le pasó al ejemplo (con su dataset)

En CreditSight (5 pliegues por cliente, costo esperado):

| Modelo | Costo (± entre pliegues) |
|---|---|
| Bosque aleatorio | 0,482 ± 0,008 |
| LightGBM | 0,489 ± 0,007 |
| XGBoost | 0,490 ± 0,008 |
| Regresión logística | 0,553 ± 0,014 |
| Línea base (clase más frecuente) | 1,049 |

- El bosque ganó a los dos modelos de boosting **en los 5 pliegues** (diferencia de 1,4 a 2,2 veces
  su ruido); LightGBM y XGBoost **empataron**, así que se prefirió el más rápido.
- Fue también el modelo **más conservador** (mayor recall de la clase más cara, menor F1 macro): lo
  correcto para esa matriz de costos, pero implica más casos para revisar. **Una métrica de control
  puede decirte algo que la principal no ve.**
- **Las métricas técnicas del mismo torneo** contaron otra parte de la historia:

  | Modelo | AUC macro | PR-AUC `Poor` | Exactitud balanceada | Recall `Poor` | Recall `Good` |
  |---|---|---|---|---|---|
  | Bosque aleatorio | 0,853 | 0,717 | 0,538 | 0,824 | 0,058 |
  | LightGBM | 0,852 | 0,707 | 0,560 | 0,808 | 0,140 |
  | XGBoost | 0,852 | 0,710 | 0,562 | 0,807 | 0,142 |
  | Regresión logística | 0,819 | 0,632 | 0,501 | 0,774 | 0,024 |
  | Línea base | 0,500 | 0,291 | 0,333 | 0,000 | 0,000 |

  Los tres modelos de árboles **ordenan igual de bien** (AUC casi idéntico): su diferencia de costo viene de
  **cómo deciden**. Y el recall de `Good` (de 0,024 a 0,142) mostró que la regla de menor costo **casi nunca
  aprueba a un cliente bueno**: el negocio se protege, pero pierde casi todas esas oportunidades. El costo solo
  no lo habría dicho.
- Los finalistas fueron el bosque y LightGBM. Eso solo vale para ese dataset.

---

## 3. Optimización

Optimizar es probar muchas configuraciones. Sin cuidado, **se ajusta al ruido** de la propia
validación. Cinco cuidados:

### 3.1 Cuándo vale la pena

La ganancia de optimizar es la **distancia entre los parámetros de partida y lo que tus datos
necesitan**. Si los parámetros de partida ya son razonables, la ganancia puede estar dentro del
ruido, y eso es un resultado válido: se documenta y se usa la configuración de partida.

### 3.2 El espacio de búsqueda es conocimiento, no azar

Acota los rangos con lo que sabes del algoritmo. Ejemplos (ajusta a tu finalista):

| Modelo | Parámetros típicos a buscar |
|---|---|
| Boosting (LightGBM / XGBoost) | `learning_rate` (escala logarítmica, ~0,01–0,2), nº de árboles, `num_leaves` o `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda` |
| Bosque aleatorio | `max_depth`, `min_samples_leaf`, `max_features` (fija el nº de árboles) |

Un espacio **plausible pero equivocado** puede dar un resultado peor que no optimizar.

### 3.3 Presupuesto acotado

Define el número de ensayos **antes** (por ejemplo, 12–20 por modelo). Cada ensayo cuesta tiempo,
memoria y tiempo de clase. Más ensayos no siempre es mejor: con pocos
ensayos, una búsqueda aleatoria y una más inteligente (Optuna) suelen llegar al mismo lugar.

### 3.4 Reserva entidades que la búsqueda nunca ve

Separa, **dentro del entrenamiento**, un 25 % de las **entidades** (no de las filas):

```python
from sklearn.model_selection import GroupShuffleSplit

if ID_COL is not None:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    i_busqueda, i_reservado = next(gss.split(train, groups=train[ID_COL]))
    busqueda, reservado = train.iloc[i_busqueda], train.iloc[i_reservado]
```

La búsqueda trabaja solo con `busqueda`. El `test.joblib` sigue cerrado.

### 3.5 El sesgo optimista

La **mejor puntuación de una búsqueda es optimista**: es el máximo de muchas estimaciones con ruido.
Por eso no se reporta: se **comprueba** con las entidades reservadas.

### 3.6 La comprobación honesta

Entrena **dos** modelos con los datos de la búsqueda (parámetros de partida y parámetros
optimizados) y compáralos **en las entidades reservadas**, con un intervalo del 95 % que remuestrea
**entidades completas**, no filas (las filas de una misma entidad no son independientes):

```python
def intervalo_por_entidad(diferencia_por_fila, entidades, n=1000, seed=42):
    """diferencia_por_fila = costo(optimizado) - costo(partida), fila a fila. Negativo: optimizado mejor."""
    por_entidad = pd.DataFrame({"g": np.asarray(entidades), "d": np.asarray(diferencia_por_fila)}).groupby("g")["d"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(por_entidad), size=(n, len(por_entidad)))
    medias = por_entidad["sum"].to_numpy()[idx].sum(axis=1) / por_entidad["count"].to_numpy()[idx].sum(axis=1)
    return np.percentile(medias, [2.5, 97.5])
```

**Regla de decisión escrita antes de mirar:** "Uso los parámetros optimizados **solo si todo el
intervalo está por debajo de cero**. Si el intervalo cruza el cero, la mejora no se distingue del
ruido y uso los de partida."

### 3.7 Lo que cuenta el ejemplo

El notebook `material-del-curso/ejemplo-creditsight/notebooks/06_optimizacion.ipynb` hace exactamente esta comprobación con
CreditSight; su resultado lo verás en la demo. Lo que **sí** debes llevarte: la decisión se toma con la
regla del intervalo, no con la sensación de que "optimizar tiene que ayudar".

---

## 4. Tracking: registrar lo que probaste

El **tracking** guarda cada ejecución con lo necesario para **comparar, reproducir y auditar**. Se
empieza a usar en la optimización, que es donde hay decenas de ensayos que comparar.

### 4.1 Qué se registra en cada ejecución

| Qué | Ejemplos |
|---|---|
| **Parámetros** | Los hiperparámetros del ensayo, el nombre del modelo |
| **Métricas** | La métrica principal, su desviación entre pliegues, las de control, el tiempo de ajuste |
| **Etiquetas** | `etapa=06_optimizacion`, `modelo=<nombre>`, versión de los datos |

### 4.2 Nombres

- **Un experimento por etapa:** `<proyecto>-optimizacion`, `<proyecto>-modelo-final`.
- **Un nombre de ejecución que se entienda solo:** `lightgbm-ensayo-07`.
- En la clase 3 el seguimiento es **local**. Desde la clase 4, en Databricks, los experimentos van dentro de la carpeta de tu `EXPERIMENT_DIR` (la defines en `.env`).

### 4.3 Usar el módulo de la plantilla

`src/proyecto/tracking/mlflow_utils.py` hace la conexión por ti. **En la clase 3 se usa en modo local** (`force_local=True`, sin Databricks);
desde la clase 4 usa Databricks si hay conexión y, si no, cae solo a un MLflow **local**.

```python
from proyecto.tracking.mlflow_utils import configure_mlflow, set_experiment, log_run, uses_databricks

donde = configure_mlflow(force_local=True)  # clase 3: siempre "local"; desde la clase 4 puedes quitar force_local
experimento = set_experiment("mi-proyecto-optimizacion")
print("MLflow:", donde, "| experimento:", experimento)

run_id = log_run(
    run_name="lightgbm-ensayo-07",
    params={"learning_rate": 0.05, "num_leaves": 31},
    metrics={"costo": 0.49, "costo_std": 0.01},
    tags={"etapa": "06_optimizacion", "modelo": "lightgbm"},
)
```

La función `registered_model_name(nombre)` se usa más adelante, en el **registro del modelo** (clase 4).

### 4.4 Tracking no es registro

| | Tracking | Registro |
|---|---|---|
| Guarda | Ejecuciones (muchas, de prueba) | Versiones del modelo (pocas, candidatas a producción) |
| Pregunta | ¿Qué probamos y qué dio? | ¿Qué modelo usamos y en qué versión? |
| Cuándo | Hoy, en la optimización | Clase 4 |

### 4.5 Respaldo local

En la clase 3 el seguimiento **ya es local**: no necesitas Databricks. Si más adelante Databricks no conecta o se agotó la cuota diaria, **no te detengas**: pon `MLFLOW_FORCE_LOCAL=1`
en `.env`. Todo se guarda en `mlflow_local.db` (en la raíz de tu repo, ignorado por Git). Para verlo:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_local.db
```

Si ves el aviso "Databricks no disponible", léelo: te dice por qué (token vencido, CLI fuera del
PATH, cuota). Detalle en `material-del-curso/guias/databricks-mlflow.md`.

### 4.6 Comprueba que quedó registrado

Abre el experimento en la interfaz local (`mlflow ui`): ¿están **todos** los ensayos?, ¿cada uno
con sus parámetros y métricas?, ¿puedes ordenar por la métrica principal y ver los mejores?

---

## 5. Las 8 preguntas que el instructor hará en los puntos de control

Prepáralas con tu grupo; **las responderá la persona que no escribió el código**.

1. ¿Cuál es tu **idea de variable más prometedora** y qué **mejora / ruido** midió? ¿La conservaste?
2. ¿Hay alguna idea que **descartaste**? ¿Con qué cifra la descartaste y dónde quedó anotada?
3. ¿Cómo sabes que **todos los modelos** se evaluaron en **los mismos pliegues**?
4. ¿Qué te dice tu **línea base** sobre la métrica, y cuánto la mejora tu mejor modelo en **dinero o volumen**?
5. ¿Gana tu mejor modelo **en todos los pliegues** o solo en el promedio? ¿Cuál es su diferencia / ruido?
6. ¿Por qué elegiste a tus **finalistas** y por qué el torneo **no** decide el modelo definitivo?
7. ¿Cuántos **ensayos** hiciste y por qué ese número? ¿Qué **entidades reservaste** y por qué?
8. ¿Qué regla usaste para decidir si **usar los parámetros optimizados**, y qué resultado dio? ¿Dónde
   están tus ensayos registrados?

---

## 6. Trampas comunes

| Trampa | Por qué falla | Qué hacer |
|---|---|---|
| Medir cada idea con una partición distinta | Comparas ruido con ruido | Los mismos pliegues en todo (`pliegues`) |
| Quedarte con la idea porque "la media subió" | Puede estar dentro del ruido | Mejora / ruido > 1 y una regla escrita antes |
| Variables con información del futuro | Fuga: el modelo no podrá usarlas al predecir | Solo lo que se conoce al decidir |
| Validación por fila con entidades repetidas | El modelo "reconoce" a la entidad | Pliegues por entidad |
| Solo guardar el promedio de los pliegues | No puedes comparar de a pares | Guarda una fila por pliegue |
| Elegir con exactitud (accuracy) con clases desbalanceadas | La clase mayoritaria ya da un número alto | Métrica de negocio y de control |
| Declarar campeón al ganador del torneo | La optimización puede cambiar el orden | Elige 1–2 finalistas |
| Reportar la mejor puntuación de la búsqueda | Es optimista (el máximo de muchas con ruido) | Comprobar con entidades reservadas |
| Remuestrear filas para el intervalo | Las filas de una entidad no son independientes | Remuestrear entidades completas |
| Mirar `test.joblib` "solo para ver" | Deja de medir generalización | No se toca hasta la clase 4 |
| Cientos de ensayos sin presupuesto | Agota memoria y cuota, y sobreajusta la validación | Ensayos acotados, definidos antes |
| Pegar el resultado de Claude sin ejecutarlo | Puede inventar columnas o cifras | Ejecuta y compara con tus propios cálculos |
| Copiar las decisiones de CreditSight | Son de otro dataset | Úsalo para ver **cómo** razonar, no **qué** decidir |

---

## Referencias del ejemplo

- `material-del-curso/ejemplo-creditsight/notebooks/04_features.ipynb` · `material-del-curso/ejemplo-creditsight/reports/04_features_tabla.csv`: tres ideas medidas con
  su mejora / ruido y las decisiones.
- `material-del-curso/ejemplo-creditsight/notebooks/05_torneo.ipynb` · `material-del-curso/ejemplo-creditsight/reports/05_torneo_tabla.csv`: torneo con comparación
  pareada y traducción de la diferencia a volumen.
- `material-del-curso/ejemplo-creditsight/notebooks/06_optimizacion.ipynb` · `material-del-curso/ejemplo-creditsight/src/creditsight/tracking/mlflow_utils.py`: búsqueda
  con registro de cada ensayo y comprobación con entidades reservadas.
- `material-del-curso/guias/databricks-mlflow.md`: conexión, experimentos y respaldo local.
- `material-del-curso/kit/checklist-revision-ia.md`: cómo revisar lo que genere Claude.
