# Guía de trabajo: validación, registro del modelo y app (clase 4 · jueves 5 de noviembre)

Esta guía es para **tu dataset**, no para el de CreditSight. El ejemplo del instructor
(sus notebooks `07_validacion_interpretacion`, `08_registro_modelo` y `09_inferencia`, **en preparación**, y su módulo
`material-del-curso/ejemplo-creditsight/src/creditsight/inference/`) te muestra **cómo se razona y cómo se arma**, pero sus decisiones y
resultados no valen para otros datos. Cada cifra de tu proyecto la tienes que medir tú.

**Regla de oro del día:** el conjunto de prueba se abre **una sola vez**, con las decisiones **ya escritas**.
Todo lo que decidas después de verlo es una decisión que la prueba ya no puede juzgar.

**Lo que se te pide al cerrar el día (hito H2):**
- la **prueba abierta una sola vez**, con **intervalo**, comparación con el **baseline** y análisis de **errores**,
- el **modelo registrado con versión** (y alias), y exportado a `models/model.joblib`,
- un **`src/` con pruebas** (`pytest` en verde),
- una **app que corre en local** y carga ese modelo.

El despliegue en internet es el viernes (clase 5).

---

## 0. Antes de empezar

- Necesitas `data/processed/train.joblib` y `test.joblib` (los deja tu notebook 03), tu **modelo final tentativo**
  (lo elegiste en el torneo y la optimización) y las secciones del reporte de las clases anteriores.
- **`test.joblib` no se toca hasta la sección 3.** Si ya lo miraste para decidir algo, **dilo** en
  `reports/07_validacion.md`: es mejor reportar esa limitación que esconderla.
- Define al inicio de cada notebook tu **ID_COL** (columna de la entidad que se repite, o `None`) y tu **TARGET**.
- Si `python scripts/verificar_entorno.py` marca algún `FALTA`, arréglalo antes de empezar.
- Los esqueletos están en `notebooks/07_validacion_interpretacion.ipynb`, `08_registro_modelo.ipynb` y
  `09_inferencia.ipynb`.

---

## 1. Decisiones congeladas

Antes de abrir la prueba escribe, en la sección 1 del notebook 07 y en `reports/07_validacion.md`:

| Decisión | Ejemplo de respuesta |
|---|---|
| Modelo final | "random forest" |
| Parámetros | "los optimizados" **o** "los de partida: la optimización no mejoró de forma distinguible" |
| ¿Calibrar? | "sí/no, decidido con entidades reservadas" |
| Métrica principal | la **de negocio** con la que elegiste finalistas |
| Regla de decisión | "clase más probable", "umbral de 0,3" o "menor costo esperado" |
| Fecha y hora | cuándo lo congelaron |

**Cómo elegir el modelo final sin mirar la prueba.** Usa lo que ya tienes: la comparación del torneo y la
comprobación de la optimización en datos reservados. Una regla simple y honesta:

1. Usa los **parámetros optimizados solo si la mejora fue distinguible del ruido** (el intervalo de la diferencia
   excluía el cero). Si no, quédate con los de partida: son más simples y no arriesgan haberse ajustado al ruido
   de la búsqueda.
2. Entre los finalistas, gana el de menor pérdida en esos datos reservados; si empatan, el más simple de operar.

> **Por qué se escribe.** Es lo que le permite al instructor (y a ti en el reporte) comprobar que la prueba midió
> generalización y no sirvió para decidir.

---

## 2. ¿Hay que calibrar las probabilidades?

Si tu decisión usa la **probabilidad** (un umbral, un costo, "revisar si dudo"), esa probabilidad tiene que ser creíble:
si el modelo dice 70 % y acierta el 55 %, la decisión sale mal. Si decides solo con la clase más probable, la
calibración importa menos, y puedes decirlo y seguir.

Compruébalo **con entidades reservadas del entrenamiento** (las mismas de la optimización), **nunca con la prueba**:

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# 1) Reserva el 25 % de las ENTIDADES dentro del entrenamiento
idx_busqueda, idx_reserva = next(GroupShuffleSplit(test_size=0.25, random_state=7).split(X, y, groups=train[ID_COL]))
X_b, y_b = X.iloc[idx_busqueda], y.iloc[idx_busqueda]
X_r, y_r = X.iloc[idx_reserva], y.iloc[idx_reserva]

# 2) Modelo sin calibrar y calibrado, ajustados solo con el 75 %
sin_calibrar = build_pipeline(modelo).fit(X_b, y_b)
pliegues = list(StratifiedGroupKFold(3, shuffle=True, random_state=1).split(X_b, y_b, groups=train[ID_COL].iloc[idx_busqueda]))
calibrado = CalibratedClassifierCV(build_pipeline(modelo), method="isotonic", cv=pliegues).fit(X_b, y_b)

# 3) Compara en el 25 % reservado: log loss y curva de calibración
from sklearn.metrics import log_loss
print(log_loss(y_r, sin_calibrar.predict_proba(X_r)), log_loss(y_r, calibrado.predict_proba(X_r)))
frec, media = calibration_curve((y_r == 1).astype(int), sin_calibrar.predict_proba(X_r)[:, 1], n_bins=10, strategy="quantile")
```

**Cómo decidir (escríbelo antes de mirar los números):** se calibra si mejora la calidad de las probabilidades **y**
no empeora tu métrica de negocio. Una calibración que mejora el log loss pero encarece la decisión **no** se usa.

---

## 3. Abrir la prueba — una sola vez

1. Entrena el **modelo final con todo el entrenamiento** (con la calibración solo si la decidiste arriba) y guárdalo
   entero (`models/final_pipeline.joblib`).
2. Abre `test.joblib`. El notebook deja una marca en `reports/prueba_abierta.txt`: **quedará constancia de cuándo**.
3. Calcula tu métrica principal y las de control, **y la del baseline** (el del torneo) en la misma prueba.
4. Pon un **intervalo** remuestreando **entidades**, no filas. Las filas de una misma entidad no son independientes:
   remuestrear filas daría un intervalo falsamente estrecho.

```python
import numpy as np
import pandas as pd

def bootstrap_por_entidad(valores, entidades, n_boot=1000, seed=42):
    """Promedio de `valores` con intervalo del 95 %, remuestreando entidades completas."""
    tabla = pd.DataFrame({"g": np.asarray(entidades), "v": np.asarray(valores, dtype=float)})
    por_entidad = tabla.groupby("g")["v"].agg(["sum", "count"])
    sumas, cuentas = por_entidad["sum"].to_numpy(), por_entidad["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(sumas), size=(n_boot, len(sumas)))
    remuestras = sumas[idx].sum(axis=1) / cuentas[idx].sum(axis=1)
    return float(sumas.sum() / cuentas.sum()), float(np.percentile(remuestras, 2.5)), float(np.percentile(remuestras, 97.5))

# perdida_modelo y perdida_baseline: UNA pérdida por fila de la prueba (p. ej. costo, o 0/1 de error)
media, bajo, alto = bootstrap_por_entidad(perdida_modelo, test[ID_COL])
diferencia = bootstrap_por_entidad(perdida_modelo - perdida_baseline, test[ID_COL])   # negativa = el modelo pierde menos
```

(Esta función ya viene en el notebook 07.) **Cómo leerla:** si todo el intervalo de la **diferencia** está por debajo de
cero, tu modelo pierde menos que el baseline con confianza del 95 %. Si el intervalo cruza el cero, **no hay evidencia**
de que el modelo sea mejor, aunque el promedio sea menor.

> **Si algo no te gusta de lo que ves en la prueba:** no la abras de nuevo para "arreglarlo". Anótalo como límite del
> modelo en el reporte. Cambiar el modelo mirando la prueba convierte la prueba en un segundo conjunto de entrenamiento.

---

## 4. Errores por segmento, carga de trabajo e interpretación

### 4.1 Errores por segmento
Un buen promedio puede esconder un grupo al que el modelo le va mal. Elige **1 o 2 variables de segmento** con sentido
para tu negocio (tramos de edad, región, tipo de cliente…) y calcula, por segmento: número de casos, tu métrica, el
**% que el modelo marca** como positivo y el **% real**. La señal de alarma es un segmento donde esas dos cifras se
separan mucho.

### 4.2 Carga de trabajo
Si tu modelo **prioriza revisión humana**, mide cuántos casos marca frente a cuántos hay realmente. Un modelo
conservador puede tener el mejor costo y a la vez **duplicar el trabajo** del analista: esa cifra no aparece en la
métrica y hay que reportarla.

### 4.3 Interpretación
Los valores **SHAP** (o la importancia por permutación) muestran qué variables mueven la predicción. Tres límites:

- **Explican al modelo, no al mundo.** No dicen si el modelo acertó.
- **Las variables correlacionadas se reparten el crédito:** quitar una puede subir el aporte de otra.
- **Una importancia alta no es causa.**

Revisa si entre las variables hay **atributos sensibles o de uso regulado** (edad, género, ubicación…): decláralo en el
reporte y en la ficha, y di qué haría falta antes de usarlo en producción.

---

## 5. Registro del modelo

El **tracking** guardó *ejecuciones* (muchas, de prueba): responde "¿qué probamos?". El **registro** guarda
*versiones del modelo* (pocas): responde "**¿qué modelo usamos y en qué versión?**".

| | Tracking | Registro |
|---|---|---|
| Guarda | Ejecuciones | Versiones del modelo |
| Pregunta | ¿Qué probamos? | ¿Cuál usamos y cuál es su versión? |

Un **alias** (`champion`) es una etiqueta móvil: quien use el modelo pide "el champion" y no le importa qué número de
versión es.

**Qué registrar junto al modelo:** parámetros y tipo de modelo, **métricas de la prueba**, una **firma** (columnas y tipos
de entrada), un **ejemplo de entrada** y etiquetas (de qué datos y de qué etapa sale).

```python
import mlflow
from proyecto.tracking.mlflow_utils import configure_mlflow, registered_model_name, set_experiment

configure_mlflow(); set_experiment("mi-proyecto-modelo-final")
with mlflow.start_run(run_name="modelo-final"):
    mlflow.log_params({"modelo": "random forest"})
    mlflow.log_metrics({"perdida_prueba": 0.0})   # tus números reales, como float
    mlflow.sklearn.log_model(
        modelo_final, name="model", input_example=ejemplo_de_entrada,
        registered_model_name=registered_model_name("mi_modelo"),
        serialization_format="cloudpickle",
    )
```

> **Por qué `serialization_format="cloudpickle"`.** MLflow guarda por defecto los modelos de scikit-learn en un formato
> que **rechaza un `Pipeline` con preprocesamiento** (error "untrusted types"). Con `cloudpickle` funciona. Pero un
> archivo *pickle* **puede ejecutar código al cargarse**: carga solo modelos que guardaste tú o tu grupo.

Después, dale el alias y **comprueba la ida y vuelta** (carga desde el registro y compara predicciones con el original):

```python
cliente = mlflow.MlflowClient()
version = max(int(v.version) for v in cliente.search_model_versions(f"name = '{nombre_completo}'"))
cliente.set_registered_model_alias(nombre_completo, "champion", str(version))
recuperado = mlflow.sklearn.load_model(f"models:/{nombre_completo}@champion")
assert np.allclose(modelo_final.predict_proba(muestra), recuperado.predict_proba(muestra))
```

**Respaldo local:** si Databricks no conecta o se agotó la cuota (`MLFLOW_FORCE_LOCAL=1` en `.env`), el registro funciona
igual en tu equipo (`mlflow_local.db`). Deja escrito en `reports/08_registro.md` cuál usaste y por qué.

---

## 6. Exportar el modelo y escribir el esquema de entrada

La app **no consulta Databricks**: no se ponen credenciales en la nube, y el despliegue no depende de la cuota de ninguna
plataforma. Por eso el modelo viaja **como archivo dentro del repositorio**, pero **siempre se exporta desde la versión
`champion` del registro**: el registro es la fuente de verdad.

```python
modelo = mlflow.sklearn.load_model(f"models:/{nombre_completo}@champion")
joblib.dump(modelo, "models/model.joblib")
```

El **esquema de entrada** (`models/input_schema.json`) dice qué columnas espera el modelo, qué rangos son plausibles y
qué categorías conoce. La app lo usa para construir su formulario y `predict` para validar:

```python
from proyecto.inference.schema import build_schema

schema = build_schema(
    train, numeric=NUMERICAS, categorical=CATEGORICAS, class_names=CLASES,
    cost_matrix=None,                      # o tu matriz k x k si decides por menor costo esperado
    valid_range={"edad": (0, 120)},        # lo POSIBLE en el mundo real, no lo observado
    path="models/input_schema.json",
)
```

> **`valid_range` es conocimiento tuyo.** Una edad de 200 es imposible aunque nunca haya aparecido en tus datos. Fuera
> de ese rango la predicción **se niega a responder** y explica por qué. El *rango observado* (de tus datos) solo
> genera una advertencia.

---

## 7. `predict` que valida, y la ficha del modelo

Prueba la función con tres filas: una **normal**, una **incompleta** (el usuario marcó "no sé") y una con un dato
**imposible**.

```python
from proyecto.inference.predict import InputValidationError, load_model, predict_with_scores
from proyecto.inference.schema import load_schema

modelo, schema = load_model("models/model.joblib"), load_schema("models/input_schema.json")
print(predict_with_scores(fila_normal, modelo, schema))
print(predict_with_scores(fila_con_un_dato_faltante, modelo, schema))
try:
    predict_with_scores(fila_con_dato_imposible, modelo, schema)
except InputValidationError as error:
    print("Error claro:", error)
```

Lo que debes ver: las dos primeras responden; la tercera **se niega y explica qué corregir**. Un modelo que contesta con
cualquier cosa es más peligroso que uno que dice "no".

**Comprueba que el exportado es el que evaluaste:** calcula tu métrica principal en la prueba con `models/model.joblib`
y compárala con la de `reports/07_metricas_test.json`. Si no coinciden, estás entregando otro modelo.

**La ficha del modelo** (`models/ficha_modelo.md`): qué es, con qué datos, rendimiento en la prueba **con su
intervalo**, variables que más pesan, **límites y usos no recomendados**, y lo que **no** se evaluó. Complétala **a mano**
con tus cifras reales.

---

## 8. De notebook a módulos, con pruebas

La regla del curso desde la clase 2: **la lógica reutilizable vive en `src/`; los notebooks la importan y solo
orquestan.** Hoy repasa que se cumpla:

| ¿Está en `src/`? | Qué |
|---|---|
| ✔ (desde la clase 2) | limpieza y partición (`src/<paquete>/data/`) |
| ✔ (clase 3) | pipeline y torneo (`src/<paquete>/models/`) |
| **hoy** | predicción y esquema (`src/<paquete>/inference/`, ya vienen en la plantilla) |

**Pruebas mínimas que se esperan:**

1. **La partición:** ninguna entidad aparece en dos particiones (ya la hiciste en la clase 2).
2. **La inferencia:** `predict` **rechaza un dato imposible** y acepta uno incompleto.
3. **El esquema:** `build_schema` guarda y recarga lo que espera.

La plantilla trae `tests/test_inference.py` con un **modelo de juguete**, como ejemplo de qué comprobar. Cuando tengas tu
modelo real, **añade una prueba con una fila de tus datos**. Ejecuta todo con:

```bash
pytest -q
```

> Si renombraste el paquete `proyecto`, cambia los `import proyecto...` de `tests/`. Las pruebas son tuyas: léelas.

---

## 9. La app mínima

La plantilla trae `app/streamlit_app.py`: un formulario que **se construye solo** desde `models/input_schema.json`,
con casilla **"No sé"** en cada campo, validación, probabilidades y un aviso de uso responsable. Con
`models/model.joblib` y `models/input_schema.json` en su sitio:

```bash
streamlit run app/streamlit_app.py
```

**Comprueba que:**
- el formulario aparece con los campos de tu modelo,
- una predicción normal responde con clase y probabilidades,
- **"No sé" no rompe nada**,
- un dato imposible muestra un **error claro**, no una predicción.

**Personalizarla** (títulos, etiquetas legibles, aviso de uso responsable): edita las claves `app_title`, `labels` y
`responsible_use` de `models/input_schema.json`, o pídele a Claude cambios **pequeños, de uno en uno**, y pruébalos
con `streamlit run` antes de pedir el siguiente.

> **Si tu problema es de regresión o de clusterización:** la plantilla está pensada para **clasificación**. En regresión la
> salida es un número (muestra una estimación y un rango de error en lugar de probabilidades); en clusterización, el
> segmento asignado y su perfil. Pide a Claude la adaptación de `predict.py` y de la app, y deja las pruebas.

**Hoy la app solo tiene que correr en local.** El despliegue en internet es la clase 5; para llegar listo, comprueba que
`requirements.txt` está en la **raíz** y que `models/model.joblib` y `models/input_schema.json` **no están en `.gitignore`**.
`models/final_pipeline.joblib` es un archivo intermedio: **no hace falta subirlo** (y GitHub rechaza archivos de más de
100 MB).

---

## 10. Las 8 preguntas que el instructor hará en los puntos de control

1. ¿Qué decisiones congelaron **antes** de abrir la prueba, y dónde están escritas?
2. ¿Por qué la prueba se abre **una sola vez**? ¿Qué pasa si la usas para ajustar el modelo?
3. ¿Qué intervalo tiene su métrica, y por qué se remuestrean **entidades** y no filas?
4. ¿Su modelo mejora al baseline **con confianza**, o el intervalo cruza el cero?
5. ¿En qué segmento le va peor al modelo? ¿Cuánta **carga de trabajo** genera frente a la real?
6. ¿Qué **no** dicen los valores SHAP? ¿Hay atributos sensibles entre sus variables?
7. ¿Qué diferencia hay entre **tracking** y **registro**? ¿Para qué sirve el alias `champion`?
8. ¿Qué responde `predict` si la edad es −500? ¿Y si el usuario marca "No sé"?

---

## 11. Trampas comunes

- **Mirar la prueba "solo para ver" antes de congelar decisiones.** Ya no es una prueba.
- **Cambiar de modelo o de umbral después de ver la prueba.** Si lo haces, decláralo.
- **Un intervalo remuestreando filas.** Sale demasiado estrecho y da falsa confianza.
- **Calibrar con la prueba** o con los mismos datos con los que ajustaste el modelo.
- **Registrar sin comprobar la ida y vuelta.** Un registro que no se puede recuperar no sirve.
- **Exportar un modelo distinto al evaluado.** Compara la métrica del `model.joblib` con la de la prueba.
- **`valid_range` copiado de los datos observados.** Es lo *posible*, no lo *visto*.
- **Pegar el token de Databricks en la app o en el repo.** La app no lo necesita; `.env` no se sube.
- **Aceptar de Claude código de la app que no entiendes.** Cada integrante debe poder explicarlo.

---

## Referencias del ejemplo

> **En preparación:** los notebooks `07`, `08` y `09` del ejemplo **todavía no existen como archivos** (se generan y se
> ejecutan después de la etapa de optimización), así que aún no hay resultados para consultar.

- `material-del-curso/ejemplo-creditsight/notebooks/07_validacion_interpretacion.ipynb`, `08_registro_modelo.ipynb` y `09_inferencia.ipynb`
- `material-del-curso/ejemplo-creditsight/src/creditsight/inference/predict.py` (el equivalente del módulo de la plantilla, con matriz de costos)
- `material-del-curso/ejemplo-creditsight/app/streamlit_app.py` (la app de CreditSight)
- `material-del-curso/guias/databricks-mlflow.md` (conexión y registro)
