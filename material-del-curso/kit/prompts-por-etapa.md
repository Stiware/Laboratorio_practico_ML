# Prompts por etapa

Cómo usarlos: abre la conversación **dentro de tu proyecto** (Claude lee `CLAUDE.md` y
`especificacion-proyecto.md`). Cambia lo que está entre `<...>`. Pide una cosa por vez y revisa el
resultado con `checklist-revision-ia.md` antes de continuar.

## Prompts para gastar poca cuota

- **Ancla el archivo:** "En `src/proyecto/data/cleaning.py`, convierte la columna `<col>` a número."
- **Una etapa, una conversación:** abre una conversación nueva al cambiar de etapa.
- **Pide el plan antes del código:** "Dime en 5 viñetas qué harías para `<tarea>`. No escribas código todavía."
- **Pide solo el cambio:** "Muéstrame solo la función que cambia, no el archivo completo."
- **Resumen para retomar:** "Resume en 10 líneas lo decidido hasta ahora para pegarlo en una conversación nueva."

---

## 1. Caso de negocio
**Prompt:** "Ayúdame a completar la sección 2 de `especificacion-proyecto.md` con mi caso: `<describe el problema>`. Hazme 5 preguntas sobre quién decide y qué cuesta equivocarse antes de escribir nada."
**Qué esperar:** preguntas concretas y una ficha redactada para no técnicos, con la decisión y los costos de error.
**Cómo revisarlo:** ¿se entiende sin saber ML? ¿dice qué error es peor? ¿la métrica elegida refleja ese costo?

## 2. EDA
**Antes de pedir nada:** abre `guia-eda-y-preprocesamiento.md` y define tu **ID_COL** (la columna que identifica a la entidad) y tu **TARGET**.

**Prompt 1 (plan, sin código):** "Lee `especificacion-proyecto.md` y mira `data/raw/<archivo>.csv` sin imprimir columnas con datos personales. Dime en 5 viñetas qué revisarías en el EDA siguiendo las 5 preguntas de `kit-estudiante/guia-eda-y-preprocesamiento.md` (unidad de análisis, objetivo, suciedad, variables que deberían ser estables, sospecha de fuga). No escribas código todavía."

**Prompt 2 (una pregunta por vez):** "En `notebooks/02_eda.ipynb`, sección `<1, 2, 3, 4 o 5>`, completa la celda de código para mi dataset (ID_COL=`<columna>`, TARGET=`<columna>`). Usa pandas, muestra cifras y no cambies otras secciones. Dime cómo reproducir cada porcentaje."

**Prompt 3 (calidad de datos):** "Revisa mi dataset con los chequeos de la sección 3 de la guía: números guardados como texto, valores centinela, valores imposibles mirando los percentiles (p90, p95, p97, p99, máximo), vacíos y qué podrían significar, duplicados y categorías raras. Dame una tabla con la cifra de cada problema. No corrijas nada todavía."

**Prompt 4 (decisiones):** "Con estos hallazgos `<pega la tabla>`, ayúdame a redactar en `reports/01_eda.md` la tabla hallazgo → decisión → motivo. Cada fila debe tener una cifra. Dime si alguna decisión es vaga o no tiene motivo."

**Qué esperar:** tablas y gráficos con cifras, una lista de problemas de datos reproducible y **al menos 5 decisiones** escritas con su motivo.

**Cómo revisarlo (con la guía):**
- ¿Pude **reproducir** los porcentajes ejecutando la celda yo mismo?
- ¿Respondí la **pregunta 1** (unidad de análisis) por escrito y dije si una entidad se repite?
- ¿Cada decisión tiene **cifra, acción y motivo**? (una decisión sin motivo no cuenta)
- ¿Revisé **valores imposibles con percentiles** y no solo con `describe()`?
- ¿Claude **inventó** alguna columna o cifra? Si dudas, ejecuta `df.columns` y compara.
- ¿Imprimió **datos personales**? Si pasó, elimínalos de la salida del notebook antes de guardarlo.

## 3. Preprocesamiento
**Antes de pedir nada:** tu tabla de decisiones del EDA (`reports/01_eda.md`) es la entrada. Una decisión = una función = una prueba.

**Prompt 1 (una decisión):** "Del `reports/01_eda.md`, decisión número `<n>`: `<pega la decisión>`. Escribe en `src/proyecto/data/cleaning.py` una función pequeña que la implemente, con docstring en español. Añade una prueba en `tests/test_cleaning.py` con un ejemplo mínimo hecho a mano (el caso normal y el caso sucio). No toques otras funciones."

**Prompt 2 (la función que une todo):** "En `src/proyecto/data/cleaning.py`, crea `clean(df)` que aplique en orden las funciones ya escritas y devuelva un `DataFrame` con el **mismo número de filas** que la entrada. Elimina las columnas personales `<lista>` al cargar los datos. No modifiques nada de `data/raw/`."

**Prompt 3 (partición por la unidad de análisis):** "En `src/proyecto/data/split.py`, escribe `split_by_group(df, group_col, test_size, seed)` con `GroupShuffleSplit`, partiendo por `<unidad>`. Añade en `tests/test_split.py` una prueba con `assert` de que **ninguna entidad aparece en entrenamiento y prueba a la vez**. Después demuéstrame que esa prueba **falla** si se parte por fila con `train_test_split`."

**Prompt 4 (orquestar):** "En `notebooks/03_preprocesamiento.ipynb`, deja que el notebook solo **llame** a `clean` y `split_by_group`, muestre las comprobaciones y guarde `data/processed/train.joblib` y `test.joblib`. Sin lógica reutilizable dentro del notebook."

**Qué esperar:** funciones pequeñas con docstring, pruebas que pasan, la limpieza corriendo de principio a fin y una verificación de que ninguna entidad cae en dos particiones.

**Cómo revisarlo (con la guía y `checklist-revision-ia.md`):**
- `pytest -q` en **verde**.
- La prueba de partición **falla** si parto por fila (compruébalo: una prueba que no puede fallar no protege nada).
- `data/raw/` **no cambió** (`git status` no lo muestra modificado).
- ¿Hay alguna estadística de **todo el dataset** (media, mediana, categorías) calculada antes de partir? Eso va dentro del pipeline, no en la limpieza.
- ¿Puedo explicar **cada línea** de las funciones? Si no, pídele que la explique antes de aceptarla.
- ¿Los datos personales quedaron **fuera** del `DataFrame` limpio?

## 4. Feature engineering
**Antes de pedir nada:** abre `guia-modelado-y-optimizacion.md` (sección 1) y ten a mano tu **ID_COL**, tu **TARGET**, tu **métrica principal** (la de tu ficha de caso) y una **regla de decisión escrita**: por ejemplo, "conservo una idea solo si mejora / ruido > 1 y se conoce al predecir".

**Prompt 1 (ideas, sin código):** "Lee `especificacion-proyecto.md` y `reports/01_eda.md`. Propón 3 ideas de variables para mi caso. Para cada una dime: cómo se calcula, por qué podría ayudar y **si se conoce al momento de predecir**. Descarta las que usen información del futuro o del objetivo. No escribas código todavía."

**Prompt 2 (validación común):** "En `notebooks/04_features.ipynb`, sección 2, crea los pliegues **una sola vez**: `StratifiedGroupKFold` (o `GroupKFold` si el objetivo es numérico) por `<ID_COL>`, 5 pliegues, semilla 42, guardados en una lista. Añade un `assert` de que ninguna entidad cae en los dos lados de un pliegue. Estos pliegues se reutilizan en **todas** las comparaciones."

**Prompt 3 (una idea por vez):** "En `src/proyecto/features/`, implementa la idea `<n>`: `<descripción>` como una función pequeña con docstring en español. Añade en `tests/test_features.py` una prueba que cubra división por cero, vacíos y que no modifica el original. Debe ejecutarse **dentro del pipeline**, ajustada solo con entrenamiento."

**Prompt 4 (medir con ruido):** "En `notebooks/04_features.ipynb`, mide la referencia y cada variante con el **mismo modelo de parámetros fijos** `<modelo>`, los pliegues del paso 2 y la métrica principal `<métrica>` más una de control `<control>`. Muestra por variante la media, la **desviación entre pliegues** y una columna `mejora / ruido` frente a la referencia. **No decidas por mí.**"

**Prompt 5 (decisiones):** "Con esta tabla `<pégala>`, redacta en `reports/04_features.md` una fila por idea: decisión (conservar o descartar), la **cifra** que la respalda y el motivo, aplicando mi regla `<regla>`. Si alguna conclusión no la respalda la cifra, dímelo."

**Qué esperar:** una tabla de ideas con su **mejora / ruido**, código reutilizable en `src/proyecto/features/` con pruebas y, probablemente, que **alguna idea no sirva**: descartarla con su cifra es un resultado válido.

**Cómo revisarlo (con la guía y `checklist-revision-ia.md`):**
- ¿Todas las variantes usan **los mismos pliegues**, partidos por entidad si hay entidades repetidas?
- ¿Alguna variable usa **información del futuro** o algo que solo se conoce después de la decisión?
- ¿La `mejora / ruido` se calculó con la desviación **entre pliegues** y no con una sola cifra?
- ¿Las ideas **descartadas** quedaron anotadas con su cifra?
- ¿Puedo **reproducir** una de las cifras ejecutando la celda yo mismo?
- ¿Todo lo que "aprende" de los datos (medias, escalas, categorías) está **dentro del pipeline**?

## 5. Torneo de modelos
**Antes de pedir nada:** abre `guia-modelado-y-optimizacion.md` (sección 2). Usa **los mismos pliegues** del notebook 04 y las variables que decidiste conservar. El torneo **descarta**: elige 1–2 finalistas, no un campeón. Reporta **métricas de negocio y técnicas**, siempre las dos.

**Prompt 1 (plan, sin código):** "Dime en 5 viñetas cómo harías el torneo en `notebooks/05_torneo.ipynb`: una línea base, un modelo simple e interpretable, un bosque y un modelo de boosting, con los **mismos pliegues** por `<ID_COL>`. La métrica **principal** es la de negocio `<métrica de negocio>`; además quiero las **métricas técnicas**: AUC macro y de la clase de interés `<clase>`, PR-AUC, exactitud balanceada, precisión y recall por clase, F1 macro y log loss (si mi problema es de regresión: RMSE, MAE y R²). No escribas código todavía."

**Prompt 2 (función reutilizable):** "En `src/proyecto/models/train.py`, escribe `build_pipeline(modelo)` (preprocesamiento + modelo dentro de un `Pipeline`) y una función que evalúe un pipeline en una lista de pliegues dada y devuelva un `DataFrame` con **una fila por pliegue**: la métrica de negocio, **todas las métricas técnicas** (AUC macro, AUC de la clase de interés, PR-AUC, exactitud balanceada, precisión y recall por clase, F1 macro, log loss) y el tiempo de ajuste. Añade una prueba en `tests/` con un modelo trivial y comprueba que con probabilidades perfectas el AUC y la exactitud dan 1."

**Prompt 3 (correr el torneo):** "En `notebooks/05_torneo.ipynb`, define los candidatos `<lista>` con parámetros razonables (sin optimizar), evalúalos con **los mismos pliegues** del notebook 04 y guarda el resultado de **cada modelo por pliegue**. Muestra **dos tablas** ordenadas con media ± desviación entre pliegues (una de **negocio** y una de **métricas técnicas**) y un `boxplot` por modelo."

**Prompt 4 (técnicas frente a negocio):** "Con las dos tablas del paso anterior, añade una tabla de **posiciones** (1 = mejor) por cada métrica y dime si el mejor por la métrica de negocio también es el mejor por AUC y por F1. Muestra el **recall de cada clase** por modelo y avísame si alguna clase casi nunca se predice. Explica en 3 frases qué significa cada diferencia que encuentres: ¿los modelos ordenan igual de bien y deciden distinto?"

**Prompt 5 (comparación pareada):** "Añade una función `comparar(mejor, otro)` que devuelva la diferencia media, en cuántos pliegues gana el mejor y la diferencia / ruido. Aplícala a los tres mejores modelos."

**Prompt 6 (dinero y finalistas):** "Traduce la diferencia entre el mejor modelo y la línea base a dinero o volumen con `<decisiones al mes>`. Luego, con las tablas `<pégalas>`, redacta en `reports/05_torneo.md` 1–2 finalistas con su motivo (consistencia, métrica de negocio, **qué dicen las métricas técnicas**, tiempo de ajuste, interpretabilidad) y explica por qué el torneo no elige un campeón."

**Qué esperar:** una tabla de negocio y una tabla de **métricas técnicas** con media y desviación entre pliegues, la comparación de rankings, la comparación pareada de los mejores, la diferencia en dinero o volumen y **1–2 finalistas** justificados.

**Cómo revisarlo (con la guía y `checklist-revision-ia.md`):**
- ¿La validación es **por entidad** y **todos los modelos** usan los mismos pliegues?
- ¿Hay una **línea base** y todos los modelos la superan con holgura? Si no, ¿por qué?
- ¿Se guardó **cada pliegue** (no solo el promedio)?
- ¿El mejor gana en **todos** los pliegues o solo en el promedio? ¿Su diferencia / ruido supera 1?
- ¿La métrica principal es **la del negocio** y hay además **métricas técnicas** (AUC, PR-AUC, exactitud balanceada, precisión y recall por clase), no solo una?
- ¿Miraste el **recall de cada clase**? ¿Alguna casi nunca se predice? El costo puede esconderlo.
- ¿Coinciden los rankings de negocio y técnico? Si no, ¿sabes explicar por qué?
- ¿Se abrió `test.joblib`? **No debe abrirse hoy.**

## 6. Optimización con tracking
**Antes de pedir nada:** abre `guia-modelado-y-optimizacion.md` (secciones 3 y 4). Comprueba que `python scripts/verificar_entorno.py` no muestra ningún `FALTA`, y recuerda que **en esta clase el seguimiento es local** (sin Databricks; entra en la clase 4). Escribe **antes de empezar** tu presupuesto de ensayos y tu regla de decisión.

**Prompt 1 (plan, sin código):** "Dime en 5 viñetas cómo optimizarías `<finalista>` con un presupuesto de `<N>` ensayos, reservando el 25 % de las **entidades** que la búsqueda no verá. Propón un rango para cada hiperparámetro y explica por qué. No escribas código todavía."

**Prompt 2 (conexión, sin reescribir nada):** "Usa el módulo que ya existe en `src/proyecto/tracking/mlflow_utils.py` (no lo reescribas). En `notebooks/06_optimizacion.ipynb`, conéctate con `configure_mlflow(force_local=True)` (en esta clase el seguimiento es local, sin Databricks), crea el experimento `<proyecto>-optimizacion` con `set_experiment` y muestra en pantalla dónde quedó (debe decir `local`) y los avisos que aparezcan."

**Prompt 3 (entidades reservadas):** "Separa el 25 % de las entidades de `data/processed/train.joblib` con `GroupShuffleSplit` (semilla 7) como `reservado`; la búsqueda usa solo el resto. Añade un `assert` de que no comparten ninguna entidad. **No abras `test.joblib`.**"

**Prompt 4 (búsqueda con registro):** "Con Optuna y como máximo `<N>` ensayos, optimiza `<finalista>` con validación cruzada por `<ID_COL>` en los datos de búsqueda, minimizando `<métrica>`. **Cada ensayo se registra** con `log_run(...)`: sus hiperparámetros, la métrica principal con su desviación entre pliegues, el tiempo y las etiquetas `etapa` y `modelo`. Rangos: `<tabla de rangos>`."

**Prompt 5 (comprobación honesta):** "Entrena dos modelos con los datos de búsqueda (parámetros de **partida** y **optimizados**) y compáralos en `reservado`. Calcula la diferencia por fila y un intervalo del 95 % **remuestreando entidades completas, no filas**. Aplica mi regla: `<regla>`. Guarda mi decisión en `reports/06_best_params.json`."

**Prompt 6 (reporte):** "Redacta `reports/06_optimizacion.md` con el espacio de búsqueda, el presupuesto, los resultados, la comprobación y la decisión. Si el intervalo cruza el cero, escribe que la mejora **no se distingue del ruido** y que se usan los parámetros de partida."

**Qué esperar:** un experimento con **una ejecución por ensayo** (en el MLflow local), una comprobación con entidades reservadas y una decisión escrita, que puede ser "no optimizar".

**Cómo revisarlo (con la guía y `checklist-revision-ia.md`):**
- Abre el experimento: ¿están **todos** los ensayos, cada uno con parámetros y métrica? ¿se respetó el **presupuesto**?
- ¿La comprobación se hizo en **entidades que la búsqueda nunca vio**?
- ¿El intervalo remuestrea **entidades** (no filas)?
- ¿Tu **regla de decisión** estaba escrita antes de mirar el resultado?
- ¿Claude reescribió `mlflow_utils.py` o inventó otro módulo? Debe **usar el de la plantilla**.
- ¿Se abrió `test.joblib`? **No debe abrirse hoy.**

## 7. Validación e interpretación
**Prompt:** "En `notebooks/07_validacion_interpretacion.ipynb`, abre el conjunto de prueba **una sola vez** con el modelo optimizado. Muestra matriz de confusión con costos, umbral, calibración, importancia de variables (SHAP) y errores por segmento."
**Qué esperar:** métricas en prueba, gráficos y una lectura en lenguaje de negocio.
**Cómo revisarlo:** ¿se usó la prueba solo aquí? ¿el resultado cae cerca del de validación? ¿se discuten riesgos y límites?

## 8. Registro del modelo
**Prompt:** "En `notebooks/08_registro_modelo.ipynb`, registra el mejor modelo en Unity Catalog con firma y ejemplo de entrada, asigna versión y alias, y explica cómo recuperarlo."
**Qué esperar:** una versión del modelo en el registro de Databricks.
**Cómo revisarlo:** ¿aparece la versión en Databricks? ¿puedes cargarla y predecir con datos nuevos?

## 9. Módulos y pruebas
**Prompt:** "Mueve a `src/proyecto/` toda la lógica que siga en los notebooks, deja los notebooks solo como orquestadores y añade pruebas en `tests/`. No cambies los resultados."
**Qué esperar:** notebooks más cortos, `src/` completo, `pytest` en verde.
**Cómo revisarlo:** ejecuta todo desde cero; ¿dan las mismas métricas que antes?

## 10. Inferencia
**Prompt:** "En `src/proyecto/inference/predict.py`, crea `predict(df)` que reciba las columnas crudas y devuelva clase y probabilidades; falla con mensaje claro si faltan columnas. Exporta `models/model.joblib` y escribe `models/ficha_modelo.md`."
**Qué esperar:** una función probada, el archivo del modelo y la ficha con métricas y límites.
**Cómo revisarlo:** prueba con un caso conocido y con datos inválidos; ¿falla con claridad?

## 11. App
**Prompt:** "Crea `app/streamlit_app.py`: un formulario con las variables de entrada, validación, predicción y probabilidades, una explicación para no técnicos y un aviso de uso responsable. Debe cargar `models/model.joblib` del repositorio y no consultar Databricks."
**Qué esperar:** una app que arranca con `streamlit run app/streamlit_app.py`.
**Cómo revisarlo:** ¿acepta datos inválidos sin avisar? ¿la predicción coincide con la función `predict`? ¿funciona en un entorno limpio?

## 12. Informe técnico
**Prompt:** "Ensambla `reports/informe_tecnico.md` a partir de las secciones de `reports/`: caso, datos, preprocesamiento, features, torneo, optimización, validación, registro y despliegue. Escribe claro, sin inventar resultados que no estén en los archivos."
**Qué esperar:** un informe con todas las secciones y las cifras de los experimentos.
**Cómo revisarlo:** compara cada cifra con el experimento original; busca afirmaciones sin respaldo.

## 13. Presentación
**Prompt:** "En `presentacion/ejecutiva.md` (Marp), escribe **máximo 5 diapositivas**: problema, enfoque, resultado y costo de negocio, demo de la app, riesgos y siguientes pasos. Lenguaje sin jerga."
**Qué esperar:** cinco diapositivas claras, una idea por diapositiva.
**Cómo revisarlo:** ¿son 5 o menos? ¿un directivo entendería el resultado y el riesgo en 1 minuto?
