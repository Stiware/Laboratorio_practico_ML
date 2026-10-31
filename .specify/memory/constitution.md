# OnTrack — Constitución

> Lo que diga aquí manda sobre las suposiciones de Claude al planear o programar.

## Principios

### I. Los datos originales no se tocan
`data/raw/` MUST NOT modificarse. Toda limpieza se hace en código (`src/olistbr/data/`) y se guarda en
`data/processed/`, que se regenera desde cero con el notebook de preprocesamiento. *Razón:* sin el original no hay
reproducibilidad.

### II. Se parte por la unidad de análisis
La unidad de análisis de este proyecto es **un pedido entregado a un cliente** y la entidad que se repite es
**`customer_unique_id`** (una persona puede hacer varios pedidos). Como esa entidad aparece en varias filas, la
partición y la validación cruzada MUST hacerse por grupo (`GroupShuffleSplit`, `GroupKFold`, `StratifiedGroupKFold`),
nunca por fila. Una prueba automática MUST verificar que ningún cliente cae en dos particiones. *Razón:* partir por fila
filtra información y infla la métrica.

### III. La prueba se abre una sola vez
El conjunto de prueba MUST NOT usarse para elegir variables, modelo, umbral ni hiperparámetros. Se abre una vez, en la
etapa de validación. Todo preprocesamiento que aprenda de los datos (medias, categorías, escalado) MUST ir dentro de un
`Pipeline` ajustado solo con entrenamiento. *Razón:* es la única medida honesta de cómo se comportará en producción.

### IV. Cada decisión de datos se escribe con su motivo
Cada decisión (qué se limpia, qué se descarta, qué se repara) MUST quedar en `reports/` con la cifra que la respalda y
su motivo, y MUST tener una función en `src/` con al menos una prueba en `tests/`. *Razón:* otra persona debe poder
discutirla sin leer el código.

### V. Nada sensible sale del equipo
`.env`, tokens y datos personales MUST NOT subirse a GitHub (el repositorio es público). Las columnas con datos personales
se eliminan al cargar, antes de cualquier análisis que las imprima. *Razón:* privacidad y seguridad de las cuentas.

### VI. El código es nuestro: lo entendemos
Cada integrante MUST poder explicar el código que generó la IA. Antes de aceptarlo se revisa con
`material-del-curso/kit/checklist-revision-ia.md`. La lógica reutilizable vive en `src/`; los notebooks solo orquestan.
Textos y comentarios en español; nombres de código en inglés. *Razón:* la autoría individual es parte de la evaluación.

### VII. Nunca se inventan datos ni resultados
Ni la IA ni el grupo MUST fabricar cifras, columnas ni métricas. Lo que no se pudo medir se dice. *Razón:* un número
inventado en el informe invalida todo lo demás.

### VIII. Uso responsable del modelo
Las predicciones de OnTrack MUST presentarse como una **estimación para planificar la operación, no como una promesa al
cliente**. La app MUST mostrar un aviso de uso responsable. El desempeño (MAE y el costo medio con el error 3 a 1)
MUST revisarse por estado o región del cliente y los resultados por segmento MUST quedar en `reports/`, indicando los
segmentos donde el modelo rinde peor. *Razón:* el EDA mostró que los plazos varían mucho por región (SP 7,2 días contra
AM 25,9); un promedio global puede ocultar que el modelo falla justo donde más se tarda.

## Restricciones del proyecto

- **Caso:** ¿cuántos días tardará en llegar un pedido de Olist y llegará antes o después de la fecha prometida? ·
  **Métrica principal:** MAE en días (se lee directo en días y no la dominan los pedidos extremos) ·
  **Tipo de problema:** regresión (regresión lineal). La probabilidad de retraso es un apartado secundario derivado de la
  predicción en días, no el objetivo de entrenamiento.
- **Entorno:** Python 3.12; dependencias fijadas en `requirements.txt`.
- **Seguimiento de experimentos:** MLflow en Databricks (o local si no hay conexión); la app **no** consulta Databricks.
- **App:** Streamlit, cargando `models/model.joblib` desde el repositorio.
- **Datos:** Olist (Kaggle), licencia CC BY-NC-SA 4.0: uso no comercial y citar a Olist; los CSV no se suben al repositorio.

## Flujo de trabajo

- Cada etapa sigue **especificación → plan → tareas → implementación** (`/speckit-specify`, `/speckit-plan`,
  `/speckit-tasks`, `/speckit-implement`). Los artefactos viven en `specs/`.
- Las guías de etapa de `especificaciones/` se llenan **antes** de pedir una especificación a Spec Kit: son su entrada.
- Los **subagentes** de `.claude/agents/` se usan para revisar (fuga, pruebas, código de la IA) antes de dar una etapa por
  terminada.
- Cada hito MUST tener criterios de aceptación verificables (`especificaciones/`, sección de cierre).

## Gobierno

Esta constitución prevalece sobre otras prácticas del proyecto. Una enmienda se hace con `/speckit-constitution`, se
justifica y se anota en la sección 7 de `especificacion-proyecto.md`. El instructor puede pedir aclaraciones sobre
cualquier principio en los puntos de control. La versión sigue el esquema semántico: MAJOR si se elimina o redefine un
principio, MINOR si se agrega uno o se amplía materialmente, PATCH para aclaraciones de redacción.

**Versión**: 1.0.0 | **Ratificada**: 2026-10-31 | **Última enmienda**: 2026-10-31
