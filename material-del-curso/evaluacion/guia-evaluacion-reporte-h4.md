# Cómo se evalúa el reporte técnico final (H4)

El reporte técnico es el **hito H4**. Se entrega **hasta el domingo 15 de noviembre** (8 días después de la última
clase, el sábado 7 de noviembre) y vale el **15 %** de la nota (criterio **A4** de la rúbrica).

No se escribe todo de golpe al final: **cada clase aportó una sección**. Lo que falta es **ensamblarlas, completarlas
y escribir la sección de despliegue**.

## Qué se entrega y dónde

- **Archivo:** `reports/informe_tecnico.md` en el repositorio **del grupo**.
- **Cómo se verifica el plazo:** por la **fecha del último commit** de ese archivo en GitHub. La **hora límite** del
  domingo la anuncia el instructor en la clase 1.

## Las 8 secciones obligatorias

| # | Sección | De dónde sale | Qué debe contener |
|---|---|---|---|
| 1 | **Caso de negocio** | `reports/00_caso.md` (clase 1 y 2) | Pregunta de negocio, quién decide, costo de equivocarse, métrica principal y por qué |
| 2 | **Datos** | `reports/01_eda.md` (clase 2) | Fuente, tamaño, unidad de análisis, problemas encontrados, las decisiones del EDA con su motivo, cómo se partieron los datos y por qué |
| 3 | **Variables (features)** | `reports/04_features.md` (clase 3) | Ideas probadas, cuáles se aceptaron y cuáles se descartaron, con la mejora frente al ruido de la validación |
| 4 | **Torneo de modelos** | `reports/05_torneo.md` (clase 3) | Modelos comparados con la misma validación, tabla con dispersión, baseline, finalistas elegidos y por qué |
| 5 | **Optimización** | `reports/06_optimizacion.md` (clase 3) | Espacio de búsqueda, presupuesto, comprobación en datos reservados y **si la mejora fue real** |
| 6 | **Validación e interpretación** | `reports/07_validacion.md` (clase 4) | Decisiones congeladas antes de abrir la prueba, resultado de la prueba (abierta **una sola vez**) con su intervalo, errores por segmento, interpretación y límites |
| 7 | **Registro del modelo** | `reports/08_registro.md` y `models/ficha_modelo.md` (clase 4) | Versión registrada, alias, cómo se recupera, ficha del modelo con usos no recomendados |
| 8 | **Despliegue** | **nueva** (clase 5) | Enlace público de la app, cómo se desplegó, cómo se valida la entrada, qué se hace si algo falla, evidencia de que funciona |

## Qué se mira en cada sección

Para cada sección, el evaluador busca estos **cuatro elementos**:

1. **Qué se decidió** (la decisión, dicha con claridad).
2. **Por qué** (el motivo; sin motivo, la decisión cuenta como no justificada).
3. **Con qué evidencia** (un número, una tabla, un resultado del notebook).
4. **Qué límite o riesgo tiene** (lo que no sabes o lo que podría salir mal).

> Una sección con decisiones pero **sin motivos** baja el nivel de A4, aunque esté completa.

## Cómo se asigna el nivel (criterio A4)

| Nivel | Qué se observó | Evidencia |
|---|---|---|
| **3** | `informe_tecnico.md` entregado **en plazo** con **las 8 secciones**; cada decisión con su motivo | informe y fecha del commit |
| **2** | Entregado en plazo con **6 o 7 secciones**, o con secciones **sin motivos** | idem |
| **1** | Entregado con **5 secciones o menos**, o **fuera de plazo hasta 3 días** | idem |
| **0** | **No entregado**, o más de **3 días de retraso** | — |

Si dos descripciones aplican a la vez (por ejemplo, 7 secciones y 2 días tarde), se asigna el **nivel menor**.
Si algo no se puede comprobar con la evidencia, el nivel es el inferior.

### La regla del plazo, día por día

| Entrega | Resultado en A4 |
|---|---|
| Hasta la hora límite del **domingo 15 de noviembre** | Se evalúa por su contenido (nivel 3, 2 o 1) |
| Lunes 16, martes 17 o miércoles 18 de noviembre | **Nivel 1** como máximo |
| Jueves 19 de noviembre o después | **Nivel 0** |

## Lista de verificación antes de entregar

**Contenido**
- [ ] El archivo está en `reports/informe_tecnico.md` del **repositorio del grupo** (no en un documento aparte).
- [ ] Están las **8 secciones** de la tabla, con el mismo orden.
- [ ] Cada sección tiene **qué se decidió, por qué, con qué evidencia y qué límite**.
- [ ] La **métrica de negocio** y la **unidad de análisis** se explican en palabras que entienda alguien sin formación técnica.
- [ ] El informe dice si el conjunto de **prueba se abrió una sola vez** y cuándo.
- [ ] Los **cambios de datos o de unidad de análisis** están documentados (por ejemplo, si cambiaron la partición).
- [ ] Hay una sección de **límites y usos no recomendados** del modelo.
- [ ] La sección de **despliegue** trae el enlace público y una captura con fecha de la app funcionando.

**Entrega**
- [ ] Hice `git push` y el archivo aparece en GitHub con su **fecha de commit**.
- [ ] El repositorio es **público**.
- [ ] No hay **credenciales ni datos personales** en ninguna parte del repo (si los hay, el criterio B1 baja a 0).
- [ ] Cada integrante **leyó** el informe y puede explicar cualquier sección (criterio D).

## Consejos

- **Empiecen por pegar** las secciones que ya tienen (`reports/00_caso.md` … `reports/08_registro.md`) y **completen lo que falta**.
- **Resuman, no copien:** si una sección es una tabla del notebook sin comentario, falta el motivo.
- **Cuenten también lo que no funcionó** (una variable descartada, una optimización sin mejora): eso demuestra criterio.
- **Entreguen un día antes:** el domingo no hay margen para un error de Git.
