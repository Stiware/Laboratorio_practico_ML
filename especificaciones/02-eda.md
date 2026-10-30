# Especificación · Etapa 2 · EDA

> Se llena **antes** de abrir `notebooks/02_eda.ipynb` (las secciones 1 y 2) y se completa **mientras**
> lo ejecutan (3 y 4). El resultado final, con las decisiones, va a `reports/01_eda.md`.
> Guía: `material-del-curso/kit/guia-eda-y-preprocesamiento.md` · Ejemplo: `material-del-curso/ejemplo-creditsight/reports/01_eda.md`.

**Grupo:** _______________   **Fecha:** _______________   **Estado:** ☐ borrador  ☐ cerrada

## 1. Qué esperamos encontrar (se escribe ANTES de mirar los datos)

Una hipótesis por pregunta. Si después no se cumple, es un hallazgo: anótenlo.

| Pregunta | Lo que creemos |
|---|---|
| ¿Qué representa una fila? | |
| ¿Alguna entidad se repite en varias filas? | |
| ¿Qué variable **debería** ser estable y no cambiar? | |
| ¿Qué significaría un vacío en nuestras columnas clave? | |
| ¿Qué columna solo se conoce **después** de la decisión que queremos apoyar? | |

## 2. Qué vamos a medir

Marquen lo que aplica a su dataset. Lo que no aplique se tacha **con una razón** (no se deja en blanco).

| Chequeo (guía, sección 3) | ¿Aplica? | Columnas a revisar | Quién lo hace |
|---|---|---|---|
| 3.1 Números guardados como texto | ☐ sí ☐ no: | | |
| 3.2 Valores centinela (`-1`, `9999`, `?`) | ☐ sí ☐ no: | | |
| 3.3 Valores imposibles (percentiles, negativos) | ☐ sí ☐ no: | | |
| 3.4 Constancia por entidad | ☐ sí ☐ no: | | |
| 3.5 Vacíos que significan algo | ☐ sí ☐ no: | | |
| 3.6 Duplicados | ☐ sí ☐ no: | | |
| 3.7 Fechas y orden temporal | ☐ sí ☐ no: | | |
| 3.8 Categorías raras / alta cardinalidad | ☐ sí ☐ no: | | |
| 3.9 Identidad y datos personales | ☐ sí ☐ no: | | |

## 3. Lo que medimos (se llena mientras corre el notebook)

- **Filas × columnas:** _______________
- **Unidad de análisis (una frase):** _______________
- **Columna que identifica a la entidad (`ID_COL`):** _______________   **¿Se repite?** ☐ no ☐ sí, ___ filas por entidad
- **Objetivo (`TARGET`):** _______________   **Distribución:** _______________
- **Línea base** (regla trivial) y su resultado: _______________
- **Vacíos en el objetivo:** _______________   **Qué haremos con esas filas:** _______________
- **Datos personales encontrados:** _______________
- **Columnas sospechosas de fuga:** _______________ (se miden con y sin ellas en la etapa 4)

## 4. Decisiones (mínimo 5)

> Cada una con **cifra**, acción **concreta** y **motivo**. Se copian tal cual a `reports/01_eda.md`.
> Cada decisión se convierte en una función de `src/olistbr/data/` y en una prueba (ver `03-preprocesamiento.md`).

| # | Hallazgo (con cifra) | Decisión | Motivo | Función que la implementa |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |

## 5. Criterios de cierre

- ☐ `notebooks/02_eda.ipynb` corre de principio a fin sin errores.
- ☐ Unidad de análisis y `ID_COL` declarados por escrito (sección 3).
- ☐ Al menos **5 decisiones** con cifra, acción y motivo.
- ☐ Datos personales identificados y plan para eliminarlos al cargar.
- ☐ Valores imposibles, centinelas y significado de los vacíos revisados.
- ☐ Fuga sospechada: lista escrita, aunque sea "ninguna, porque…".
- ☐ `reports/01_eda.md` completado.
- ☐ Quien **no** escribió una decisión sabe explicar por qué se tomó.
