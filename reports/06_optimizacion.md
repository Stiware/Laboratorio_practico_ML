# Informe de la optimización

> Plantilla en blanco. Resultado de `notebooks/06_optimizacion.ipynb`. La mejor puntuación de una
> búsqueda es **optimista**: se comprueba en entidades que la búsqueda no vio. Guía:
> `material-del-curso/kit/guia-modelado-y-optimizacion.md`, secciones 3 y 4.

## Diseño de la búsqueda

- **Finalista(s) optimizado(s):**
- **Espacio de búsqueda** (parámetro → rango y por qué ese rango):

| Parámetro | Rango | Motivo |
|---|---|---|
| | | |
| | | |

- **Presupuesto** (ensayos por modelo, fijado antes de empezar) y **por qué ese número:**
- **Entidades reservadas** (qué porcentaje, partidas por qué entidad, semilla):

## Seguimiento (tracking)

- **Dónde quedó registrado** (MLflow local en la clase 3) y **nombre del experimento:**
- **¿Está cada ensayo como una ejecución con parámetros, métrica y etiquetas?** ☐ sí ☐ no
- **Mejor ejecución** (identificador y métrica):

## Resultados de la búsqueda

<!-- Mejor valor encontrado en la validación de la búsqueda (recuerda: es optimista) y los
     parámetros que más influyeron. -->

-

## Comprobación honesta (entidades reservadas)

> Modelo con parámetros de **partida** frente al de parámetros **optimizados**, evaluados en las
> entidades que la búsqueda no vio. Intervalo del 95 % remuestreando **entidades completas**.

| Modelo | Métrica con parámetros de partida | Métrica con parámetros optimizados | Diferencia (optimizado − partida) | Intervalo 95 % |
|---|---|---|---|---|
| | | | | [ ; ] |
| | | | | [ ; ] |

**Regla de decisión escrita antes de mirar:**

**Resultado de la regla** (¿todo el intervalo por debajo de cero?):

## Decisión

- **Parámetros que se usarán a partir de ahora** (de partida u optimizados) **y por qué:**
- **Lo que no pudimos concluir:**

## Para la clase siguiente

<!-- Modelo final que pasa a la validación (prueba abierta una sola vez) y a su registro. -->

-
