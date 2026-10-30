# Ficha del modelo

> Plantilla en blanco. Complétala a mano con **tus** resultados (los de la prueba de `reports/07_metricas_test.json`).
> Un documento corto que cualquiera pueda leer antes de usar el modelo: qué es, con qué datos, qué rendimiento tiene
> y **qué no hay que hacer con él**.

## Qué es

_______________  (en 2 frases: qué predice el modelo, para quién y para apoyar qué decisión)

- **Modelo:** _______________
- **Registro:** _______________ versión _______________, alias `champion` (Databricks o local)
- **Regla de decisión:** _______________  (clase más probable, umbral o menor costo esperado)

## Datos de entrenamiento

- **Fuente y licencia:** _______________
- **Tamaño:** _______________ filas de _______________ entidades
- **Partición:** por _______________  (la unidad de análisis); la prueba se abrió **una sola vez**
- **Datos personales:** _______________  (qué se eliminó o anonimizó)

## Rendimiento (prueba, entidades nunca vistas)

| Métrica | Valor | Intervalo del 95 % |
|---|---|---|
| Métrica principal | | |
| Baseline | | |
| Control 1 | | |

## Variables que más pesan

_______________

## Límites y usos no recomendados

- Datos con los que se entrenó (¿sintéticos, de una época, de una región?): _______________
- Supuestos de negocio de los que depende (p. ej. una variable que debe conocerse al decidir): _______________
- Atributos sensibles o de uso regulado: _______________
- Qué **no** debe hacerse con este modelo: _______________
- Lo que **no se evaluó** (estabilidad en el tiempo, otros grupos, deriva): _______________

## Reproducibilidad

Código en `src/`, pruebas en `tests/`, notebooks en `notebooks/`.
