# Validación e interpretación

> Plantilla en blanco. Complétala **después** de ejecutar `notebooks/07_validacion_interpretacion.ipynb`.
> Todo lo que escribas aquí lo tienes que poder sostener ante una pregunta del instructor.

## 1. Decisiones congeladas (escritas ANTES de abrir la prueba)

| Decisión | Valor | Cómo se decidió (con qué datos) |
|---|---|---|
| Modelo final | | |
| Parámetros (optimizados o de partida) | | |
| ¿Calibrar las probabilidades? | | |
| Métrica principal | | |
| Regla de decisión (clase más probable, umbral, menor costo…) | | |
| **Fecha y hora en que se congelaron** | | |

## 2. Calibración (con entidades reservadas, no con la prueba)

- ¿Se evaluó? ☐ sí  ☐ no → si no, por qué no hace falta: _______________
- Resultado y decisión: _______________

## 3. La prueba, abierta **una sola vez**

- Fecha y hora en que se abrió (la marca está en `reports/prueba_abierta.txt`): _______________
- ¿Se cambió algo del modelo después de verla? ☐ no  ☐ sí → qué y por qué (hay que declararlo): _______________

| Métrica | Modelo final | Baseline | Intervalo del 95 % (entidades) |
|---|---|---|---|
| Métrica principal | | | |
| Métrica de control 1 | | | |
| Métrica de control 2 | | | |

**¿Mejora al baseline con confianza?** (¿el intervalo de la diferencia excluye el cero?): _______________

## 4. Errores y carga de trabajo

| Segmento | Casos | Métrica | % que el modelo marca | % real |
|---|---|---|---|---|
| | | | | |

- ¿Hay un segmento al que le va peor? ¿Por qué podría ser?: _______________
- Si el modelo prioriza revisión humana: carga que genera frente a la carga real: _______________

## 5. Interpretación

- Variables que más pesan (y qué significa para el negocio): _______________
- **Límite de la explicación:** explica al modelo, no al mundo; no dice si el modelo acertó: _______________
- Atributos sensibles o de uso regulado entre las variables (¿cuáles, y qué se hace?): _______________

## 6. Conclusión

- ¿El modelo sirve para la decisión del caso? ¿Con qué condiciones?: _______________
- Qué NO debe hacerse con él: _______________
