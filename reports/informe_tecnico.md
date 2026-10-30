# Informe técnico — TODO: nombre de tu startup

> Esqueleto. Aquí **ensamblas** las secciones que ya escribiste en `reports/`; no las reescribas desde cero: tómalas, ordénalas
> y cuenta una historia coherente. **Se entrega el domingo 15 de noviembre** (hito H4).

> **Regla del informe:** **cada decisión va con su motivo.** Un resultado sin la razón de la decisión que lo produjo es solo un número.

**Grupo:** TODO (integrantes y roles) · **Repositorio:** TODO · **App:** TODO (enlace público)

## Resumen ejecutivo

TODO: 5 a 8 líneas para alguien que no leerá el resto. La decisión de negocio, el resultado en unidades de negocio, el
principal límite y el enlace a la app. *(La conclusión primero.)*

## 1. Caso de negocio

> Fuente: `reports/00_caso.md` (ficha de caso aprobada).

TODO: quién decide, qué decide, con qué información, qué cuesta equivocarse y cómo se midió el éxito.

## 2. Datos y limpieza

> Fuente: `reports/01_eda.md` y `src/<paquete>/data/`.

- **Unidad de análisis y partición** (y por qué respeta la entidad que se repite): TODO
- **Hallazgos y decisiones** (cada una con su motivo): TODO
- **Datos personales o sensibles y cómo se trataron:** TODO

## 3. Variables

> Fuente: `reports/04_features.md`.

TODO: ideas probadas, la mejora medida frente al ruido de la validación y qué se **descartó** y por qué.

## 4. Torneo de modelos

> Fuente: `reports/05_torneo.md`.

TODO: candidatos, validación común, métrica principal y de control, comparación con el punto de partida, finalistas y motivo.

## 5. Optimización y seguimiento

> Fuente: `reports/06_optimizacion.md`.

TODO: espacio de búsqueda, presupuesto, comprobación con datos reservados (¿la mejora fue real o está dentro del ruido?) y qué se
registró en el seguimiento (Databricks o MLflow local).

## 6. Validación e interpretación

> Fuente: `reports/07_validacion.md`.

TODO: decisiones congeladas **antes** de abrir la prueba, resultado en la prueba con su intervalo, comparación con el punto de
partida, errores por segmento o carga de trabajo, interpretación y su límite.

## 7. Registro del modelo

> Fuente: `reports/08_registro.md` y `models/ficha_modelo.md`.

TODO: versión y alias, ida y vuelta, exportación a `models/model.joblib` y esquema de entrada.

## 8. Despliegue

> Fuente: `reports/09_despliegue.md`.

TODO: enlace público, decisiones de despliegue y su motivo, pruebas del enlace, plan B y, si hubo, errores encontrados.

## 9. Riesgos, límites y siguientes pasos

TODO: qué **no** hace el modelo, límites de los datos, uso responsable (variables sensibles o reguladas) y una mejora concreta.

## Anexo: cómo reproducir el proyecto

TODO: pasos mínimos desde un repositorio recién clonado (entorno, datos, orden de los notebooks, `pytest`, `streamlit run`).

---

### Lista de comprobación antes de entregar

- [ ] Las **9 secciones** tienen contenido y **cada decisión lleva su motivo**.
- [ ] Las cifras **coinciden** con `reports/07_validacion.md` y con la presentación.
- [ ] No hay credenciales, `.env` ni datos personales.
- [ ] El enlace de la app abre en una ventana privada.
- [ ] Un tercero puede reproducir el proyecto con el anexo.
