# Informe del torneo de modelos

> Plantilla en blanco. Resultado de `notebooks/05_torneo.ipynb`. El torneo **descarta**: elige 1–2
> finalistas con evidencia, no un campeón definitivo. Guía:
> `material-del-curso/kit/guia-modelado-y-optimizacion.md`, sección 2.

## Reglas del torneo

- **Mismos datos y mismos pliegues para todos** (partidos por qué entidad):
- **Línea base** (qué es y cuánto da):
- **Métrica principal** (la del negocio) y **métricas técnicas** que reportas (AUC, PR-AUC, exactitud balanceada,
  precisión y recall por clase, F1 macro, log loss; en regresión: RMSE, MAE, R²):
- **Variables usadas** (las decididas en `reports/04_features.md`):

## Tabla comparativa

> Media ± desviación entre pliegues. Ordena por la métrica principal.

| Modelo | Métrica de negocio (± ruido) | Segundos de ajuste |
|---|---|---|
| Línea base | | |
| | | |
| | | |
| | | |

## Métricas técnicas

> Media entre pliegues, con la misma validación. Las de ranking (AUC, PR-AUC) usan las probabilidades; las de
> clasificación usan la decisión de tu negocio. Guía: sección 2.2.

| Modelo | AUC macro | AUC clase de interés | PR-AUC clase de interés | Exactitud balanceada | Precisión clase de interés | Recall por clase | F1 macro | Log loss |
|---|---|---|---|---|---|---|---|---|
| Línea base | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

## ¿Coinciden los rankings de negocio y técnico?

> Posición de cada modelo (1 = mejor) en la métrica de negocio y en las técnicas.

| Modelo | Negocio | AUC macro | F1 macro | Recall de la clase de interés |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |

- **¿El mejor por la métrica de negocio también es el mejor por AUC?**
- **¿Hay modelos con AUC casi igual pero costo distinto? ¿Por qué (cómo deciden)?**
- **¿Hay alguna clase con recall casi nulo en algún modelo? ¿Cuál y qué le cuesta al negocio?**

## Comparación pareada de los mejores

> Todos se evaluaron en los mismos pliegues: ¿gana siempre el mismo?

| Mejor | Contra | Diferencia media | Pliegues que gana | Diferencia / ruido |
|---|---|---|---|---|
| | | | | |
| | | | | |

## ¿Cuánto vale la diferencia?

<!-- Traducción a dinero o volumen: métrica por decisión × decisiones al mes. -->

- **Decisiones estimadas al mes:**
- **Ahorro frente a la línea base:**
- **Diferencia entre el mejor y el siguiente:**

## Finalistas

| Finalista | Por qué (consistencia, métrica, tiempo, interpretabilidad) | Qué podría cambiar con la optimización |
|---|---|---|
| | | |
| | | |

**Modelos descartados y por qué:**

**Por qué el torneo no elige un campeón** (con tus palabras):

## Efectos que la métrica principal no ve

<!-- Por ejemplo: un modelo más conservador exige más revisión humana, o una clase casi nunca se predice
     (recall casi nulo) aunque la métrica de negocio se vea bien. -->

-
