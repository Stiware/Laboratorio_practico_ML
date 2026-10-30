# Especificaciones por etapa

Una **especificación** es una guía que **se llena antes de programar**. Dice qué van a hacer, con qué
criterio y cómo sabrán que quedó bien. Claude las lee antes de generar código, igual que
`especificacion-proyecto.md`: **lo que escriban aquí manda**.

| Archivo | Cuándo se llena | Cierra con |
|---|---|---|
| [`02-eda.md`](02-eda.md) | **Clase 2 · viernes 30 de octubre**, bloque de EDA | `reports/01_eda.md` con ≥ 5 decisiones |
| [`03-preprocesamiento.md`](03-preprocesamiento.md) | **Clase 2 · viernes 30 de octubre**, bloque de limpieza y partición | `src/olistbr/data/` + `pytest` en verde |

**Relación con Spec Kit.** Estas guías son **la entrada**: lo que escriben aquí se pasa a `/speckit-specify`, que lo convierte
en `specs/<nº>/spec.md`; después `/speckit-plan` y `/speckit-tasks` producen el plan y la lista de tareas (`tasks.md`).
Ver `material-del-curso/guias/entorno-ia-specify.md`.

Las especificaciones de las etapas siguientes se agregan en las clases 3 y 4.

## Cómo se usan

1. **Llénenla en grupo**, con lo que miden en **su** dataset. No copien las cifras de CreditSight.
2. **Primero la especificación, después el código.** Si Claude propone algo que la contradice, se corrige el pedido, no la especificación (salvo que descubran algo nuevo: entonces actualícenla y anoten el cambio en la sección 7 de `especificacion-proyecto.md`).
3. **Cada casilla `☐` se marca solo cuando la ven funcionando**, no cuando alguien dice que la hizo.
4. Se suben a GitHub junto con el resto del repo. El instructor las revisa en los puntos de control.
