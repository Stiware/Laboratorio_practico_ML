# Punto de reanudación · OnTrack (al 31 oct 2026)

> Nota para retomar el trabajo en otra sesión sin repetir nada. Las decisiones del grupo están en `especificacion-proyecto.md` §7 y, resumidas para la
> exposición, al inicio de `reports/bitacora_decisiones.md`.

## Dónde estamos

| Etapa | Estado | Dónde está |
|---|---|---|
| 1 Caso de negocio, 2 EDA | Hecho y validado por el grupo | `reports/00_caso.md`, `reports/01_eda.md` |
| 3 Preprocesamiento | Hecho (96.139 pedidos; 76.924 para entrenar y 19.215 de prueba cerrada) | `reports/02_preprocesamiento.md` |
| 4 Variables | Hecho: 8 de 9 ideas conservadas | `reports/04_features.md` |
| 5 Torneo de modelos (Fase 1) | **Hecho y revisado.** Gana LightGBM cuantil 0,75; la regresión lineal quedó 7.ª | `reports/05_torneo.md` |
| 5 Optimización (Fase 2, lo de `tuning.py` del ejemplo) | **PENDIENTE: se hace en otra sesión** | — |
| 6 en adelante (validación final con la prueba, registro del modelo, app, presentación, informe) | Pendiente | — |

Todo el trabajo va en la rama `main` (decisión del grupo: no se crean más ramas).

## Cómo retomar (desde la raíz del repositorio)

```bash
.venv/bin/python -m pytest tests -q            # 166 pruebas en verde
```

- `data/processed/` no se versiona (la guía lo indica). Si falta, se regenera con `notebooks/03_preprocesamiento.ipynb` (necesita los CSV de Olist en `data/raw/`).
- Los notebooks se ejecutan con `.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/0N_nombre.ipynb --output 0N_nombre.ipynb` (el 04 tarda ≈ 8 min y el 05 ≈ 10 min).
- Los notebooks y reportes 03 a 05 se generan con scripts guardados en `scripts/generadores/` (ver su `README.md`).
- `pwsh` no está instalado: los scripts de PowerShell de Spec Kit no corren; los documentos de `specs/` se arman a mano con las plantillas.
- Para Spec Kit: `/speckit-specify` → `/speckit-plan` → (el grupo aprueba el plan) → `/speckit-tasks` → (el grupo aprueba las tareas) → `/speckit-implement`. **Sin plan y tareas aprobados no se escribe código.**
- Commit y push solo cuando el grupo lo pida; antes se revisa `git status` (`.env`, `data/raw/` y `data/processed/` están ignorados).

## Lo que ya decidió el grupo (resumen del torneo)

- El **costo 3 a 1 decide al ganador**; el MAE se reporta siempre. El plazo óptimo es el **percentil 75**.
- Dos familias: A (valor típico + margen de clientes de calibración) y B (percentil 75 directo). Torneo **sin** la marca de fin de año y **sin** el historial por vendedor; el plazo prometido por Olist nunca entra.
- **El modelo final no tiene que ser una regresión lineal: se busca el mejor modelo** (decisión del 31 oct 2026).
- Prueba final (`data/processed/test.parquet`): **no se ha abierto** y no se abre hasta la validación final.

## Pendientes de decisión del grupo

1. **Finalistas.** La regla aprobada da **LightGBM cuantil 0,75** y **HistGradientBoosting cuantil 0,75**; LightGBM de valor típico es una alternativa fuera de la regla.
2. **Recalibrar el margen con datos recientes** y definir cómo se vigila la cobertura (hacia adelante va de 61 % en marzo de 2018 a 93 % en agosto).
3. **Regiones:** ¿indicador por región en la validación final? (El Norte y el Nordeste tienen el mayor error.)
4. **La marca de fin de año (F6):** una sola temporada; ¿se mide en los finalistas?
5. **Umbral mínimo de mejora en días** (variables que casi no aportan) y **retomar el historial por vendedor (F8)** con validación por bloques de fechas.
6. **Formato Parquet:** las plantillas de los notebooks 06 a 09 leen `train.joblib`; el proyecto guarda Parquet. Hay que cambiar la lectura o confirmar con el profesor.
7. **Umbral de éxito** con el instructor, y **cómo calcular la probabilidad de retraso** (apartado secundario).
8. **Borrar la rama vieja `etapa-3-preprocesamiento` en GitHub** (ya está fusionada en `main`; el sistema no me dejó borrarla): `git push origin --delete etapa-3-preprocesamiento`.

## Siguiente paso: Fase 2, optimización (otra sesión)

1. Confirmar con el grupo los finalistas (punto 1 de arriba).
2. Escribir el borrador de `especificaciones/06-optimizacion.md` a partir de `tuning.py` del ejemplo (`material-del-curso/ejemplo-creditsight/src/creditsight/models/tuning.py`) y de la guía (`kit/guia-modelado-y-optimizacion.md`, secciones 3 y 4), adaptado a **regresión**:
   - búsqueda con **Optuna** y presupuesto acotado (≈ 15 ensayos por finalista); tiempo por ajuste medido: LightGBM cuantil ≈ 3,6 s, HistGradientBoosting cuantil ≈ 10 s;
   - **25 % de los clientes reservados** que la búsqueda no ve (`GroupShuffleSplit`, semilla 7);
   - comprobación honesta con **intervalo del 95 % remuestreando clientes completos** y regla escrita antes de mirar («uso los parámetros optimizados solo si todo el intervalo está por debajo de cero»);
   - **MLflow**: `src/olistbr/tracking/mlflow_utils.py` ya existe (cae a local si Databricks no conecta); el `.env` existe (no se abre ni se imprime);
   - incluir el **control de datos entre familias** (la A se ajusta con el 80 %, la B con el 100 %) y la **recalibración temporal**.
3. Spec Kit (`specs/004-…`): especificación → plan → aprobación → tareas → aprobación → código con pruebas → revisión con `auditor-de-fuga` y `revisor-de-codigo-ia`.
4. Actualizar `reports/bitacora_decisiones.md` (tabla «Decisiones clave para la exposición») y la sección 7 de `especificacion-proyecto.md`.

## Cifras para recordar

| Dato | Valor |
|---|---|
| Costo 3 a 1 del ganador / de la regresión lineal / del plazo fijo p75 / de la mediana | 9,367 / 9,664 / 11,840 / 14,334 |
| Error medio de las variables (etapa 4): referencia → conjunto conservado | 5,211 → 5,033 días |
| Cobertura de los plazos en los pliegues / hacia adelante | 74 % a 75 % / 82 % a 83 % (por mes: 61 % a 93 %) |
| Tiempo de ajuste de un modelo (un pliegue) | 1,5 a 14 s (el bosque de 200 árboles tardaba 254 s) |
