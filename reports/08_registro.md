# Registro del modelo

> Las cifras salen de `notebooks/08_registro_modelo.ipynb` y de `reports/08_registro.json`. Reglas: `especificaciones/08-registro.md`
> (cerrada el 5 nov 2026) · Código: `src/olistbr/tracking/registry.py` · Pruebas: `tests/test_registry.py`.

## 1. Dónde quedó registrado

| Dato | Valor |
|---|---|
| Plataforma | ☒ Databricks (Unity Catalog)  ☐ MLflow local (respaldo) → no hizo falta el respaldo |
| Experimento | `ontrack-modelo-final` (una ejecución, `modelo-final`) |
| Nombre del modelo registrado | `workspace.default.ontrack_plazo_entrega` |
| Versión | 1 (la única) |
| Alias | `champion` → versión 1, estado `READY`, única versión registrada (consulta de solo lectura guardada en `reports/08_verificacion_registro.json`) |
| Quién lo registró y cuándo | El grupo (cuenta de Databricks del `.env`), el 2026-11-05 a las 20:47 (UTC−5), con `notebooks/08_registro_modelo.ipynb` |

## 2. Qué se registró junto al modelo

- **Modelo:** exactamente `models/final_pipeline.joblib` de la etapa 7 (preprocesador + LightGBM cuantil 0,75), sin reentrenar. Guardado con `cloudpickle`, porque MLflow
  rechaza por defecto un `Pipeline` con preprocesamiento. **Aviso:** un archivo pickle puede ejecutar código al cargarse; solo se cargan modelos del propio grupo.
- **Parámetros:** tipo de modelo, percentil 0,75, los 7 grupos de variables del torneo y los 7 hiperparámetros optimizados (418 árboles, tasa 0,049, 70 hojas, mínimo
  73 pedidos por hoja, `subsample` 0,95, `colsample_bytree` 0,65, `reg_lambda` 4,43).
- **Métricas de la prueba:** costo 3 a 1 (9,216 con su intervalo [9,005 ; 9,422]), % tarde (26,4 %), cobertura (73,6 %), MAE del plazo (5,66), diferencia contra el plazo
  fijo del percentil 75 (−2,493 [−2,608 ; −2,373]) y el costo del plazo fijo (11,709).
- **Firma y ejemplo de entrada:** 25 columnas (18 numéricas y 7 categóricas: las que produce `add_features` para los 7 grupos de variables) y 3 pedidos del entrenamiento.
  El ejemplo **no tiene** el cliente, el pedido, el plazo prometido ni el objetivo (prueba `test_input_example_has_no_identifiers`); sí tiene `seller_id`, que es el
  código anónimo de Olist para el vendedor (una empresa) y una variable del modelo. Las columnas enteras se registran como decimales, para que la firma acepte datos
  faltantes al predecir (la app permitirá responder «No sé»); se comprobó que eso no cambia la predicción.
- **Etiquetas:** `etapa=08_registro`, `datos=train.parquet`, la fecha del congelamiento (2026-11-05 20:44:38) y la de la apertura de la prueba (20:44:44).

## 3. Comprobación de ida y vuelta

- ¿El modelo recuperado del registro predice igual que el local? ☒ sí  ☐ no
- Cómo se comprobó: se cargó `models:/workspace.default.ontrack_plazo_entrega@champion` y (1) sus plazos coinciden con los del modelo local en **500 pedidos de la prueba**
  (`np.allclose`); (2) su costo 3 a 1 en **toda la prueba** es **9,216262**, igual al de la etapa 7 (`reports/07_metricas_test.json`). La prueba se leyó **solo** para esta
  comprobación (decisión del grupo); no cambió nada del modelo.

## 4. Tracking frente a registro (en tus palabras)

- **¿Qué guarda el tracking en tu proyecto y para qué lo usaron?** En la etapa 6, 46 ejecuciones en el experimento `ontrack-optimizacion`: 2 principales (una por
  finalista), sus 40 ensayos (parámetros, costo con su desviación y demás métricas) y 4 comprobaciones honestas (2 de la primera corrida y 2 iguales de la segunda,
  que solo recuperó las salidas del notebook). Sirvió para comparar ensayos, ver cuál fue el mejor y auditar que se
  respetó el presupuesto. Responde «¿qué probamos?».
- **¿Qué guarda el registro y por qué el alias `champion` ayuda?** Guarda **versiones del modelo** listas para usar, cada una con su firma, ejemplo y métricas.
  Responde «¿qué modelo usamos y en qué versión?». El alias `champion` es una etiqueta móvil: la etapa 9 exporta el modelo de la app pidiendo «el champion», sin
  saber el número; si mañana se registra una versión 2 mejor, basta con mover el alias.

## 5. Reproducibilidad

- ¿Alguien de otro grupo podría recuperar tu modelo con lo que dejaste escrito? ☒ sí (con acceso al workspace)  ☐ no → qué falta: con permiso de lectura sobre
  `workspace.default`, basta `mlflow.sklearn.load_model("models:/workspace.default.ontrack_plazo_entrega@champion")`. Sin acceso a Databricks, el modelo se reconstruye
  con el código: los notebooks 03 (datos), 06 (parámetros) y 07 (ajuste final) con las mismas semillas.

## Revisiones

**Riesgo técnico declarado:** el modelo se guardó con Python 3.14 (el `.venv` de esta máquina) y el proyecto pide 3.12; un archivo `cloudpickle` hecho con una versión
puede no cargar en otra. Antes de exportar `models/model.joblib` (etapa 9) hay que comprobar que carga en el entorno de la app.

La prueba se leyó aquí una segunda vez, **después** de registrar y **sin decidir nada**, solo para la ida y vuelta (decisión del grupo); no pasó por `open_test_once` y
no dejó constancia propia (ver `reports/07_validacion.md` §3).

### Revisión de código (`revisor-de-codigo-ia`) y auditoría (`auditor-de-fuga`)

| Hallazgo | Qué se hizo |
|---|---|
| «Estado READY» y «única versión» no estaban guardados | Consulta de solo lectura guardada en `reports/08_verificacion_registro.json` (versión 1, `READY`, versiones registradas: [1]) |
| Conteo de ejecuciones del tracking (40 + 2 ≠ 44) | Corregido: 2 principales + 40 ensayos + 2 comprobaciones |
| `set_champion` tomaba siempre la última versión | Ahora acepta la versión explícita; el generador pasa la recién registrada (prueba nueva). En esta ejecución coincidían (había una sola) |
| `except Exception` en el notebook podía esconder un error de programación como «respaldo local» | El generador ahora solo captura errores de MLflow o de conexión. En esta ejecución no hubo respaldo (`motivo_respaldo: null`) |
| Nada probaba que la firma aceptara datos faltantes | Prueba nueva con `mlflow.pyfunc` (que sí aplica la firma) y dos datos faltantes: pasa |
| No se comprobaba que se guardaran la firma y el ejemplo | Prueba nueva: la firma registrada tiene las 25 columnas |
| Python 3.14 en lugar de 3.12 | Declarado arriba |
