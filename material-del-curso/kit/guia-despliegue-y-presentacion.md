# Guía de trabajo: despliegue y presentación ejecutiva (clase 5 · viernes 6 de noviembre)

Esta guía es para **tu proyecto**, no para el de CreditSight. El ejemplo del instructor te muestra **cómo se razona**, pero
cada cifra, cada enlace y cada diapositiva de tu entrega los tienes que producir tú con **tus** datos.

**Lo que se te pide al cerrar el día (hito H3):**
- una **app desplegada** con **enlace público**,
- una **presentación ejecutiva de máximo 5 diapositivas**,
- un **ensayo cronometrado de 10 minutos**.

Mañana (clase 6, sábado 7 de noviembre) expones **desde tu enlace público**. Si no hay enlace, la rúbrica da como máximo el
nivel 1 del criterio A3, con evidencia local.

**Referencias:** `material-del-curso/guias/streamlit-despliegue.md` (el paso a paso del despliegue), `material-del-curso/evaluacion/guia-exposicion.md` y
`material-del-curso/evaluacion/rubrica.md` (criterios A3 y C).

---

## 0. Antes de empezar

- Necesitas `models/model.joblib`, `models/input_schema.json` y `app/streamlit_app.py` **en tu repo** y la app **corriendo en
  local** (`streamlit run app/streamlit_app.py`).
- Tu repo debe ser **público** y tener `requirements.txt` en la **raíz**.
- Cuenta en **Streamlit Community Cloud**, conectada a tu GitHub.
- `python scripts/verificar_entorno.py` sin ningún `FALTA`.

## 1. Despliegue

Sigue **`material-del-curso/guias/streamlit-despliegue.md`** apartado por apartado. Lo esencial:

1. **`requirements.txt` en la raíz**, con versiones fijadas (las mismas del entrenamiento) y Python **3.12**.
2. **`models/model.joblib` dentro del repo** y **no ignorado** por `.gitignore`. No subas `models/final_pipeline.joblib` ni
   los datos de entrenamiento.
3. `git add`, `commit`, `push`, y **comprueba en el navegador** que los archivos aparecen en GitHub.
4. **Crea la app** en Streamlit Community Cloud: repo, rama y archivo principal `app/streamlit_app.py`; Python 3.12; *Deploy*.
   *Los nombres exactos de los botones pueden cambiar: guíate por la documentación oficial.*
5. **Comprueba el enlace en una ventana privada** con tres casos: **normal**, **límite** e **inválido**.
6. Pega el enlace en `README.md` y en `reports/09_despliegue.md`.

> **Regla:** la app **no consulta Databricks** ni lleva credenciales. Si Claude propone poner un token «para que funcione», la
> respuesta es no.

**Si no sale:** lee los **registros (logs)** de abajo hacia arriba, copia el mensaje **completo** a Claude y dile en qué paso
estabas. Si a pesar de eso no hay enlace, prepara el **plan B**: captura de pantalla con fecha y repo, app corriendo en local y
un **plan de despliegue escrito** (qué intentaron, qué error dio, qué harían). La rúbrica da máximo el nivel 1 de A3 y **no
afecta** a los demás criterios.

## 2. La presentación: 5 diapositivas

Usa `presentacion/ejecutiva.md` (trae **exactamente 5 diapositivas**, con preguntas guía). **No agregues una sexta.**

| # | Diapositiva | Responde |
|---|---|---|
| 1 | **Problema** | ¿Quién decide, qué decide y cuánto cuesta equivocarse? |
| 2 | **Enfoque** | ¿Con qué datos y cómo se validó, **una frase por idea**? |
| 3 | **Resultado y costo de negocio** | ¿Cuánto mejora la decisión frente al punto de partida? |
| 4 | **Demo de la app** | ¿Cómo se usa? Un caso normal, uno límite y uno inválido |
| 5 | **Riesgos y siguientes pasos** | ¿Qué no hace el modelo y qué viene? |

**Cómo hablarle a un ejecutivo (tres reglas):**
1. **La conclusión primero.**
2. **Un mensaje por diapositiva** (si no cabe en una frase, son dos diapositivas).
3. **Cada número con su unidad de negocio:** no «F1 = 0,68», sino «detectamos 8 de cada 10 clientes riesgosos».

**De dónde salen las cifras:** de `reports/07_validacion.md` (tu prueba, abierta una sola vez). **No inventes ni redondees a tu
favor.** Si falta una cifra, vuelve a ese reporte.

**La diapositiva de riesgos:** qué **no** hace el modelo, límites de tus datos, uso responsable y una mejora concreta. «No hay
riesgos» no es una respuesta.

## 3. El informe técnico se va armando

`reports/informe_tecnico.md` (plantilla en el repo) ensambla las secciones que ya escribiste: caso, EDA, variables, torneo,
optimización, validación, registro y, hoy, **despliegue** (`reports/09_despliegue.md`). **Cada decisión va con su motivo.** Se
entrega el **domingo 15 de noviembre** (H4).

## 4. El ensayo (clase 5, bloque final)

Se hace **por parejas de grupos**:
- El grupo A expone **máximo 10 minutos**, con un cronómetro visible, **desde el enlace público** y en un navegador distinto al
  de desarrollo.
- El grupo B **toma el tiempo**, avisa a los 8 y 10 minutos y llena la hoja de abajo (5 minutos). Luego se intercambian.
- Quedan unos 10 minutos para **hacer una corrección concreta** por lo que dijo el otro grupo.

**El reparto de la exposición: 2 · 4 · 3 · 1.**

| Parte | Min | Qué debe mostrar |
|---|---|---|
| Caso de negocio | 2 | Quién decide, qué decide, cuánto cuesta equivocarse |
| App | 4 | Demo en vivo: normal, límite, inválido |
| Resultados | 3 | Métrica de negocio, comparación con el punto de partida, límites y riesgos |
| Cierre | 1 | Qué harían con más tiempo; una decisión que cambiarían |

Anota en `reports/09_despliegue.md` el **tiempo real** de cada parte y la **corrección** que hiciste.

### Hoja de retroalimentación (la llena el grupo que escucha)

| Pregunta | Sí / No | Comentario |
|---|---|---|
| ¿Duró **10 minutos o menos**? Tiempo real: ___ min ___ s | | |
| ¿Se entendió **quién decide y qué decide** en los primeros 2 minutos? | | |
| ¿El resultado trae una **cifra de negocio** (no solo F1 o accuracy) y su comparación con el punto de partida? | | |
| ¿La demo mostró un caso **normal**, uno **límite** y uno **inválido**? | | |
| ¿La app abrió **desde el enlace público**? | | |
| ¿Hubo **riesgos y límites honestos**? | | |
| ¿Hablaron **varias personas**? ¿Quién respondería las preguntas técnicas? | | |
| **Lo mejor** de la presentación (una cosa) | | |
| **Una corrección concreta** que haría el grupo antes de mañana | | |

## 5. Las 8 preguntas que el instructor hará hoy

Prepáralas: se hacen a un integrante **distinto** de quien escribió el código o habló.

1. ¿Por qué tu app **no consulta Databricks**?
2. ¿Dónde está tu `requirements.txt` y por qué en la **raíz**?
3. Si `model.joblib` no aparece en GitHub, ¿qué revisas primero?
4. ¿Qué cosas **nunca** deben ir en un repo público?
5. ¿Qué hace tu app con una **edad de −500**? ¿Y con un campo en «No sé»?
6. Cuenta tu proyecto **en una frase**.
7. ¿Cuál es el resultado **en dinero o en decisiones**, no en F1?
8. ¿Qué **plan B** tienes si la app falla en vivo mañana?

## 6. Trampas comunes

| Trampa | Cómo evitarla |
|---|---|
| Probar el enlace solo en tu sesión de GitHub o en tu navegador de siempre | Ventana privada, siempre |
| `models/model.joblib` ignorado por `.gitignore` | Comprueba en GitHub, en el navegador, que aparece |
| Versiones distintas de las del entrenamiento en `requirements.txt` | Copia las versiones con `pip freeze` y quédate solo con las que la app usa |
| Poner un token o un `.env` «solo para que funcione» | Nunca: la app no consulta Databricks |
| Cargar un modelo (`pickle`) de otra persona o de una fuente desconocida | Solo modelos propios: un pickle puede ejecutar código al cargarse |
| Presentación de 8 diapositivas con texto diminuto | Quita detalle técnico: va al informe |
| Abrir con el nombre del algoritmo | Abre con la decisión del cliente |
| Resultado solo en F1 o accuracy | Tradúcelo a costo, dinero o casos |
| Demo solo con el caso feliz | Normal, límite e inválido |
| La app «dormida» el día de la exposición | Ábrela 10 minutos antes; ten captura y app local como plan B |
| No cronometrar el ensayo | Si no lo mides, mañana te pasas |
| Hablar una sola persona | Repartan partes; todos explican su código |
| Aceptar todo lo que propone Claude | Revisa con `kit-estudiante/checklist-revision-ia.md` y pregúntate si podrías explicarlo |

## 7. Qué NO tienes que dominar

Servidores, contenedores, Docker ni infraestructura de nube. Con el enlace de Streamlit Community Cloud basta.
