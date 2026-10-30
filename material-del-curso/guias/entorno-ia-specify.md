# Entorno de desarrollo con IA: Spec Kit, tareas y subagentes (clase 2 · ~20 minutos)

Al terminar esta guía, tu repositorio **le explica a Claude cómo trabaja tu grupo**: tiene una **constitución**
(las reglas del proyecto), sabe pasar de **especificación → plan → tareas → código**, y cuenta con **subagentes** que
revisan tu trabajo. Todo viene **ya instalado** en la rama `feature/estudiantes`, así que viaja con tu repo cuando lo
apuntas al del grupo: no hay nada que instalar.

**Parte de:** el montaje terminado (pasos 1 a 11 de `montaje-proyecto.md`), con Claude Code abierto **en la carpeta del
repo**.

## Las cuatro piezas

| Pieza | Dónde está | Qué hace | Quién la escribe |
|---|---|---|---|
| **`CLAUDE.md`** | raíz | **Cómo** trabajamos: estilo, estructura y reglas que no se negocian | Viene en el kit; solo se ajusta el paquete |
| **Constitución** | `.specify/memory/constitution.md` | Los **principios** del proyecto, en lenguaje de reglas (`MUST`) | Viene sembrada; **el grupo la ajusta** (paso 12.1) |
| **Spec Kit** | `.claude/skills/speckit-*` y `.specify/` | Los comandos `/speckit-…` que llevan de la idea al código | Ya instalado |
| **Subagentes** | `.claude/agents/` | Ayudantes con una sola tarea: revisar fuga, escribir pruebas, revisar código de la IA | Vienen tres; **el grupo puede crear los suyos** (paso 12.4) |

Y las **guías de etapa** (`especificaciones/`) son la **entrada** de Spec Kit: lo que ustedes escriben ahí se convierte en
la especificación formal.

## El flujo de Spec Kit en una línea

```text
/speckit-constitution → /speckit-specify → /speckit-clarify → /speckit-plan → /speckit-tasks → /speckit-analyze → /speckit-implement
   reglas del grupo        qué queremos       dudas              cómo           pasos pequeños     ¿es coherente?      escribir el código
```

Cada comando **deja un archivo** en `specs/<nº>-<nombre>/`: `spec.md`, `plan.md` y `tasks.md`. Ustedes **leen y aprueban
cada uno antes de pasar al siguiente**. Si Claude escribe código sin que hayan leído el plan, se perdió el sentido del flujo.

---

## Paso 12.1 · Ajustar la constitución (10 min · todo el grupo, una pantalla)

1. Abre `.specify/memory/constitution.md` y lee los principios I a VII. Son las reglas del curso, ya escritas como reglas del
   proyecto.
2. En Claude Code, ejecuta `/speckit-constitution` y pídele, con **tus** datos:

   > Ajusta esta constitución a mi proyecto. Nombre: `<proyecto>`. Caso: `<una frase>`. Métrica principal: `<métrica>`.
   > Una fila es `<qué es una fila>` y la entidad que se repite es `<columna o ninguna>`. Conserva los principios I a VII
   > y agrega un principio VIII con esta regla propia del grupo: `<regla>`.

3. **Léela entera** antes de aceptar. Ningún `[corchete]` debe quedar sin llenar, salvo los de fecha.
4. Anota el cambio en la sección 7 de `especificacion-proyecto.md` (fecha, decisión, motivo).

✔ **Comprobación:** al buscar `[` en el archivo (Ctrl+F en VS Code) solo aparecen las fechas, y el
nombre del proyecto aparece en el título.

## Paso 12.2 · Comprobar que Spec Kit está disponible (2 min)

1. En Claude Code, escribe `/speckit` y verás la lista: `speckit-constitution`, `speckit-specify`, `speckit-clarify`,
   `speckit-plan`, `speckit-tasks`, `speckit-analyze`, `speckit-implement`, `speckit-checklist` y `speckit-converge`.
2. Los comandos usan **PowerShell** (`.specify/scripts/powershell/`). En Windows ya lo tienen. **En Mac o Linux** instala
   PowerShell una vez: `brew install powershell` (Mac) y comprueba con `pwsh --version`.

✔ **Comprobación:** ves los comandos `/speckit-…` y, en Mac, `pwsh --version` responde.

## Paso 12.3 · Conocer los subagentes (3 min)

Un **subagente** es un ayudante con **una sola tarea**, que trabaja aparte y devuelve un informe. Vienen tres en
`.claude/agents/`:

| Subagente | Para qué | ¿Modifica archivos? |
|---|---|---|
| `auditor-de-fuga` | Busca partición por fila, preprocesamiento antes de partir, prueba usada para decidir y columnas sospechosas | No, solo lee |
| `escritor-de-pruebas` | Escribe `pytest` para las funciones de `src/` a partir de `especificaciones/03-preprocesamiento.md` | Solo `tests/` |
| `revisor-de-codigo-ia` | Aplica el checklist del curso y **explica cada función** en palabras simples | No, solo lee |

**Cómo se llaman** (pídelos por su nombre, en una frase):

> Usa el subagente `revisor-de-codigo-ia` para revisar el cambio que acabas de hacer en `src/proyecto/data/cleaning.py`.

> Usa el subagente `auditor-de-fuga` sobre `src/` y `notebooks/03_preprocesamiento.ipynb`.

**Cuándo usarlos hoy:** `escritor-de-pruebas` y `revisor-de-codigo-ia` en el bloque de limpieza; `auditor-de-fuga` antes del
punto de control 3. **Regla del curso:** el informe de un subagente **no reemplaza** que ustedes entiendan el código: el
instructor le pregunta a quien no lo escribió.

✔ **Comprobación:** escribe `/agents` y aparecen los tres en la lista.

## Paso 12.4 · Crear un subagente propio (5 min · opcional hoy, obligatorio antes de la clase 4)

1. Ejecuta `/agents` → **Create new agent** → *Project*, y descríbelo: *«Revisa que cada decisión de reports/ tenga cifra,
   acción y motivo»*. O copia uno de `.claude/agents/` y cámbiale `name`, `description` y las instrucciones.
2. Reglas de oro: **una sola tarea**; `tools` con lo **mínimo** (si solo revisa, `Read, Grep, Glob`); en la `description`,
   **cuándo** usarlo.
3. Pruébalo con un caso real y ajusta lo que diga. Guárdalo en `.claude/agents/` y haz commit.

Ideas: un subagente que revise la tabla de decisiones del EDA, uno que verifique que no hay datos personales en `reports/`,
uno que prepare las preguntas orales para el punto de control.

---

## Cómo se usa durante el curso: una especificación por pieza

Para cada pieza de trabajo (por ejemplo, **«limpieza y partición reproducibles»** hoy):

| Paso | Comando | Entrada | Salida (léanla antes de seguir) |
|---|---|---|---|
| 1 | Llenar la guía de etapa | `especificaciones/03-preprocesamiento.md` | Sus reglas, escritas por ustedes |
| 2 | `/speckit-specify` | Esa guía: *«Especifica la limpieza y partición según especificaciones/03-preprocesamiento.md»* | `specs/001-…/spec.md` |
| 3 | `/speckit-clarify` | Dudas de Claude (responden ustedes) | `spec.md` actualizado |
| 4 | `/speckit-plan` | La especificación | `plan.md`: qué archivos y funciones se crean |
| 5 | `/speckit-tasks` | El plan | **`tasks.md`**: lista de tareas pequeñas, con su archivo y su prueba |
| 6 | `/speckit-analyze` | Los tres archivos | Incoherencias entre `spec.md`, `plan.md` y `tasks.md` |
| 7 | `/speckit-implement` | `tasks.md` | El código; cada tarea hecha se marca `[X]` |
| 8 | Subagentes | El código | Informe de `revisor-de-codigo-ia` y `auditor-de-fuga` |

**El `tasks.md` es su lista de trabajo del día.** Cada línea es una tarea pequeña (`T001`, `T002`…) con su archivo. Se leen
en voz alta en el grupo, se reparten por rol y se marcan `[X]` al terminar. Una tarea sin prueba que la verifique no está
terminada.

> **Hoy no tienen que recorrer los ocho pasos completos.** El mínimo para H1 es el 1 (guía llena) y el 8 (revisión con
> subagentes). Los pasos 2 a 7 los practican con la limpieza si el tiempo alcanza, y son **obligatorios desde la clase 3**
> (features y torneo).

## Qué revisa el instructor

- **Constitución** ajustada: nombre, caso, unidad de análisis y al menos un principio propio.
- **Guías de `especificaciones/`** llenas antes del código.
- Si hay `specs/`: `spec.md`, `plan.md` y `tasks.md` **leídos y entendidos** (pregunta a quien no los escribió).
- **Al menos un subagente** probado sobre código real (el grupo muestra el informe y qué corrigió).

## Listo cuando…

- [ ] **1.** La constitución tiene el nombre del proyecto, el caso y la unidad de análisis; sin `[corchetes]` pendientes.
- [ ] **2.** `/speckit-…` aparece en Claude Code (y `pwsh` funciona en Mac/Linux).
- [ ] **3.** `/agents` muestra los tres subagentes y el grupo probó **uno** sobre código real.
- [ ] **4.** Todos saben decir qué hace cada pieza: `CLAUDE.md`, constitución, `especificaciones/`, `specs/`, subagentes.
- [ ] **5.** Los cambios (constitución, subagentes propios) están en un commit.

## Errores comunes

| Síntoma | Causa y solución |
|---|---|
| `/speckit-…` **no aparece** | Claude Code se abrió fuera de la carpeta del repo, o antes de clonar. Cierra y ábrelo **dentro** de la raíz del repo (donde está `.claude/`) |
| `pwsh no se reconoce` / el comando falla al ejecutar un script (Mac/Linux) | Falta PowerShell: `brew install powershell` y vuelve a intentar |
| `/speckit-specify` creó una carpeta `specs/001-…` y no una rama | Es lo esperado: cada pieza de trabajo vive en su carpeta de `specs/` y se sigue trabajando en `main`. Súbanla con el resto del repo |
| `/agents` **no muestra** los subagentes | Mismo caso: Claude Code abierto fuera del repo. O el archivo `.md` perdió el bloque `---` del inicio |
| Claude **escribió código sin plan** | Se saltó el flujo. Pídele: *«Detente. Ejecuta /speckit-plan y espera mi aprobación antes de programar»* |
| La constitución quedó genérica (*«escribir buen código»*) | Una regla sin criterio medible no sirve. Reescríbela como: **qué** se prohíbe o exige, **cómo se comprueba** y **por qué** |
| El subagente **inventó** un hallazgo | Revisa su cita `archivo:línea`. Un hallazgo sin línea que lo pruebe se descarta |
| Subí `.env` por error al hacer commit del entorno de IA | Avisa al instructor y **genera un token nuevo**. `.claude/` y `.specify/` sí se suben; `.env` nunca |
