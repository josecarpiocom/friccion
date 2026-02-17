# Fricción – Construye agentes de IA entendiendo cada pieza

![FRICCIÓN – Aprende construyendo. Sin magia.](assets/logo.png)

> El valor en agentes de IA no está en la IA — está en la ingeniería de sistemas. Memoria, coordinación, aprendizaje, entrada sin fricción, salida accionable. Si no resuelves esas 5 cosas, tienes chatbots sofisticados.

Todo el mundo "trabaja con agentes de IA". Instalan un framework, copian un ejemplo y dicen que saben.

Pero pregúntales qué es un bucle de agente. Cómo persiste la memoria entre sesiones. Qué pasa cuando el contexto crece demasiado. Cómo controlas lo que el agente puede ejecutar. Silencio.

**Porque no hubo fricción. Y sin fricción no hay aprendizaje.**

Fricción es una librería educativa en Python (~800 líneas) que te obliga a entender cada pieza de un agente de IA. Sin frameworks de 47 capas de abstracción. Sin magia. Código que puedes leer de arriba a abajo.

---

## Por qué Fricción

La industria converge en una idea: **la infraestructura es el producto, los agentes son solo la interfaz.** El modelo es commodity — lo que importa es lo que construyes alrededor: memoria, permisos, sesiones, compactación, herramientas.

Los frameworks populares resuelven esto por ti. Y eso es exactamente el problema si quieres **aprender**.

Fricción toma el camino opuesto:

- **No abstrae, expone.** Cada componente (cliente LLM, gestor de sesiones, registro de herramientas, gestor de permisos, almacén de memoria, bucle de agente) está visible y es modificable.
- **Markdown > código.** La personalidad, las reglas, las herramientas y la memoria del agente viven en archivos `.md`. Cambias el comportamiento editando texto, no Python. La misma filosofía que `CLAUDE.md` o los Agent Skills: el plan y las preferencias van en archivos que el agente lee, no en lógica fija en el código.
- **El modelo es lo de menos.** Todo usa OpenRouter (API compatible con OpenAI). Cambias de modelo con una variable de entorno. Porque lo que importa es lo que construyes alrededor.
- **Redes de seguridad, no confianza ciega.** Sistema de permisos granular para comandos shell: lista segura, patrones peligrosos, aprobaciones persistentes.
- **Aprendizaje compuesto.** 5 pasos donde cada uno hace el siguiente más fácil. No es un tutorial lineal — es un sistema que acumula capas de la misma forma que un codebase nativo de agentes reduce complejidad con cada ciclo.

---

## Qué vas a aprender

No solo "cómo usar" — sino **cómo funciona por dentro**:

- Bucle de agente
- Workspace como configuración (archivos `.md`)
- Sesiones persistentes
- Memoria a largo plazo
- Compactación de contexto
- Permisos granulares
- Multi-canal (Telegram + HTTP API)
- Tareas programadas (heartbeats)

Cada concepto se introduce en los 5 pasos y se define en el [glosario](#conceptos-clave).

---

## Dos formas de usar el proyecto

| Enfoque | Archivos | Para qué |
|--------|----------|----------|
| **Aprender paso a paso** | `steps/paso01_bot_basico.py` … `paso05_sistema_completo.py` | Cada script es autocontenido y añade una capa. Ideal para leer, romper y rehacer. |
| **Usar la librería** | `friccion_lib.py` + `mini_friccionIA.py` | La librería concentra toda la lógica; el mini es un ejemplo "todo en uno" (Telegram + HTTP + heartbeats) en pocas líneas. |

Recomendación: empieza por **paso01** y sigue en orden. Cuando entiendas los pasos, usa **friccion_lib** y **mini_friccionIA.py** para montar tu propio agente sin duplicar código.

---

## Los 5 pasos

Cada paso es un script ejecutable que añade una capa al anterior. La fricción aumenta — y con ella, lo que entiendes.

| Paso | Qué añade | Qué aprendes |
|------|-----------|--------------|
| **01** | Bot mínimo por Telegram | Una llamada al LLM por mensaje, sin historial. Sin fricción: no recuerda ni tiene personalidad. Aquí ves lo poco que vale un chatbot sin infraestructura. |
| **02** | Sesiones + SOUL | **Sesiones**: historial por usuario en JSONL. **SOUL**: el system prompt se construye leyendo `IDENTITY.md` y `SOUL.md` del workspace. La personalidad y los límites viven en archivos, no en el código. Primer contacto con la idea de que **markdown > código**. |
| **03** | Herramientas + bucle de agente + permisos | **Herramientas**: el modelo puede llamar a funciones (run_command, read_file, write_file, web_search). **Bucle de agente**: mientras el modelo pida herramientas, se ejecutan y se le devuelve el resultado. **Permisos**: lista segura, patrones peligrosos y aprobaciones persistentes. Aquí es donde la mayoría de frameworks te ocultan la magia. Aquí la ves entera. |
| **04** | Compactación + memoria + USER.md | **Compactación**: cuando el historial supera un umbral de tokens, se resume la parte antigua y se mantiene la reciente. **Memoria a largo plazo**: save_memory / search_memory en archivos .md. **USER.md**: lo que el agente sabe del usuario. Aquí el agente deja de ser sin estado. |
| **05** | Gateway HTTP + cola + heartbeats | Mismo agente, pero: **HTTP** (API REST + página web), **cola** (locks por usuario), **heartbeats** (tareas programadas desde HEARTBEAT.md). Un sistema completo con un solo agente por workspace. |

> El primer intento de entender esto será confuso. El segundo, menos. El tercero, aterriza. Esa es la fricción haciendo su trabajo.

---

## Workspace: por qué archivos .md

El "cerebro" del agente no es un único prompt fijo en el código. Se **compone** leyendo varios `.md` del workspace.

Esto no es una decisión arbitraria. Es la misma filosofía que está emergiendo como estándar en la industria: los Agent Skills son Markdown + YAML, no código. `CLAUDE.md` define preferencias y reglas. Los archivos `.md` son el sistema operativo del agente.

Con este diseño puedes:

- Cambiar instrucciones o personalidad sin tocar Python.
- Versionar y revisar la configuración como texto plano.
- Separar responsabilidades: identidad, reglas, herramientas, datos del usuario.
- Editar el comportamiento del agente desde el propio chat (write_file sobre el workspace).

Orden en que se concatenan para formar el system prompt:

```
AGENTS.md + IDENTITY.md + SOUL.md + TOOLS.md + USER.md
```

| Archivo | Rol |
|---------|-----|
| **AGENTS.md** | Reglas de operación, proceso, límites, uso de memoria. El manual del agente. |
| **IDENTITY.md** | Quién es el agente (nombre, rol, propósito). |
| **SOUL.md** | Tono, estilo, qué no hará. La personalidad. |
| **TOOLS.md** | Qué herramientas tiene y cuándo usarlas. Guía para el modelo. |
| **USER.md** | Lo que sabe del usuario. Se actualiza con cada conversación. |
| **HEARTBEAT.md** | Tareas programadas (hora, prompt). El agente actúa sin que le pidas. |

Un workspace = un agente. La identidad viene del contenido, no de la estructura de carpetas.

---

## Estructura del repositorio

```
friccion/
  .env.example              Plantilla de variables (copiar a .env)
  requirements.txt          Dependencias (openai, telegram, flask, schedule, etc.)
  friccion_lib.py           Librería: todo el motor en un solo módulo (~800 líneas)
  mini_friccionIA.py        Ejemplo completo usando solo friccion_lib

  workspace/                Configuración del agente (archivos .md)
    AGENTS.md               Cómo operar, reglas, límites
    IDENTITY.md             Quién es el agente (nombre, rol, propósito)
    SOUL.md                 Personalidad y tono
    TOOLS.md                Qué herramientas tiene y cuándo usarlas
    USER.md                 Lo que sabe del usuario
    HEARTBEAT.md            Tareas programadas (opcional)

  steps/                    Scripts educativos (ejecutables en orden)
    paso01_bot_basico.py    Bot mínimo sin memoria
    paso02_sesiones_y_alma.py   Sesiones + personalidad desde IDENTITY/SOUL
    paso03_herramientas.py  Herramientas + bucle de agente + permisos
    paso04_memoria.py       Compactación + memoria a largo plazo + USER.md
    paso05_sistema_completo.py  Telegram + HTTP + cola + heartbeats
```

---

## Inicio rápido

```bash
pip install -r requirements.txt
cp .env.example .env    # Rellena OPENROUTER_API_KEY y TELEGRAM_BOT_TOKEN
python steps/paso01_bot_basico.py
```

Luego prueba en orden: paso02, paso03, paso04, paso05. Para usar la web de chat en paso05 necesitas Flask; el puerto por defecto es **5001** (en macOS el 5000 suele usarlo AirPlay). Abre `http://localhost:5001/`.

---

## La librería: friccion_lib.py

Todo el motor en un solo módulo importable. ~800 líneas, sin dependencias ocultas, sin herencia profunda.

**Clases principales:**

- **LLMClient** – Envoltorio para OpenRouter / cualquier API compatible con OpenAI.
- **SessionManager** – Sesiones persistentes en JSONL por usuario.
- **PermissionManager** – Control de permisos para comandos shell (ask/record/ignore, aprobaciones, patrones).
- **ToolRegistry** – Registro y ejecución de herramientas (run_command, read_file, write_file, web_search, save_memory, search_memory).
- **MemoryStore** – Memoria a largo plazo en archivos .md con búsqueda por relevancia.
- **SessionQueue** – Locks por sesión (sync y async) para no procesar dos mensajes del mismo usuario a la vez.
- **Agent** – Bucle de agente con soporte de herramientas y compactación.
- **MultiAgentRouter** – Enrutar mensajes por prefijo de comando.
- **TaskScheduler** – Heartbeats desde HEARTBEAT.md.

**Funciones de workspace:**

- `compose_prompt(workspace_dir)` – Devuelve el system prompt concatenando los .md del workspace.
- `discover_agents(workspace_dir)` – Detecta si hay un agente configurado (busca IDENTITY.md).
- `parse_heartbeats(workspace_dir)` – Parsea HEARTBEAT.md.

**Ejemplo mínimo:**

```python
from friccion_lib import (
    LLMClient, SessionManager, Agent,
    create_base_tools, PermissionManager, MemoryStore,
    compose_prompt, discover_agents,
)

client = LLMClient()
sessions = SessionManager()
permissions = PermissionManager()
tools = create_base_tools(permission_checker=permissions.evaluate)
memory = MemoryStore()
memory.register_tools(tools)

soul = compose_prompt("./workspace")
agent = Agent(client=client, tools=tools, soul=soul, sessions=sessions)
print(agent.run_turn("user1", "Hola!"))
```

---

## mini_friccionIA.py: ejemplo "todo en uno"

Un solo script que usa **solo** la librería para tener Telegram + HTTP + heartbeats, sin reimplementar nada.

En pocas líneas:

1. Carga configuración desde `.env`.
2. Crea LLM, sesiones, permisos, herramientas, memoria, cola de sesiones.
3. Descubre el agente con `discover_agents(WORKSPACE_DIR)`.
4. Crea un `Agent` y un `MultiAgentRouter`.
5. Arranca el gateway HTTP y el TaskScheduler (heartbeats).
6. Conecta Telegram: `/nuevo`, mensajes normales, `/investigar <texto>`.

```bash
python mini_friccionIA.py
```

Úsalo cuando ya entiendas los pasos y quieras un bot listo. Si quieres aprender cómo está hecho cada capa, usa los scripts en `steps/`.

---

## Conceptos clave

- **Bucle de agente**: el modelo responde o pide ejecutar una herramienta; el programa ejecuta, devuelve el resultado y repite hasta obtener texto.
- **Workspace**: configuración completa del agente en archivos `.md`. Editable, versionable, legible.
- **Sesión**: historial de mensajes por usuario en `data/sessions/<clave>.jsonl`. Persiste entre reinicios.
- **Compactación**: resumen automático del historial antiguo cuando supera un umbral de tokens. Sin pérdida perceptible de conocimiento.
- **Memoria a largo plazo**: archivos `.md` en `data/memory/` que el agente escribe y consulta. Sobreviven a resets de sesión.
- **Permisos**: cada comando se evalúa contra lista segura, patrones peligrosos y aprobaciones previas. Modos: `ask`, `record`, `ignore`.
- **Heartbeats**: tareas programadas que el agente ejecuta automáticamente desde HEARTBEAT.md.

---

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| OPENROUTER_API_KEY | (requerido) | API key de OpenRouter |
| OPENROUTER_MODEL | minimax/minimax-m2.5 | Modelo a usar (cambia aquí, no en el código) |
| TELEGRAM_BOT_TOKEN | (requerido) | Token del bot de Telegram |
| WORKSPACE_DIR | ./workspace | Directorio del workspace |
| DATA_DIR | ./data | Directorio de datos (sesiones, memoria, aprobaciones) |
| PERMISSION_MODE | ask | ask / record / ignore |
| MAX_ITERATIONS | 10 | Límite de llamadas a herramientas por turno |
| COMMAND_TIMEOUT | 30 | Timeout en segundos para run_command |
| COMPACTION_THRESHOLD | 100000 | Tokens a partir de los cuales se compacta (pruebas: 5000) |
| HTTP_PORT | 5001 | Puerto del gateway HTTP |

---

## Seguridad

- **No subas `.env`** a ningún repositorio (contiene claves).
- Los permisos bloquean por defecto comandos peligrosos; en producción revisa la lista y los patrones.
- Sesiones y memoria se guardan en tu máquina; si expones el HTTP, protege el acceso y usa HTTPS en producción.

---

## Disclaimer

Fricción es un proyecto **educativo** cuyo único propósito es enseñar cómo funcionan los agentes de IA por dentro. No está pensado para entornos de producción ni para manejar datos sensibles.

El autor no se hace responsable del uso que se le dé a este código ni de los daños que pueda causar su uso indebido. Úsalo bajo tu propia responsabilidad.

Si decides ejecutar el agente con `run_command` habilitado, ten en cuenta que estás permitiendo que un LLM ejecute comandos en tu máquina. Revisa los permisos, limita el acceso y no lo expongas a redes públicas sin protección.

---

## Licencia

MIT License. Puedes usar, copiar, modificar y distribuir este código libremente, con o sin modificaciones, siempre que incluyas el aviso de licencia original. **Se proporciona "tal cual", sin garantías de ningún tipo.**
