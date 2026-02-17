# Fricción – Construye agentes de IA entendiendo cada pieza

![FRICCIÓN – Aprende construyendo. Sin magia.](assets/logo.png)

> "El valor en agentes IA no esta en la IA — esta en la ingenieria de sistemas. Memory, coordinacion, learning, input sin friccion, output accionable. Si no resuelves esas 5 cosas, tienes chatbots sofisticados."

Todo el mundo "trabaja con agentes de IA". Instalan un framework, copian un ejemplo, y dicen que saben.

Pero preguntales que es un agent loop. Como persiste la memoria entre sesiones. Que pasa cuando el contexto crece demasiado. Como controlas lo que el agente puede ejecutar. Silencio.

**Porque no hubo friccion. Y sin friccion no hay aprendizaje.**

Friccion es una libreria educativa en Python (~800 lineas) que te obliga a entender cada pieza de un agente de IA. Sin frameworks de 47 capas de abstraccion. Sin magia. Codigo que puedes leer de arriba a abajo.

---

## Por que Friccion

La industria ha convergido en una idea: **la infraestructura es el producto, los agentes son solo la interfaz.** El modelo es commodity — lo que compone es el stack alrededor: memoria, permisos, sesiones, compaction, herramientas.

Los frameworks populares resuelven esto por ti. Y eso es exactamente el problema si quieres **aprender**.

Friccion toma el camino opuesto:

- **No abstrae, expone.** Cada componente (LLM client, session manager, tool registry, permission manager, memory store, agent loop) esta visible y es modificable.
- **Markdown > codigo.** La personalidad, las reglas, las herramientas y la memoria del agente viven en archivos `.md`. Cambias el comportamiento editando texto, no Python. La misma filosofia que `CLAUDE.md` o los Agent Skills: el plan y las preferencias van en archivos que el agente lee, no en logica hardcodeada.
- **El modelo es lo de menos.** Todo usa OpenRouter (API compatible con OpenAI). Cambias de modelo con una variable de entorno. Porque lo que importa es lo que construyes alrededor.
- **Safety nets, no confianza ciega.** Sistema de permisos granular para comandos shell: lista segura, patrones peligrosos, aprobaciones persistentes. Trust the process, build safety nets.
- **Compound learning.** 5 pasos donde cada uno hace el siguiente mas facil. No es un tutorial lineal — es un sistema que acumula capas de la misma forma que un codebase agent-native reduce complejidad con cada ciclo.

---

## Que vas a aprender

No solo "como usar" — sino **como funciona por dentro**:

- **Agent loop**: el modelo decide si responder o llamar a una herramienta. El programa ejecuta la herramienta, devuelve el resultado, y vuelve a preguntar. Este ciclo es el corazon de todo agente — y la mayoria de frameworks te lo ocultan.
- **Workspace como configuracion**: la identidad, personalidad, reglas, herramientas y datos del usuario se componen desde archivos `.md`. Un workspace = un agente. Sin tocar Python.
- **Sesiones persistentes**: historial por usuario en JSONL. El agente recuerda la conversacion.
- **Memoria a largo plazo**: datos que sobreviven a resets de sesion y reinicios. Archivos `.md` con busqueda por relevancia.
- **Compaction**: cuando el contexto crece demasiado, el agente resume la parte antigua y mantiene la reciente. Zero noticeable knowledge loss.
- **Permisos**: control granular de lo que el agente puede ejecutar. Porque un agente sin restricciones no es un agente — es un riesgo.
- **Multi-canal**: Telegram + HTTP API compartiendo la misma logica y sesiones.
- **Heartbeats**: tareas programadas que el agente ejecuta sin que le pidas nada.

---

## Dos formas de usar el proyecto

| Enfoque | Archivos | Para que |
|--------|----------|----------|
| **Aprender paso a paso** | `steps/paso01_bot_basico.py` … `paso05_sistema_completo.py` | Cada script es autocontenido y anade una capa. Ideal para leer, romper y rehacer. |
| **Usar la libreria** | `friccion_lib.py` + `mini_friccionIApy` | La libreria concentra toda la logica; el mini es un ejemplo "todo en uno" (Telegram + HTTP + heartbeats) en pocas lineas. |

Recomendacion: empieza por **paso01** y sigue en orden. Cuando entiendas los pasos, usa **friccion_lib** y **mini_friccionIApy** para montar tu propio agente sin duplicar codigo.

---

## Los 5 pasos

Cada paso es un script ejecutable que anade una capa al anterior. La friccion aumenta — y con ella, lo que entiendes.

| Paso | Que anade | Que aprendes |
|------|-----------|--------------|
| **01** | Bot minimo por Telegram | Una llamada al LLM por mensaje, sin historial. Sin friccion: no recuerda ni tiene personalidad. Aqui ves lo poco que vale un chatbot sin infraestructura. |
| **02** | Sesiones + SOUL | **Sesiones**: historial por usuario en JSONL. **SOUL**: el system prompt se construye leyendo `IDENTITY.md` y `SOUL.md` del workspace. La personalidad y los limites viven en archivos, no en el codigo. Primer contacto con la idea de que **markdown > codigo**. |
| **03** | Tools + agent loop + permisos | **Tools**: el modelo puede llamar a funciones (run_command, read_file, write_file, web_search). **Agent loop**: mientras el modelo pida tools, se ejecutan y se le devuelve el resultado. **Permisos**: lista segura, patrones peligrosos y aprobaciones persistentes. Aqui es donde la mayoria de frameworks te ocultan la magia. Aqui la ves entera. |
| **04** | Compaction + memoria + USER.md | **Compaction**: cuando el historial supera un umbral de tokens, se resume la parte antigua y se mantiene la reciente. **Memoria a largo plazo**: save_memory / search_memory en archivos .md. **USER.md**: lo que el agente sabe del usuario. Aqui el agente deja de ser stateless. |
| **05** | Gateway HTTP + cola + heartbeats | Mismo agente, pero: **HTTP** (API REST + pagina web), **cola** (locks por usuario), **heartbeats** (tareas programadas desde HEARTBEAT.md). Un sistema completo con un solo agente por workspace. |

> El primer intento de entender esto sera confuso. El segundo, menos. El tercero, aterriza. Esa es la friccion haciendo su trabajo.

---

## Workspace: por que archivos .md

El "cerebro" del agente no es un unico prompt fijo en el codigo. Se **compone** leyendo varios `.md` del workspace.

Esto no es una decision arbitraria. Es la misma filosofia que esta emergiendo como estandar en la industria: Agent Skills son Markdown + YAML, no codigo. `CLAUDE.md` define preferencias y reglas. Los archivos `.md` son el sistema operativo del agente.

Con este diseno puedes:

- Cambiar instrucciones o personalidad sin tocar Python.
- Versionar y revisar la configuracion como texto plano.
- Separar responsabilidades: identidad, reglas, herramientas, datos del usuario.
- Editar el comportamiento del agente desde el propio chat (write_file sobre el workspace).

Orden en que se concatenan para formar el system prompt:

```
AGENTS.md + IDENTITY.md + SOUL.md + TOOLS.md + USER.md
```

| Archivo | Rol |
|---------|-----|
| **AGENTS.md** | Reglas de operacion, proceso, limites, uso de memoria. El manual del agente. |
| **IDENTITY.md** | Quien es el agente (nombre, rol, proposito). |
| **SOUL.md** | Tono, estilo, que no hara. La personalidad. |
| **TOOLS.md** | Que tools tiene y cuando usarlas. Guia para el modelo. |
| **USER.md** | Lo que sabe del usuario. Se actualiza con cada conversacion. |
| **HEARTBEAT.md** | Tareas programadas (hora, prompt). El agente actua sin que le pidas. |

Un workspace = un agente. La identidad viene del contenido, no de la estructura de carpetas.

---

## Estructura del repositorio

```
friccion/
  .env.example              Plantilla de variables (copiar a .env)
  requirements.txt          Dependencias (openai, telegram, flask, schedule, etc.)
  friccion_lib.py           Libreria: todo el motor en un solo modulo (~800 lineas)
  mini_friccionIApy         Ejemplo completo usando solo friccion_lib

  workspace/                Configuracion del agente (archivos .md)
    AGENTS.md               Como operar, reglas, limites
    IDENTITY.md             Quien es el agente (nombre, rol, proposito)
    SOUL.md                 Personalidad y tono
    TOOLS.md                Que herramientas tiene y cuando usarlas
    USER.md                 Lo que sabe del usuario
    HEARTBEAT.md            Tareas programadas (opcional)

  steps/                    Scripts educativos (ejecutables en orden)
    paso01_bot_basico.py    Bot minimo sin memoria
    paso02_sesiones_y_alma.py   Sesiones + personalidad desde IDENTITY/SOUL
    paso03_herramientas.py  Tools + agent loop + permisos
    paso04_memoria.py       Compaction + memoria a largo plazo + USER.md
    paso05_sistema_completo.py  Telegram + HTTP + cola + heartbeats
```

---

## Inicio rapido

```bash
pip install -r requirements.txt
cp .env.example .env    # Rellena OPENROUTER_API_KEY y TELEGRAM_BOT_TOKEN
python steps/paso01_bot_basico.py
```

Luego prueba en orden: paso02, paso03, paso04, paso05. Para usar la web de chat en paso05 necesitas Flask; el puerto por defecto es **5001** (en macOS el 5000 suele usarlo AirPlay). Abre `http://localhost:5001/`.

---

## La libreria: friccion_lib.py

Todo el motor en un solo modulo importable. ~800 lineas, sin dependencias ocultas, sin herencia profunda.

**Clases principales:**

- **LLMClient** – Wrapper para OpenRouter / cualquier API compatible con OpenAI.
- **SessionManager** – Sesiones persistentes en JSONL por usuario.
- **PermissionManager** – Control de permisos para comandos shell (ask/record/ignore, approvals, globs).
- **ToolRegistry** – Registro y ejecucion de tools (run_command, read_file, write_file, web_search, save_memory, search_memory).
- **MemoryStore** – Memoria a largo plazo en archivos .md con busqueda por relevancia.
- **SessionQueue** – Locks por sesion (sync y async) para no procesar dos mensajes del mismo usuario a la vez.
- **Agent** – Agent loop con soporte de tools y compaction.
- **MultiAgentRouter** – Enrutar mensajes por prefijo de comando.
- **TaskScheduler** – Heartbeats desde HEARTBEAT.md.

**Funciones de workspace:**

- `compose_prompt(workspace_dir)` – Devuelve el system prompt concatenando los .md del workspace.
- `discover_agents(workspace_dir)` – Detecta si hay un agente configurado (busca IDENTITY.md).
- `parse_heartbeats(workspace_dir)` – Parsea HEARTBEAT.md.

**Ejemplo minimo:**

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

Un solo script que usa **solo** la libreria para tener Telegram + HTTP + heartbeats, sin reimplementar nada.

En pocas lineas:

1. Carga configuracion desde `.env`.
2. Crea LLM, sessions, permissions, tools, memory, SessionQueue.
3. Descubre el agente con `discover_agents(WORKSPACE_DIR)`.
4. Crea un `Agent` y un `MultiAgentRouter`.
5. Arranca el gateway HTTP y el TaskScheduler (heartbeats).
6. Conecta Telegram: `/nuevo`, mensajes normales, `/investigar <texto>`.

```bash
python mini_friccionIA.py
```

Usalo cuando ya entiendas los pasos y quieras un bot listo. Si quieres aprender como esta hecho cada capa, usa los scripts en `steps/`.

---

## Conceptos clave

- **Agent loop**: el modelo devuelve "llamar a esta tool"; el programa ejecuta, anade el resultado al historial y vuelve a llamar al modelo hasta que responde en texto. Es el patron central de cualquier agente.
- **Workspace**: la configuracion completa del agente en archivos `.md`. Un workspace = un agente. Editable, versionable, legible.
- **Sesion**: historial de mensajes de un usuario. Se guarda en `data/sessions/<clave>.jsonl`. Persiste entre reinicios.
- **Compaction**: cuando el historial supera un umbral de tokens, se pide al modelo un resumen de la parte antigua y se sustituye por ese resumen + mensajes recientes.
- **Memoria a largo plazo**: archivos `.md` en `data/memory/` que el agente puede escribir (save_memory) y consultar (search_memory). Sobreviven a resets de sesion.
- **Permisos**: cada comando se evalua contra una lista segura, patrones peligrosos y aprobaciones previas. Modos: `ask` (bloquear), `record` (permitir y registrar), `ignore` (solo pruebas).
- **Heartbeats**: tareas programadas que el agente ejecuta automaticamente. Configuradas en HEARTBEAT.md.

---

## Variables de entorno

| Variable | Default | Descripcion |
|----------|---------|-------------|
| OPENROUTER_API_KEY | (requerido) | API key de OpenRouter |
| OPENROUTER_MODEL | minimax/minimax-m2.5 | Modelo a usar (cambia aqui, no en el codigo) |
| TELEGRAM_BOT_TOKEN | (requerido) | Token del bot de Telegram |
| WORKSPACE_DIR | ./workspace | Directorio del workspace |
| DATA_DIR | ./data | Directorio de datos (sessions, memory, approvals) |
| PERMISSION_MODE | ask | ask / record / ignore |
| MAX_ITERATIONS | 10 | Limite de llamadas a tools por turno |
| COMMAND_TIMEOUT | 30 | Timeout en segundos para run_command |
| COMPACTION_THRESHOLD | 100000 | Tokens a partir de los cuales se compacta (pruebas: 5000) |
| HTTP_PORT | 5001 | Puerto del HTTP gateway |

---

## Seguridad

- **No subas `.env`** a ningun repositorio (contiene claves).
- Los permisos bloquean por defecto comandos peligrosos; en produccion revisa la lista y los patrones.
- Sessions y memoria se guardan en tu maquina; si expones el HTTP, protege acceso y usa HTTPS en produccion.

---

## Disclaimer

Friccion es un proyecto **educativo** cuyo unico proposito es ensenar como funcionan los agentes de IA por dentro. No esta pensado para entornos de produccion ni para manejar datos sensibles.

El autor no se hace responsable del uso que se le de a este codigo ni de los danos que pueda causar su uso indebido. Usalo bajo tu propia responsabilidad.

Si decides ejecutar el agente con `run_command` habilitado, ten en cuenta que estas permitiendo que un LLM ejecute comandos en tu maquina. Revisa los permisos, limita el acceso, y no lo expongas a redes publicas sin proteccion.

---

## Licencia

MIT License. Puedes usar, copiar, modificar y distribuir este codigo libremente, con o sin modificaciones, siempre que incluyas el aviso de licencia original. **Se proporciona "tal cual", sin garantias de ningun tipo.**
