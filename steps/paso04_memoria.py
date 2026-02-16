"""
================================================================================
PASO 4 - CONTEXT COMPACTION + LONG-TERM MEMORY + USER.md
================================================================================

Aqui el agente deja de ser stateless. Y eso cambia todo.

Dos problemas que nadie te cuenta cuando empiezas con agentes:

  1. CONTEXTO QUE CRECE SIN LIMITE: despues de semanas chateando, el historial
     supera la ventana de contexto del modelo. No puedes meter todo.
     Solucion: compaction. El agente resume los mensajes antiguos y mantiene
     los recientes intactos. Aprende a decidir que preservar y que soltar.
     Como la memoria humana: no recuerdas cada palabra, pero recuerdas
     lo que importa.

  2. MEMORY QUE NO SOBREVIVE AL RESET: si reseteas la session o reinicias
     el bot, todo se pierde. Solucion: long-term memory basada en archivos
     .md que persisten para siempre. El agente puede escribir y buscar
     en su propia memoria. Datos que sobreviven a todo.

WORKSPACE (en este paso se suma USER.md):
  El system prompt se compone desde 5 archivos en la raiz:

    workspace/
      AGENTS.md            <- Instrucciones de operacion
      IDENTITY.md          <- Quien es
      SOUL.md              <- Personalidad
      TOOLS.md             <- Tools disponibles
      USER.md              <- Lo que el bot sabe sobre el humano

  USER.md es la pieza que cierra el circulo: el agente no solo recuerda
  la conversacion — sabe QUIEN eres. Nombre, preferencias, proyectos.
  Siempre en el system prompt, siempre disponible.

FRICCION SUTIL:
  La compaction decide QUE preservar y que se puede resumir. El resumen
  prioriza datos clave del usuario, decisiones tomadas y tareas pendientes.
  No es borrar contexto — es destilarlo. El mismo patron que los mejores
  sistemas de agentes usan en produccion: memory compounding.

COMO FUNCIONA LA MEMORY:
  - save_memory: escribe un archivo .md en ./data/memory/
  - search_memory: busca por keywords con scoring TF-IDF simplificado
  - La memory sobrevive a resets de session y reinicios del bot
  - Los agents (paso 5) comparten la memory
  - El patron: no consumir pasivamente — estructurar para recuperar

COMO EJECUTAR:
  python steps/paso04_memoria.py

  Para probar compaction rapido, baja el umbral:
    COMPACTION_THRESHOLD=1000

PRUEBALO:
  Tu: Recuerda que mi bar favorito es Casa Gonzalez y siempre pido
      tortilla con cebolla y una cana
  Bot: [usa save_memory: "preferencias-usuario"]
       Apuntado. Casa Gonzalez, tortilla con cebolla y cana.

  [Reseteas la session o reinicias el bot]

  Tu: Oye, donde vamos de canas esta tarde?
  Bot: [usa search_memory: "bar canas favorito"]
       Que tal Casa Gonzalez? Se que te gusta la tortilla con cebolla
       de alli. Vamos y pides tu cana de siempre?

  Tu: Recomiendame algo para comer en el centro de Madrid
  Bot: [usa search_memory] Sabiendo que te va la tortilla de patatas...
       Te recomiendo: Casa Lucio (huevos rotos miticos), La Barraca
       (paella de las buenas), o El Brillante (bocata de calamares).

  El agente no busco en Google. Busco en lo que sabe de TI.
  Esa es la diferencia entre un buscador y un asistente personal.

================================================================================
"""

import fnmatch
import json
import math
import os
import re
import subprocess
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _web_search_impl(query: str, max_results: int = 5) -> str:
    """Busca en la web con DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No se encontraron resultados."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            href = r.get("href", "")
            body = (r.get("body") or "").strip()
            lines.append(f"{i}. {title}\n   {href}\n   {body}")
        return "\n\n".join(lines)
    except ImportError:
        return "Para buscar en la web instala: pip install duckduckgo-search"
    except Exception as e:
        return f"Error en la busqueda: {e}"


client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")
COMMAND_TIMEOUT = int(os.getenv("COMMAND_TIMEOUT", "30"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "100000"))

DATA_DIR = os.getenv("DATA_DIR", "./data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
MEMORY_DIR = os.path.join(DATA_DIR, "memory")
APPROVALS_FILE = os.path.join(DATA_DIR, "approvals.json")
PERMISSION_MODE = os.getenv("PERMISSION_MODE", "ask")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
if not os.path.exists(APPROVALS_FILE):
    with open(APPROVALS_FILE, "w", encoding="utf-8") as f:
        json.dump({"allowed": [], "denied": [], "globs": []}, f, indent=2, ensure_ascii=False)


# =============================================================================
# SOUL - Ahora con USER.md
# =============================================================================

def load_workspace_file(relative_path: str) -> str:
    full_path = os.path.join(WORKSPACE_DIR, relative_path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def compose_soul() -> str:
    """Compone el system prompt desde 5 archivos del workspace (raiz).

    Orden: AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md, USER.md.
    """
    parts = [
        load_workspace_file("AGENTS.md"),
        load_workspace_file("IDENTITY.md"),
        load_workspace_file("SOUL.md"),
        load_workspace_file("TOOLS.md"),
        load_workspace_file("USER.md"),
    ]
    return "\n\n".join(p for p in parts if p)


SOUL = compose_soul()


# =============================================================================
# CONTEXT COMPACTION
# =============================================================================
# El contexto del LLM es finito. Despues de muchos mensajes, no cabe todo.
# La solucion naive: borrar los mensajes viejos. La solucion real: resumirlos.
#
# Compaction = pedirle al propio modelo que resuma la parte antigua del
# historial, preservando datos del usuario, decisiones y tareas pendientes.
# El resumen sustituye a los mensajes originales. Zero noticeable knowledge loss.
#
# Ademas, el resumen se guarda en long-term memory. Asi, si la session se
# resetea, el conocimiento destilado sigue disponible via search_memory.

def estimate_tokens(messages: list[dict]) -> int:
    text = json.dumps(messages)
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except (ImportError, Exception):
        return len(text) // 4


def compact_session(session_id: str, messages: list[dict]) -> list[dict]:
    """Compacta la session si supera el umbral de tokens."""
    tokens = estimate_tokens(messages)
    if tokens < COMPACTION_THRESHOLD:
        logger.info("Session %s: %d tokens (umbral %d) -> sin compactar", session_id, tokens, COMPACTION_THRESHOLD)
        return messages

    logger.info("Session %s: %d tokens (umbral %d) -> compactando...", session_id, tokens, COMPACTION_THRESHOLD)
    half = len(messages) // 2
    old_msgs = messages[:half]
    recent_msgs = messages[half:]

    summary_parts = []
    for m in old_msgs:
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            summary_parts.append(f"[{m.get('role', '?')}]: {content[:500]}")

    try:
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=2000,
            messages=[{"role": "user", "content":
                "Resume esta conversacion. Preserva:\n"
                "- Datos del usuario (nombre, preferencias, ciudad)\n"
                "- Decisiones tomadas\n"
                "- Tareas pendientes\n\n"
                + "\n".join(summary_parts)}],
        )
        summary = resp.choices[0].message.content
    except Exception as e:
        logger.error("Error al compactar: %s", e)
        return messages

    compacted = [
        {"role": "user", "content": f"[Resumen de conversacion anterior]\n{summary}"},
        {"role": "assistant", "content": "Entendido, tengo el contexto."},
    ] + recent_msgs
    save_session(session_id, compacted)
    logger.info("Compactado: %d -> %d mensajes", len(messages), len(compacted))

    # Guardar resumen en memoria a largo plazo para que search_memory lo encuentre
    try:
        import time
        key = f"compaction-{session_id}-{int(time.time())}"
        save_memory_file(key, summary)
        logger.info("Resumen guardado en memory: %s", key)
    except Exception as e:
        logger.warning("No se pudo guardar resumen en memory: %s", e)

    return compacted


# =============================================================================
# LONG-TERM MEMORY
# =============================================================================
# La memoria a largo plazo es la infraestructura que separa un chatbot de
# un asistente. Archivos .md que el agente puede escribir y consultar.
# Sobreviven a resets, reinicios, y al paso del tiempo.
#
# save_memory: el agente decide que merece recordarse y lo escribe.
# search_memory: busca por keywords con scoring TF-IDF simplificado.
# No necesitas embeddings ni vectores — para la mayoria de casos de uso,
# una busqueda por terminos con un scoring decente es suficiente.

def save_memory_file(key: str, content: str) -> str:
    safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
    path = os.path.join(MEMORY_DIR, f"{safe_key}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Memory guardada: %s", key)
    return f"Guardado en memory: {key}"


def search_memory(query: str) -> str:
    """Busca en memory con scoring TF-IDF simplificado."""
    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words or not os.path.exists(MEMORY_DIR):
        return "No se encontraron memorias relevantes."

    results: list[tuple[float, str, str]] = []
    for filename in os.listdir(MEMORY_DIR):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(MEMORY_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()
        cl = content.lower()
        dl = max(len(cl.split()), 1)
        score = sum((1 + math.log(max(cl.count(w), 1))) / math.sqrt(dl)
                     for w in words if w in cl)
        if score > 0:
            results.append((score, filename, content))

    if not results:
        return "No se encontraron memorias relevantes."
    results.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(f"--- {n} (relevancia: {s:.2f}) ---\n{c}"
                       for s, n, c in results[:5])


# =============================================================================
# PERMISSIONS
# =============================================================================

SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami",
                 "echo", "pwd", "env", "python", "python3", "node"}

def load_approvals() -> dict:
    if os.path.exists(APPROVALS_FILE):
        try:
            with open(APPROVALS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"allowed": [], "denied": [], "globs": []}

def evaluate_permission(command: str) -> str:
    if PERMISSION_MODE == "ignore":
        return "safe"
    cmd_base = command.strip().split()[0] if command.strip() else ""
    if cmd_base in SAFE_COMMANDS:
        return "safe"
    approvals = load_approvals()
    if command in approvals["allowed"]:
        return "approved"
    for pattern in approvals.get("globs", []):
        if fnmatch.fnmatch(command, pattern):
            return "approved"
    if PERMISSION_MODE == "record":
        return "approved"
    return "requires_approval"


# =============================================================================
# TOOLS - Con save_memory y search_memory
# =============================================================================

TOOLS = [
    {"type": "function", "function": {"name": "run_command",
     "description": "Ejecutar un comando de shell",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file",
     "description": "Leer un archivo",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "Escribir un archivo",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "web_search",
     "description": "Buscar en la web",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "save_memory",
     "description": "Guardar info importante en long-term memory",
     "parameters": {"type": "object", "properties": {
         "key": {"type": "string", "description": "Etiqueta corta descriptiva"},
         "content": {"type": "string", "description": "Informacion a recordar"}
     }, "required": ["key", "content"]}}},
    {"type": "function", "function": {"name": "search_memory",
     "description": "Buscar en la long-term memory",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Que buscar"}
     }, "required": ["query"]}}},
]


def execute_tool(name: str, arguments: dict) -> str:
    try:
        if name == "run_command":
            cmd = arguments["command"]
            if evaluate_permission(cmd) == "requires_approval":
                return "Permiso denegado."
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=COMMAND_TIMEOUT)
            return (r.stdout + r.stderr).strip() or "(ok)"
        elif name == "read_file":
            with open(arguments["path"], "r", encoding="utf-8") as f:
                return f.read()
        elif name == "write_file":
            d = os.path.dirname(arguments["path"])
            if d:
                os.makedirs(d, exist_ok=True)
            with open(arguments["path"], "w", encoding="utf-8") as f:
                f.write(arguments["content"])
            return f"Escrito en {arguments['path']}"
        elif name == "web_search":
            return _web_search_impl(arguments["query"])
        elif name == "save_memory":
            return save_memory_file(arguments["key"], arguments["content"])
        elif name == "search_memory":
            return search_memory(arguments["query"])
        return f"Tool desconocida: {name}"
    except Exception as e:
        return f"Error ejecutando {name}: {e}"


# =============================================================================
# AGENT LOOP + SESSIONS
# =============================================================================

def run_agent_turn(messages: list[dict], system_prompt: str) -> tuple[str, list[dict]]:
    api_messages = [{"role": "system", "content": system_prompt}] + messages
    for _ in range(MAX_ITERATIONS):
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=4096,
            messages=api_messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages
        assistant_msg: dict = {
            "role": "assistant", "content": msg.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls],
        }
        messages.append(assistant_msg)
        api_messages.append(assistant_msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            logger.info("  Tool: %s(%s)", tc.function.name, json.dumps(args))
            result = execute_tool(tc.function.name, args)
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            messages.append(tool_msg)
            api_messages.append(tool_msg)
    return "Limite de iteraciones alcanzado.", messages

def session_path(uid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{uid}.jsonl")

def load_session(uid: str) -> list[dict]:
    path = session_path(uid)
    msgs: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        msgs.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Linea %d corrupta en %s", i, uid)
    return msgs

def save_session(uid: str, msgs: list[dict]) -> None:
    with open(session_path(uid), "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


# =============================================================================
# HANDLER DE TELEGRAM (con compaction)
# =============================================================================

async def handle_message(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    messages = load_session(user_id)
    messages = compact_session(user_id, messages)
    messages.append({"role": "user", "content": update.message.text})

    try:
        text, messages = run_agent_turn(messages, SOUL)
    except Exception as e:
        logger.error("Error: %s", e)
        text = f"Error: {e}"

    save_session(user_id, messages)
    await update.message.reply_text(text)


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN")

    logger.info("Iniciando bot con memory (modelo: %s)", MODEL)
    logger.info("  Workspace: %s", WORKSPACE_DIR)
    logger.info("  Memory:    %s", MEMORY_DIR)
    logger.info("  Compaction: umbral %d tokens (para probar con poca conversacion: COMPACTION_THRESHOLD=5000)", COMPACTION_THRESHOLD)

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
