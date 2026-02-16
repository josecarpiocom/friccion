"""
================================================================================
PASO 5 - SISTEMA COMPLETO (un agente)
================================================================================

Todo junto. Este es el resultado de los 4 pasos anteriores: un agente
completo con tools, memoria, permisos, compaction, multi-canal y tareas
programadas. Todo en un solo script, sin dependencias externas de frameworks.

Si has llegado hasta aqui leyendo los pasos en orden, ya entiendes CADA
pieza de lo que ves abajo. No hay magia. No hay abstracciones que no
hayas visto por dentro. Esa era la idea desde el principio.

  1. GATEWAY: Telegram + API HTTP comparten la misma session por usuario.
     Hablas por Telegram, abres la web, y el agente sabe quien eres.

  2. SESSION QUEUE: locks por usuario. Si mandas dos mensajes seguidos,
     el segundo espera a que termine el primero. Sin esto, dos respuestas
     simultaneas corrompen el historial.

  3. HEARTBEATS: tareas programadas leidas de HEARTBEAT.md. El agente
     actua sin que le pidas nada. Ejemplo: briefing matutino a las 7:30.
     Configurado en un archivo .md, no en codigo.

  4. COMPACTION: igual que paso 4. Contexto que crece -> se resume.

ESTE ES EL NIVEL "INFRAESTRUCTURA":
  La infraestructura es el producto. Los agentes son solo la interfaz.
  Lo que convierte un chatbot en algo util no es el modelo — es todo
  lo que hay alrededor: sesiones, memoria, permisos, locks, heartbeats.
  Este paso es la prueba de que puedes tener todo eso en ~700 lineas
  de Python sin frameworks.

WORKSPACE (raiz, MAYÚSCULAS):
  workspace/
    AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md, USER.md, HEARTBEAT.md

COMANDOS:
  /nuevo             -> resetea tu session
  /investigar <texto> -> mismo agente, quita el prefijo
  (resto)            -> mensaje normal

COMO EJECUTAR:
  python steps/paso05_sistema_completo.py

PRUEBALO:
  Telegram: "Me llamo Jose" -> el agente recuerda.
  HTTP:     curl -X POST http://127.0.0.1:5001/chat \\
              -H "Content-Type: application/json" \\
              -d '{"user_id": "TU_ID_TELEGRAM", "message": "Como me llamo?"}'
  /investigar mejores arrocerias -> responde y puede guardar en memory.

  Abre la web y Telegram a la vez con el mismo user_id. El agente
  mantiene el contexto en ambos canales. Un agente, multiples interfaces.

  Y cuando domines esto, mira friccion_lib.py + mini_friccionIApy:
  todo lo de este script empaquetado en una libreria importable.

================================================================================
"""

import asyncio
import fnmatch
import json
import math
import os
import re
import subprocess
import threading
import time
import logging
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import schedule
except ImportError:
    schedule = None  # heartbeats deshabilitados sin pip install schedule

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

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
HTTP_PORT = int(os.getenv("HTTP_PORT", "5001"))  # 5000 en macOS suele usarlo AirPlay
LOCK_TIMEOUT = int(os.getenv("LOCK_TIMEOUT", "60"))

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
# WORKSPACE - Carga de archivos .md
# =============================================================================

def load_workspace_file(relative_path: str) -> str:
    """Carga un archivo .md del workspace. Devuelve '' si no existe."""
    full_path = os.path.join(WORKSPACE_DIR, relative_path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def compose_soul() -> str:
    """Compone el system prompt desde los archivos del workspace (raiz)."""
    parts = [
        load_workspace_file("AGENTS.md"),
        load_workspace_file("IDENTITY.md"),
        load_workspace_file("SOUL.md"),
        load_workspace_file("TOOLS.md"),
        load_workspace_file("USER.md"),
    ]
    return "\n\n".join(p for p in parts if p)


def parse_heartbeats() -> list[dict]:
    """Lee HEARTBEAT.md del workspace (raiz) y parsea los heartbeats.

    Formato:
      ## nombre_del_heartbeat
      - hora: 07:30
      - prompt: Buenos dias! Dame la fecha de hoy.
    """
    content = load_workspace_file("HEARTBEAT.md")
    if not content:
        return []

    heartbeats = []
    current: dict | None = None

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("## "):
            if current:
                heartbeats.append(current)
            current = {"name": line[3:].strip()}
        elif current and line.startswith("- hora:"):
            current["time"] = line[len("- hora:"):].strip()
        elif current and line.startswith("- prompt:"):
            current["prompt"] = line[len("- prompt:"):].strip()

    if current:
        heartbeats.append(current)
    return heartbeats


# =============================================================================
# SESSION QUEUE (Locks)
# =============================================================================
# Sin locks, si un usuario manda dos mensajes rapido, ambos se procesan
# a la vez y compiten por el historial. Resultado: corrupcion.
# Con locks, el segundo espera a que termine el primero.
# Parece un detalle. Pero es el tipo de infraestructura que separa
# un prototipo de algo que funciona en el mundo real.

async_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
sync_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


# =============================================================================
# AGENT - Un workspace = un agente
# =============================================================================
# Un workspace = un agente. No hay routing complejo, no hay jerarquias
# de agentes, no hay orchestrators. Un directorio con archivos .md
# y un agente que los lee. Simple, predecible, debuggeable.
# IDENTITY.md en la raiz define quien es el agente (nombre en el contenido).

def discover_agents() -> dict[str, dict]:
    """Un solo agente por workspace: IDENTITY.md en la raiz."""
    agents = {}
    if not os.path.isdir(WORKSPACE_DIR):
        return agents
    identity_path = os.path.join(WORKSPACE_DIR, "IDENTITY.md")
    if not os.path.isfile(identity_path):
        return agents

    with open(identity_path, "r", encoding="utf-8") as f:
        identity_text = f.read()
    name = "Agent"
    for line in identity_text.split("\n"):
        if "**Nombre:**" in line:
            name = line.split("**Nombre:**")[1].strip()
            break

    agents["default"] = {
        "name": name,
        "soul": compose_soul(),
        "session_prefix": "agent:default",
    }
    logger.info("Agent descubierto: default (%s)", name)
    return agents


AGENTS = discover_agents()
# Un solo agente; la clave "default" y su session_prefix
AGENT = list(AGENTS.values())[0] if AGENTS else None
SESSION_PREFIX = AGENT["session_prefix"] if AGENT else "agent:default"


def route_message(text: str) -> str:
    """Quita el prefijo /investigar si existe; devuelve el texto a procesar."""
    if text.startswith("/investigar "):
        return text[len("/investigar "):].strip()
    return text


# =============================================================================
# LONG-TERM MEMORY
# =============================================================================

def save_memory_file(key: str, content: str) -> str:
    safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
    with open(os.path.join(MEMORY_DIR, f"{safe_key}.md"), "w", encoding="utf-8") as f:
        f.write(content)
    return f"Guardado en memory: {key}"

def search_memory(query: str) -> str:
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
    return "\n\n".join(f"--- {n} ---\n{c}" for _, n, c in results[:5])


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
# TOOLS
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
     "description": "Guardar info en long-term memory",
     "parameters": {"type": "object", "properties": {"key": {"type": "string"}, "content": {"type": "string"}}, "required": ["key", "content"]}}},
    {"type": "function", "function": {"name": "search_memory",
     "description": "Buscar en long-term memory",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]

def execute_tool(name: str, arguments: dict) -> str:
    try:
        if name == "run_command":
            cmd = arguments["command"]
            if evaluate_permission(cmd) == "requires_approval":
                return "Permiso denegado."
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
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
        return f"Error: {e}"


# =============================================================================
# COMPACTION + AGENT LOOP + SESSIONS
# =============================================================================

def estimate_tokens(msgs: list[dict]) -> int:
    text = json.dumps(msgs)
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except Exception:
        return len(text) // 4

def compact_session(session_id: str, msgs: list[dict]) -> list[dict]:
    """Compacta si supera el umbral; guarda resumen en memory (como paso 4)."""
    tokens = estimate_tokens(msgs)
    if tokens < COMPACTION_THRESHOLD:
        logger.info("Session %s: %d tokens (umbral %d) -> sin compactar", session_id, tokens, COMPACTION_THRESHOLD)
        return msgs
    logger.info("Session %s: %d tokens (umbral %d) -> compactando...", session_id, tokens, COMPACTION_THRESHOLD)
    half = len(msgs) // 2
    old, recent = msgs[:half], msgs[half:]
    summary_parts = [f"[{m.get('role', '?')}]: {(m.get('content') or '')[:500]}"
                     for m in old if isinstance(m.get("content"), str) and (m.get("content") or "").strip()]
    try:
        r = client.chat.completions.create(
            model=MODEL, max_tokens=2000,
            messages=[{"role": "user", "content":
                "Resume esta conversacion. Preserva datos del usuario, decisiones, tareas pendientes.\n\n"
                + "\n".join(summary_parts)}])
        summary = r.choices[0].message.content or ""
    except Exception as e:
        logger.error("Error al compactar: %s", e)
        return msgs
    compacted = [
        {"role": "user", "content": f"[Resumen de conversacion anterior]\n{summary}"},
        {"role": "assistant", "content": "Entendido, tengo el contexto."},
    ] + recent
    save_session(session_id, compacted)
    logger.info("Compactado: %d -> %d mensajes", len(msgs), len(compacted))
    try:
        key = f"compaction-{session_id}-{int(time.time())}"
        save_memory_file(key, summary)
        logger.info("Resumen guardado en memory: %s", key)
    except Exception as e:
        logger.warning("No se pudo guardar resumen en memory: %s", e)
    return compacted

def run_agent_turn(messages: list[dict], system_prompt: str) -> tuple[str, list[dict]]:
    api_messages = [{"role": "system", "content": system_prompt}] + messages
    for _ in range(MAX_ITERATIONS):
        resp = client.chat.completions.create(
            model=MODEL, max_tokens=4096, messages=api_messages, tools=TOOLS)
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
    return os.path.join(SESSIONS_DIR, f"{uid.replace('/', '_').replace(chr(92), '_')}.jsonl")

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

def reset_session(uid: str) -> None:
    path = session_path(uid)
    if os.path.exists(path):
        os.remove(path)


# =============================================================================
# PROCESADOR (un agente)
# =============================================================================

def process_message_sync(user_id: str, text: str) -> tuple[str, str]:
    """Procesa un mensaje (sync). Devuelve (nombre_agente, response)."""
    if not AGENT:
        return "Sistema", "No hay agente configurado (revisa workspace/IDENTITY.md)."
    clean_text = route_message(text)
    session_key = f"{SESSION_PREFIX}:{user_id}"
    with sync_locks[session_key]:
        messages = load_session(session_key)
        messages = compact_session(session_key, messages)
        messages.append({"role": "user", "content": clean_text})
        try:
            response, messages = run_agent_turn(messages, AGENT["soul"])
        except Exception as e:
            response = f"Error: {e}"
        save_session(session_key, messages)
    return AGENT["name"], response


# =============================================================================
# HANDLERS DE TELEGRAM
# =============================================================================

async def handle_new(update: Update, context) -> None:
    """Comando /nuevo: resetea la session del usuario."""
    user_id = str(update.effective_user.id)
    reset_session(f"{SESSION_PREFIX}:{user_id}")
    await update.message.reply_text("Session reseteada. Empezamos de cero!")


async def handle_message(update: Update, context) -> None:
    """Handler principal: un agente, misma session por usuario."""
    if not AGENT:
        await update.message.reply_text("No hay agente configurado.")
        return
    user_id = str(update.effective_user.id)
    text = route_message(update.message.text)
    session_key = f"{SESSION_PREFIX}:{user_id}"
    lock = async_locks[session_key]
    try:
        await asyncio.wait_for(lock.acquire(), timeout=LOCK_TIMEOUT)
    except asyncio.TimeoutError:
        await update.message.reply_text("Timeout: otro mensaje se esta procesando.")
        return
    try:
        messages = load_session(session_key)
        messages = compact_session(session_key, messages)
        messages.append({"role": "user", "content": text})
        response, messages = run_agent_turn(messages, AGENT["soul"])
        save_session(session_key, messages)
    except Exception as e:
        logger.error("Error: %s", e)
        response = f"Error: {e}"
    finally:
        lock.release()
    await update.message.reply_text(response)


# =============================================================================
# HTTP GATEWAY
# =============================================================================
# El mismo agente, accesible desde otro canal. Telegram + HTTP comparten
# sesiones. Esto demuestra que el agente es independiente del canal:
# la logica vive en el agent loop y las sessions, no en el handler.

def start_http_gateway() -> None:
    try:
        from flask import Flask, request, jsonify, Response
        from flask_cors import CORS
    except ImportError:
        logger.warning("Flask no instalado. HTTP gateway deshabilitado.")
        return

    http_app = Flask(__name__)
    CORS(http_app)

    @http_app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": MODEL, "agent": AGENT["name"] if AGENT else None})

    @http_app.route("/", methods=["GET"])
    def index():
        """Pagina web para chatear desde el navegador."""
        html = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chat - FriccionIA</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; }
    label { display: block; margin-top: 1rem; font-weight: 500; }
    input, textarea { width: 100%; padding: 0.5rem; margin-top: 0.25rem; }
    textarea { min-height: 80px; resize: vertical; }
    button { margin-top: 1rem; padding: 0.5rem 1.5rem; cursor: pointer; background: #333; color: #fff; border: none; border-radius: 4px; }
    button:hover { background: #555; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #respuesta { margin-top: 1rem; padding: 1rem; background: #f5f5f5; border-radius: 4px; white-space: pre-wrap; min-height: 2rem; }
    .error { color: #c00; }
  </style>
</head>
<body>
  <h1>Chat (FriccionIA)</h1>
  <form id="form">
    <label>Tu ID (ej: web o tu Telegram ID)</label>
    <input type="text" id="user_id" value="web" required>
    <label>Mensaje</label>
    <textarea id="message" placeholder="Escribe aqui..." required></textarea>
    <button type="submit" id="btn">Enviar</button>
  </form>
  <div id="respuesta"></div>
  <script>
    const form = document.getElementById("form");
    const btn = document.getElementById("btn");
    const respuesta = document.getElementById("respuesta");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const user_id = document.getElementById("user_id").value.trim();
      const message = document.getElementById("message").value.trim();
      if (!message) return;
      btn.disabled = true;
      respuesta.textContent = "Enviando...";
      try {
        const r = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ user_id, message })
        });
        const data = await r.json();
        if (!r.ok) {
          respuesta.innerHTML = '<span class="error">' + (data.error || r.status) + "</span>";
          return;
        }
        respuesta.textContent = data.response || "(sin respuesta)";
      } catch (err) {
        respuesta.innerHTML = '<span class="error">Error: ' + err.message + "</span>";
      }
      btn.disabled = false;
    });
  </script>
</body>
</html>"""
        return Response(html, mimetype="text/html; charset=utf-8")

    @http_app.route("/chat", methods=["POST"])
    def chat():
        data = request.json
        if not data or "user_id" not in data or "message" not in data:
            return jsonify({"error": "Campos requeridos: user_id, message"}), 400
        name, response = process_message_sync(data["user_id"], data["message"])
        return jsonify({"agent": name, "response": response})

    threading.Thread(
        target=lambda: http_app.run(host="0.0.0.0", port=HTTP_PORT, debug=False),
        daemon=True,
    ).start()
    logger.info("HTTP Gateway en http://0.0.0.0:%d", HTTP_PORT)


# =============================================================================
# HEARTBEATS - Leidos desde HEARTBEAT.md del workspace
# =============================================================================
# El agente hace cosas sin que se lo pidas. Tareas programadas leidas
# de un archivo .md: la hora, el prompt, y el agente ejecuta.
# Ejemplo: briefing matutino, verificar emails, generar reportes.
# 10 minutos de configuracion en un archivo de texto, no en codigo.

def setup_heartbeats() -> None:
    """Configura heartbeats leyendo HEARTBEAT.md del workspace (raiz)."""
    if not schedule:
        logger.warning("Heartbeats deshabilitados: pip install schedule")
        return
    if not AGENT:
        return
    heartbeats = parse_heartbeats()
    for hb in heartbeats:
        hb_name = hb.get("name", "unnamed")
        hb_prompt = hb.get("prompt", "")
        hb_time = hb.get("time", "")
        if not hb_prompt or not hb_time:
            continue
        session_key = f"cron:{hb_name}"
        soul = AGENT["soul"]

        def create_fn(n, p, sk, s):
            def run():
                logger.info("Heartbeat: %s", n)
                with sync_locks[sk]:
                    msgs = load_session(sk)
                    msgs.append({"role": "user", "content": p})
                    try:
                        resp, msgs_out = run_agent_turn(msgs, s)
                    except Exception as e:
                        resp = f"Error: {e}"
                        msgs_out = msgs
                    save_session(sk, msgs_out)
                logger.info("Heartbeat [%s]: %s", n, resp[:200])
            return run

        schedule.every().day.at(hb_time).do(create_fn(hb_name, hb_prompt, session_key, soul))
        logger.info("Heartbeat '%s' (%s) a las %s", hb_name, AGENT["name"], hb_time)

    def scheduler_loop():
        while True:
            schedule.run_pending()
            time.sleep(30)

    threading.Thread(target=scheduler_loop, daemon=True).start()
    logger.info("Scheduler de heartbeats iniciado")


# =============================================================================
# ARRANQUE
# =============================================================================

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN")

    start_http_gateway()
    setup_heartbeats()

    logger.info("=" * 60)
    logger.info("FriccionIA (un agente)")
    logger.info("  Workspace: %s", WORKSPACE_DIR)
    logger.info("  Agent:     %s", AGENT["name"] if AGENT else "ninguno")
    logger.info("  Model:     %s", MODEL)
    logger.info("  HTTP:      http://0.0.0.0:%d", HTTP_PORT)
    logger.info("  Permisos:  %s", PERMISSION_MODE)
    logger.info("  Compaction: umbral %d tokens", COMPACTION_THRESHOLD)
    logger.info("  Comandos:  /nuevo, /investigar <texto>")
    logger.info("=" * 60)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("nuevo", handle_new))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
