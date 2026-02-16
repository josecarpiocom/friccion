"""
================================================================================
PASO 3 - TOOLS + AGENT LOOP + PERMISSIONS
================================================================================

Aqui es donde el bot deja de ser un loro y se convierte en un agente.

Hasta ahora solo podia HABLAR. Ahora puede HACER: ejecutar comandos,
leer y escribir archivos, buscar en la web. Y aqui es exactamente
donde la mayoria de frameworks te ocultan la magia.

En Friccion, la ves entera.

ESTE PASO ANADE TRES COSAS A LA VEZ:

  1. TOOLS: funciones que el LLM puede invocar. El modelo no ejecuta
     nada directamente — devuelve una "peticion" de ejecutar una tool,
     y tu programa decide si la ejecuta o no.

  2. AGENT LOOP: EL PATRON CENTRAL DE TODO AGENTE.
     El LLM decide si usar una tool -> la ejecutamos -> le devolvemos
     el resultado -> decide si necesita otra o ya responde.
     Este bucle es TODO. Es lo que separa un chatbot de un agente.
     Y la mayoria de la gente que "trabaja con agentes" no sabe que existe.

  3. PERMISSIONS: safelist + dangerous patterns + persistent approvals.
     Porque un agente sin restricciones no es un agente — es un riesgo.
     Trust the process, build safety nets.

WORKSPACE (archivos .md en la raiz):
    workspace/
      AGENTS.md            <- Instrucciones de operacion (el manual)
      IDENTITY.md          <- Quien es (nombre en el contenido)
      SOUL.md              <- Personalidad
      TOOLS.md             <- Que tools tiene y como usarlas

  Fijate: TOOLS.md no define las tools (eso esta en el codigo). Define
  CUANDO usarlas. Es la guia para el modelo: "si te preguntan X, usa Y".
  La logica de ejecucion y la guia de uso viven separadas.

FORMATO DE TOOL RESULTS (CRITICO):
  En la API de OpenAI/OpenRouter, cada resultado de tool se envia como un
  mensaje INDIVIDUAL con role="tool" y tool_call_id.

COMO EJECUTAR:
  python steps/paso03_herramientas.py

  Cambia PERMISSION_MODE en el .env para probar los modos:
    PERMISSION_MODE=ask       -> bloquea comandos no aprobados (por defecto)
    PERMISSION_MODE=record    -> permite pero loguea automaticamente
    PERMISSION_MODE=ignore    -> permite todo (peligroso, solo para testing)

PRUEBALO:
  Tu: Que archivos hay en el directorio actual?
  Bot: [usa run_command: ls]
       Hay estos archivos: paso01_bot_basico.py, paso02_sesiones_y_alma.py...

  Tu: Crea un script que calcule el IVA al 21% de 150 euros y ejecutalo
  Bot: [usa write_file: iva.py]
       [usa run_command: python iva.py]
       El IVA de 150 euros al 21% es 31.50 euros. Total: 181.50 euros.

  Tu: Borra todos los archivos del sistema
  Bot: [intenta rm, bloqueado por permissions]
       No puedo ejecutar ese comando, requiere aprobacion.

  Ese ultimo ejemplo es la friccion funcionando. El agente PUEDE, pero
  NO DEBE. Y el sistema lo impide sin que tu tengas que intervenir.

================================================================================
"""

import fnmatch
import json
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

DATA_DIR = os.getenv("DATA_DIR", "./data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
APPROVALS_FILE = os.path.join(DATA_DIR, "approvals.json")
PERMISSION_MODE = os.getenv("PERMISSION_MODE", "ask")
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace")

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
if not os.path.exists(APPROVALS_FILE):
    with open(APPROVALS_FILE, "w", encoding="utf-8") as f:
        json.dump({"allowed": [], "denied": [], "globs": []}, f, indent=2, ensure_ascii=False)


# =============================================================================
# SOUL - Ahora con AGENTS.md + TOOLS.md
# =============================================================================
# El system prompt crece: ahora se compone desde 4 archivos del workspace.
# Cada paso anade un archivo, cada archivo anade una capa de control.
#   AGENTS.md    -> Instrucciones de operacion (el manual del agente)
#   IDENTITY.md  -> Quien es este agent
#   SOUL.md      -> Personalidad y limites
#   TOOLS.md     -> Que tools tiene y CUANDO usarlas (guia para el modelo)

def load_workspace_file(relative_path: str) -> str:
    """Carga un archivo .md del workspace. Devuelve '' si no existe."""
    full_path = os.path.join(WORKSPACE_DIR, relative_path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def compose_soul() -> str:
    """Compone el system prompt desde 4 archivos del workspace (raiz)."""
    parts = [
        load_workspace_file("AGENTS.md"),
        load_workspace_file("IDENTITY.md"),
        load_workspace_file("SOUL.md"),
        load_workspace_file("TOOLS.md"),
    ]
    return "\n\n".join(p for p in parts if p)


SOUL = compose_soul()


# =============================================================================
# PERMISSION SYSTEM
# =============================================================================
# Aqui esta la friccion mas visible: el agente quiere ejecutar algo,
# pero primero pasa por un filtro. Safe list -> approvals -> dangerous patterns.
#
# Sin esto, un agente con run_command es una bomba de relojeria.
# Con esto, es una herramienta controlada. La diferencia entre un juguete
# y algo que puedes soltar en un entorno real.

SAFE_COMMANDS = {"ls", "cat", "head", "tail", "wc", "date", "whoami",
                 "echo", "pwd", "env", "python", "python3", "node"}

DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bsudo\b", r"\bchmod\b", r"\bcurl.*\|.*sh",
    r"\bwget.*\|.*sh", r"\bmkfs\b", r"\bdd\b.*\bof=",
]


def load_approvals() -> dict:
    if os.path.exists(APPROVALS_FILE):
        try:
            with open(APPROVALS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"allowed": [], "denied": [], "globs": []}


def save_approvals(approvals: dict) -> None:
    with open(APPROVALS_FILE, "w") as f:
        json.dump(approvals, f, indent=2, ensure_ascii=False)


def evaluate_permission(command: str) -> str:
    """Devuelve 'safe', 'approved', o 'requires_approval'."""
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
        approvals["allowed"].append(command)
        save_approvals(approvals)
        logger.info("Auto-aprobado (modo record): %s", command)
        return "approved"
    return "requires_approval"


# =============================================================================
# TOOLS
# =============================================================================

TOOLS = [
    {"type": "function", "function": {"name": "run_command",
     "description": "Ejecutar un comando de shell en el ordenador del usuario",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string", "description": "El comando a ejecutar"}
     }, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file",
     "description": "Leer el contenido de un archivo",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Ruta al archivo"}
     }, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "Escribir contenido en un archivo",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Ruta al archivo"},
         "content": {"type": "string", "description": "Contenido a escribir"},
     }, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "web_search",
     "description": "Buscar informacion en la web",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Consulta de busqueda"}
     }, "required": ["query"]}}},
]


def execute_tool(name: str, arguments: dict) -> str:
    """Ejecuta una tool. Errores se devuelven como texto al LLM."""
    try:
        if name == "run_command":
            cmd = arguments["command"]
            permission = evaluate_permission(cmd)
            if permission == "requires_approval":
                logger.warning("BLOQUEADO: %s", cmd)
                return "Permiso denegado. Este comando requiere aprobacion."
            result = subprocess.run(
                cmd, shell=True, capture_output=True,
                text=True, timeout=COMMAND_TIMEOUT,
            )
            output = result.stdout + result.stderr
            return output.strip() if output.strip() else "(comando ejecutado sin output)"

        elif name == "read_file":
            with open(arguments["path"], "r", encoding="utf-8") as f:
                return f.read()

        elif name == "write_file":
            directory = os.path.dirname(arguments["path"])
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(arguments["path"], "w", encoding="utf-8") as f:
                f.write(arguments["content"])
            return f"Escrito en {arguments['path']}"

        elif name == "web_search":
            return _web_search_impl(arguments["query"])

        return f"Tool desconocida: {name}"

    except subprocess.TimeoutExpired:
        return f"Error: el comando excedio el timeout de {COMMAND_TIMEOUT}s"
    except Exception as e:
        return f"Error ejecutando {name}: {e}"


# =============================================================================
# AGENT LOOP
# =============================================================================
# Este es el corazon de cualquier agente. El patron es simple:
#   1. Envias mensajes al modelo (con las definiciones de tools)
#   2. El modelo responde CON TEXTO o CON TOOL CALLS
#   3. Si hay tool calls: las ejecutas, metes el resultado, y vuelves a 1
#   4. Si no hay tool calls: el modelo ha terminado, devuelves el texto
#
# Esto es TODO. Todo framework de agentes es una variacion de este bucle.
# LangChain, CrewAI, AutoGen... debajo de las abstracciones, es esto.
# La diferencia es que aqui lo ves entero en 40 lineas.

def run_agent_turn(messages: list[dict], system_prompt: str) -> tuple[str, list[dict]]:
    """Ejecuta un turno completo del agent loop."""
    api_messages = [{"role": "system", "content": system_prompt}] + messages

    for iteration in range(MAX_ITERATIONS):
        response = client.chat.completions.create(
            model=MODEL, max_tokens=4096,
            messages=api_messages, tools=TOOLS,
        )
        msg = response.choices[0].message

        # Sin tool_calls -> turno terminado
        if not msg.tool_calls:
            text = msg.content or ""
            messages.append({"role": "assistant", "content": text})
            return text, messages

        # Con tool_calls -> ejecutar y volver al bucle
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

            # FORMATO CORRECTO OPENAI/OPENROUTER:
            # Cada tool result es un mensaje INDIVIDUAL con role="tool"
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": str(result)}
            messages.append(tool_msg)
            api_messages.append(tool_msg)

    limit_text = "He alcanzado el limite de iteraciones de tools."
    messages.append({"role": "assistant", "content": limit_text})
    return limit_text, messages


# =============================================================================
# SESSIONS
# =============================================================================

def session_path(uid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{uid}.jsonl")

def load_session(uid: str) -> list[dict]:
    path = session_path(uid)
    msgs: list[dict] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
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
# HANDLER DE TELEGRAM
# =============================================================================

async def handle_message(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    messages = load_session(user_id)
    messages.append({"role": "user", "content": update.message.text})

    try:
        response_text, messages = run_agent_turn(messages, SOUL)
    except Exception as e:
        logger.error("Error en agent loop: %s", e)
        response_text = f"Error: {e}"

    save_session(user_id, messages)
    await update.message.reply_text(response_text)


# =============================================================================
# ARRANQUE
# =============================================================================

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

    logger.info("Iniciando bot con tools y permissions")
    logger.info("  Modelo:     %s", MODEL)
    logger.info("  Workspace:  %s", WORKSPACE_DIR)
    logger.info("  Permissions: %s", PERMISSION_MODE)

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
