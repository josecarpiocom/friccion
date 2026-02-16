"""
================================================================================
FriccionIA - Ejemplo completo usando friccion_lib.py + workspace
================================================================================

Este script demuestra como usar la libreria friccion_lib con la
arquitectura de workspace de FriccionIA para montar un asistente AI
completo en pocas lineas. Incluye:

  - Workspace plano: un workspace = un agente, archivos .md en la raiz
  - AGENTS, IDENTITY, SOUL, TOOLS, USER, HEARTBEAT
  - Telegram + API HTTP compartiendo sessions
  - Permissions, memory, compaction, heartbeats, session locks

WORKSPACE (archivos en la raiz, MAYÚSCULAS):
  workspace/
    AGENTS.md          Instrucciones de operacion
    IDENTITY.md        Nombre, rol (quien es el agente)
    SOUL.md            Personalidad
    TOOLS.md           Tools disponibles
    USER.md            Info sobre el usuario
    HEARTBEAT.md       Tareas programadas

COMO EJECUTAR:
  1. cp .env.example .env  (y rellena tus claves)
  2. pip install -r requirements.txt
  3. python mini_friccionIA.py

COMANDOS:
  /nuevo              Resetea la session del agente
  /investigar <q>     Manda la consulta al agente (mismo agente, prefijo opcional)
  (cualquier otro)    Mensaje normal al agente

================================================================================
"""

import asyncio
import logging
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from friccion_lib import (
    LLMClient, SessionManager, ToolRegistry, create_base_tools,
    PermissionManager, MemoryStore, SessionQueue, TaskScheduler,
    Agent, MultiAgentRouter,
    compose_prompt, discover_agents,
    create_http_app, start_http_gateway,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("friccion_ia")

# --- Configuracion ---
DATA_DIR = os.getenv("DATA_DIR", "./data")
HTTP_PORT = int(os.getenv("HTTP_PORT", "5000"))
PERMISSION_MODE = os.getenv("PERMISSION_MODE", "ask")
COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "100000"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace")

# --- Componentes ---
llm = LLMClient()
sessions = SessionManager(os.path.join(DATA_DIR, "sessions"))
permissions = PermissionManager(approvals_file=os.path.join(DATA_DIR, "approvals.json"),
                                mode=PERMISSION_MODE)
tools = create_base_tools(permission_checker=permissions.evaluate)
memory = MemoryStore(os.path.join(DATA_DIR, "memory"))
memory.register_tools(tools)
queue = SessionQueue(timeout=60)

# --- Agents (descubiertos desde el workspace) ---
agent_configs = discover_agents(WORKSPACE_DIR)

# Crear Agent instances desde las configs del workspace
agents: dict[str, Agent] = {}
for agent_id, config in agent_configs.items():
    agents[agent_id] = Agent(
        client=llm, tools=tools, soul=config["soul"],
        sessions=sessions, compaction_threshold=COMPACTION_THRESHOLD,
        max_iterations=MAX_ITERATIONS,
    )

# Router (un agente por workspace)
router = MultiAgentRouter()
for agent_id, config in agent_configs.items():
    router.register(agent_id, agents[agent_id], name=config["name"],
                    is_default=True, session_prefix=config["session_prefix"],
                    prefix="/investigar")
    break  # solo hay uno


# --- Procesamiento ---
def process_sync(uid: str, text: str) -> tuple[str, str]:
    with queue.lock_sync(uid):
        return router.execute(uid, text)


# --- Telegram ---
async def cmd_new(update: Update, context) -> None:
    uid = str(update.effective_user.id)
    session_prefix = next((c["session_prefix"] for c in agent_configs.values()), "agent:default")
    sessions.reset(f"{session_prefix}:{uid}")
    await update.message.reply_text("Session reseteada. Empezamos de cero!")

async def handle_message(update: Update, context) -> None:
    uid = str(update.effective_user.id)
    try:
        async with queue.lock_async(uid):
            name, response = router.execute(uid, update.message.text)
    except asyncio.TimeoutError:
        await update.message.reply_text("Timeout: otro mensaje se esta procesando.")
        return
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        return
    await update.message.reply_text(f"[{name}] {response}")


# --- Arranque ---
def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Falta TELEGRAM_BOT_TOKEN"); sys.exit(1)

    # HTTP gateway
    try:
        http_app = create_http_app(process_sync, llm.model, router.get_info())
        start_http_gateway(http_app, port=HTTP_PORT)
    except ImportError:
        logger.warning("Flask no instalado. HTTP gateway deshabilitado.")

    # Heartbeats (leidos desde workspace)
    scheduler = TaskScheduler(lambda sk, p: process_sync(sk, p)[1])
    scheduler.add_from_workspace(WORKSPACE_DIR)
    scheduler.start()

    logger.info("=" * 50)
    logger.info("FriccionIA")
    logger.info("  Workspace: %s", WORKSPACE_DIR)
    logger.info("  Agents:    %s", ", ".join(router.get_info().values()))
    logger.info("  Model:     %s", llm.model)
    logger.info("  HTTP:      http://0.0.0.0:%d", HTTP_PORT)
    logger.info("  Cmds:      /nuevo, /investigar <consulta>")
    logger.info("=" * 50)

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("nuevo", cmd_new))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()

if __name__ == "__main__":
    main()
