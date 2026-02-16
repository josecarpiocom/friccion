"""
================================================================================
PASO 2 - SESSIONS PERSISTENTES + SOUL (IDENTITY.md + SOUL.md)
================================================================================

El bot del paso 1 tenia amnesia. Ahora le damos las dos cosas que lo
convierten en algo que se parece a "alguien":

  1. SESSIONS: El historial de conversacion se guarda en archivos JSONL
     (una linea JSON por mensaje). El bot recuerda lo que dijiste antes.

  2. SOUL: La personalidad no es un string hardcodeado en el codigo.
     Se compone desde archivos .md en la raiz del workspace:

       workspace/
         IDENTITY.md   <- Quien es: nombre, rol, proposito
         SOUL.md       <- Personalidad: estilo, limites, caracter

     En este paso usamos solo estos dos archivos.

AQUI EMPIEZA LA IDEA CENTRAL DE FRICCION: MARKDOWN > CODIGO.
  La industria ha convergido en esto. CLAUDE.md define preferencias y
  reglas. Los Agent Skills son Markdown + YAML, no codigo. Los archivos
  .md son el sistema operativo del agente.

  Cambias la personalidad editando un archivo de texto. Sin tocar Python.
  Esto es lo que separa un prototipo de algo mantenible: la configuracion
  vive fuera del codigo, es versionable, y la puede editar alguien que
  no programa.

PRIMERA FRICCION:
  El SOUL.md define limites y caracter. No es un asistente generico:
  tiene nombre (Pepe), tiene estilo (directo, sin rodeos), y tiene
  cosas que NO hara. Es la primera capa de control deliberado.

FORMATO JSONL (por que no JSON normal):
  - Cada linea es un mensaje independiente: {"role": "user", "content": "..."}
  - Es append-only: si el proceso crashea, pierdes como mucho un mensaje
  - Se puede leer linea por linea sin cargar todo en memoria

COMO EJECUTAR:
  python steps/paso02_sesiones_y_alma.py

PRUEBALO:
  Tu: Me llamo Jose y soy de Sevilla
  Bot: Encantado, Jose! Sevilla, que tierra mas bonita.

  [... horas despues, o incluso reinicias el bot ...]

  Tu: Como me llamo?
  Bot: Te llamas Jose, de Sevilla!   <-- ahora SI recuerda

  Tu: Que opinas de la tortilla de patatas sin cebolla?
  Bot: [responde con personalidad, no como un manual]  <-- tiene SOUL

  La diferencia con el paso 1 es brutal. Y solo hemos anadido dos cosas:
  un archivo por usuario y dos archivos .md. Esa es la infraestructura
  minima que convierte un chatbot en algo util.

================================================================================
"""

import json
import os
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

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")


# =============================================================================
# SOUL - Compuesto desde archivos del workspace
# =============================================================================
# El "cerebro" del agente no es un prompt fijo en el codigo. Se COMPONE
# leyendo archivos .md del workspace. Cada archivo tiene una responsabilidad:
#   - IDENTITY.md: el "quien soy" (nombre, rol, proposito)
#   - SOUL.md: la personalidad (estilo, limites, caracter)
#
# En este paso solo usamos estos dos. En pasos posteriores anadimos
# AGENTS.md (instrucciones de operacion) y TOOLS.md (skills).
#
# Esta separacion no es estetica. Es la misma filosofia que hace que
# CLAUDE.md funcione: las reglas del agente viven donde son editables,
# no donde estan enterradas en logica de negocio.

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", "./workspace")

# Valores por defecto: se crean si no existen los archivos
DEFAULT_IDENTITY = """\
# Identidad

**Nombre:** Pepe
**Rol:** Asistente personal de IA
**Proposito:** Ayudar al usuario con lo que necesite en su dia a dia.
"""

DEFAULT_SOUL = """\
# Personalidad

- Se genuinamente util, no teatralmente util
- Nada de "Excelente pregunta!" - simplemente ayuda
- Ten opiniones propias. Puedes discrepar educadamente
- Se conciso cuando toca, profundo cuando importa
- Si no sabes algo, dilo. Mejor un "no lo se" que inventarse la respuesta
- Un poco de humor no viene mal, pero sin forzar

# Estilo
- Directo, sin rodeos innecesarios
- Responde en el idioma en que te hablen

# Limites
- Las cosas privadas son privadas
- Ante la duda, pregunta antes de actuar en el mundo exterior
- No eres la voz del usuario: cuidado con actuar en su nombre
"""


def load_workspace_file(relative_path: str, default: str = "") -> str:
    """Carga un archivo .md del workspace. Si no existe, crea el default."""
    full_path = os.path.join(WORKSPACE_DIR, relative_path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            logger.info("Cargado: %s", relative_path)
            return content

    # No existe o esta vacio -> crear con default
    if default:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(default)
        logger.info("Creado por defecto: %s", relative_path)
        return default.strip()

    return ""


def compose_soul() -> str:
    """Compone el system prompt desde los archivos del workspace (raiz).

    En este paso solo usa IDENTITY.md + SOUL.md (en la raiz del workspace).
    Los pasos siguientes anaden AGENTS.md, TOOLS.md, USER.md.
    """
    parts = [
        load_workspace_file("IDENTITY.md", DEFAULT_IDENTITY),
        load_workspace_file("SOUL.md", DEFAULT_SOUL),
    ]
    return "\n\n".join(p for p in parts if p)


# Componer el SOUL al arrancar
SOUL = compose_soul()


# =============================================================================
# SESSIONS - Memoria conversacional persistente
# =============================================================================
# Sin sesiones, el agente tiene amnesia. Con sesiones, recuerda.
# Parece obvio, pero es la primera pieza de infraestructura que separa
# un chatbot de un asistente. Y la mayoria de tutoriales la ignoran.

SESSIONS_DIR = os.getenv("SESSIONS_DIR", "./data/sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_path(user_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{user_id}.jsonl")


def load_session(user_id: str) -> list[dict]:
    """Carga el historial. Lineas corruptas se ignoran."""
    path = session_path(user_id)
    messages: list[dict] = []
    if not os.path.exists(path):
        return messages
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Linea %d corrupta en session %s, ignorada", line_num, user_id)
    return messages


def append_to_session(user_id: str, message: dict) -> None:
    """Anade un mensaje (append-only). Lo mas seguro ante crashes."""
    with open(session_path(user_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")


def save_session(user_id: str, messages: list[dict]) -> None:
    """Sobreescribe la session completa. Para compactacion (paso 4)."""
    with open(session_path(user_id), "w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")


# =============================================================================
# HANDLER DE MENSAJES
# =============================================================================

async def handle_message(update: Update, context) -> None:
    user_id = str(update.effective_user.id)
    user_text = update.message.text

    # 1. Cargar historial existente
    messages = load_session(user_id)

    # 2. Agregar el mensaje del usuario
    user_msg = {"role": "user", "content": user_text}
    messages.append(user_msg)
    append_to_session(user_id, user_msg)

    # 3. Construir mensajes para la API.
    #    El SOUL (compuesto desde IDENTITY.md + SOUL.md) va como system.
    api_messages = [{"role": "system", "content": SOUL}] + messages

    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=4096,
            messages=api_messages,
        )
        text = response.choices[0].message.content
    except Exception as e:
        logger.error("Error LLM: %s", e)
        text = f"Error: {e}"

    # 4. Guardar respuesta del asistente
    assistant_msg = {"role": "assistant", "content": text}
    append_to_session(user_id, assistant_msg)

    await update.message.reply_text(text)


# =============================================================================
# ARRANQUE
# =============================================================================

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en el archivo .env")

    logger.info("Iniciando bot con sessions y SOUL (modelo: %s)...", MODEL)
    logger.info("Workspace: %s", WORKSPACE_DIR)
    logger.info("Sessions:  %s", SESSIONS_DIR)

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
