"""
================================================================================
PASO 1 - EL BOT MAS SIMPLE POSIBLE
================================================================================

Este es el punto de partida. Y a proposito, es decepcionante.

Un bot de Telegram que recibe un mensaje, se lo envia a un LLM a traves
de OpenRouter, y devuelve la respuesta. Sin mas. Es el equivalente a
abrir ChatGPT y escribir algo, pero en Telegram.

Esto es lo que el 90% de la gente llama "tener un agente de IA".
No lo es. Es un chatbot stateless. Y este paso existe para que lo veas
con tus propios ojos antes de construir algo mejor.

QUE CUBRE ESTE PASO:
  - Conectar con OpenRouter (compatible con la API de OpenAI)
  - Recibir mensajes de Telegram y responder
  - Configurar el modelo via variable de entorno

QUE NO TIENE TODAVIA (y por que importa):
  - Sin memoria: cada mensaje es independiente, como hablar con alguien
    que tiene amnesia. "Como me llamo?" -> no tiene ni idea.
  - Sin personalidad: responde como un asistente generico.
  - Sin tools: solo puede hablar, no puede HACER nada.
  - Sin friccion: no hay ningun control sobre lo que dice o hace.

  Esto es lo que pasa cuando no hay infraestructura alrededor del modelo.
  El modelo es commodity. Lo que importa es lo que construyes alrededor.

POR QUE OPENROUTER:
  OpenRouter es un gateway que te da acceso a cientos de modelos (Claude,
  GPT, Llama, Mistral, etc.) con una sola API key y formato OpenAI.
  Cambiar de modelo es cambiar una variable de entorno, nada mas.
  Porque en esta libreria el modelo es lo de menos.

COMO EJECUTAR:
  1. Copia .env.example a .env y rellena OPENROUTER_API_KEY y TELEGRAM_BOT_TOKEN
  2. pip install -r requirements.txt
  3. python steps/paso01_bot_basico.py

PRUEBALO:
  Tu: Oye, que tiempo va a hacer mañana en Sevilla?
  Bot: No puedo consultar el tiempo en tiempo real, pero...

  Tu: Como me llamo?
  Bot: No tengo esa informacion...  (no recuerda nada; cada mensaje es independiente)

  Tu: Recomiendame un sitio para comer en Madrid
  Bot: Te recomiendo...  <-- pero no sabe tus gustos ni te recuerda

  Frustrante, verdad? Bien. Esa frustracion es la friccion del paso 1.
  Ahora ya sabes lo que falta. Vamos al paso 2.

================================================================================
"""

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

# =============================================================================
# CONFIGURACION
# =============================================================================
# OpenRouter es compatible con la API de OpenAI. Solo hay que cambiar
# la base_url y usar tu api_key de OpenRouter.
# Aqui eliges modelo. Pero fijate: el modelo no cambia que este bot
# sea inutil sin memoria ni personalidad. Da igual GPT-4, Claude o Llama.
# Sin infraestructura, todos son el mismo chatbot amnésico.

client = OpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")


# =============================================================================
# HANDLER DE MENSAJES
# =============================================================================
# Fijate en lo que pasa aqui: UN mensaje entra, UN mensaje sale.
# No hay historial. No hay system prompt. No hay nada.
# Esto es lo minimo que se puede hacer con un LLM. Y es lo maximo
# que muchos llegan a construir.

async def handle_message(update: Update, context) -> None:
    """Recibe un mensaje del usuario, lo envia al LLM, y devuelve la respuesta."""
    user_text = update.message.text
    logger.info("Mensaje recibido: %s", user_text[:80])

    try:
        # Solo enviamos UN mensaje, sin historial.
        # Por eso el bot no recuerda nada entre mensajes.
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_text}],
        )
        text = response.choices[0].message.content

    except Exception as e:
        logger.error("Error al llamar al LLM: %s", e)
        text = f"Error al procesar tu mensaje: {e}"

    await update.message.reply_text(text)


# =============================================================================
# ARRANQUE
# =============================================================================

def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Falta TELEGRAM_BOT_TOKEN. Crea un bot con @BotFather en Telegram "
            "y pon el token en tu archivo .env"
        )

    logger.info("Iniciando bot basico (modelo: %s)...", MODEL)
    logger.info("Este bot NO tiene memoria. Cada mensaje es independiente.")

    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
