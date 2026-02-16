"""
================================================================================
friccion_lib.py - Libreria completa de AI Agents con friccion
================================================================================

~800 lineas. Todo el motor de un agente de IA en un solo archivo importable.
Sin frameworks encima, sin abstracciones innecesarias, sin magia.

La idea: el modelo es commodity. Lo que importa es la infraestructura
alrededor — memoria, permisos, sesiones, compaction, herramientas.
Esta libreria es esa infraestructura, visible y modificable.

El system prompt se compone desde archivos .md en un workspace:

  workspace/
    AGENTS.md         Instrucciones de operacion (el manual del agente)
    IDENTITY.md       Nombre, rol, proposito (quien es)
    SOUL.md           Personalidad, limites (como habla)
    TOOLS.md          Tools y cuando usarlas (guia para el modelo)
    USER.md           Info sobre el usuario (lo que sabe de ti)
    HEARTBEAT.md      Tareas programadas (el agente actua sin pedirlo)

  Markdown > codigo. Cambias el comportamiento editando texto, no Python.

Clases principales:

  - LLMClient:          Wrapper para OpenRouter (API OpenAI-compatible)
  - SessionManager:     Sessions persistentes en archivos JSONL
  - PermissionManager:  Control de permissions para comandos shell
  - ToolRegistry:       Registro y ejecucion de tools
  - MemoryStore:        Long-term memory basada en archivos
  - SessionQueue:       Locks por session (async + sync)
  - Agent:              Agent loop con soporte de tools y compaction
  - MultiAgentRouter:   Routing de mensajes entre agents
  - TaskScheduler:      Tareas periodicas (heartbeats)

Uso rapido:
    from friccion_lib import (
        LLMClient, SessionManager, Agent, ToolRegistry,
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
    response = agent.run_turn("user123", "Hola!")

================================================================================
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import math
import os
import re
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Callable, Iterator

logger = logging.getLogger(__name__)

__version__ = "0.4.0"


# =============================================================================
# WORKSPACE - Carga y composicion de archivos .md
# =============================================================================

def load_workspace_file(workspace_dir: str, relative_path: str) -> str:
    """Carga un archivo .md del workspace. Devuelve '' si no existe."""
    full_path = os.path.join(workspace_dir, relative_path)
    if os.path.isfile(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def compose_prompt(workspace_dir: str, agent_id: str | None = None) -> str:
    """Compone el system prompt desde los archivos .md del workspace (todos en la raiz).

    Orden: AGENTS.md, IDENTITY.md, SOUL.md, TOOLS.md, USER.md.
    agent_id se ignora; la identidad del agente se lee de IDENTITY.md.
    """
    parts = [
        load_workspace_file(workspace_dir, "AGENTS.md"),
        load_workspace_file(workspace_dir, "IDENTITY.md"),
        load_workspace_file(workspace_dir, "SOUL.md"),
        load_workspace_file(workspace_dir, "TOOLS.md"),
        load_workspace_file(workspace_dir, "USER.md"),
    ]
    return "\n\n".join(p for p in parts if p)


def parse_heartbeats(workspace_dir: str, agent_id: str | None = None) -> list[dict]:
    """Lee HEARTBEAT.md del workspace (raiz) y parsea los heartbeats configurados.

    Formato esperado:
      ## nombre_del_heartbeat
      - hora: 07:30
      - prompt: Buenos dias!
    """
    content = load_workspace_file(workspace_dir, "HEARTBEAT.md")
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


def discover_agents(workspace_dir: str) -> dict[str, dict]:
    """Un solo agente por workspace. Busca IDENTITY.md en la raiz.

    Si existe, devuelve un dict con un agente (id "default").
    Devuelve dict[agent_id, {"name", "soul", "session_prefix"}].
    """
    agents = {}
    if not os.path.isdir(workspace_dir):
        return agents
    identity_path = os.path.join(workspace_dir, "IDENTITY.md")
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
        "soul": compose_prompt(workspace_dir),
        "session_prefix": "agent:default",
    }
    logger.info("Agent descubierto: default (%s)", name)
    return agents


# =============================================================================
# LLM CLIENT
# =============================================================================

class LLMClient:
    """Wrapper para OpenRouter / cualquier API compatible con OpenAI."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.5")
        self.max_tokens = max_tokens

        key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("open_router_api_key")
        url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        if not key:
            raise ValueError("API key requerida. Define OPENROUTER_API_KEY en el entorno.")

        self._client = OpenAI(base_url=url, api_key=key)
        logger.info("LLMClient: model=%s, url=%s", self.model, url)

    def chat(self, messages: list[dict], system: str | None = None,
             tools: list[dict] | None = None,
             max_tokens: int | None = None, **kwargs) -> Any:
        api_msgs = list(messages)
        if system:
            api_msgs.insert(0, {"role": "system", "content": system})
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": api_msgs,
            **kwargs,
        }
        if tools:
            params["tools"] = tools
        return self._client.chat.completions.create(**params)

    def chat_text(self, messages: list[dict], system: str | None = None,
                  max_tokens: int | None = None) -> str:
        resp = self.chat(messages=messages, system=system, max_tokens=max_tokens)
        return resp.choices[0].message.content or ""


# =============================================================================
# SESSION MANAGER
# =============================================================================

class SessionManager:
    """Sessions persistentes en archivos JSONL."""

    def __init__(self, directory: str = "./data/sessions") -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_id: str) -> str:
        safe = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return os.path.join(self.directory, f"{safe}.jsonl")

    def load(self, session_id: str) -> list[dict]:
        path = self._path(session_id)
        messages: list[dict] = []
        if not os.path.exists(path):
            return messages
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Linea %d corrupta en session '%s'", i, session_id)
        return messages

    def save(self, session_id: str, messages: list[dict]) -> None:
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def append(self, session_id: str, message: dict) -> None:
        with open(self._path(session_id), "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    def reset(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)

    def list_sessions(self) -> list[str]:
        return sorted(f[:-6] for f in os.listdir(self.directory) if f.endswith(".jsonl"))


# =============================================================================
# PERMISSION MANAGER
# =============================================================================

DEFAULT_SAFE_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "date", "whoami", "echo",
    "pwd", "env", "python", "python3", "node", "printenv", "uname",
})

DEFAULT_DANGEROUS_PATTERNS = [
    r"\brm\b", r"\bsudo\b", r"\bchmod\b", r"\bcurl.*\|.*sh",
    r"\bwget.*\|.*sh", r"\bmkfs\b", r"\bdd\b.*\bof=",
]


class PermissionManager:
    """Control de permissions. Modos: ask, record, ignore."""

    def __init__(
        self,
        safe_commands: frozenset[str] | None = None,
        dangerous_patterns: list[str] | None = None,
        approvals_file: str = "./data/approvals.json",
        mode: str = "ask",
    ) -> None:
        self.safe_commands = safe_commands or DEFAULT_SAFE_COMMANDS
        self.dangerous_patterns = dangerous_patterns or DEFAULT_DANGEROUS_PATTERNS
        self.approvals_file = approvals_file
        self.mode = mode
        os.makedirs(os.path.dirname(approvals_file) or ".", exist_ok=True)
        if not os.path.exists(self.approvals_file):
            self._save({"allowed": [], "denied": [], "globs": []})

    def _load(self) -> dict:
        if os.path.exists(self.approvals_file):
            try:
                with open(self.approvals_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"allowed": [], "denied": [], "globs": []}

    def _save(self, data: dict) -> None:
        with open(self.approvals_file, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def check(self, command: str) -> str:
        base = command.strip().split()[0] if command.strip() else ""
        if base in self.safe_commands:
            return "safe"
        data = self._load()
        if command in data.get("allowed", []):
            return "approved"
        for pattern in data.get("globs", []):
            if fnmatch.fnmatch(command, pattern):
                return "approved"
        return "requires_approval"

    def evaluate(self, command: str) -> str:
        if self.mode == "ignore":
            return "safe"
        result = self.check(command)
        if result == "requires_approval" and self.mode == "record":
            self.approve(command)
            return "approved"
        return result

    def approve(self, command: str) -> None:
        data = self._load()
        if command not in data["allowed"]:
            data["allowed"].append(command)
            self._save(data)

    def approve_glob(self, pattern: str) -> None:
        data = self._load()
        if pattern not in data.get("globs", []):
            data.setdefault("globs", []).append(pattern)
            self._save(data)


# =============================================================================
# TOOL REGISTRY
# =============================================================================

ToolHandler = Callable[[dict[str, Any]], str]


class ToolRegistry:
    """Registro central de tools para el agent."""

    def __init__(self, command_timeout: int = 30) -> None:
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self.command_timeout = command_timeout

    def register(self, name: str, description: str,
                 parameters: dict, handler: ToolHandler) -> None:
        self._tools[name] = {
            "type": "function",
            "function": {"name": name, "description": description, "parameters": parameters},
        }
        self._handlers[name] = handler

    def get_definitions(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict) -> str:
        handler = self._handlers.get(name)
        if not handler:
            return f"Tool desconocida: {name}"
        try:
            return handler(arguments) or "(ejecutado sin output)"
        except subprocess.TimeoutExpired:
            return f"Error: timeout de {self.command_timeout}s"
        except Exception as e:
            logger.error("Error en %s: %s", name, e)
            return f"Error ejecutando {name}: {e}"


def do_web_search(query: str, max_results: int = 5) -> str:
    """Busca en la web con DuckDuckGo. Devuelve titulo, URL y fragmento por resultado."""
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
        logger.exception("web_search")
        return f"Error en la busqueda: {e}"


def create_base_tools(
    command_timeout: int = 30,
    permission_checker: Callable[[str], str] | None = None,
) -> ToolRegistry:
    """Crea un registro con las 4 tools basicas."""
    registry = ToolRegistry(command_timeout=command_timeout)

    def run_command(args: dict) -> str:
        cmd = args["command"]
        if permission_checker and permission_checker(cmd) == "requires_approval":
            return "Permiso denegado. Este comando requiere aprobacion."
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=command_timeout)
        return (r.stdout + r.stderr).strip() or "(ok)"

    def read_file(args: dict) -> str:
        with open(args["path"], "r", encoding="utf-8") as f:
            return f.read()

    def write_file(args: dict) -> str:
        d = os.path.dirname(args["path"])
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args["path"], "w", encoding="utf-8") as f:
            f.write(args["content"])
        return f"Escrito en {args['path']}"

    def web_search(args: dict) -> str:
        return do_web_search(args["query"])

    registry.register("run_command", "Ejecutar un comando de shell",
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}, run_command)
    registry.register("read_file", "Leer un archivo",
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, read_file)
    registry.register("write_file", "Escribir un archivo",
        {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}, write_file)
    registry.register("web_search", "Buscar en la web",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}, web_search)

    return registry


# =============================================================================
# MEMORY STORE
# =============================================================================

class MemoryStore:
    """Long-term memory basada en archivos .md con busqueda por relevancia."""

    def __init__(self, directory: str = "./data/memory") -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save(self, key: str, content: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        with open(os.path.join(self.directory, f"{safe_key}.md"), "w", encoding="utf-8") as f:
            f.write(content)
        return f"Guardado en memory: {key}"

    def search(self, query: str) -> str:
        words = [w.lower() for w in query.split() if len(w) > 2]
        if not words or not os.path.exists(self.directory):
            return "No se encontraron memorias relevantes."
        results: list[tuple[float, str, str]] = []
        for filename in os.listdir(self.directory):
            if not filename.endswith(".md"):
                continue
            with open(os.path.join(self.directory, filename), "r", encoding="utf-8") as f:
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

    def list_keys(self) -> list[str]:
        if not os.path.exists(self.directory):
            return []
        return sorted(f[:-3] for f in os.listdir(self.directory) if f.endswith(".md"))

    def register_tools(self, registry: ToolRegistry) -> None:
        store = self

        def save_handler(args: dict) -> str:
            return store.save(args["key"], args["content"])

        def search_handler(args: dict) -> str:
            return store.search(args["query"])

        registry.register("save_memory",
            "Guardar info importante en long-term memory",
            {"type": "object", "properties": {
                "key": {"type": "string", "description": "Etiqueta descriptiva"},
                "content": {"type": "string", "description": "Informacion a recordar"}
            }, "required": ["key", "content"]}, save_handler)

        registry.register("search_memory",
            "Buscar en la long-term memory",
            {"type": "object", "properties": {
                "query": {"type": "string", "description": "Que buscar"}
            }, "required": ["query"]}, search_handler)


# =============================================================================
# CONTEXT COMPACTION
# =============================================================================

def estimate_tokens(messages: list[dict]) -> int:
    text = json.dumps(messages)
    try:
        import tiktoken
        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    except (ImportError, Exception):
        return len(text) // 4


def compact_messages(messages: list[dict], client: LLMClient,
                     threshold: int = 100_000) -> list[dict]:
    if estimate_tokens(messages) < threshold:
        return messages
    half = len(messages) // 2
    old, recent = messages[:half], messages[half:]
    parts = [f"[{m.get('role','?')}]: {m.get('content','')[:500]}"
             for m in old if isinstance(m.get("content"), str) and m["content"].strip()]
    try:
        summary = client.chat_text(
            messages=[{"role": "user", "content":
                "Resume esta conversacion. Preserva datos del usuario, decisiones y tareas.\n\n"
                + "\n".join(parts)}],
            max_tokens=2000)
    except Exception:
        return messages
    return [
        {"role": "user", "content": f"[Resumen anterior]\n{summary}"},
        {"role": "assistant", "content": "Entendido."},
    ] + recent


# =============================================================================
# SESSION QUEUE
# =============================================================================

class SessionQueue:
    """Locks por session: async para Telegram, sync para Flask/HTTP."""

    def __init__(self, timeout: float = 60.0) -> None:
        self.timeout = timeout
        self._async: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._sync: dict[str, threading.Lock] = defaultdict(threading.Lock)

    @asynccontextmanager
    async def lock_async(self, session_id: str) -> AsyncIterator[None]:
        lock = self._async[session_id]
        await asyncio.wait_for(lock.acquire(), timeout=self.timeout)
        try:
            yield
        finally:
            lock.release()

    @contextmanager
    def lock_sync(self, session_id: str) -> Iterator[None]:
        lock = self._sync[session_id]
        if not lock.acquire(timeout=self.timeout):
            raise TimeoutError(f"Timeout en lock para '{session_id}'")
        try:
            yield
        finally:
            lock.release()


# =============================================================================
# AGENT (Agent Loop)
# =============================================================================

class Agent:
    """Agent loop con soporte de tools, sessions y compaction."""

    def __init__(
        self,
        client: LLMClient,
        tools: ToolRegistry | None = None,
        soul: str = "",
        sessions: SessionManager | None = None,
        compaction_threshold: int = 100_000,
        max_iterations: int = 10,
    ) -> None:
        self.client = client
        self.tools = tools
        self.soul = soul
        self.sessions = sessions
        self.compaction_threshold = compaction_threshold
        self.max_iterations = max_iterations

    def run_turn(self, session_id: str, user_message: str) -> str:
        if not self.sessions:
            msgs: list[dict] = [{"role": "user", "content": user_message}]
            text, _ = self._loop(msgs)
            return text

        messages = self.sessions.load(session_id)
        if estimate_tokens(messages) >= self.compaction_threshold:
            messages = compact_messages(messages, self.client, self.compaction_threshold)
            self.sessions.save(session_id, messages)

        messages.append({"role": "user", "content": user_message})
        try:
            text, messages = self._loop(messages)
        except Exception as e:
            logger.error("Error en agent: %s", e)
            text = f"Error: {e}"
        self.sessions.save(session_id, messages)
        return text

    def _loop(self, messages: list[dict]) -> tuple[str, list[dict]]:
        defs = self.tools.get_definitions() if self.tools else None

        for _ in range(self.max_iterations):
            resp = self.client.chat(messages=messages, system=self.soul, tools=defs)
            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = msg.content or ""
                messages.append({"role": "assistant", "content": text})
                return text, messages

            assistant_msg: dict[str, Any] = {
                "role": "assistant", "content": msg.content or "",
                "tool_calls": [{"id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls],
            }
            messages.append(assistant_msg)

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                logger.info("  Tool: %s(%s)", tc.function.name, json.dumps(args))
                result = self.tools.execute(tc.function.name, args) if self.tools else "Sin tools"
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

        return "Limite de iteraciones alcanzado.", messages


# =============================================================================
# MULTI-AGENT ROUTER
# =============================================================================

class MultiAgentRouter:
    """Enruta mensajes al agent correcto segun prefijo de comando."""

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}
        self._prefixes: list[tuple[str, str]] = []
        self._default: str | None = None

    def register(self, agent_id: str, agent: Agent, name: str = "",
                 prefix: str | None = None, is_default: bool = False,
                 session_prefix: str | None = None) -> None:
        self._agents[agent_id] = {
            "agent": agent, "name": name or agent_id,
            "session_prefix": session_prefix or f"agent:{agent_id}",
        }
        if prefix:
            self._prefixes.append((prefix, agent_id))
            self._prefixes.sort(key=lambda x: len(x[0]), reverse=True)
        if is_default or self._default is None:
            self._default = agent_id

    def route(self, text: str) -> tuple[str, str]:
        for prefix, agent_id in self._prefixes:
            if text.startswith(prefix + " "):
                return agent_id, text[len(prefix) + 1:]
            if text == prefix:
                return agent_id, ""
        return self._default or "pepe", text

    def execute(self, user_id: str, text: str) -> tuple[str, str]:
        agent_id, clean_text = self.route(text)
        info = self._agents.get(agent_id)
        if not info:
            return "Sistema", f"Agent '{agent_id}' no encontrado."
        key = f"{info['session_prefix']}:{user_id}"
        response = info["agent"].run_turn(key, clean_text)
        return info["name"], response

    def get_info(self) -> dict[str, str]:
        return {k: v["name"] for k, v in self._agents.items()}


# =============================================================================
# TASK SCHEDULER (Heartbeats)
# =============================================================================

class TaskScheduler:
    """Tareas periodicas leidas desde HEARTBEAT.md."""

    def __init__(self, execution_fn: Callable[[str, str], str],
                 poll_interval: int = 30) -> None:
        self.fn = execution_fn
        self.interval = poll_interval
        self._running = False

    def add(self, name: str, prompt: str,
            daily_time: str | None = None,
            every_minutes: int | None = None) -> None:
        import schedule as sched
        sk = f"cron:{name}"

        def run():
            logger.info("Heartbeat: %s", name)
            try:
                resp = self.fn(sk, prompt)
                logger.info("Heartbeat [%s]: %s", name, resp[:200])
            except Exception as e:
                logger.error("Error en heartbeat %s: %s", name, e)

        if daily_time:
            sched.every().day.at(daily_time).do(run)
        elif every_minutes:
            sched.every(every_minutes).minutes.do(run)

    def add_from_workspace(self, workspace_dir: str, agent_id: str | None = None) -> None:
        """Carga heartbeats desde HEARTBEAT.md del workspace (raiz). agent_id se ignora."""
        for hb in parse_heartbeats(workspace_dir):
            hb_time = hb.get("time")
            hb_prompt = hb.get("prompt")
            hb_name = hb.get("name", "unnamed")
            if hb_time and hb_prompt:
                self.add(hb_name, hb_prompt, daily_time=hb_time)
                logger.info("Heartbeat '%s' a las %s", hb_name, hb_time)

    def start(self) -> None:
        import schedule as sched
        if self._running:
            return
        self._running = True

        def loop():
            while self._running:
                sched.run_pending()
                time.sleep(self.interval)

        threading.Thread(target=loop, daemon=True).start()
        logger.info("TaskScheduler iniciado")

    def stop(self) -> None:
        self._running = False


# =============================================================================
# HTTP GATEWAY
# =============================================================================

def create_http_app(process_fn: Callable, model: str = "",
                    agents: dict[str, str] | None = None) -> Any:
    """Crea una Flask app con /health y /chat."""
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        r: dict[str, Any] = {"status": "ok", "model": model}
        if agents:
            r["agents"] = agents
        return jsonify(r)

    @app.route("/chat", methods=["POST"])
    def chat():
        data = request.json
        if not data or "user_id" not in data or "message" not in data:
            return jsonify({"error": "Campos requeridos: user_id, message"}), 400
        result = process_fn(data["user_id"], data["message"])
        if isinstance(result, tuple):
            return jsonify({"agent": result[0], "response": result[1]})
        return jsonify({"response": result})

    return app


def start_http_gateway(app: Any, host: str = "0.0.0.0", port: int = 5000) -> None:
    threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False),
        daemon=True,
    ).start()
    logger.info("HTTP Gateway en http://%s:%d", host, port)
