"""本地聊天 WebUI：复用现有 agent，并通过 NDJSON 推送执行步骤。"""

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv

from config import Config
from my_main import build_agent
from session_store import SessionStore


ROOT = Path(__file__).resolve().parent
MAX_BODY_BYTES = 64 * 1024
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def build_memory_tool():
    """删除旧会话只需要记忆组件，不依赖模型服务的配置。"""
    memory_directory = str(ROOT / "memory")
    if memory_directory not in sys.path:
        sys.path.insert(0, memory_directory)
    from memory.memory_tool import MemoryTool
    return MemoryTool()


class WebApplication:
    def __init__(
            self, config: Config, store: SessionStore,
            agent_factory: Callable = build_agent,
            memory_tool_factory: Callable = build_memory_tool,
    ):
        self.config = config
        self.store = store
        self.agent_factory = agent_factory
        self.memory_tool_factory = memory_tool_factory
        self.memory_tool = None
        self.agent = None
        # 共用 agent 的会话绑定和记忆工具，不允许并发切换它的 session。
        self.run_lock = Lock()
        self.state_lock = Lock()
        self.active_run: Optional[tuple[str, Event]] = None

    def start_run(self, session_id: str, message: str):
        if not self.run_lock.acquire(blocking=False):
            raise BlockingIOError("The agent is busy. Wait for the current run to finish.")
        try:
            self.store.require_session(session_id)
        except Exception:
            self.run_lock.release()
            raise
        run_id = uuid4().hex
        cancelled = Event()
        events: Queue = Queue()
        with self.state_lock:
            self.active_run = (run_id, cancelled)

        def worker() -> None:
            try:
                events.put({"type": "started", "run_id": run_id})
                if self.agent is None:
                    events.put({"type": "status", "content": "Initializing agent..."})
                    self.agent = self.agent_factory(self.config, self.store, session_id)
                else:
                    self.agent.select_session(session_id)
                self.agent.run(message, event_callback=events.put, cancel_event=cancelled)
            except Exception as exc:
                if cancelled.is_set():
                    events.put({"type": "cancelled"})
                else:
                    events.put({"type": "error", "content": str(exc)})
            finally:
                with self.state_lock:
                    self.active_run = None
                self.run_lock.release()
                events.put({"type": "done"})

        Thread(target=worker, daemon=True).start()
        return events, cancelled

    def _get_memory_tool(self):
        if self.agent is not None:
            registry = getattr(self.agent, "tool_registry", None)
            if registry is not None:
                function = registry.get_tool("memory_tool")
                tool = getattr(function, "__self__", None)
                if tool is not None and callable(getattr(tool, "_forget_with_session", None)):
                    return tool
        if self.memory_tool is None:
            self.memory_tool = self.memory_tool_factory()
        return self.memory_tool

    def delete_session(self, session_id: str) -> str:
        """先清空记忆，再删除 ID；失败时保留 ID，允许重试清理。"""
        if not self.run_lock.acquire(blocking=False):
            raise BlockingIOError("The agent is busy. Wait for the current operation to finish.")
        try:
            self.store.require_session(session_id)
            tool = self._get_memory_tool()
            result = tool._forget_with_session(session_id=session_id)
            if not isinstance(result, str) or not result.startswith("Successfully forgot "):
                raise RuntimeError(result or "Memory cleanup did not complete.")
            # ReAct 还有自己的 MemoryManager，需要同时清掉其内存中的工作记忆。
            manager = getattr(self.agent, "memory_manager", None)
            if manager is not None and manager is not tool.memory_manager:
                working = manager.memory_types.get("working")
                if working is not None:
                    working.remove_memories_by_session(session_id)
            self.store.delete_session(session_id)
            return result
        finally:
            self.run_lock.release()

    def cancel(self, run_id: str) -> bool:
        with self.state_lock:
            if self.active_run is None or self.active_run[0] != run_id:
                return False
            self.active_run[1].set()
            return True


class WebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app: WebApplication):
        self.app = app
        super().__init__(address, WebHandler)


class WebHandler(BaseHTTPRequestHandler):
    def _allowed_request(self) -> bool:
        expected_host = f"127.0.0.1:{self.server.server_port}"
        alternate_host = f"localhost:{self.server.server_port}"
        if self.headers.get("Host") not in {expected_host, alternate_host}:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {
            f"http://{expected_host}", f"http://{alternate_host}",
        }

    def _headers(self, status: int, content_type: str, size: Optional[int] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; object-src 'none'")
        if size is not None:
            self.send_header("Content-Length", str(size))
        self.end_headers()

    def _json(self, data: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_BODY_BYTES:
            raise ValueError("Request body must be between 1 and 65536 bytes.")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def do_GET(self) -> None:
        if not self._allowed_request():
            self._json({"error": "Request origin is not allowed."}, HTTPStatus.FORBIDDEN)
            return
        path = urlsplit(self.path).path
        if path == "/api/sessions":
            self._json({
                "sessions": [session.session_id for session in self.server.app.store.list_sessions()],
            })
        elif path in STATIC_FILES:
            filename, content_type = STATIC_FILES[path]
            body = (ROOT / "web" / filename).read_bytes()
            self._headers(HTTPStatus.OK, content_type, len(body))
            self.wfile.write(body)
        else:
            self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._allowed_request():
            self._json({"error": "Request origin is not allowed."}, HTTPStatus.FORBIDDEN)
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/sessions":
                self._json({"session_id": self.server.app.store.create_session()}, HTTPStatus.CREATED)
            elif path == "/api/sessions/delete":
                data = self._body()
                session_id = data.get("session_id")
                if not isinstance(session_id, str) or not session_id.strip():
                    raise ValueError("A non-empty session_id is required.")
                result = self.server.app.delete_session(session_id)
                self._json({"deleted": True, "session_id": session_id, "message": result})
            elif path == "/api/cancel":
                data = self._body()
                self._json({"cancelled": self.server.app.cancel(data.get("run_id"))})
            elif path == "/api/chat":
                data = self._body()
                session_id = data.get("session_id")
                message = data.get("message")
                if not isinstance(session_id, str) or not isinstance(message, str) or not message.strip():
                    raise ValueError("A session_id and a non-empty message are required.")
                if len(message) > 12000:
                    raise ValueError("Message must not exceed 12000 characters.")
                events, cancelled = self.server.app.start_run(session_id, message.strip())
                try:
                    self._headers(HTTPStatus.OK, "application/x-ndjson; charset=utf-8")
                    while True:
                        try:
                            event = events.get(timeout=5)
                        except Empty:
                            event = {"type": "heartbeat"}
                        body = json.dumps(event, ensure_ascii=False).encode("utf-8") + b"\n"
                        self.wfile.write(body)
                        self.wfile.flush()
                        if event["type"] == "done":
                            break
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    # 页面关闭时停止后续步骤；不能中断已发出的模型或工具请求。
                    cancelled.set()
            else:
                self._json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
        except (ValueError, UnicodeDecodeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except BlockingIOError as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    load_dotenv(ROOT / ".env")
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Run the local agent WebUI.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--session-db", default=config.session_database_path)
    args = parser.parse_args()
    app = WebApplication(config, SessionStore(args.session_db))
    with WebServer(("127.0.0.1", args.port), app) as server:
        print(f"Agent WebUI: http://127.0.0.1:{server.server_port}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
