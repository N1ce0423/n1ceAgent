"""验证真实 ReAct 事件顺序和 WebUI 的 HTTP 流、会话与取消行为。"""

import http.client
import json
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))

from config import Config
from my_react_agent import AgentRunCancelled, MyReactAgent
from session_store import SessionStore
from tool_registry import ToolRegistry
from webui import WebApplication, WebServer


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.kwargs = []

    def think(self, messages, **kwargs):
        self.kwargs.append(kwargs)
        return next(self.responses)


class ReactEventTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore(str(Path(directory.name) / "sessions.db"))
        self.memory_patch = patch("my_react_agent.MemoryManager", return_value=SimpleNamespace(
            get_all_working_memory=lambda: [],
        ))
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)

    def agent(self, *responses, registry=None):
        return MyReactAgent("Test", FakeLLM(*responses), tool_registry=registry, session_store=self.store)

    def test_tool_events_arrive_before_next_step_and_final_answer(self):
        registry = ToolRegistry()
        registry.register_function("lookup", "test", lambda _: "工具结果 <script>")
        agent = self.agent("Thought: 查询资料\nAction: lookup[问题]",
                           "Thought: 根据结果回答\nAction: Finish[最终答案]", registry=registry)
        events = []
        answer = agent.run("问题", event_callback=events.append, temperature=0.2)
        self.assertEqual(answer, "最终答案")
        self.assertEqual([event["type"] for event in events],
                         ["step", "thought", "action", "observation", "step", "thought", "final"])
        self.assertEqual(events[2]["tool"], "lookup")
        self.assertEqual(events[3]["content"], "工具结果 <script>")
        self.assertEqual(agent.llm_client.kwargs, [{"temperature": 0.2}, {"temperature": 0.2}])

    def test_cancel_after_model_response_prevents_tool_execution(self):
        cancelled = Event()
        registry = ToolRegistry()
        calls = []
        registry.register_function("write", "test", calls.append)
        agent = self.agent("Action: write[data]", registry=registry)
        def think(*args, **kwargs):
            cancelled.set()
            return "Action: write[data]"
        agent.llm_client.think = think
        with self.assertRaises(AgentRunCancelled):
            agent.run("question", cancel_event=cancelled)
        self.assertEqual(calls, [])

    def test_cancel_after_tool_prevents_another_model_request(self):
        cancelled = Event()
        registry = ToolRegistry()
        registry.register_function("stop", "test", lambda _: cancelled.set() or "done")
        agent = self.agent("Action: stop[input]", registry=registry)
        events = []
        with self.assertRaises(AgentRunCancelled):
            agent.run("question", event_callback=events.append, cancel_event=cancelled)
        self.assertEqual([event["type"] for event in events], ["step", "action", "observation"])
        self.assertEqual(len(agent.llm_client.kwargs), 1)

    def test_empty_model_response_is_an_explicit_error(self):
        with self.assertRaisesRegex(RuntimeError, "LLM returned no response"):
            self.agent(None).run("question")

    def test_exhausted_step_budget_is_reported_as_incomplete(self):
        agent = self.agent("Thought: malformed response")
        agent.max_steps = 1
        events = []
        agent.run("question", event_callback=events.append)
        self.assertEqual(events[-1]["status"], "exhausted")


class FakeAgent:
    def __init__(self, config, store, session_id):
        self.session_id = session_id
        self.calls = []

    def select_session(self, session_id):
        self.session_id = session_id

    def run(self, message, event_callback, cancel_event):
        self.calls.append((self.session_id, message))
        event_callback({"type": "step", "step": 1, "max_steps": 2})
        event_callback({"type": "thought", "step": 1, "content": "公开的动作摘要"})
        event_callback({"type": "action", "step": 1, "tool": "lookup", "input": message})
        if message == "wait":
            if not cancel_event.wait(5):
                raise RuntimeError("Cancellation did not arrive.")
            raise AgentRunCancelled("Agent run cancelled.")
        if message == "error":
            raise RuntimeError("Test service unavailable")
        event_callback({"type": "observation", "step": 1, "content": "observed result"})
        event_callback({"type": "final", "step": 2, "content": "测试回答", "status": "completed"})
        return "测试回答"


class HTTPTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "sessions.db"
        self.store = SessionStore(str(self.path))
        self.first_id = self.store.create_session()
        self.memory_tool = MagicMock()
        self.memory_tool._forget_with_session.return_value = "Successfully forgot 3 memories."
        self.app = WebApplication(Config(), self.store, FakeAgent, lambda: self.memory_tool)
        self.server = WebServer(("127.0.0.1", 0), self.app)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        connection.request(method, path, body=json.dumps(data) if data is not None else None,
                           headers=headers or {})
        return connection.getresponse()

    def chat(self, message, session_id=None):
        response = self.request("POST", "/api/chat", {
            "session_id": session_id or self.first_id, "message": message,
        })
        self.assertEqual(response.status, 200)
        return [json.loads(line) for line in response.read().splitlines()]

    def test_page_and_assets_are_served_but_project_files_are_private(self):
        for path in ("/", "/app.js", "/style.css"):
            with self.subTest(path=path):
                response = self.request("GET", path)
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 100)
        for path in ("/.env", "/webui.py", "/../.env", "/memory.db"):
            self.assertEqual(self.request("GET", path).status, 404)

    def test_sessions_are_listed_and_created_without_initializing_agent(self):
        response = self.request("POST", "/api/sessions", {})
        self.assertEqual(response.status, 201)
        second_id = json.loads(response.read())["session_id"]
        result = json.loads(self.request("GET", "/api/sessions").read())
        self.assertEqual(set(result["sessions"]), {self.first_id, second_id})
        self.assertIsNone(self.app.agent)

    def test_stream_contains_steps_and_final_and_does_not_persist_chat(self):
        events = self.chat("unique private chat marker")
        self.assertEqual([event["type"] for event in events],
                         ["started", "status", "step", "thought", "action", "observation", "final", "done"])
        self.assertEqual(events[-2]["content"], "测试回答")
        self.assertNotIn(b"unique private chat marker", self.path.read_bytes())

    def test_switch_rebinds_same_agent_to_selected_session(self):
        self.chat("first message")
        agent = self.app.agent
        second_id = self.store.create_session()
        self.chat("second message", second_id)
        self.assertIs(self.app.agent, agent)
        self.assertEqual(agent.calls, [(self.first_id, "first message"), (second_id, "second message")])

    def test_invalid_session_or_body_does_not_start_agent(self):
        for data in ({"session_id": "unknown", "message": "question"},
                     {"session_id": self.first_id, "message": " "}, [],
                     {"session_id": self.first_id, "message": "x" * 12001}):
            self.assertEqual(self.request("POST", "/api/chat", data).status, 400)
        self.assertIsNone(self.app.agent)

    def test_host_and_cross_origin_requests_are_rejected(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.test"}).status, 403)
        self.assertEqual(self.request("POST", "/api/sessions", {},
                                      {"Origin": "https://evil.test"}).status, 403)

    def test_cancel_is_scoped_to_run_and_concurrent_run_is_rejected(self):
        response = self.request("POST", "/api/chat", {"session_id": self.first_id, "message": "wait"})
        started = json.loads(response.readline())
        self.assertEqual(started["type"], "started")
        wrong = self.request("POST", "/api/cancel", {"run_id": "wrong"})
        self.assertFalse(json.loads(wrong.read())["cancelled"])
        busy = self.request("POST", "/api/chat", {"session_id": self.first_id, "message": "other"})
        self.assertEqual(busy.status, 409)
        cancelled = self.request("POST", "/api/cancel", {"run_id": started["run_id"]})
        self.assertTrue(json.loads(cancelled.read())["cancelled"])
        events = [json.loads(line) for line in response.read().splitlines()]
        self.assertEqual([event["type"] for event in events][-2:], ["cancelled", "done"])
        self.assertEqual(self.chat("next")[-2]["content"], "测试回答")

    def test_error_is_streamed_and_lock_is_released(self):
        events = self.chat("error")
        self.assertEqual(events[-2], {"type": "error", "content": "Test service unavailable"})
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(self.chat("retry")[-2]["type"], "final")

    def test_delete_clears_memories_before_id_and_preserves_other_session(self):
        second_id = self.store.create_session()
        def forget(*, session_id):
            self.store.require_session(session_id)
            return "Successfully forgot 3 memories."
        self.memory_tool._forget_with_session.side_effect = forget
        response = self.request("POST", "/api/sessions/delete", {"session_id": self.first_id})
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(response.read())["deleted"])
        self.assertEqual(self.store.list_sessions()[0].session_id, second_id)
        with self.assertRaises(ValueError):
            self.store.require_session(self.first_id)
        self.memory_tool._forget_with_session.assert_called_once_with(session_id=self.first_id)
        self.assertIsNone(self.app.agent)

    def test_delete_cleanup_failure_keeps_id_and_allows_retry(self):
        self.memory_tool._forget_with_session.return_value = "Error forgetting memories: vector unavailable"
        response = self.request("POST", "/api/sessions/delete", {"session_id": self.first_id})
        self.assertEqual(response.status, 500)
        self.assertIn("vector unavailable", json.loads(response.read())["error"])
        self.store.require_session(self.first_id)
        self.memory_tool._forget_with_session.return_value = "Successfully forgot 0 memories."
        self.assertEqual(self.request("POST", "/api/sessions/delete", {"session_id": self.first_id}).status, 200)
        self.assertEqual(self.store.list_sessions(), [])

    def test_invalid_delete_does_not_touch_memory(self):
        for data in ({}, {"session_id": ""}, {"session_id": "unknown"}, {"session_id": None}):
            self.assertEqual(self.request("POST", "/api/sessions/delete", data).status, 400)
        self.memory_tool._forget_with_session.assert_not_called()
        self.assertEqual(self.chat("valid request after invalid delete")[-2]["type"], "final")

    def test_delete_during_chat_is_rejected_without_clearing_memories(self):
        response = self.request("POST", "/api/chat", {"session_id": self.first_id, "message": "wait"})
        run_id = json.loads(response.readline())["run_id"]
        result = self.request("POST", "/api/sessions/delete", {"session_id": self.first_id})
        self.assertEqual(result.status, 409)
        self.memory_tool._forget_with_session.assert_not_called()
        self.store.require_session(self.first_id)
        self.request("POST", "/api/cancel", {"run_id": run_id}).read()
        response.read()

    def test_delete_blocks_chat_until_cleanup_completes(self):
        entered, release = Event(), Event()
        def forget(*, session_id):
            entered.set()
            release.wait(5)
            return "Successfully forgot 0 memories."
        self.memory_tool._forget_with_session.side_effect = forget
        thread = Thread(target=self.app.delete_session, args=(self.first_id,))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            result = self.request("POST", "/api/chat", {"session_id": self.first_id, "message": "question"})
            self.assertEqual(result.status, 409)
            self.assertIsNone(self.app.agent)
        finally:
            release.set()
            thread.join(5)


if __name__ == "__main__":
    unittest.main()
