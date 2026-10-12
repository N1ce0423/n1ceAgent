"""验证会话删除通过 MemoryTool 清理各类记忆，并保留其他会话。"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory"))

from config import Config
from memory.document_store import SQLiteDocumentStore
from memory.episodic_memory import EpisodicMemory
from memory.memory_config import MemoryConfig
from memory.memory_item import MemoryItem
from memory.memory_manager import MemoryManager
from memory.memory_tool import MemoryTool
from memory.semantic_memory import SemanticMemory
from memory.working_memory import WorkingMemory
from session_store import SessionStore
from tool_registry import ToolRegistry
from webui import WebApplication


class SessionDeletionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = SessionStore(str(Path(directory.name) / "sessions.db"))
        self.target = self.store.create_session()
        self.other = self.store.create_session()
        self.working = WorkingMemory(MemoryConfig())
        self.episodic = EpisodicMemory.__new__(EpisodicMemory)
        self.episodic.config = SimpleNamespace(qdrant_url=None)
        self.episodic.doc_store = SQLiteDocumentStore(str(Path(directory.name) / "episodes.db"), 100, 30)
        self.episodic.sessions = {}
        self.episodic.vector_store = MagicMock()
        self.semantic = SemanticMemory.__new__(SemanticMemory)
        self.semantic_items = {}
        self.semantic.graph_store = MagicMock()
        self.semantic.graph_store.get_all.side_effect = lambda: list(self.semantic_items.values())
        self.semantic.remove = MagicMock(side_effect=lambda memory_id: self.semantic_items.pop(memory_id, None) is not None)
        self.manager = MemoryManager.__new__(MemoryManager)
        self.manager.memory_types = {
            "working": self.working, "episodic": self.episodic, "semantic": self.semantic,
        }
        self.tool = MemoryTool()
        self.tool.__dict__["memory_manager"] = self.manager
        self.app = WebApplication(Config(), self.store, memory_tool_factory=lambda: self.tool)
        for session_id in (self.target, self.other):
            for memory_type in self.manager.memory_types:
                item = MemoryItem(f"{memory_type}_{session_id}", "private memory", memory_type, 0.5,
                                  metadata={"session_id": session_id})
                if memory_type == "working":
                    self.working.add(item)
                elif memory_type == "episodic":
                    self.episodic.doc_store.add(item)
                else:
                    self.semantic_items[item.id] = item
        self.episodic._refresh_sessions()

    def test_real_tool_clears_all_types_and_does_not_remove_other_session(self):
        result = self.app.delete_session(self.target)
        self.assertIn("Successfully forgot 3 memories", result)
        self.assertEqual([item.metadata["session_id"] for item in self.working.get_all()], [self.other])
        self.assertEqual([item.metadata["session_id"] for item in self.episodic.doc_store.get_all()], [self.other])
        self.assertEqual([item.metadata["session_id"] for item in self.semantic.get_all()], [self.other])
        self.semantic.remove.assert_called_once_with(f"semantic_{self.target}")
        self.episodic.vector_store.remove.assert_not_called()
        self.assertNotIn(self.target, self.episodic.sessions)
        self.assertEqual([session.session_id for session in self.store.list_sessions()], [self.other])

    def test_episodic_deletes_sqlite_and_vector_when_vector_store_enabled(self):
        self.episodic.config.qdrant_url = ":memory:"
        count = self.episodic.remove_memories_by_session(self.target)
        self.assertEqual(count, 1)
        self.episodic.vector_store.remove.assert_called_once_with(f"episodic_{self.target}")
        self.assertEqual([item.metadata["session_id"] for item in self.episodic.doc_store.get_all()], [self.other])

    def test_vector_failure_keeps_sqlite_and_session_id_for_retry(self):
        self.episodic.config.qdrant_url = ":memory:"
        self.episodic.vector_store.remove.side_effect = RuntimeError("Vector service unavailable")
        with self.assertRaisesRegex(RuntimeError, "Vector service unavailable"):
            self.app.delete_session(self.target)
        self.store.require_session(self.target)
        self.assertTrue(self.episodic.doc_store.contains(f"episodic_{self.target}"))
        self.episodic.vector_store.remove.side_effect = None
        self.assertIn("Successfully forgot", self.app.delete_session(self.target))

    def test_registered_tool_and_agent_working_cache_are_both_cleared(self):
        agent_working = WorkingMemory(MemoryConfig())
        for session_id in (self.target, self.other):
            agent_working.add(MemoryItem(session_id, "cached", "working", 0.5,
                                         metadata={"session_id": session_id}))
        registry = ToolRegistry()
        registry.register_function(self.tool.name, self.tool.description, self.tool.run)
        self.tool.set_session(self.other)
        self.app.agent = SimpleNamespace(
            tool_registry=registry, memory_manager=SimpleNamespace(memory_types={"working": agent_working}),
        )
        with patch.object(self.app, "memory_tool_factory", side_effect=AssertionError("Must reuse the registered tool")):
            self.app.delete_session(self.target)
        self.assertEqual(self.tool.current_session_id, self.other)
        self.assertEqual([item.metadata["session_id"] for item in agent_working.get_all()], [self.other])

    def test_unknown_or_empty_session_does_not_delete_memories(self):
        for session_id in ("", "missing", None):
            with self.assertRaises(ValueError):
                self.manager.forget_memories_by_session(session_id) if not session_id else self.app.delete_session(session_id)
        self.assertEqual(len(self.working.get_all()), 2)
        self.assertEqual(len(self.episodic.doc_store.get_all()), 2)
        self.semantic.remove.assert_not_called()


if __name__ == "__main__":
    unittest.main()
