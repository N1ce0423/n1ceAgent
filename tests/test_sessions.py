"""验证 session 仅保存 ID，模型上下文不包含旧问答，记忆按当前会话隔离。"""

import copy
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config import Config
from message import Message
from my_main import AGENT_NAME, __main__, chat_loop, choose_session
from my_react_agent import MyReactAgent
from session_store import SessionInfo, SessionStore
from tool_registry import ToolRegistry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'memory'))
from document_store import SQLiteDocumentStore
from episodic_memory import EpisodicMemory
from memory.memory_item import MemoryItem
from memory.memory_tool import MemoryTool
from memory.graph_store import Neo4jGraphStore
from memory_manager import MemoryManager
from semantic_memory import SemanticMemory


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def think(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return next(self.responses)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database_path = str(Path(self.directory.name) / 'conversations.db')
        self.store = SessionStore(self.database_path)
        self.config = Config(session_database_path=self.database_path)
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def agent(self, *responses, session_id=None, registry=None):
        return MyReactAgent(
            name=AGENT_NAME, llm_client=FakeLLM(*responses), tool_registry=registry,
            config=self.config, session_store=self.store, session_id=session_id,
        )

    def test_session_schema_contains_only_id_and_no_messages_table(self):
        with closing(sqlite3.connect(self.database_path)) as connection:
            columns = connection.execute('PRAGMA table_info(chat_sessions)').fetchall()
            tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        self.assertEqual([row[1] for row in columns], ['session_id'])
        self.assertEqual(tables, [('chat_sessions',)])
        self.assertEqual([field.name for field in fields(SessionInfo)], ['session_id'])

    def test_legacy_migration_preserves_ids_and_removes_body_and_summary(self):
        legacy_path = str(Path(self.directory.name) / 'legacy.db')
        marker = 'OLD_HISTORY_SECRET_MARKER_462809'
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.executescript('''
                CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, agent_name TEXT,
                    title TEXT, created_at TEXT, updated_at TEXT);
                CREATE TABLE chat_messages (message_id INTEGER PRIMARY KEY,
                    session_id TEXT REFERENCES chat_sessions(session_id), message_json TEXT);
            ''')
            connection.execute('INSERT INTO chat_sessions VALUES (?, ?, ?, ?, ?)',
                               ('saved_id', AGENT_NAME, marker, 'old', 'old'))
            connection.execute('INSERT INTO chat_messages VALUES (1, ?, ?)', ('saved_id', marker))
        migrated = SessionStore(legacy_path)
        self.assertEqual([session.session_id for session in migrated.list_sessions()], ['saved_id'])
        migrated.require_session('saved_id')
        with closing(sqlite3.connect(legacy_path)) as connection:
            self.assertEqual(connection.execute('PRAGMA table_info(chat_sessions)').fetchall()[0][1], 'session_id')
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name='chat_messages'").fetchone())
        self.assertNotIn(marker.encode(), Path(legacy_path).read_bytes())
        self.assertEqual(SessionStore(legacy_path).list_sessions(), migrated.list_sessions())

    def test_same_session_next_turn_receives_no_old_chat(self):
        agent = self.agent('Action: Finish[old answer]', 'Action: Finish[new answer]')
        agent.run('old preference marker')
        agent.run('new question')
        second_call = agent.llm_client.calls[1]
        self.assertEqual([message['role'] for message in second_call], ['system', 'user'])
        self.assertNotIn('old preference marker', str(second_call))
        self.assertNotIn('old answer', str(second_call))
        self.assertEqual(agent.get_history(), [])
        self.assertNotIn('history', vars(agent))
        self.assertNotIn('current_history', vars(agent))

    def test_restart_keeps_id_and_does_not_restore_chat(self):
        agent = self.agent('Action: Finish[old answer]')
        agent.run('old preference marker')
        reopened = MyReactAgent(name=AGENT_NAME, llm_client=FakeLLM('Action: Finish[answer]'),
                                config=self.config, session_store=SessionStore(self.database_path),
                                session_id=agent.session_id)
        reopened.run('Who am I?')
        self.assertEqual(reopened.session_id, agent.session_id)
        self.assertEqual(len(reopened.list_sessions()), 1)
        self.assertNotIn('old preference marker', str(reopened.llm_client.calls[0]))
        self.assertEqual(reopened.get_history(), [])

    def test_new_and_selected_sessions_only_change_id_and_tool_binding(self):
        registry = ToolRegistry()
        tool = MemoryTool(memory_types=['working'])
        registry.register_function(tool.name, tool.description, tool.run)
        agent = self.agent(registry=registry)
        first_id = agent.session_id
        second_id = agent.new_session()
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(tool.current_session_id, second_id)
        agent.select_session(first_id)
        self.assertEqual(tool.current_session_id, first_id)
        self.assertEqual(len(agent.list_sessions()), 2)

    def test_unknown_session_preserves_current_id_and_binding(self):
        registry = ToolRegistry()
        agent = self.agent(registry=registry)
        original_id = agent.session_id
        with self.assertRaisesRegex(ValueError, 'Session not found'):
            agent.select_session('missing')
        self.assertEqual(agent.session_id, original_id)
        self.assertEqual(registry.current_session_id, original_id)

    def test_legacy_history_helpers_do_not_save_messages(self):
        agent = self.agent()
        agent.add_message(Message('secret marker', 'user'))
        agent.record_turn('secret marker', 'secret response')
        agent.clear_history()
        self.assertEqual(agent.get_history(), [])
        self.assertNotIn(b'secret marker', Path(self.database_path).read_bytes())
        self.assertNotIn(b'secret response', Path(self.database_path).read_bytes())

    def test_react_keeps_only_current_turn_observations(self):
        registry = ToolRegistry()
        registry.register_function('lookup', 'test tool', lambda _: 'current observation')
        agent = self.agent('Action: lookup[input]', 'Action: Finish[finished]',
                           'Action: Finish[next answer]', registry=registry)
        agent.run('first question')
        self.assertIn('current observation', agent.llm_client.calls[1][-1]['content'])
        agent.run('second question')
        self.assertNotIn('current observation', str(agent.llm_client.calls[2]))
        self.assertNotIn('first question', str(agent.llm_client.calls[2]))
        self.assertNotIn('current_history', vars(agent))

    def test_failed_and_budget_exhausted_turns_do_not_save_history(self):
        agent = self.agent('Thought: no action')
        agent.max_steps = 1
        self.assertIn('maximum steps', agent.run('unsaved question'))
        self.assertEqual(agent.get_history(), [])
        self.assertNotIn(b'unsaved question', Path(self.database_path).read_bytes())

    def test_working_memory_search_is_scoped_before_limit(self):
        tool = MemoryTool(memory_types=['working'])
        tool.set_session('first')
        tool.run('add', content='preference alpha', memory_type='working', importance=0.3)
        tool.set_session('second')
        tool.run('add', content='preference beta', memory_type='working', importance=0.99)
        tool.set_session('first')
        response = tool.run('search', query='preference', memory_types=['working'], limit=1)
        self.assertIn('preference alpha', response)
        self.assertNotIn('preference beta', response)
        tool.set_session('empty')
        self.assertEqual(tool.run('search', query='preference'), 'No memories found matching the query.')

    def test_saved_memory_belongs_to_selected_session(self):
        tool = MemoryTool(memory_types=['working'])
        registry = ToolRegistry()
        registry.register_function(tool.name, tool.description, tool.run)
        agent = self.agent(registry=registry)
        response = registry.execute_tool(tool.name, json.dumps({
            'action': 'add', 'content': 'preference alpha', 'memory_type': 'working',
            'session_id': 'model supplied wrong id',
        }))
        self.assertIn('successfully', response)
        item, = tool.memory_manager.memory_types['working'].get_all()
        self.assertEqual(item.metadata['session_id'], agent.session_id)
        self.assertEqual(item.type, 'working')

    def test_memory_type_stays_english_from_storage_through_search_and_reuse(self):
        for memory_type in ('working', 'episodic', 'semantic'):
            with self.subTest(memory_type=memory_type):
                tool = MemoryTool(memory_types=[memory_type])
                manager = MemoryManager.__new__(MemoryManager)
                store = MagicMock()
                stored = []
                def save(item):
                    stored.append(item)
                    return item.id
                store.add.side_effect = save
                store.retrieve.side_effect = lambda *args, **kwargs: [
                    {"memory": item, "score": 0.9} for item in stored
                ]
                manager.memory_types = {memory_type: store}
                tool.__dict__['memory_manager'] = manager

                result = tool.run('add', content='preference alpha', memory_type=memory_type)
                self.assertIn('successfully', result)
                self.assertEqual(stored[0].type, memory_type)
                response = tool.run('search', query='preference', memory_type=memory_type)
                self.assertIn(f'Type: {memory_type},', response)
                returned_type = response.split('Type: ', 1)[1].split(',', 1)[0]
                repeated = tool.run('search', query='preference', memory_type=returned_type)
                self.assertIn('preference alpha', repeated)
                store.retrieve.assert_called_with(
                    'preference', limit=5, threshold=0.7, min_importance=0.1,
                    session_id=tool.current_session_id,
                )

    def test_later_registered_tool_receives_session_id(self):
        registry = ToolRegistry()
        agent = self.agent(registry=registry)
        tool = MemoryTool(memory_types=['working'])
        registry.register_function(tool.name, tool.description, tool.run)
        self.assertEqual(tool.current_session_id, agent.session_id)

    def test_invalid_memory_json_returns_error(self):
        tool = MemoryTool(memory_types=['working'])
        self.assertIn('Error parsing', tool.run('{"action":'))
        self.assertIn('Error parsing', tool.run('{"content":"missing action"}'))

    def test_switching_does_not_initialize_memory_services(self):
        with patch('memory.memory_tool.MemoryManager', side_effect=RuntimeError('unavailable')) as manager:
            tool = MemoryTool()
            registry = ToolRegistry()
            registry.register_function(tool.name, tool.description, tool.run)
            agent = self.agent(registry=registry)
            first_id = agent.session_id
            agent.new_session()
            agent.select_session(first_id)
            manager.assert_not_called()
            self.assertIn('Error searching', tool.run('search', query='test'))

    def test_startup_choices_and_explicit_id_validation(self):
        agent = self.agent()
        with patch('builtins.input', side_effect=['invalid', agent.session_id]):
            self.assertEqual(choose_session(self.store, None, False), agent.session_id)
        with patch('builtins.input', return_value=''):
            self.assertIsNone(choose_session(self.store, None, False))
        self.assertIsNone(choose_session(self.store, None, True))
        self.assertEqual(choose_session(self.store, agent.session_id, False), agent.session_id)
        with self.assertRaises(ValueError):
            choose_session(self.store, 'unknown', False)

    def test_cli_commands_only_switch_ids(self):
        agent = self.agent()
        first_id = agent.session_id
        with patch('builtins.input', side_effect=['/sessions', '/new', '/use unknown',
                                                 f'/use {first_id}', '/exit']):
            chat_loop(agent, max_round=1)
        self.assertEqual(len(agent.list_sessions()), 2)
        self.assertEqual(agent.session_id, first_id)
        self.assertEqual(agent.llm_client.calls, [])
        self.assertNotIn('messages restored', self.output.getvalue())

    def test_list_only_does_not_initialize_llm_or_memory(self):
        agent = self.agent()
        with patch('my_main.build_agent') as build:
            __main__(['--list-sessions', '--session-db', self.database_path])
            build.assert_not_called()
        self.assertIn(agent.session_id, self.output.getvalue())
        self.assertNotIn('messages |', self.output.getvalue())

    def test_entry_point_passes_selected_id(self):
        agent = self.agent()
        with patch('my_main.build_agent', return_value=agent) as build:
            with patch('my_main.chat_loop'):
                __main__(['--session', agent.session_id, '--session-db', self.database_path])
        self.assertEqual(build.call_args.args[2], agent.session_id)

    def test_episodic_memory_session_filter_uses_original_metadata(self):
        memory = EpisodicMemory.__new__(EpisodicMemory)
        memory.config = SimpleNamespace(qdrant_url=None)
        memory.ttl_days = 30
        memory.doc_store = SQLiteDocumentStore(str(Path(self.directory.name) / 'episodic.db'), 100, 30)
        memory.sessions = {}
        memory.add(MemoryItem('one', 'preference alpha', 'episodic', 0.3, metadata={'session_id': 'first'}))
        memory.add(MemoryItem('two', 'preference beta', 'episodic', 0.9, metadata={'session_id': 'second'}))
        manager = MemoryManager.__new__(MemoryManager)
        manager.memory_types = {'episodic': memory}
        hits = manager.retrieve_memories('preference', ['episodic'], limit=1, session_id='first')
        self.assertEqual([hit['memory'].id for hit in hits], ['one'])

    def test_semantic_session_ids_limit_both_channels_before_retrieval(self):
        memory = SemanticMemory.__new__(SemanticMemory)
        memory.graph_store = MagicMock()
        memory.vector_store = MagicMock()
        memory._graph_query_terms = lambda query: ([], [query])
        memory.graph_store.get_session_ids.return_value = ['one', 'two']
        memory.graph_store.retrieve.return_value = []
        item = MemoryItem('one', 'preference', 'semantic', 0.8, metadata={'session_id': 'first'})
        memory.vector_store.retrieve.return_value = [{'memory_id': 'one', 'score': 0.9, 'memory': item}]
        hits = memory.retrieve('preference', session_id='first', allowed_ids=['one', 'foreign'])
        self.assertEqual([hit['memory'].id for hit in hits], ['one'])
        self.assertEqual(memory.vector_store.retrieve.call_args.kwargs['allowed_ids'], ['one'])
        self.assertEqual(memory.graph_store.retrieve.call_args.kwargs['allowed_ids'], ['one'])
        memory.graph_store.get_session_ids.assert_called_once_with('first', 0)

    def test_empty_semantic_session_skips_vector_and_graph_retrieval(self):
        memory = SemanticMemory.__new__(SemanticMemory)
        memory.graph_store = MagicMock()
        memory.vector_store = MagicMock()
        memory.graph_store.get_session_ids.return_value = []
        self.assertEqual(memory.retrieve('preference', session_id='empty'), [])
        memory.vector_store.retrieve.assert_not_called()
        memory.graph_store.retrieve.assert_not_called()

    def test_graph_session_filter_parses_legacy_json_exactly(self):
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store.query = MagicMock(return_value=[
            {'id': 'one', 'metadata_json': '{ "session_id" : "first" }'},
            {'id': 'two', 'metadata_json': '{"session_id":"first-other"}'},
            {'id': 'three', 'metadata_json': '{"description":"first"}'},
            {'id': 'four', 'metadata_json': None},
        ])
        self.assertEqual(store.get_session_ids('first'), ['one'])
        self.assertTrue(store.query.call_args.kwargs['read_only'])


if __name__ == '__main__':
    unittest.main()
