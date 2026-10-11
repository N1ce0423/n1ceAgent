"""命令行聊天入口，支持选择和新建记忆会话 ID。"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from config import Config
from session_store import SessionStore


AGENT_NAME = "EnhancedAgent"


def build_agent(config: Config, store: SessionStore, session_id: Optional[str] = None):
    # 现有 memory 模块仍使用顶层导入；保持项目根目录启动时的兼容性。
    memory_directory = str(Path(__file__).resolve().parent / "memory")
    if memory_directory not in sys.path:
        sys.path.insert(0, memory_directory)

    from HelloAgentsLLM import HelloAgentsLLM
    from memory.memory_tool import MemoryTool
    from my_react_agent import MyReactAgent
    from tools.advanced_search_tool import MyAdvancedSearchTool
    from tools.calculator import calculator

    llm = HelloAgentsLLM()
    tool_registry = calculator().create_calculator_registry()
    advanced_search_tool = MyAdvancedSearchTool()
    tool_registry.register_function(
        name=advanced_search_tool.name,
        description=advanced_search_tool.description,
        func=advanced_search_tool.run,
    )
    memory_tool = MemoryTool()
    tool_registry.register_function(
        name=memory_tool.name, description=memory_tool.description, func=memory_tool.run,
    )
    return MyReactAgent(
        name=AGENT_NAME, llm_client=llm, tool_registry=tool_registry,
        config=config, session_id=session_id, session_store=store,
    )


def print_sessions(store: SessionStore, current_session_id: Optional[str] = None) -> None:
    sessions = store.list_sessions()
    if not sessions:
        print("No saved sessions.")
        return
    for session in sessions:
        marker = "*" if session.session_id == current_session_id else " "
        print(f"{marker} {session.session_id}")


def choose_session(store: SessionStore, requested_id: Optional[str], new: bool) -> Optional[str]:
    """启动时选择已有对话；None 表示创建新对话。"""
    if requested_id is not None:
        store.require_session(requested_id)
        return requested_id
    if new or not store.list_sessions():
        return None
    print_sessions(store)
    while True:
        selected = input("Select a session_id for memory search, or press Enter to create one: ").strip()
        if not selected or selected.lower() == "new":
            return None
        try:
            store.require_session(selected)
            return selected
        except ValueError as exc:
            print(f"Error: {exc}")


def chat_loop(agent, max_round: int) -> None:
    print(f"Current session_id: {agent.session_id}")
    print("Commands: /sessions, /use <session_id>, /new, /help, /exit")
    completed_rounds = 0
    while completed_rounds < max_round:
        try:
            user_input = input(f"[{agent.session_id}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "/exit"}:
            print("Exiting.")
            break
        if user_input == "/sessions":
            print_sessions(agent.session_store, agent.session_id)
            continue
        if user_input == "/new":
            print(f"New session_id: {agent.new_session()}")
            continue
        if user_input.startswith("/use") and user_input.split(maxsplit=1)[0] == "/use":
            parts = user_input.split(maxsplit=1)
            if len(parts) != 2:
                print("Usage: /use <session_id>")
                continue
            try:
                agent.select_session(parts[1])
                print(f"Current session_id: {agent.session_id}")
            except ValueError as exc:
                print(f"Error: {exc}")
            continue
        if user_input.startswith("/"):
            print("Commands: /sessions, /use <session_id>, /new, /help, /exit")
            continue
        try:
            response = agent.run(user_input)
        except Exception as exc:
            print(f"Error running agent: {exc}")
            continue
        completed_rounds += 1
        print(f"Agent: {response}")


def __main__(argv: Optional[list[str]] = None) -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    config = Config.from_env()
    parser = argparse.ArgumentParser(description="Select a saved or new memory session ID.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--session", help="Select a saved session_id for memory search.")
    selection.add_argument("--new", action="store_true", help="Start a new conversation.")
    parser.add_argument("--list-sessions", action="store_true", help="List saved conversations and exit.")
    parser.add_argument("--session-db", default=config.session_database_path, help="Session ID SQLite path.")
    args = parser.parse_args(argv)
    store = SessionStore(args.session_db)
    if args.list_sessions:
        print_sessions(store)
        return
    try:
        session_id = choose_session(store, args.session, args.new)
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return
    except ValueError as exc:
        parser.error(str(exc))
    try:
        agent = build_agent(config, store, session_id)
    except Exception as exc:
        raise SystemExit(f"Error initializing agent: {exc}") from exc
    chat_loop(agent, config.max_round)


if __name__ == "__main__":
    __main__()
