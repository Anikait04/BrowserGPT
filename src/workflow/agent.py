# agent.py — public entry points (backward-compat facade).
#
# Graph wiring lives in graph.py, run loops in runner.py, state construction in
# state_factory.py. This module preserves the original imports
# (graph, get_app, run_agent_server, resume_agent_server, run_agent) so
# routers and external callers keep working.

from __future__ import annotations

import os
import uuid

import aiosqlite
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END
from langgraph.types import Command

from src.config import thread_dir_name
from logs import logger, log_separator
from src.workflow.state import AgentState
from src.workflow.browser.manager import close_browser
from src.workflow.graph import build_graph
from src.workflow.llm import reset_session_id, set_session_id
from src.workflow.runner import (
    artifact_info as _artifact_info_impl,
    pause_prompt as _pause_prompt_impl,
)
from src.workflow.state_factory import initial_state

load_dotenv()


# Module-level compiled graph (built once via build_graph for testability).
graph = build_graph()

# ── Checkpointer is async, so compile happens inside get_app() ──
_checkpointer = None
_app = None


async def get_app():
    global _checkpointer, _app
    if _app is None:
        if not os.path.exists(thread_dir_name):
            os.makedirs(thread_dir_name)
        conn = await aiosqlite.connect(f"{thread_dir_name}/checkpoints.db")
        _checkpointer = AsyncSqliteSaver(conn)
        _app = graph.compile(checkpointer=_checkpointer)

        # Opt-in graph diagram export (disabled by default; see constants).
        from src.workflow.constants import EXPORT_GRAPH_PNG, GRAPH_PNG_PATH

        if EXPORT_GRAPH_PNG:
            try:
                png_bytes = _app.get_graph().draw_mermaid_png()
                with open(GRAPH_PNG_PATH, "wb") as f:
                    f.write(png_bytes)
            except Exception as e:
                logger.warning(f"Could not generate graph PNG: {e}")

    return _app


def _build_initial_state(goal: str, max_steps: int, task_id: str | None) -> AgentState:
    """Deprecated wrapper — use state_factory.initial_state directly."""
    return initial_state(goal, max_steps, task_id)


def _artifact_info(values: dict) -> dict:
    """Deprecated wrapper — use runner.artifact_info directly."""
    return _artifact_info_impl(values)


def _pause_prompt(snapshot) -> str:
    """Deprecated wrapper — use runner.pause_prompt directly."""
    return _pause_prompt_impl(snapshot)


async def run_agent_server(
    goal: str, max_steps: int = 30, task_id: str | None = None
) -> dict:
    """Non-interactive start of a persistent run (for API servers).

    Runs until the first user interrupt or END. NEVER blocks on input().
    - paused -> browser stays open, caller must call resume_agent_server().
    - done (explicit user exit — the only END path) -> browser closed.
    """
    from src.workflow.runner import AgentRunner

    runner = AgentRunner(get_app)
    return await runner.start(goal, max_steps=max_steps, task_id=task_id)


async def resume_agent_server(
    thread_id: str, human_input: str, task_id: str | None = None
) -> dict:
    """Resume a paused persistent run with user input (for API servers).

    - exit command -> graph routes to END, browser closed, status done.
    - new task -> replanned via planner, runs until next pause/END.
    - otherwise -> resumes current goal until next pause/END.
    """
    from src.workflow.runner import AgentRunner

    runner = AgentRunner(get_app)
    return await runner.resume(thread_id, human_input, task_id=task_id)


async def run_agent(goal: str, max_steps: int = 30, thread_id: str = None, task_id: str = None):
    """Interactive CLI run: persistent loop, ends ONLY on explicit user exit."""
    log_separator("AGENT RUN START")

    app = await get_app()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Thread ID: {thread_id}")

    # Every LLM call in this run carries thread_id as its session id.
    session_token = set_session_id(thread_id)

    state: AgentState = initial_state(goal, max_steps, task_id)

    try:
        while True:
            async for _ in app.astream(state, config=config, stream_mode="values"):
                pass

            current_state = await app.aget_state(config)

            if current_state.next:  # graph is paused (interrupt)
                logger.info("=" * 60)
                logger.info(f"AGENT PAUSED | thread_id: {thread_id}")
                logger.info("=" * 60)
                # Persistent loop: the run only ends on an explicit exit
                # command. Anything else (new task / continue) resumes.
                try:
                    human_input = input("Your instruction ('exit' to end, new task, or 'continue'): ").strip()
                except EOFError:
                    # No stdin (e.g. server mode without a resume endpoint):
                    # end cleanly instead of hanging forever.
                    logger.warning("[RUN] No stdin available for user input, ending run")
                    human_input = "exit"

                async for _ in app.astream(
                    Command(resume=human_input),
                    config=config,
                    stream_mode="values"
                ):
                    pass

                # Sync state from checkpointer for next loop iteration
                state = (await app.aget_state(config)).values
                # If the user explicitly exited, the wait node routed to END:
                # next is empty and exit_requested is set -> fall through to break.
                if not (await app.aget_state(config)).next and state.get("exit_requested"):
                    break
            else:
                # Graph reached END — with the persistent loop this happens
                # ONLY after an explicit user exit (see route_from_wait_for_user).
                break
            # Signal SSE stream: task finished successfully
        if task_id:
            from src.routers.common import push_done
            await push_done(task_id, "Task completed successfully")
    except Exception as e:
        # Signal SSE stream: task failed
        if task_id:
            from src.routers.common import push_error
            await push_error(task_id, str(e))
        raise

    finally:
        reset_session_id(session_token)
        await close_browser()
        log_separator("AGENT RUN END")


__all__ = [
    "graph",
    "get_app",
    "run_agent",
    "run_agent_server",
    "resume_agent_server",
]
