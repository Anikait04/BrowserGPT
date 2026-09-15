import os
import uuid

import aiosqlite
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command

from config import thread_dir_name
from logs import logger, log_separator
from src.workflow.agent_state import AgentState
from src.workflow.browsertools import close_browser
from src.workflow.delegation import delegation_node
from src.workflow.extract_information import extract_information_node
from src.workflow.llm import reset_session_id, set_session_id
from src.workflow.navigation import navigation_node
from src.workflow.planner import planner_node
from src.workflow.routing import route_from_delegation, route_from_verification, route_from_wait_for_user
from src.workflow.verify import verify_node
from src.workflow.wait_for_user import wait_for_user_node

load_dotenv()


graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("delegation", delegation_node)
graph.add_node("navigation", navigation_node)
graph.add_node("verify", verify_node)
graph.add_node("extract_information", extract_information_node)
graph.add_node("wait_for_user", wait_for_user_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "delegation")
# Persistent loop: delegation NEVER routes to END directly. "finish"
# (goal done, max steps, failure cap, unknown) routes to wait_for_user so
# the user reviews the result and issues the next task. Only an explicit
# user exit command (handled in wait_for_user) reaches END.
graph.add_conditional_edges(
    "delegation",
    route_from_delegation,
    {
        "navigation": "navigation",
        "extract_information": "extract_information",
        "wait_for_user": "wait_for_user",
    },
)

graph.add_edge("navigation", "verify")
graph.add_conditional_edges(
    "verify",
    route_from_verification,
    {
        "navigation": "navigation",
        "delegation": "delegation",
    },
)
graph.add_edge("extract_information", "delegation")
# wait_for_user is the ONLY gateway to END (explicit user exit), otherwise
# it replans a fresh user task (planner) or resumes the goal (delegation).
graph.add_conditional_edges(
    "wait_for_user",
    route_from_wait_for_user,
    {
        "planner": "planner",
        "delegation": "delegation",
        END: END,
    },
)

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

        # Generate graph PNG once after compile
        try:
            png_bytes = _app.get_graph().draw_mermaid_png()
            with open("agent_flow.png", "wb") as f:
                f.write(png_bytes)
        except Exception as e:
            logger.warning(f"Could not generate graph PNG: {e}")

    return _app


def _build_initial_state(goal: str, max_steps: int, task_id: str | None) -> AgentState:
    return {
        "goal": goal,
        "entire_plan": [],
        "step_count": 0,
        "agent_decision": "",
        "task_id": task_id,   # ← passed into every node via state
        "steps": 0,
        "progress_verification": "",
        "max_steps": max_steps,
        "current_url": "",
        "messages": [],
        "current_delegated_task": "",
        "delegation_decision": None,
        "success_criteria": "",
        "navigation_result": "",
        "verification_result": None,
        "extracted_information": None,
        "waiting_for_user": False,
        "exit_requested": False,
        "final_response": "",
        "navigation_iterations": 0,
        "consecutive_failures": 0,
        "all_actions": [],
    }


def _pause_prompt(snapshot) -> str:
    """Extract the human-readable interrupt prompt from a paused snapshot."""
    try:
        for task in getattr(snapshot, "tasks", []) or []:
            for intr in getattr(task, "interrupts", []) or []:
                value = getattr(intr, "value", "")
                if value:
                    return str(value)
    except Exception:
        pass
    return "Agent is waiting for input. Type your next task or 'exit' to end."


async def run_agent_server(
    goal: str, max_steps: int = 30, thread_id: str | None = None, task_id: str | None = None
) -> dict:
    """Non-interactive start of a persistent run (for API servers).

    Runs until the first user interrupt or END. NEVER blocks on input().
    - paused -> browser stays open, caller must call resume_agent_server().
    - done (explicit user exit — the only END path) -> browser closed.
    """
    from src.workflow.browsertools import close_browser as _close_browser

    log_separator("AGENT RUN START (server)")
    app = await get_app()
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Thread ID: {thread_id}")
    session_token = set_session_id(thread_id)
    try:
        state = _build_initial_state(goal, max_steps, task_id)
        async for _ in app.astream(state, config=config, stream_mode="values"):
            pass
        snapshot = await app.aget_state(config)
        values = snapshot.values or {}
        if snapshot.next:  # paused for user input
            return {
                "status": "paused",
                "thread_id": thread_id,
                "prompt": _pause_prompt(snapshot),
                "goal": values.get("goal", goal),
                "final_response": values.get("final_response", ""),
                "current_url": values.get("current_url", ""),
            }
        # END — only reachable after explicit user exit.
        if task_id:
            from src.routers.agent_router import push_done
            await push_done(task_id, values.get("final_response") or "Task completed successfully")
        await _close_browser()
        return {
            "status": "done",
            "thread_id": thread_id,
            "final_response": values.get("final_response", ""),
            "current_url": values.get("current_url", ""),
        }
    except Exception as e:
        if task_id:
            from src.routers.agent_router import push_error
            await push_error(task_id, str(e))
        raise
    finally:
        reset_session_id(session_token)
        log_separator("AGENT RUN PAUSED/END (server)")


async def resume_agent_server(
    thread_id: str, human_input: str, task_id: str | None = None
) -> dict:
    """Resume a paused persistent run with user input (for API servers).

    - exit command -> graph routes to END, browser closed, status done.
    - new task -> replanned via planner, runs until next pause/END.
    - otherwise -> resumes current goal until next pause/END.
    """
    from src.workflow.browsertools import close_browser as _close_browser

    app = await get_app()
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Resuming thread: {thread_id}")
    session_token = set_session_id(thread_id)
    try:
        # Keep SSE streaming coherent across resumes.
        if task_id:
            snapshot_before = await app.aget_state(config)
            before_values = snapshot_before.values or {}
            if before_values.get("task_id") != task_id and task_id:
                # Rebind the run to the new stream's queue so screenshots flow.
                try:
                    await app.aupdate_state(config, {"task_id": task_id})
                except Exception as e:
                    logger.warning(f"[RUN] Could not rebind task_id: {e}")
        async for _ in app.astream(Command(resume=str(human_input or "").strip()), config=config, stream_mode="values"):
            pass
        snapshot = await app.aget_state(config)
        values = snapshot.values or {}
        if snapshot.next:  # paused again
            return {
                "status": "paused",
                "thread_id": thread_id,
                "prompt": _pause_prompt(snapshot),
                "goal": values.get("goal", ""),
                "final_response": values.get("final_response", ""),
                "current_url": values.get("current_url", ""),
            }
        if task_id:
            from src.routers.agent_router import push_done
            await push_done(task_id, values.get("final_response") or "Session ended")
        await _close_browser()
        return {
            "status": "done",
            "thread_id": thread_id,
            "final_response": values.get("final_response", ""),
            "current_url": values.get("current_url", ""),
        }
    except Exception as e:
        if task_id:
            from src.routers.agent_router import push_error
            await push_error(task_id, str(e))
        raise
    finally:
        reset_session_id(session_token)


async def run_agent(goal: str, max_steps: int = 30, thread_id: str = None, task_id: str = None):
    """Interactive CLI run: persistent loop, ends ONLY on explicit user exit."""
    log_separator("AGENT RUN START")

    app = await get_app()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Thread ID: {thread_id}")

    # Every LLM call in this run carries thread_id as its session id.
    session_token = set_session_id(thread_id)

    state: AgentState = {
        "goal": goal,
        "entire_plan": [],
        "step_count": 0,
        "agent_decision": "",
        "task_id": task_id,   # ← passed into every node via state
        "steps": 0,
        "progress_verification": "",
        "max_steps": max_steps,
        "current_url": "",
        "messages": [],
        "current_delegated_task": "",
        "delegation_decision": None,
        "success_criteria": "",
        "navigation_result": "",
        "verification_result": None,
        "extracted_information": None,
        "waiting_for_user": False,
        "exit_requested": False,
        "final_response": "",
        "navigation_iterations": 0,
        "consecutive_failures": 0,
        "all_actions": [],
    }

    try:
        while True:
            async for _ in app.astream(state, config=config, stream_mode="values"):
                pass


            current_state = await app.aget_state(config)

            if current_state.next:  # graph is paused (interrupt)
                print("\n" + "="*60)
                print(f"⏸  AGENT PAUSED | thread_id: {thread_id}")
                print("="*60)
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
            from src.routers.agent_router import push_done
            await push_done(task_id, "Task completed successfully")
    except Exception as e:
        # Signal SSE stream: task failed
        if task_id:
            from src.routers.agent_router import push_error
            await push_error(task_id, str(e))
        raise

    finally:
        reset_session_id(session_token)
        await close_browser()
        log_separator("AGENT RUN END")
