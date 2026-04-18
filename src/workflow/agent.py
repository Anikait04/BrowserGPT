import os

import aiosqlite
import uuid
from dotenv import load_dotenv
from src.workflow.agent_state import AgentState
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END
from src.workflow.browsertools import get_browser,close_browser
from logs import logger, log_separator
from langgraph.types import Command
load_dotenv()

from src.workflow.nodes import *
from config import thread_dir_name 

def agent_router(state: AgentState):
    route = state["agent_decision"]
    if route == "finish":
        return END
    if state["steps"] >= state["max_steps"]:
        return END
    if route == "tools":
        return "tools"
    if route == "read_page":
        return "read_page"
    if route == "wait":
        return "human_wait"
    return END


graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("read_page", observe_and_choose_node)
graph.add_node("verifier", verifier_node)
graph.add_node("human_wait", human_wait_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "agent")
graph.add_conditional_edges(
    "agent",
    agent_router,
    {
        "tools": "tools",
        "read_page": "read_page",
        "human_wait": "human_wait",
        END: END,
    },
)
graph.add_edge("tools", "verifier")
graph.add_edge("verifier", "agent")
graph.add_edge("read_page", "agent")
graph.add_edge("human_wait", "agent")

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


async def run_agent(goal: str, max_steps: int = 30, thread_id: str = None,task_id: str = None):
    log_separator("AGENT RUN START")

    app = await get_app()

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Thread ID: {thread_id}")

    state: AgentState = {
        "goal": goal,
        "entire_plan": [],
        "step_count": 0,
        "current_action": "",
        "agent_decision": "",
        "task_id": task_id, 
        "steps": 0,
        "progress_verification": "",
        "max_steps": max_steps,
        "last_action": "none",
        "current_url": "",
        "chosen_element": [],
        "tool_name": "",
        "tool_input": "",
        "tool_selector": "",
        "messages": [],
        "task_id": task_id,   # ← passed into every node via state
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
                human_input = input("Your instruction (or 'continue'): ").strip()

                async for _ in app.astream(
                    Command(resume=human_input),
                    config=config,
                    stream_mode="values"
                ):
                    pass

                # Sync state from checkpointer for next loop iteration
                state = (await app.aget_state(config)).values
            else:
                break  # agent finished normally
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
        await close_browser()
        log_separator("AGENT RUN END")