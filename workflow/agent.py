
from dotenv import load_dotenv
from workflow.agent_state import AgentState
from langgraph.graph import StateGraph, END
from workflow.browsertools import get_browser
from logs import logger, log_separator
load_dotenv()
from workflow.nodes import *


def agent_router(state: AgentState):
    route = state["agent_decision"]
    # 1. Explicit finish
    if route == "finish":
        return END

    # 2. Step budget exhausted
    if state["steps"] >= state["max_steps"]:
        return END

    # 3. Tool execution
    if route == "tools":
        return "tools"

    # 4. Page read
    if route == "read_page":
        return "read_page"

    # 5. Controlled loop
    if route == "wait":
        return "agent"

    # Safety fallback (should never happen)
    return END

def tool_router(state: AgentState):
    return "verifier"

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("read_page", observe_and_choose_node)
graph.add_node("verifier", verifier_node)

# Entry
graph.set_entry_point("planner")
graph.add_edge("planner", "agent")

graph.add_conditional_edges(
    "agent",
    agent_router,
    {
        "tools": "tools",
        "read_page": "read_page",
        "agent": "agent",
        END: END,
    },
)
graph.add_edge("tools", "verifier")
graph.add_edge("verifier", "agent")
graph.add_edge("read_page", "agent")

app = graph.compile()
try:
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("agent_flow.png", "wb") as f:
        f.write(png_bytes)
except Exception as e:
    logger.warning(f"Could not generate graph PNG: {e}")

async def run_agent(goal: str, max_steps: int = 30):
    log_separator("AGENT RUN START")

    state: AgentState = {
        "goal": goal,
        "entire_plan": [],
        "step_count": 0,
        "current_action":"",
        "agent_decision": "",
        "steps": 0,
        "progress_verification":"",
        "max_steps": max_steps,
        "last_action": "none",
        "current_url": "",
        "chosen_element": [],
        "tool_name": "",
        "tool_input": "",
        "tool_selector": "",
        "messages": []
    }

    try:
        async for _ in app.astream(state, stream_mode="values"):
            # print("\n" + "="*80)
            # print("FULL STATE AFTER STEP")
            # print("=======", _, "===========")
            pass  
    finally:
        browser = await get_browser()
        await browser.close()
        log_separator("AGENT RUN END")
