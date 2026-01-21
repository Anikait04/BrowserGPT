import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from browsertools import tools, get_browser
from prompt import get_prompt
from logs import logger, log_separator

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b:free"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def save_successful_run(goal: str, actions: list[str]):
    os.makedirs("success_prev_run", exist_ok=True)
    safe_goal = goal.replace(" ", "_")[:50]
    filename = f"success_prev_run/{safe_goal}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"GOAL:\n{goal}\n\n")
        f.write("SUCCESSFUL ACTIONS:\n")
        for i, action in enumerate(actions, 1):
            f.write(f"{i}. {action}\n")

    logger.info(f"Saved successful run  {filename}")


def create_llm(model_name: str):
    logger.info(f"Initializing LLM: {model_name}")

    llm = ChatOpenAI(
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2,
    )

    return llm.bind_tools(tools)


llm = create_llm(MODEL_NAME)


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class AgentState(TypedDict):
    goal: str
    plan: List[str]
    current_plan_step: int

    messages: Annotated[Sequence[BaseMessage], "Conversation history"]
    current_url: str
    steps: int
    max_steps: int

    last_action: str
    page_content: str

    all_actions: List[str]
    visited: set


# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------

async def planner_node(state: AgentState):
    logger.info("Planning high-level steps")

    prompt = get_prompt("planner_prompt")

    response = await llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Goal: {state['goal']}"),
        ]
    )

    plan = [
        line.strip("- ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return {
        **state,                     # PRESERVE EVERYTHING
        "plan": plan,
        "current_plan_step": 0,
        "messages": state["messages"] + [response],
    }

# ---------------------------------------------------------------------
# Agent / Executor
# ---------------------------------------------------------------------

async def agent_node(state: AgentState):
    if state["steps"] >= state["max_steps"]:
        logger.warning("Max steps reached")
        return {
            **state,
            "messages": state["messages"]
            + [AIMessage(content="Max steps reached, stopping")],
            "last_action": "max_steps_reached",
            "steps": state["steps"] + 1,
        }

    plan_step = (
        state["plan"][state["current_plan_step"]]
        if state["current_plan_step"] < len(state["plan"])
        else "Finish the task"
    )

    prompt = get_prompt("navigate_prompt")
    page_preview = state["page_content"][:600]

    user_prompt = f"""
GOAL:
{state['goal']}

CURRENT PLAN STEP:
{plan_step}

STATUS:
- Step: {state['steps']} / {state['max_steps']}
- Last action: {state['last_action']}
- Current URL: {state['current_url'] or 'none'}

PAGE PREVIEW:
{page_preview}

Rules:
- Call EXACTLY ONE tool OR finish_task
- Do not explain
"""

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_prompt),
    ] + list(state["messages"][-4:])

    logger.info(f"Agent deciding action (step {state['steps'] + 1})")

    response = await llm.ainvoke(messages)

    return {
        **state,
        "messages": state["messages"] + [response],
        "steps": state["steps"] + 1,
    }


# ---------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------

async def tool_execution_node(state: AgentState):
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        logger.warning("No tool call detected")
        return state

    if len(last_msg.tool_calls) != 1:
        raise RuntimeError("More than one tool call detected")

    tool_call = last_msg.tool_calls[0]
    tool_name = tool_call["name"]

    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [last_msg]})

    tool_msg: ToolMessage = result["messages"][-1]
    tool_output = tool_msg.content or ""

    logger.info(f"Tool executed: {tool_name}")

    # --- URL tracking ---
    current_url = state["current_url"]

    if tool_name == "navigate":
        for line in tool_output.splitlines():
            if line.lower().startswith("current url"):
                current_url = line.split(":", 1)[1].strip()

    # --- Page content ---
    page_content = state["page_content"]
    if tool_name == "read_page":
        page_content = tool_output

    # --- Loop detection ---
    visited = set(state["visited"])
    visit_key = (tool_name, current_url or tool_output[:100])
    visited.add(visit_key)

    action_log = f"{tool_name}: {tool_output[:200].replace(chr(10), ' ')}"

    return {
        **state,   # ← THIS IS CRITICAL
        "messages": state["messages"] + result["messages"],
        "last_action": tool_name,
        "page_content": page_content,
        "current_url": current_url,
        "all_actions": state["all_actions"] + [action_log],
        "visited": visited,
    }


# ---------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------

async def verifier_node(state: AgentState):
    logger.info("Verifying last action")

    verdict = await llm.ainvoke(
        [
            SystemMessage(content="Answer only yes or no."),
            HumanMessage(
                content=f"""
Goal: {state['goal']}
Last action: {state['last_action']}
Page preview:
{state['page_content'][:500]}

Did this move us closer to the goal?
"""
            ),
        ]
    )

    if "yes" in verdict.content.lower():
        return {
            "current_plan_step": min(
                state["current_plan_step"] + 1, len(state["plan"])
            )
        }

    # Penalize bad action
    return {
        "steps": state["steps"] + 1
    }


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------

def router(state: AgentState):
    last_action = state.get("last_action")

    if last_action == "finish_task":
        save_successful_run(state["goal"], state["all_actions"])
        return END

    if state["steps"] >= state["max_steps"]:
        return END

    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    return END


# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("verifier", verifier_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "agent")
graph.add_conditional_edges("agent", router, {"tools": "tools", END: END})
graph.add_edge("tools", "verifier")
graph.add_edge("verifier", "agent")

app = graph.compile()

png_bytes = app.get_graph().draw_mermaid_png()
with open("agent_flow.png", "wb") as f:
    f.write(png_bytes)

logger.info("Agent flow graph saved as agent_flow.png")


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

async def run_agent(goal: str, max_steps: int = 30):
    log_separator("AGENT RUN START")

    initial_state: AgentState = {
        "goal": goal,
        "plan": [],
        "current_plan_step": 0,
        "messages": [],
        "steps": 0,
        "max_steps": max_steps,
        "last_action": "none",
        "current_url": "",  
        "page_content": "",
        "all_actions": [],
        "visited": set(),
    }

    try:
        async for _ in app.astream(initial_state, stream_mode="values"):
            pass
    finally:
        try:
            browser = await get_browser()
            await browser.close()
        except Exception:
            logger.warning("Browser cleanup failed", exc_info=True)

        log_separator("AGENT RUN END")
