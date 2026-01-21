import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated, List, Dict, Any, Optional

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
# LLM
# ---------------------------------------------------------------------

def create_llm():
    return ChatOpenAI(
        model_name=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2,
    ).bind_tools(tools)

llm = create_llm()

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class AgentState(TypedDict):
    goal: str
    plan: List[str]
    current_plan_step: int

    messages: Annotated[Sequence[BaseMessage], "Conversation history"]

    steps: int
    max_steps: int
    last_action: str

    current_url: str
    page_content: str

    dom_candidates: List[Dict[str, Any]]
    chosen_element: Optional[Dict[str, Any]]

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
        **state,
        "plan": plan,
        "current_plan_step": 0,
        "messages": state["messages"] + [response],
    }

# ---------------------------------------------------------------------
# Agent (navigation / read only)
# ---------------------------------------------------------------------

async def agent_node(state: AgentState):
    if state["steps"] >= state["max_steps"]:
        return {
            **state,
            "last_action": "max_steps_reached",
            "steps": state["steps"] + 1,
        }

    plan_step = (
        state["plan"][state["current_plan_step"]]
        if state["current_plan_step"] < len(state["plan"])
        else "Finish the task"
    )

    prompt = get_prompt("navigate_prompt")

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
{state['page_content'][:600]}

Rules:
- ONLY navigate or read
- NEVER click or type
- Call EXACTLY ONE tool or finish_task
"""

    response = await llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=user_prompt),
        ]
    )

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
        return state

    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [last_msg]})

    tool_msg: ToolMessage = result["messages"][-1]
    output = tool_msg.content or ""

    tool_name = last_msg.tool_calls[0]["name"]

    current_url = state["current_url"]
    if tool_name == "navigate":
        for line in output.splitlines():
            if line.lower().startswith("current url"):
                current_url = line.split(":", 1)[1].strip()

    page_content = state["page_content"]
    if tool_name == "read_page":
        page_content = output

    return {
        **state,
        "messages": state["messages"] + result["messages"],
        "last_action": tool_name,
        "current_url": current_url,
        "page_content": page_content,
    }

# ---------------------------------------------------------------------
# DOM extractor (deterministic)
# ---------------------------------------------------------------------

async def get_stable_selector(el):
    for attr in ["aria-label", "name", "id"]:
        val = await el.get_attribute(attr)
        if val:
            return f"[{attr}='{val}']"
    tag = await el.evaluate("el => el.tagName.toLowerCase()")
    return tag

async def extract_dom_node(state: AgentState):
    browser = await get_browser()
    page = browser.page

    candidates = []
    idx = 1

    # Inputs
    for el in await page.locator("input, textarea").all():
        try:
            if not await el.is_visible():
                continue
            selector = await get_stable_selector(el)
            label = await el.get_attribute("aria-label") or ""
            name = await el.get_attribute("name") or ""
            candidates.append({
                "id": idx,
                "type": "input",
                "label": label or name,
                "selector": selector,
            })
            idx += 1
        except Exception:
            pass

    # Buttons
    for el in await page.locator("button,[role='button']").all():
        try:
            if not await el.is_visible():
                continue
            text = (await el.inner_text()).strip()
            if not text:
                continue
            selector = await get_stable_selector(el)
            candidates.append({
                "id": idx,
                "type": "button",
                "label": text,
                "selector": selector,
            })
            idx += 1
        except Exception:
            pass

    # Links (cap)
    for el in (await page.locator("a").all())[:15]:
        try:
            if not await el.is_visible():
                continue
            text = (await el.inner_text()).strip()
            href = await el.get_attribute("href") or ""
            if not text or not href:
                continue
            candidates.append({
                "id": idx,
                "type": "link",
                "label": text,
                "selector": f"a[href='{href}']",
            })
            idx += 1
        except Exception:
            pass

    return {
        **state,
        "dom_candidates": candidates,
    }

# ---------------------------------------------------------------------
# LLM chooser (decision only)
# ---------------------------------------------------------------------

async def choose_element_node(state: AgentState):
    if not state["dom_candidates"]:
        return {**state, "chosen_element": None}

    candidates_text = "\n".join(
        f"[{c['id']}] {c['type']} - {c['label']}"
        for c in state["dom_candidates"]
    )

    response = await llm.ainvoke(
        [
            SystemMessage(
                content="""
Choose ONE element ID that best matches the goal.
Reply ONLY with a number or NONE.
"""
            ),
            HumanMessage(
                content=f"""
GOAL:
{state['goal']}

ELEMENTS:
{candidates_text}
"""
            ),
        ]
    )

    choice = response.content.strip()
    chosen = None

    if choice.isdigit():
        chosen = next(
            (c for c in state["dom_candidates"] if c["id"] == int(choice)),
            None,
        )

    return {
        **state,
        "chosen_element": chosen,
        "messages": state["messages"] + [response],
    }

# ---------------------------------------------------------------------
# Executor (only place that clicks/types)
# ---------------------------------------------------------------------

async def executor_node(state: AgentState):
    chosen = state["chosen_element"]
    if not chosen:
        return state

    browser = await get_browser()

    if chosen["type"] == "input":
        await browser.type(
            chosen["selector"],
            state["goal"],
            press_enter=True,
        )

    elif chosen["type"] in ("button", "link"):
        await browser.click(chosen["selector"])

    return {
        **state,
        "last_action": f"interacted_{chosen['type']}",
        "chosen_element": None,
    }

# ---------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------

async def verifier_node(state: AgentState):
    verdict = await llm.ainvoke(
        [
            SystemMessage(content="Answer only yes or no."),
            HumanMessage(
                content=f"""
Goal: {state['goal']}
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
                state["current_plan_step"] + 1,
                len(state["plan"]),
            )
        }

    return {
        "steps": state["steps"] + 1
    }

# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------

def router(state: AgentState):
    if state["last_action"] == "finish_task":
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
graph.add_node("extract_dom", extract_dom_node)
graph.add_node("choose_element", choose_element_node)
graph.add_node("executor", executor_node)
graph.add_node("verifier", verifier_node)

graph.set_entry_point("planner")

graph.add_edge("planner", "agent")
graph.add_conditional_edges("agent", router, {"tools": "tools", END: END})
graph.add_edge("tools", "extract_dom")
graph.add_edge("extract_dom", "choose_element")
graph.add_edge("choose_element", "executor")
graph.add_edge("executor", "verifier")
graph.add_edge("verifier", "agent")

app = graph.compile()

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

async def run_agent(goal: str, max_steps: int = 30):
    log_separator("AGENT RUN START")

    state: AgentState = {
        "goal": goal,
        "plan": [],
        "current_plan_step": 0,
        "messages": [],
        "steps": 0,
        "max_steps": max_steps,
        "last_action": "none",
        "current_url": "",
        "page_content": "",
        "dom_candidates": [],
        "chosen_element": None,
        "all_actions": [],
        "visited": set(),
    }

    try:
        async for _ in app.astream(state, stream_mode="values"):
            pass
    finally:
        browser = await get_browser()
        await browser.close()
        log_separator("AGENT RUN END")
