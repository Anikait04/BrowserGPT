import os
import asyncio
import uuid
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated, List, Dict, Any, Optional
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from utils import plan_steps_update
from pydantic import BaseModel, Field
from typing import List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from browsertools import tools, get_browser
from prompt import get_prompt
from logs import logger, log_separator
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from structured import AgentDecision, PlanOutput
import re
from utils import plan_steps_update
load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b:free"
_PAGE_CACHE: Dict[str, str] = {}


# llm = ChatOpenAI(
#         model_name=MODEL_NAME,
#         base_url="https://openrouter.ai/api/v1",
#         temperature=0.0,
#         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#         max_retries=2,
#     )
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0,
    max_retries=2,
    model_kwargs={"format": "json"}
)

class AgentState(TypedDict):
    goal: str
    entire_plan: List[str]
    step_count: int
    current_action:str
    agent_decision:str
    steps: int
    max_steps: int
    progress_verification: str
    last_action: str
    current_url: str
    tool_name:str
    tool_input:str
    element_id:int
    messages: Annotated[Sequence[BaseMessage], "node remarks messages exchanged so far"]
    chosen_element:List[str]
    



async def planner_node(state: AgentState):
    logger.info("Planning high-level steps")
    prompt = get_prompt("planner_prompt")
    structured_llm = llm.with_structured_output(PlanOutput)
    result= await structured_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Goal: {state['goal']}"),
        ])
    return {
        **state,
        "entire_plan": result.plan,
        "step_count": 0,
        "messages": result.messages,
    }

async def agent_node(state: AgentState):
    logger.info("Started Agent Node")
    if state["steps"] >= state["max_steps"]:
        return {
            **state,
            "agent_decision": {
                "route_decision": "finish",
                "tool_input": "",
            },
            "last_action": "max_steps_reached",
            "steps": state["steps"] + 1,
        }
    updated_plan_step_info = plan_steps_update(state["step_count"],state["entire_plan"])
    state["current_action"]=updated_plan_step_info
    elements_info = ""
    if state.get("chosen_element"):
        elements_info = "\n\nWebsite Elements Currently in Use (These will be passed into the tools as input):\n"
        for el in state["chosen_element"]:
            elements_info += f"[{el['id']}] {el['type']} - {el['label']} - {el['selector']}\n"
    print(elements_info)
    user_prompt = f"""
        GOAL:
        {state['goal']}

        CURRENT PLAN STEP:
        {state["current_action"]}

        TOOL EXECUTION VEIFICATION:
        {state['progress_verification']}

        STATUS:
        - Step: {state['steps']} / {state['max_steps']}
        - Current URL: {state['current_url'] or 'none'}

        {elements_info}
        """ 
    structured_llm=llm.with_structured_output(AgentDecision)
    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=get_prompt("navigate_prompt")),
            HumanMessage(content=user_prompt),
        ]
    )
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        existing_messages = [existing_messages]
    existing_messages.append(response.get("message", ""))
    return {
            **state,
            "messages": existing_messages,
            "agent_decision": response.get("route_decision",""),
            "tool_name": response.get("tool_name", ""),
            "tool_input": response.get("tool_input", ""),
            "element_id":response.get("element_id",""),
            "steps": state.get("steps", 0) + 1,
            "current_action": state["current_action"],
        }


async def tool_execution_node(state: dict):
    logger.info("Started Tool Execution Node")
    tool_name = state.get("tool_name")
    if not tool_name:
        return state

    tool_input = state.get("tool_input")
    element_id = state.get("element_id")

    tool_selector = None

    if element_id is not None:
        selected = next(
            (el for el in state.get("chosen_element", [])
             if el["id"] == element_id),
            None
        )
        if selected:
            tool_selector = selected["selector"]
        else:
            return state


    if tool_name == "navigate":
        args = {"url": tool_input}

    elif tool_name == "type_text":
        args = {
            "selector": tool_selector,
            "value": tool_input,
            # "press_enter": True
        }

    elif tool_name == "click_element":
        args = {
            "selector": tool_selector,
        }

    else:
        args = {}

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "id": uuid.uuid4().hex,
                "name": tool_name,
                "args": args,
            }
        ],
    )
    print("tool_call input:::",ai_message)
    tool_node = ToolNode(tools)


    result = await tool_node.ainvoke({"messages": [ai_message]})
    tool_msg: ToolMessage = result["messages"][-1]

    existing_messages = state.get("messages", [])

    if isinstance(existing_messages, list):
        updated_messages = existing_messages + [tool_msg.content]
    else:
        updated_messages = [tool_msg.content]


    if tool_name == "navigate" and tool_input:
        state["current_url"] = tool_input

    browser = await get_browser()
    current_page_url = browser.page.url
    return {
        **state,
        "messages": updated_messages,
        "last_action": tool_name,
        "current_url":current_page_url
    }

async def observe_and_choose_node(state: AgentState):
    logger.info("Started Observe and Choose Node")

    browser = await get_browser()
    page = browser.page


    page_content = await browser.read()

    if state.get("current_url"):
        _PAGE_CACHE[state["current_url"]] = page_content

    elements = []
    idx = 1

    async def safe_text(el):
        try:
            return (await el.inner_text()).strip()
        except Exception:
            return ""

    async def visible(el):
        try:
            return await el.is_visible()
        except Exception:
            return False

    async def get_stable_selector(el):
        el_id = await el.get_attribute("id")
        if el_id:
            return f"#{el_id}"

        aria = await el.get_attribute("aria-label")
        if aria:
            return f"[aria-label='{aria}']"

        name = await el.get_attribute("name")
        if name:
            return f"[name='{name}']"

        tag = await el.evaluate("e => e.tagName.toLowerCase()")
        return tag

    async def build_candidate(el, el_type):
        nonlocal idx

        label = (
            await el.get_attribute("aria-label")
            or await el.get_attribute("placeholder")
            or await el.get_attribute("name")
            or await safe_text(el)
        )

        if not label:
            return None

        label = re.sub(r"\s+", " ", label).strip()

        selector = await get_stable_selector(el)

        candidate = {
            "id": idx,
            "type": el_type,
            "label": label,
            "selector": selector,
        }

        idx += 1
        return candidate

    # Inputs
    for el in await page.locator("input, textarea").all():
        if await visible(el):
            c = await build_candidate(el, "input")
            if c:
                elements.append(c)

    # Buttons
    for el in await page.locator("button, [role='button'], [onclick]").all():
        if await visible(el):
            c = await build_candidate(el, "button")
            if c:
                elements.append(c)

    # Links
    for el in await page.locator("a[href]").all():
        if await visible(el):
            c = await build_candidate(el, "link")
            if c:
                elements.append(c)
    labels_text = "\n".join(f"- {c['label']}" for c in elements)
    with open("label.txt", "w",encoding="utf-8") as f:
        f.write(labels_text)
    
    plan_step = state.get("current_action", "")
    PROMPTY=get_prompt("choose_and_observe_prompt")
    # print("elements::::",elements)
    response = await llm.ainvoke(
        [
            SystemMessage(content=PROMPTY),
                        HumanMessage(
                            content=f"""
            GOAL: {state['goal']}
            STEP: {plan_step}

            AVAILABLE ELEMENTS:
            {elements}
            """
            ),
        ]
    )

    chosen_label = response.content.strip()
    print("chosen_label",chosen_label)

    if chosen_label.upper() == "NONE":
        return {
            **state,
            "dom_candidates": elements,
            "chosen_element": None,
            "top_candidates": [],
            "messages": state.get("messages", []) + [
                AIMessage(content="No relevant interactive elements found.")
            ],
        }

    def tokenize(text):
        return set(text.lower().split())

    chosen_tokens = tokenize(chosen_label)

    def score(el):
        label_tokens = tokenize(el["label"])
        overlap = len(chosen_tokens & label_tokens)
        s = overlap * 5

        if el["type"] == "button":
            s += 3
        if el["type"] == "input":
            s += 2

        return s
    with open("elements.txt", "w",encoding="utf-8") as f:
        f.write(str(elements))
    ranked = sorted(elements, key=score, reverse=True)

    top_5 = ranked[:20]
    chosen_label+="lable is chosen"

    return {
        **state,
        "chosen_element": top_5,
        "messages": state["messages"] + [chosen_label],
    }

# ---------------------------------------------------------------------
# Verifier - IMPROVED
# ---------------------------------------------------------------------

async def verifier_node(state: AgentState):
    logger.info("Started Verifier Node")
    plan_step = state.get("current_action", "")

    # page_preview = ""
    current_url = state.get("current_url")
    # if current_url and current_url in _PAGE_CACHE:
    #     page_preview = _PAGE_CACHE[current_url][:600]
    verdict = await llm.ainvoke(
        [
            SystemMessage(
                content="Answer only 'yes' or 'no'. Be strict — only say yes if real progress was made."
            ),
            HumanMessage(
                content=f"""
GOAL: {state['goal']}

ENTIRE PLAN: {state['entire_plan']}

CURRENT PLAN STEP: {plan_step}

LAST ACTION: {state['last_action']}

CURRENT URL: {current_url}

MESSAGES LOG OF ENTIRE PROCESS TILL NOW {state["messages"]}

Did the last action successfully complete or make progress on the plan?

Examples:
- If step is "search for X" and search was performed → yes
- If step is "click video" and video page opened → yes
- If step is "search" but search was done again → no (redundant)
- If an error occurred → no
"""
            ),
        ]
    )

    verdict_text = verdict.content.lower().strip()
    logger.info(f"verifier_node verdict: {verdict_text}")
    verification = state.get("current_action", "")
    if verdict_text.startswith("yes"):
        next_step = state["step_count"] + 1

        logger.info(
            f"Plan step {state['step_count']} completed, moving to step {next_step}"
        )

        verification += " was completed and verified successfully."

        return {
            **state,
            "step_count": next_step,
            "progress_verification": verification
        }

    logger.info(
        f"Plan step {state['step_count']} not completed, retrying"
    )

    last_message = ""
    if state.get("messages") and isinstance(state["messages"], list):
        last_message = state["messages"][-1]

    verification += f" has failed due to: {last_message}. Please try a different approach."

    return {
        **state,
        "progress_verification": verification
    }
# ---------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------

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

# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("read_page", observe_and_choose_node)
graph.add_node("verifier", verifier_node)

# Entry
graph.set_entry_point("planner")
graph.add_edge("planner", "agent")

# agent → tools | read_page | agent | END
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

# tools → verify
graph.add_edge("tools", "verifier")

# verifier → agent
graph.add_edge("verifier", "agent")
# read_page → extract_dom → choose_element → agent
graph.add_edge("read_page", "agent")

app = graph.compile()
try:
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("agent_flow.png", "wb") as f:
        f.write(png_bytes)
except Exception as e:
    logger.warning(f"Could not generate graph PNG: {e}")

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

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
