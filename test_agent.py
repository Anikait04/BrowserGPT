import os
import asyncio
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
from pydantic import BaseModel, Field
from typing import List
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

llm = ChatOpenAI(
        model_name=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2,
    )
llm_tools=ChatOpenAI(
        model_name=MODEL_NAME,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2,
    ).bind_tools(tools)

# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------
class AgentDecision(TypedDict):
    route_decision: Literal["tools", "read_page", "finish", "wait"]
    tool_input: str
    tool_name: Optional[str]
    message: str = Field(...,description="Short description of the action taken by the agent")
class AgentState(TypedDict):
    goal: str
    plan: List[str]
    current_plan_step: int
    agent_router: str
    messages: Annotated[Sequence[BaseMessage], "Conversation history"]
    agent_decision:AgentDecision
    steps: int
    max_steps: int
    last_action: str

    current_url: str
    page_content: str

    dom_candidates: List[Dict[str, Any]]
    chosen_element: Optional[Dict[str, Any]]

    all_actions: List[str]
    visited: set
    
    needs_interaction: bool

# ---------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------
class PlanOutput(BaseModel):
    plan: List[str] = Field(
        description="Ordered list of high-level browser actions"
    )
async def planner_node(state: AgentState):
    logger.info("Planning high-level steps")

    prompt = get_prompt("planner_prompt")

    structured_llm = llm.with_structured_output(PlanOutput)

    result: PlanOutput = await structured_llm.ainvoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Goal: {state['goal']}"),
        ]
    )
    state["plan"]=result.plan
    state["current_plan_step"]=0
    state["messages"]=state["messages"] + [result]
    print("planner_node:::::", result)

    return state

# ---------------------------------------------------------------------
# Agent (handles ALL interactions via tool calls)
# ---------------------------------------------------------------------
from typing import TypedDict, Literal


import json

async def agent_node(state: AgentState):
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

    plan_step = (
        state["plan"][state["current_plan_step"]]
        if state["current_plan_step"] < len(state["plan"])
        else "Finish the task"
    )

    elements_info = ""
    if state.get("dom_candidates"):
        elements_info = "\n\nAVAILABLE INTERACTIVE ELEMENTS:\n"
        for el in state["dom_candidates"][:10]:
            elements_info += f"[{el['id']}] {el['type']} - {el['label'][:50]}\n"

    chosen_info = ""
    if state.get("chosen_element"):
        el = state["chosen_element"]
        chosen_info = f"""
CHOSEN ELEMENT TO INTERACT WITH:
- ID: {el['id']}
- Type: {el['type']}
- Label: {el['label']}
- Selector: {el['selector']}
"""

    user_prompt = f"""
GOAL:
{state['goal']}

CURRENT PLAN STEP:
{plan_step}

STATUS:
- Step: {state['steps']} / {state['max_steps']}
- Current URL: {state['current_url'] or 'none'}

{chosen_info}
{elements_info}

PAGE PREVIEW:
{state['page_content'][:800]}
"""
    structured_llm=llm.with_structured_output(AgentDecision)
    response = await structured_llm.ainvoke(
        [
            SystemMessage(content=get_prompt("navigate_prompt")),
            HumanMessage(content=user_prompt),
        ]
    )

    try:
        decision = response
        print("agent_node:::::", decision)
        yo={
        **state,
        "messages": state["messages"] + [response],
        "agent_decision": decision,
        "steps": state["steps"] + 1,
        "chosen_element": None,
    }
        print("agent_node state:::::", state)
    except Exception as e:
        raise ValueError(f"Invalid agent decision JSON: {response.content}") from e

    return yo

# ---------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------

async def tool_execution_node(state: AgentState):
    decision = state["agent_decision"]

    # Guard: no tool requested
    if decision.get("route_decision") != "tools":
        return state

    tool_calls = decision.get("tools")
    if not tool_calls:
        return state

    tool_node = ToolNode(tools)

    # Execute tool
    result = await tool_node.ainvoke(
        {"messages": state["messages"] + [decision]}
    )
    print("tool_execution_node:::::", result)
    tool_msg: ToolMessage = result["messages"][-1]
    tool_name = tool_calls[0]["name"]

    # Generic state updates
    print("tool_execution_node state:::::", state)
    next_state = {
        **state,
        "messages": state["messages"] + result["messages"],
        "last_action": tool_name,
        "needs_interaction": tool_name in {
            "navigate",
            "click_element",
            "type_text",
            "scroll",
        },
    }
    print("tool_execution_node state:::::", state)
    # Update URL ONLY if tool exposes it explicitly
    if hasattr(tool_msg, "metadata"):
        url = tool_msg.metadata.get("current_url")
        if url:
            next_state["current_url"] = url

    return next_state

# ---------------------------------------------------------------------
# Read page
# ---------------------------------------------------------------------

async def read_page_node(state: AgentState):
    """Reads the current page content after navigation or interaction"""
    browser = await get_browser()
    page_content = await browser.read()
    
    logger.info("Page content read")
    
    return {
        **state,
        "page_content": page_content,
        "needs_interaction": False,
    }

# ---------------------------------------------------------------------
# DOM extractor - IMPROVED
# ---------------------------------------------------------------------

async def get_stable_selector(el):
    """Generate unique selector for element"""
    # Try aria-label first (most reliable)
    aria_label = await el.get_attribute("aria-label")
    if aria_label:
        return f"[aria-label='{aria_label}']"
    
    # Try href for links
    href = await el.get_attribute("href")
    if href and href.startswith("/"):
        # Use href as selector
        escaped_href = href.replace("'", "\\'")
        return f"a[href='{escaped_href}']"
    
    # Try id (but only if unique)
    el_id = await el.get_attribute("id")
    if el_id and el_id not in ["video-title"]:  # Skip common non-unique IDs
        return f"#{el_id}"
    
    # Try name
    name = await el.get_attribute("name")
    if name:
        return f"[name='{name}']"
    
    # Fallback to tag
    tag = await el.evaluate("el => el.tagName.toLowerCase()")
    return tag

async def extract_dom_node(state: AgentState):
    browser = await get_browser()
    page = browser.page

    candidates = []
    idx = 1

    # Inputs (only if not already filled)
    for el in await page.locator("input, textarea").all():
        try:
            if not await el.is_visible():
                continue
            
            # Skip if already has value (search already done)
            value = await el.input_value()
            if value and len(value) > 0:
                continue
                
            selector = await get_stable_selector(el)
            label = await el.get_attribute("aria-label") or await el.get_attribute("placeholder") or ""
            name = await el.get_attribute("name") or ""
            
            candidates.append({
                "id": idx,
                "type": "input",
                "label": label or name or "input field",
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
            if not text or len(text) > 100:  # Skip very long text
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

    # Links - prioritize video links on YouTube
    link_count = 0
    for el in await page.locator("a").all():
        try:
            if link_count >= 20:  # Limit total links
                break
                
            if not await el.is_visible():
                continue
            
            text = (await el.inner_text()).strip()
            href = await el.get_attribute("href") or ""
            aria_label = await el.get_attribute("aria-label") or ""
            
            if not (text or aria_label) or not href:
                continue
            
            # Prioritize video links on YouTube
            if "/watch?" in href and "youtube.com" in state.get("current_url", ""):
                # This is a video link
                display_text = aria_label if aria_label else text
                if len(display_text) > 100:
                    display_text = display_text[:100] + "..."
                
                # Use aria-label as selector since it's unique
                if aria_label:
                    candidates.append({
                        "id": idx,
                        "type": "link",
                        "label": display_text,
                        "selector": f"a[aria-label='{aria_label}']",
                    })
                    idx += 1
                    link_count += 1
            elif text and href and len(text) < 100:
                # Regular link
                candidates.append({
                    "id": idx,
                    "type": "link",
                    "label": text[:100],
                    "selector": f"a[href='{href}']" if href.startswith("/") else "a",
                })
                idx += 1
                link_count += 1
        except Exception:
            pass

    logger.info(f"Extracted {len(candidates)} interactive elements")
    
    return {
        **state,
        "dom_candidates": candidates,
    }

# ---------------------------------------------------------------------
# LLM chooser - IMPROVED
# ---------------------------------------------------------------------

async def choose_element_node(state: AgentState):
    if not state["dom_candidates"]:
        logger.info("No DOM candidates to choose from")
        return {**state, "chosen_element": None}

    # Build detailed candidate list
    candidates_text = "\n".join(
        f"[{c['id']}] {c['type']:8} | {c['label']}"
        for c in state["dom_candidates"]
    )

    plan_step = (
        state["plan"][state["current_plan_step"]]
        if state["current_plan_step"] < len(state["plan"])
        else "Complete the task"
    )

    response = await llm.ainvoke(
        [
            SystemMessage(
                content="""
You are choosing which element to interact with next.

Analyze the goal and current plan step carefully.
Choose the element ID that BEST matches what needs to be done NOW.

IMPORTANT:
- If the search is already done (URL shows search results), DO NOT choose the search box
- Choose video links, buttons, or other interactive elements instead
- Reply with ONLY a number (the element ID) or NONE
- No explanation, just the number
"""
            ),
            HumanMessage(
                content=f"""
GOAL: {state['goal']}

CURRENT PLAN STEP: {plan_step}

CURRENT URL: {state['current_url']}

LAST ACTION: {state['last_action']}

AVAILABLE ELEMENTS:
{candidates_text}

Which element ID should be interacted with next?
"""
            ),
        ]
    )

    choice = response.content.strip().upper()
    chosen = None
    
    print("choose_element_node:::::", choice)
    
    # Parse the choice
    if choice == "NONE":
        chosen = None
    elif choice.isdigit():
        chosen = next(
            (c for c in state["dom_candidates"] if c["id"] == int(choice)),
            None,
        )
    
    if chosen:
        logger.info(f"Chose element: [{chosen['id']}] {chosen['type']} - {chosen['label'][:50]}")
    else:
        logger.info("No element chosen")

    return {
        **state,
        "chosen_element": chosen,
        "messages": state["messages"] + [response],
    }

# ---------------------------------------------------------------------
# Verifier - IMPROVED
# ---------------------------------------------------------------------

async def verifier_node(state: AgentState):
    plan_step = (
        state["plan"][state["current_plan_step"]]
        if state["current_plan_step"] < len(state["plan"])
        else "Complete the task"
    )
    
    verdict = await llm.ainvoke(
        [
            SystemMessage(content="Answer only 'yes' or 'no'. Be strict - only say yes if real progress was made."),
            HumanMessage(
                content=f"""
GOAL: {state['goal']}

CURRENT PLAN STEP: {plan_step}

LAST ACTION: {state['last_action']}

CURRENT URL: {state['current_url']}

PAGE PREVIEW:
{state['page_content'][:600]}

Did the last action successfully complete or make progress on the current plan step?

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
    print("verifier_node:::::", verdict_text)
    
    if "yes" in verdict_text:
        # Success: move to next plan step
        new_step = min(state["current_plan_step"] + 1, len(state["plan"]))
        logger.info(f"Plan step {state['current_plan_step']} completed, moving to step {new_step}")
        return {
            **state,
            "current_plan_step": new_step,
        }
    else:
        # Failure: don't advance plan, but increment step counter
        logger.info(f"Plan step {state['current_plan_step']} not completed, retrying")
        return {
            **state,
        }

# ---------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------

def agent_router(state: AgentState):
    route = state["agent_decision"]
    route= route["route_decision"]
    print("agent_router:::::", route)
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

def verifier_router(state: AgentState):
    if state["current_plan_step"] >= len(state["plan"]):
        return END
    return "agent"
# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

graph = StateGraph(AgentState)

graph.add_node("planner", planner_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)
graph.add_node("read_page", read_page_node)
graph.add_node("verifier", verifier_node)
graph.add_node("extract_dom", extract_dom_node)
graph.add_node("choose_element", choose_element_node)

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
graph.add_edge("read_page", "extract_dom")
graph.add_edge("extract_dom", "choose_element")
graph.add_edge("choose_element", "agent")

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
        "needs_interaction": False,
    }

    try:
        async for _ in app.astream(state, stream_mode="values"):
            print("\n" + "="*80)
            print("FULL STATE AFTER STEP")
            print(state)
    finally:
        browser = await get_browser()
        await browser.close()
        log_separator("AGENT RUN END")