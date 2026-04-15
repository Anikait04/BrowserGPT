
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage, SystemMessage,ToolMessage
from src.workflow.agent_state import AgentState
from src.workflow.utils import plan_steps_update
from src.workflow.browsertools import tools, get_browser
from src.workflow.prompt import get_prompt
from logs import logger
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from src.workflow.structured import AgentDecision, PlanOutput
import re
from src.workflow.utils import plan_steps_update
from config import _PAGE_CACHE
from src.workflow.llm import llm_call
load_dotenv()


async def planner_node(state: AgentState):
    logger.info("Planning high-level steps") 
    llm=llm_call()
    prompt = get_prompt("planner_prompt")
    structured_llm = llm.with_structured_output(PlanOutput)
    result= await structured_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Goal: {state['goal']}"),
        ])
    print("planner result:::",result)
    return {
        **state,
        "entire_plan": result.plan,
        "step_count": 0,
        "messages": result.messages,
    }

async def agent_node(state: AgentState):
    logger.info("Started Agent Node")

    llm=llm_call()
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
    print("agent_node response:::",response)
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
    llm=llm_call()
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
    llm=llm_call()
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