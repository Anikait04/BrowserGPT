#nodes.py

import uuid
from dotenv import load_dotenv
from langgraph.types import interrupt
from langchain_core.messages import HumanMessage,AIMessage, SystemMessage, SystemMessage,ToolMessage
from src.workflow.agent_state import AgentState
from src.workflow.utils import plan_steps_update
from src.workflow.browsertools import tools, get_browser
from src.workflow.prompt import get_prompt
from logs import logger
from pydantic import TypeAdapter 
from typing import cast
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from src.workflow.structured import AgentDecision, DOMElement, PlanOutput
import re
from src.workflow.utils import plan_steps_update
from config import _PAGE_CACHE
from src.workflow.llm import CustomLLMClient
load_dotenv()
from langchain_core.runnables import RunnableConfig
# singleton
client = CustomLLMClient()
async def planner_node(state: AgentState):
    logger.info("Planning high-level steps")

    system_prompt = get_prompt("planner_prompt")
    user_prompt = f"Goal: {state['goal']}".strip()



    result = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured=True,
        schema=PlanOutput.model_json_schema()
    )
    print("Raw planner_node result:::", result)
    # result is a dict from res.json() — parse it into PlanOutput
    result=result.get("response",result)
    result = PlanOutput(**result)

    print("planner result:::", result)

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
    schema = TypeAdapter(AgentDecision).json_schema()
    system_prompt=get_prompt("navigate_prompt")
    result = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured=True,
        schema=schema
    )
    
    result=result.get("response",result)
    response = AgentDecision(**result)
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

 
async def tool_execution_node(state: AgentState, config: RunnableConfig):
    logger.info("Started Tool Execution Node")
    tool_name = state.get("tool_name")
    if not tool_name:
        return state
 
    tool_input  = state.get("tool_input")
    element_id  = state.get("element_id")
    task_id     = state.get("task_id")       # None when not streaming
    step        = state.get("steps", 0)
    max_steps   = state.get("max_steps", 30)
 
    tool_selector = None
    if element_id is not None:
        selected = next(
            (el for el in state.get("chosen_element", []) if el["id"] == element_id),
            None
        )
        if selected:
            tool_selector = selected["selector"]
        else:
            return state
 
    if tool_name == "navigate":
        args = {"url": tool_input}
    elif tool_name == "type_text":
        args = {"selector": tool_selector, "value": tool_input}
    elif tool_name == "type_and_enter":
        args = {"selector": tool_selector, "value": tool_input}
    elif tool_name == "click_element":
        args = {"selector": tool_selector}
    else:
        args = {}
 
    ai_message = AIMessage(
        content="",
        tool_calls=[{"id": uuid.uuid4().hex, "name": tool_name, "args": args}],
    )
    print("tool_call input:::", ai_message)
    tool_node = ToolNode(tools)
 
    result = await tool_node.ainvoke({"messages": [ai_message]}, config)
    tool_msg: ToolMessage = result["messages"][-1]
 
    existing_messages = state.get("messages", [])
    updated_messages = (
        existing_messages + [tool_msg.content]
        if isinstance(existing_messages, list)
        else [tool_msg.content]
    )
 
    if tool_name == "navigate" and tool_input:
        state["current_url"] = tool_input
 
    browser = await get_browser()
    current_page_url = browser.page.url
 
    # ── Screenshot → SSE push (only when a task_id exists i.e. streaming mode) ──
    if task_id:
        try:
            import base64
            from src.routers.agent_router import push_screenshot
 
            screenshot_bytes = await browser.page.screenshot(type="jpeg", quality=70)
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
 
            await push_screenshot(
                task_id=task_id,
                screenshot_b64=screenshot_b64,
                step=step,
                max_steps=max_steps,
                action=tool_name,
                url=current_page_url,
                message=f"{tool_name} — {tool_input or tool_selector or ''}".strip(" —"),
            )
        except Exception as e:
            logger.warning(f"Screenshot push failed (non-fatal): {e}")
    # ─────────────────────────────────────────────────────────────────────────
 
    return {
        **state,
        "messages": updated_messages,
        "last_action": tool_name,
        "current_url": current_page_url,
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
        # 1. aria-label (most stable)
        aria = await el.get_attribute("aria-label")
        if aria:
            return f"[aria-label='{aria}']"

        # 2. name attribute
        name = await el.get_attribute("name")
        if name:
            return f"[name='{name}']"

        # 3. placeholder
        placeholder = await el.get_attribute("placeholder")
        if placeholder:
            return f"[placeholder='{placeholder}']"

        # 4. data-testid
        testid = await el.get_attribute("data-testid")
        if testid:
            return f"[data-testid='{testid}']"

        # 5. type + role combo
        el_type = await el.get_attribute("type")
        role = await el.get_attribute("role")
        if el_type:
            return f"input[type='{el_type}']"
        if role:
            return f"[role='{role}']"

        # 6. Last resort: escape the ID properly
        el_id = await el.get_attribute("id")
        if el_id:
            return f'[id="{el_id}"]'  # attribute selector, not #id — avoids CSS parsing issues

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
    with open("elements.txt", "w",encoding="utf-8") as f:
        f.write(str(elements))
    
    plan_step = state.get("current_action", "")
    system_prompt=get_prompt("choose_and_observe_prompt")
    user_prompt = f"""
    GOAL: {state['goal']}
    STEP: {plan_step}

    AVAILABLE ELEMENTS:
    {elements}
    """.strip()
    # print("elements::::",elements)
    result = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured=True,
        schema=DOMElement.model_json_schema()
    )
    
    result=result.get("response",result)
    response = DOMElement(**result)

    print("planner result:::", result)
    # structured_llm_doms=llm.with_structured_output(DOMElement)

    # response = await structured_llm_doms.ainvoke(
    #     [
    #         SystemMessage(content=PROMPTY),
    #                     HumanMessage(
    #                         content=f"""
    #         GOAL: {state['goal']}
    #         STEP: {plan_step}

    #         AVAILABLE ELEMENTS:
    #         {elements}
    #         """
    #         ),
    #     ]
    # )
    selected_element = next(
    (el for el in elements if el["id"] == response.id),
    None
)
    message_decision = response.message
    if not selected_element:
        return {
            **state,
            "chosen_element": None,
            "messages": state.get("messages", []) + [
                AIMessage(content="Selected ID not found in candidates.")
            ],
        }
    # def tokenize(text):
    #     return set(text.lower().split())
    # doms_selected=chosen_label
    # chosen_tokens = tokenize(chosen_label)

    # def score(el):
    #     label_tokens = tokenize(el["label"])
    #     overlap = len(chosen_tokens & label_tokens)
    #     s = overlap * 5

    #     if el["type"] == "button":
    #         s += 3
    #     if el["type"] == "input":
    #         s += 2

    #     return s
    # # with open("elements.txt", "w",encoding="utf-8") as f:
    # #     f.write(str(elements))
    # ranked = sorted(elements, key=score, reverse=True)

    # top_5 = ranked[:20]
    print("Selected element:::",selected_element)
    return {
        **state,
        "chosen_element": [selected_element],
        "messages": state["messages"] + [message_decision],
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
    
    system_prompt = (
    "Answer only 'yes' or 'no'. "
    "Be strict — only say yes if real progress was made."
)

    user_prompt = f"""
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
    """.strip()
    verdict = await client.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured=False
    )
    
#     verdict = await llm.ainvoke(
#         [
#             SystemMessage(
#                 content="Answer only 'yes' or 'no'. Be strict — only say yes if real progress was made."
#             ),
#             HumanMessage(
#                 content=f"""
# GOAL: {state['goal']}

# ENTIRE PLAN: {state['entire_plan']}

# CURRENT PLAN STEP: {plan_step}

# LAST ACTION: {state['last_action']}

# CURRENT URL: {current_url}

# MESSAGES LOG OF ENTIRE PROCESS TILL NOW {state["messages"]}

# Did the last action successfully complete or make progress on the plan?

# Examples:
# - If step is "search for X" and search was performed → yes
# - If step is "click video" and video page opened → yes
# - If step is "search" but search was done again → no (redundant)
# - If an error occurred → no
# """
#             ),
#         ]
#     )
    verdict=verdict.get("response",verdict)
    verdict_text = verdict.lower()
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


async def human_wait_node(state: AgentState):
    logger.info("Started Human Wait Node")
    
    # Show the agent's current status to the human
    summary = f"""
Agent is pausing for human input.

Goal: {state['goal']}
Current Step: {state['step_count']} / {len(state['entire_plan'])}
Current Action: {state.get('current_action', 'N/A')}
Current URL: {state.get('current_url', 'N/A')}
Last Action: {state.get('last_action', 'N/A')}
Progress Verification: {state.get('progress_verification', 'N/A')}

Please provide instruction or type 'continue' to proceed.
"""
    # This pauses execution and waits for human input
    human_input = interrupt(summary)
    
    logger.info(f"Human provided input: {human_input}")
    
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        existing_messages = [existing_messages]
    
    return {
        **state,
        "messages": existing_messages + [HumanMessage(content=str(human_input))],
        "progress_verification": f"Human instruction: {human_input}",
    }