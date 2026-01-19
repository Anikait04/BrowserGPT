import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from browsertools import tools, get_browser
from prompt import get_prompt
from logs import logger, log_separator

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b:free"


def save_successful_run(goal: str, actions: list[str]):
    os.makedirs("success_prev_run", exist_ok=True)

    safe_goal = goal.replace(" ", "_")[:50]
    filename = f"success_prev_run/{safe_goal}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"GOAL:\n{goal}\n\n")
        f.write("SUCCESSFUL ACTIONS:\n")
        for i, action in enumerate(actions, 1):
            f.write(f"{i}. {action}\n")

    logger.info(f"Saved successful run → {filename}")


def create_llm(model_name: str):
    """Create ChatOpenAI LLM instance for OpenRouter"""
    logger.info(f"Initializing LLM: {model_name}")

    llm = ChatOpenAI(
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2
    )

    logger.debug("Binding browser tools to LLM")
    return llm.bind_tools(tools)


llm = create_llm(MODEL_NAME)


class AgentState(TypedDict):
    goal: str
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    steps: int
    max_steps: int
    last_action: str
    page_content: str
    all_actions: Annotated[Sequence[str], "All actions taken so far"]


async def agent_node(state: AgentState):
    """Agent decision node"""
    if state["steps"] >= state["max_steps"]:
        logger.warning(f"Max steps ({state['max_steps']}) reached")
        return {
            "messages": state["messages"] + [AIMessage(content="Max steps reached, stopping")],
            "steps": state["steps"] + 1,
            "last_action": "max_steps_reached"
        }

    prompt_navigate = get_prompt("navigate_prompt")

    recent_actions = state.get("last_action", "none")
    page_preview = state.get("page_content", "No content yet")[:600]

    user_prompt = f"""🎯 GOAL: {state['goal']}

📊 STATUS:
- Step: {state['steps']}/{state['max_steps']}
- Last action: {recent_actions}
- Page preview: {page_preview}

❓ What is the ONE tool you should call now?"""

    messages = [
        SystemMessage(content=prompt_navigate),
        HumanMessage(content=user_prompt)
    ] + list(state["messages"][-4:])

    logger.info(f"Agent thinking (Step {state['steps'] + 1}/{state['max_steps']})")

    try:
        response = await llm.ainvoke(messages)
    except Exception:
        logger.exception("LLM invocation failed")
        raise

    return {
        "messages": state["messages"] + [response],
        "steps": state["steps"] + 1,
        "last_action": state.get("last_action", ""),
        "page_content": state.get("page_content", "")
    }


async def tool_execution_node(state: AgentState):
    """Execute tools and capture results"""
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        logger.debug("No tool calls detected")
        return state

    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [last_msg]})

    tool_name = last_msg.tool_calls[0]["name"]
    tool_output_msg = result["messages"][-1]
    tool_result = tool_output_msg.content

    logger.info(f"Tool executed: {tool_name}")
    logger.debug(f"Tool output preview: {tool_result[:200]}")

    page_content = state.get("page_content", "")
    if tool_name == "read_page":
        page_content = tool_result

    clean_result = tool_result[:200].replace("\n", " ")
    action_log = f"{tool_name}: {clean_result}"

    return {
        "messages": state["messages"] + result["messages"],
        "steps": state["steps"],
        "last_action": tool_name,
        "page_content": page_content,
        "all_actions": state["all_actions"] + [action_log]
    }


def router(state: AgentState):
    """Route graph execution"""
    last_msg = state["messages"][-1]

    if state["steps"] >= state["max_steps"]:
        logger.warning("Stopping graph: max steps reached")
        return END

    if state.get("last_action") == "finish_task":
        save_successful_run(state["goal"], state["all_actions"])
        logger.info("Task completed successfully")
        return END

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_name = last_msg.tool_calls[0]["name"]
        logger.info(f"Routing to tool: {tool_name}")
        return "tools"

    logger.warning("No tool called, ending execution")
    return END


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", router, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile()

png_bytes = app.get_graph().draw_mermaid_png()
with open("agent_flow.png", "wb") as f:
    f.write(png_bytes)

logger.info("Agent flow graph saved as agent_flow.png")


async def run_agent(goal: str, max_steps: int = 30):
    """Run the browser agent"""
    log_separator("AGENT RUN START")

    logger.info(f"Goal: {goal}")
    logger.info(f"Max steps: {max_steps}")
    logger.info(f"Model: {MODEL_NAME}")

    try:
        initial_state = {
            "goal": goal,
            "messages": [],
            "steps": 0,
            "max_steps": max_steps,
            "last_action": "none",
            "page_content": "",
            "all_actions": []
        }

        async for _ in app.astream(initial_state, stream_mode="values"):
            pass

        logger.info("Agent execution finished successfully")

    except Exception:
        logger.exception("Agent execution failed")

    finally:
        try:
            browser = await get_browser()
            await browser.close()
            logger.info("Browser closed cleanly")
        except Exception:
            logger.warning("Browser cleanup failed", exc_info=True)

        log_separator("AGENT RUN END")
