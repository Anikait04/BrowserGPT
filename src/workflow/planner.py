# planner.py — planner node for the delegated architecture.

from langchain_core.prompts import ChatPromptTemplate

from logs import logger
from src.workflow.agent_state import AgentState
from src.workflow.llm import get_llm
from src.workflow.prompt import PLANNER_PROMPT_V2
from src.workflow.schemas import Plan


async def planner_node(state: AgentState) -> dict:
    """Break the user's goal into an ordered list of high-level steps."""
    logger.info("[PLANNER] Creating execution plan")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_PROMPT_V2),
            ("human", "Goal: {goal}"),
        ]
    )

    chain = prompt | get_llm().with_structured_output(Plan)

    result: Plan = await chain.ainvoke({"goal": state["goal"].strip()})
    logger.info(f"[PLANNER] Plan created with {len(result.steps)} step(s): {result.steps}")
    logger.info(f"[PLANNER] Goal restated: {result.goal}")

    existing_messages = list(state.get("messages", []) or [])

    return {
        "entire_plan": result.steps,
        "step_count": 0,
        "messages": existing_messages + [result.goal],
    }
