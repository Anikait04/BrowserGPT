import os
import asyncio
from dotenv import load_dotenv
from typing import Any, Dict, TypedDict, Sequence, Annotated, List, Optional, Literal
from pydantic import BaseModel, Field, validator
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from enum import Enum
import json
from browsertools import tools, get_browser
from prompt import get_prompt
from logs import logger, log_separator

load_dotenv()

# --- LLM Setup ---
MODEL_NAME = "openai/gpt-oss-20b:free"

def create_llm(model_name: str, bind_tools: bool = False) -> ChatOpenAI:
    """Create LLM instance"""
    logger.info(f"Initializing LLM: {model_name}")
    llm = ChatOpenAI(
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.5,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2
    )
    return llm.bind_tools(tools) if bind_tools else llm

llm = create_llm(MODEL_NAME, bind_tools=False)
llm_tools = create_llm(MODEL_NAME, bind_tools=True)

# --- Pydantic Models ---
class BrowserAction(BaseModel):
    step_description: str = Field(..., description="Simple step description like 'navigate to google'")

class IntentParseResult(BaseModel):
    actions: List[BrowserAction] = Field(..., description="Simple step description like 'navigate to google'")
    current_step: list[str] = Field(None, description="Current Step to take step to take")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    high_level_goal: str = Field(..., description="Simplified goal summary")

class ManageParseResult(BaseModel):
    action: List[BrowserAction] = Field(..., description="Simple step description like 'navigate to google'")
    current_step: list[str] = Field(None, description="Current Step to take step to take")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    high_level_goal: str = Field(..., description="Simplified goal summary")

class PageRelevanceResult(BaseModel):
    """Determines if page content is relevant to the task"""
    is_relevant: bool = Field(..., description="Does this page contain needed data?")
    relevant_content: str = Field(..., description="Only relevant portions of the page")
    confidence: float = Field(..., ge=0.0, le=1.0)
    next_action: Literal["extract", "replan", "continue"] = Field(
        ..., 
        description="What to do next"
    )
    reasoning: str = Field(..., description="Why this decision was made")

class ExtractedDataResult(BaseModel):
    success: bool = Field(..., description="Was extraction successful?")
    extracted_data: Dict[str, Any] = Field(..., description="The actual data")
    missing_fields: List[str] = Field(default_factory=list)
    summary: str = Field(..., description="Summary of extraction")

# --- Agent State ---
class AgentState(TypedDict, total=False):
    user_query: str
    high_level_goal: str
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    steps: int
    max_steps: int
    last_action: str
    page_content: str
    all_actions: Annotated[Sequence[str], "All actions taken so far"]
    current_plan: Annotated[Sequence[str], "Current execution plan"]
    initial_step:str
    extracted_data: Annotated[Dict[str, Any], "Data collected so far"]
    task_complete: bool
    validation_result: Dict[str, Any]


# --- Node Functions ---
async def intent_parser_node(state: AgentState) -> AgentState:
    """Parse user goal into structured plan"""
    goal = state.get("user_query", "")
    logger.info(f"\n{'='*60}\nINTENT PARSER: {goal}\n{'='*60}")
    
    try:
        structured_llm = llm.with_structured_output(IntentParseResult)
        messages = [
            SystemMessage(content=get_prompt("intent_system_prompt")),
            HumanMessage(content=f"User goal: {goal}")
        ]
        
        parsed = await structured_llm.ainvoke(messages)
        
        state["high_level_goal"] = parsed.high_level_goal
        state["current_plan"] = [action.step_description for action in parsed.actions]
        state["initial_step"] = parsed.initial_step
        state["steps"] = 0
        state["parsed_actions"] = parsed.actions
        logger.info(f" Parsed {len(parsed.actions)} actions (confidence: {parsed.confidence:.2f})")
        logger.info(f" Plan: {'  '.join(state['current_plan'])}")
        print("parsed:::", parsed)
        print("state:::", state)
        return state
        
    except Exception as e:
        logger.exception(f"Intent parsing failed: {e}")
        state["last_action"] = "intent_parse_failed"
        state["task_complete"] = False
        return state
    
class PageDecisionResult(BaseModel):
    decision: Literal["read_page", "navigate", "finish"]
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)

async def decide_read_or_navigate_node(state: AgentState) -> AgentState:
    """
    Decide whether to read the current page or navigate to a new one.
    """
    logger.info(f"\n{'='*60}\nDECIDE READ VS NAVIGATE\n{'='*60}")

    plan = state.get("current_plan", [])
    steps = state.get("steps", 0)
    page_content = state.get("page_content")
    last_action = state.get("last_action")

    # Hard stops
    if not plan or steps >= state.get("max_steps", 10):
        state["decision"] = "finish"
        return state

    current_step = plan[0].lower()

    # Heuristic routing (fast + cheap)
    if any(k in current_step for k in ["read", "extract", "analyze", "scrape"]):
        state["decision"] = "read_page"
        state["decision_reason"] = "Current plan step requires reading page content"
        return state

    if any(k in current_step for k in ["navigate", "open", "go to", "search"]):
        state["decision"] = "navigate"
        state["decision_reason"] = "Current plan step requires navigation"
        return state

    # Fallback: use LLM if ambiguous
    structured_llm = llm_tools.with_structured_output(PageDecisionResult)

    messages = [
        SystemMessage(
            content=(
                "You are deciding the next browser action.\n"
                "Choose 'read_page' if current page content is sufficient.\n"
                "Choose 'navigate' if a new page is needed.\n"
                "Choose 'finish' if task is complete."
            )
        ),
        HumanMessage(
            content=f"""
High-level goal: {state.get("high_level_goal")}
Current step: {current_step}
Has page content: {bool(page_content)}
Last action: {last_action}
"""
        ),
    ]

    result = await structured_llm.ainvoke(messages)

    state["decision"] = result.decision
    state["decision_reason"] = result.reason
    state["decision_confidence"] = result.confidence

    return state

async def read_page_node(state: AgentState) -> AgentState:
    """
    Read and store current page content.
    """
    logger.info(f"\n{'='*60}\nREAD PAGE NODE\n{'='*60}")

    browser = get_browser()

    try:
        page_text = await browser.read()  # or read_dom(), read_text(), etc.
        current_step = state.get("current_plan", [])[0] if state.get("current_plan") else "N/A"
        last_action = state.get("last_action", "N/A")
        messages = [
            SystemMessage(
                content=(
                    "You are deciding the next browser action.\n"
                    "Choose 'read_page' if current page content is sufficient.\n"
                    "Choose 'navigate' if a new page is needed.\n"
                    "Choose 'finish' if task is complete."
                )
            ),
            HumanMessage(
                content=f"""
    Current step: {current_step}
    Has page content: {page_text}
    Last action: {last_action}
    """
            ),
        ]

        useful_page_content = await llm.ainvoke(messages)

        state["last_action"] = "read_page"
        state["steps"] += 1

        # Remove completed step
        if state.get("current_plan"):
            state["current_plan"] = state["current_plan"][1:]

        state.setdefault("all_actions", []).append("read_page")

    except Exception as e:
        logger.exception(f"Read page failed: {e}")
        state["last_action"] = "read_failed"

    return state

async def navigate_node(state: AgentState) -> AgentState:
    """
    Navigate to a new page based on the current plan step.
    """
    logger.info(f"\n{'='*60}\nNAVIGATE NODE\n{'='*60}")

    browser = get_browser()
    plan = state.get("current_plan", [])

    if not plan:
        state["last_action"] = "no_navigation_needed"
        return state

    step = plan[0]

    try:
        await browser.navigate(step)
        state["last_action"] = "navigate"
        state["steps"] += 1

        # Clear stale page content
        state["page_content"] = ""

        # Remove completed step
        state["current_plan"] = plan[1:]

        state.setdefault("all_actions", []).append(f"navigate: {step}")

    except Exception as e:
        logger.exception(f"Navigation failed: {e}")
        state["last_action"] = "navigate_failed"

    return state


def create_agent_graph(max_steps: int = 10) -> StateGraph:
    """Create graph with intelligent routing"""
    logger.info("Building enhanced agent graph...")
    
    workflow = StateGraph(AgentState)
    
    # FIX: Use consistent node names
    workflow.add_node("intent_parser", intent_parser_node)
    # workflow.add_node("execute_step", execute_step_node)
    # workflow.add_node("read_and_validate", read_and_filter_node)
    # workflow.add_node("data_extraction", final_extraction_node)
    # workflow.add_node("replanning", replanning_node)
    # workflow.add_node("update_progress", update_progress_node)  # FIX: Now defined!
    
    # Define base flow
    workflow.add_edge(START, "intent_parser")
    workflow.add_edge("intent_parser", END)  # TEMP: Direct to END for now
    # workflow.add_edge("intent_parser", "execute_step")
    # workflow.add_edge("execute_step", "read_and_validate")
    # workflow.add_edge("replanning", "read_and_validate")
    # workflow.add_edge("data_extraction", "update_progress")
    
    # # FIX: Correct conditional edges with matching names
    # workflow.add_conditional_edges(
    #     "read_and_validate",
    #     intelligent_router,
    #     {
    #         "execute_step": "execute_step",
    #         "data_extraction": "data_extraction",  # FIX: Now matches node name!
    #         "replanning": "replanning",
    #         "update_progress": "update_progress",
    #         END: END
    #     }
    # )
    
    # workflow.add_conditional_edges(
    #     "update_progress",
    #     intelligent_router,
    #     {
    #         "execute_step": "execute_step",
    #         "data_extraction": "data_extraction",  # FIX: Now matches node name!
    #         "replanning": "replanning",
    #         END: END
    #     }
    # )
    
    graph = workflow.compile()
    logger.info(" Enhanced graph compiled successfully")
    
    return graph

# --- Main Execution ---
async def run_agent(goal: str, max_steps: int = 10) -> AgentState:
    """Run agent with intelligent data validation"""
    logger.info(f"\n{'='*60}")
    logger.info(f"AGENT START: {goal}")
    logger.info(f"{'='*60}")
    
    graph = create_agent_graph(max_steps)
    
    initial_state: AgentState = {
        "user_query": goal,
        "messages": [],
        "steps": 0,
        "max_steps": max_steps,
        "all_actions": [],
        "current_plan": [],
        "extracted_data": {},
        "task_complete": False,
        "validation_result": {},
        "high_level_goal": "",
        "page_content": ""
    }
    
    try:
        final_state = await graph.ainvoke(initial_state)
        
        logger.info(f"\n{'='*60}")
        logger.info("AGENT COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Steps: {final_state.get('steps')}")
        logger.info(f"Actions: {len(final_state.get('all_actions', []))}")
        logger.info(f"Data extracted: {len(final_state.get('extracted_data', {}))} fields")
        logger.info(f"Task complete: {final_state.get('task_complete')}")
        
        if final_state.get("extracted_data"):
            logger.info(f"\nExtracted Data:")
            for key, value in final_state["extracted_data"].items():
                logger.info(f"  {key}: {str(value)[:100]}...")
        
        return final_state
        
    except Exception as e:
        logger.exception(f"Agent failed: {e}")
        return initial_state

# Example usage
if __name__ == "__main__":
    async def main():
        goal = "summaries of the latest articles on bbc.com technology section"
        result = await run_agent(goal, max_steps=8)
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        if result.get("extracted_data"):
            print(json.dumps(result["extracted_data"], indent=2))
        else:
            print("No data extracted")
    
    asyncio.run(main())