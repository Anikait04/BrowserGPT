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

# --- Pydantic Models ---
class BrowserAction(BaseModel):
    tool: str = Field(..., description="Tool name (navigate, type, click, read, etc.)")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    step_description: str = Field(..., description="Simple step description like 'navigate to google'")

class IntentParseResult(BaseModel):
    actions: List[BrowserAction] = Field(..., description="Sequential browser actions")
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
    extracted_data: Annotated[Dict[str, Any], "Data collected so far"]
    task_complete: bool
    validation_result: Dict[str, Any]

# --- LLM Setup ---
MODEL_NAME = "openai/gpt-oss-20b:free"

def create_llm(model_name: str, bind_tools: bool = False) -> ChatOpenAI:
    """Create LLM instance"""
    logger.info(f"Initializing LLM: {model_name}")
    llm = ChatOpenAI(
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=2
    )
    return llm.bind_tools(tools) if bind_tools else llm

intent_llm = create_llm(MODEL_NAME, bind_tools=False)
execution_llm = create_llm(MODEL_NAME, bind_tools=True)

# --- Node Functions ---
async def intent_parser_node(state: AgentState) -> AgentState:
    """Parse user goal into structured plan"""
    goal = state.get("user_query", "")
    logger.info(f"\n{'='*60}\nINTENT PARSER: {goal}\n{'='*60}")
    
    try:
        structured_llm = intent_llm.with_structured_output(IntentParseResult)
        messages = [
            SystemMessage(content=get_prompt("intent_system_prompt")),
            HumanMessage(content=f"User goal: {goal}")
        ]
        
        parsed = await structured_llm.ainvoke(messages)
        
        state["high_level_goal"] = parsed.high_level_goal
        state["current_plan"] = [action.step_description for action in parsed.actions]
        state["steps"] = 0
        state["parsed_actions"] = parsed.actions
        
        logger.info(f" Parsed {len(parsed.actions)} actions (confidence: {parsed.confidence:.2f})")
        logger.info(f" Plan: {'  '.join(state['current_plan'])}")
        
        return state
        
    except Exception as e:
        logger.exception(f"Intent parsing failed: {e}")
        state["last_action"] = "intent_parse_failed"
        state["task_complete"] = False
        return state

async def execute_step_node(state: AgentState) -> AgentState:
    """Execute next step in plan using tool-calling LLM"""
    plan = state.get("current_plan", [])
    if not plan:
        logger.warning("No steps remaining")
        state["task_complete"] = True
        return state
    
    next_step = plan[0]
    logger.info(f"\n{'='*60}\nEXECUTE STEP: {next_step}\n{'='*60}")
    
    try:
        context = f"""
        Current Step: {next_step}
        Plan: {'  '.join(plan)}
        Progress: {state.get('steps', 0)}/{state.get('max_steps', 10)}
        """
        
        messages = [
            SystemMessage(content=get_prompt("execution_system_prompt")),
            HumanMessage(content=context),
            HumanMessage(content=f"Execute: {next_step}")
        ]
        
        response = await execution_llm.ainvoke(messages)
        
        # Process tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                state["steps"] += 1
                state["last_action"] = tool_name
                state["all_actions"] = state.get("all_actions", []) + [
                    f"{tool_name}({tool_args})"
                ]
                
                # Execute tool
                tool = next((t for t in tools if t.name == tool_name), None)
                if tool:
                    try:
                        result = await tool.ainvoke(tool_args)
                        state["messages"] = state.get("messages", []) + [
                            ToolMessage(content=str(result), tool_call_id=tool_call['id'])
                        ]
                        logger.info(f" {tool_name} executed")
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
        
        state["current_plan"] = plan[1:]  # Remove executed step
        return state
        
    except Exception as e:
        logger.exception(f"Step execution failed: {e}")
        state["task_complete"] = True
        return state

async def read_and_filter_node(state: AgentState) -> AgentState:
    """Read page and filter only relevant content"""
    logger.info(f"\n{'='*60}\nREAD & FILTER PAGE\n{'='*60}")
    
    try:
        # FIX: Add await and use correct method
        browser = await get_browser()
        raw_content = await browser.read()
        
        structured_llm = intent_llm.with_structured_output(PageRelevanceResult)
        messages = [
            SystemMessage(content=get_prompt("content_filter_prompt")),
            HumanMessage(content=f"""
            Query: {state.get('user_query')}
            Goal: {state.get('high_level_goal')}
            
            Page Content:
            {raw_content[:8000]}
            """)
        ]
        
        analysis = await structured_llm.ainvoke(messages)
        
        state["page_content"] = analysis.relevant_content
        state["validation_result"] = {
            "is_relevant": analysis.is_relevant,
            "confidence": analysis.confidence,
            "next_action": analysis.next_action
        }
        
        logger.info(f" Content filtered ({len(analysis.relevant_content)} chars)")
        logger.info(f" Relevant: {analysis.is_relevant}")
        logger.info(f" Next: {analysis.next_action}")
        print("state::::", state)
        return state
        
    except Exception as e:
        logger.exception(f"Read & filter failed: {e}")
        return state

async def validate_data_node(state: AgentState) -> AgentState:
    """Validate extracted data is correct and complete"""
    logger.info(f"\n{'='*60}\nVALIDATE DATA\n{'='*60}")
    
    extracted_data = state.get("extracted_data", {})
    validation_tools = [{
        "name": "validate_data_complete",
        "description": "Validate extracted data matches requirements",
        "parameters": {
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "confidence": {"type": "number"},
                "missing_fields": {"type": "array", "items": {"type": "string"}}
            }
        }
    }]
    
    try:
        validation_llm = execution_llm.bind_tools(validation_tools)
        messages = [
            SystemMessage(content=get_prompt("validation_prompt")),
            HumanMessage(content=f"""
            Query: {state.get('user_query')}
            Extracted: {json.dumps(extracted_data, indent=2)}
            """)
        ]
        
        response = await validation_llm.ainvoke(messages)
        
        # Determine if validation passed
        is_valid = any(
            tool_call['name'] == 'validate_data_complete' and tool_call['args'].get('is_valid')
            for tool_call in response.tool_calls
        )
        
        state["validation_result"] = {
            "is_valid": is_valid,
            "timestamp": state.get("steps", 0)
        }
        
        if is_valid:
            logger.info(" Data validation PASSED")
            state["last_action"] = "data_validated"
        else:
            logger.warning("✗ Data validation FAILED")
            state["last_action"] = "data_invalid"
        
        return state
        
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        state["last_action"] = "validation_error"
        return state

async def replanning_node(state: AgentState) -> AgentState:
    """Generate new plan when data is wrong or incomplete"""
    logger.info(f"\n{'='*60}\nREPLANNING\n{'='*60}")
    
    try:
        structured_llm = intent_llm.with_structured_output(IntentParseResult)
        messages = [
            SystemMessage(content=get_prompt("replanning_prompt")),
            HumanMessage(content=f"""
            Original: {state.get('user_query')}
            Previous Plan: {'  '.join(state.get('all_actions', [])[-5:])}
            Error: Data extraction failed or was incorrect
            
            Create a new plan that:
            1. Targets different pages/elements
            2. Uses alternative strategies
            3. Handles edge cases
            """)
        ]
        
        new_plan = await structured_llm.ainvoke(messages)
        
        state["current_plan"] = [action.step_description for action in new_plan.actions]
        state["steps"] = 0  # Reset step count for new plan
        
        logger.info(f" New plan: {len(new_plan.actions)} steps")
        logger.info(f" New plan: {'  '.join(state['current_plan'])}")
        
        return state
        
    except Exception as e:
        logger.exception(f"Replanning failed: {e}")
        state["task_complete"] = True
        return state

async def final_extraction_node(state: AgentState) -> AgentState:
    """Perform final structured data extraction"""
    logger.info(f"\n{'='*60}\nFINAL EXTRACTION\n{'='*60}")
    
    try:
        structured_llm = execution_llm.with_structured_output(ExtractedDataResult)
        messages = [
            SystemMessage(content=get_prompt("final_extraction_prompt")),
            HumanMessage(content=f"""
            Query: {state.get('user_query')}
            Page Content: {state.get('page_content', '')[:6000]}
            
            Extract data exactly as specified in the query.
            Format it properly and note any missing fields.
            """)
        ]
        
        result = await structured_llm.ainvoke(messages)
        
        state["extracted_data"] = result.extracted_data
        state["task_complete"] = result.success
        
        logger.info(f" Extraction success: {result.success}")
        logger.info(f" Data keys: {list(result.extracted_data.keys())}")
        
        if result.missing_fields:
            logger.warning(f"✗ Missing fields: {result.missing_fields}")
        
        return state
        
    except Exception as e:
        logger.exception(f"Final extraction failed: {e}")
        state["task_complete"] = True
        return state

# FIX: Add missing update_progress_node
async def update_progress_node(state: AgentState) -> AgentState:
    """Update progress and check if task should continue"""
    logger.info(f"\n{'='*60}\nUPDATE PROGRESS\n{'='*60}")
    
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", 10)
    
    logger.info(f"Progress: {steps}/{max_steps} steps")
    logger.info(f"Actions taken: {len(state.get('all_actions', []))}")
    
    # Check if we should continue or finish
    if steps >= max_steps:
        state["task_complete"] = True
        logger.warning("Max steps reached")
    
    return state

# --- Routing Logic ---
def intelligent_router(state: AgentState) -> str:
    """Routes to appropriate node based on state"""
    logger.info(f"\n--- ROUTING: last_action={state.get('last_action')} ---")
    
    # Termination checks
    if state.get("task_complete"):
        logger.info(" END (task complete)")
        return END
    
    if state.get("steps", 0) >= state.get("max_steps", 10):
        logger.warning(f" data_extraction (max steps {state.get('max_steps', 10)} reached)")
        return "data_extraction"
    
    # Data quality routing
    validation = state.get("validation_result", {})
    last_action = state.get("last_action", "")
    
    if last_action == "data_invalid":
        logger.info(" replanning (data invalid)")
        return "replanning"
    
    if last_action == "data_validated" and not state.get("extracted_data"):
        logger.info(" data_extraction (data validated, ready to extract)")
        return "data_extraction"
    
    if last_action == "data_validated" and state.get("extracted_data"):
        logger.info(" END (data extracted and validated)")
        return END
    
    # Progress routing
    if not state.get("current_plan"):
        logger.info(" data_extraction (plan complete)")
        return "data_extraction"
    
    # Default
    logger.info(" execute_step (continue plan)")
    return "execute_step"

# --- Build Graph ---
def create_agent_graph(max_steps: int = 10) -> StateGraph:
    """Create graph with intelligent routing"""
    logger.info("Building enhanced agent graph...")
    
    workflow = StateGraph(AgentState)
    
    # FIX: Use consistent node names
    workflow.add_node("intent_parser", intent_parser_node)
    workflow.add_node("execute_step", execute_step_node)
    workflow.add_node("read_and_validate", read_and_filter_node)
    workflow.add_node("data_extraction", final_extraction_node)
    workflow.add_node("replanning", replanning_node)
    workflow.add_node("update_progress", update_progress_node)  # FIX: Now defined!
    
    # Define base flow
    workflow.add_edge(START, "intent_parser")
    workflow.add_edge("intent_parser", "execute_step")
    workflow.add_edge("execute_step", "read_and_validate")
    workflow.add_edge("replanning", "read_and_validate")
    workflow.add_edge("data_extraction", "update_progress")
    
    # FIX: Correct conditional edges with matching names
    workflow.add_conditional_edges(
        "read_and_validate",
        intelligent_router,
        {
            "execute_step": "execute_step",
            "data_extraction": "data_extraction",  # FIX: Now matches node name!
            "replanning": "replanning",
            "update_progress": "update_progress",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "update_progress",
        intelligent_router,
        {
            "execute_step": "execute_step",
            "data_extraction": "data_extraction",  # FIX: Now matches node name!
            "replanning": "replanning",
            END: END
        }
    )
    
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
        goal = "Find the latest Python release version and download URL from python.org"
        result = await run_agent(goal, max_steps=8)
        
        print("\n" + "="*60)
        print("FINAL RESULTS:")
        print("="*60)
        if result.get("extracted_data"):
            print(json.dumps(result["extracted_data"], indent=2))
        else:
            print("No data extracted")
    
    asyncio.run(main())