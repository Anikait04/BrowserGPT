import os
import asyncio
from enum import Enum
from typing import TypedDict, Sequence, Annotated, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from browsertools import tools, get_browser
from logs import logger

load_dotenv()

MODEL_NAME = "openai/gpt-oss-20b:free"  # More capable model

# ============================================================================
# ENUMS FOR TYPE SAFETY
# ============================================================================

class ActionStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"

class FailureType(str, Enum):
    ELEMENT_NOT_FOUND = "element_not_found"
    TIMEOUT = "timeout"
    NAVIGATION_ERROR = "navigation_error"
    SELECTOR_ERROR = "selector_error"
    BLOCKED_BY_CAPTCHA = "blocked_by_captcha"
    UNKNOWN = "unknown"

# ============================================================================
# IMPROVED STATE
# ============================================================================

class AgentState(TypedDict):
    goal: str
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    
    # Execution tracking
    steps: int
    max_steps: int
    consecutive_failures: int  # NEW: Track failure streaks
    
    # Page state
    current_url: str
    page_title: str
    page_content: str  # Store more content
    
    # Error handling
    last_error: str | None
    failure_type: FailureType | None
    
    # Success tracking
    all_actions: Annotated[list[str], "All actions taken"]
    successful_actions: Annotated[list[str], "Only successful actions"]

# ============================================================================
# IMPROVED PROMPTS
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are a browser automation agent with access to browser control tools.

**YOUR OBJECTIVE**: Accomplish the user's goal by calling the appropriate tools.

**IMPORTANT RULES**:
1. You have direct access to browser tools - just call them, no intent resolution needed
2. Always call read_page() after navigation or page changes to understand the page
3. Use specific CSS selectors from the page content you've read
4. If an action fails, try an alternative approach (different selector, different tool)
5. Call finish_task() ONLY when the goal is fully achieved

**AVAILABLE TOOLS**: {tool_names}

**STRATEGY**:
- Navigate → Read → Interact → Verify → Complete
- Prefer stable selectors (id, name, data-* attributes)
- Use click_text() when CSS selectors are unreliable
- Wait for page loads using wait_for_navigation() after clicks

**CURRENT GOAL**: {goal}
**CURRENT URL**: {current_url}
**STEPS TAKEN**: {steps}/{max_steps}
"""

# ============================================================================
# IMPROVED LLM SETUP
# ============================================================================

def create_llm(model_name: str):
    """Create LLM with better configuration"""
    logger.info(f"Initializing LLM: {model_name}")
    
    llm = ChatOpenAI(
        model_name=model_name,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.1,  # Lower temperature for more consistent behavior
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=3,
        request_timeout=60
    )
    
    # Bind tools directly - no need for separate intent resolution!
    return llm.bind_tools(tools)

llm = create_llm(MODEL_NAME)

# ============================================================================
# SIMPLIFIED AGENT NODE
# ============================================================================

async def agent_node(state: AgentState):
    """Main agent decision node - simplified!"""
    
    # Check limits
    if state["steps"] >= state["max_steps"]:
        logger.warning(f"Max steps ({state['max_steps']}) reached")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content="Max steps reached")],
        }
    
    # Check failure streak
    if state["consecutive_failures"] >= 3:
        logger.error("Too many consecutive failures, stopping")
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content="Stopping due to repeated failures")
            ],
        }
    
    # Build context-aware prompt
    tool_names = ", ".join([t.name for t in tools])
    system_prompt = AGENT_SYSTEM_PROMPT.format(
        tool_names=tool_names,
        goal=state["goal"],
        current_url=state.get("current_url", "unknown"),
        steps=state["steps"],
        max_steps=state["max_steps"]
    )
    
    # Build message history with context
    context_msg = f"""
CURRENT SITUATION:
- Page: {state.get('page_title', 'Unknown')}
- URL: {state.get('current_url', 'None')}
- Last Error: {state.get('last_error', 'None')}

PAGE CONTENT (last 1000 chars):
{state.get('page_content', '')[-1000:]}

What action should you take next to achieve the goal?
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_msg)
    ] + list(state["messages"][-6:])  # Keep last 6 messages for context
    
    logger.info(f"Agent thinking (Step {state['steps'] + 1}/{state['max_steps']})")
    
    try:
        response = await llm.ainvoke(messages)
        logger.info(f"Agent decided: {response.tool_calls[0]['name'] if response.tool_calls else 'no tool'}")
    except Exception as e:
        logger.exception("LLM invocation failed")
        raise
    
    return {
        **state,
        "messages": state["messages"] + [response],
        "steps": state["steps"] + 1,
    }

# ============================================================================
# IMPROVED TOOL EXECUTION
# ============================================================================

async def tool_execution_node(state: AgentState):
    """Execute tool and analyze result"""
    last_msg = state["messages"][-1]
    
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return state
    
    # Execute tool
    tool_node = ToolNode(tools)
    result = await tool_node.ainvoke({"messages": [last_msg]})
    
    tool_name = last_msg.tool_calls[0]["name"]
    tool_output = result["messages"][-1].content
    
    logger.info(f"Tool '{tool_name}' executed: {tool_output[:100]}")
    
    # Analyze result
    status = analyze_tool_result(tool_output)
    failure_type = None
    last_error = None
    consecutive_failures = state.get("consecutive_failures", 0)
    
    if status == ActionStatus.FAILURE:
        failure_type = categorize_failure(tool_output)
        last_error = tool_output
        consecutive_failures += 1
        logger.warning(f"Tool failed: {failure_type}")
    else:
        consecutive_failures = 0  # Reset on success
    
    # Update page state if we read the page
    page_content = state.get("page_content", "")
    page_title = state.get("page_title", "")
    current_url = state.get("current_url", "")
    
    if tool_name == "read_page":
        page_content = tool_output
        # Extract title and URL from output if present
        if "Title:" in tool_output:
            lines = tool_output.split("\n")
            for line in lines:
                if line.startswith("Title:"):
                    page_title = line.replace("Title:", "").strip()
                if line.startswith("URL:"):
                    current_url = line.replace("URL:", "").strip()
    
    # Track successful actions
    successful_actions = state.get("successful_actions", [])
    if status == ActionStatus.SUCCESS:
        successful_actions = successful_actions + [tool_name]
    
    return {
        **state,
        "messages": state["messages"] + result["messages"],
        "all_actions": state["all_actions"] + [tool_name],
        "successful_actions": successful_actions,
        "consecutive_failures": consecutive_failures,
        "failure_type": failure_type,
        "last_error": last_error,
        "page_content": page_content,
        "page_title": page_title,
        "current_url": current_url,
    }

# ============================================================================
# RESULT ANALYSIS
# ============================================================================

def analyze_tool_result(output: str) -> ActionStatus:
    """Determine if tool execution succeeded"""
    output_lower = output.lower()
    
    # Success indicators
    if any(word in output_lower for word in ["successfully", "completed", "navigated to"]):
        return ActionStatus.SUCCESS
    
    # Failure indicators
    if any(word in output_lower for word in ["error", "failed", "could not", "timeout", "not find"]):
        return ActionStatus.FAILURE
    
    # Blocked indicators
    if any(word in output_lower for word in ["captcha", "blocked", "access denied"]):
        return ActionStatus.BLOCKED
    
    return ActionStatus.UNKNOWN

def categorize_failure(output: str) -> FailureType:
    """Categorize the type of failure"""
    output_lower = output.lower()
    
    if "timeout" in output_lower:
        return FailureType.TIMEOUT
    if any(word in output_lower for word in ["not find", "no such element"]):
        return FailureType.ELEMENT_NOT_FOUND
    if any(word in output_lower for word in ["selector", "invalid selector"]):
        return FailureType.SELECTOR_ERROR
    if "navigation" in output_lower:
        return FailureType.NAVIGATION_ERROR
    if "captcha" in output_lower:
        return FailureType.BLOCKED_BY_CAPTCHA
    
    return FailureType.UNKNOWN

# ============================================================================
# ROUTER
# ============================================================================

def router(state: AgentState) -> Literal["tools", "end"]:
    """Route graph execution"""
    last_msg = state["messages"][-1]
    
    # Check termination conditions
    if state["steps"] >= state["max_steps"]:
        logger.warning("Max steps reached")
        return "end"
    
    if state.get("consecutive_failures", 0) >= 3:
        logger.error("Too many failures, ending")
        return "end"
    
    # Check if tool should be called
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_name = last_msg.tool_calls[0]["name"]
        
        # Check for finish
        if tool_name == "finish_task":
            logger.info("Task completed")
            return "end"
        
        logger.info(f"Routing to tool: {tool_name}")
        return "tools"
    
    logger.warning("No tool called, ending")
    return "end"

# ============================================================================
# BUILD GRAPH
# ============================================================================

graph = StateGraph(AgentState)

# Simple 2-node graph!
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_execution_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # Loop back after tool execution

app = graph.compile()

# ============================================================================
# RUN AGENT
# ============================================================================

async def run_agent(goal: str, max_steps: int = 20):
    """Run the improved browser agent"""
    logger.info(f"Starting agent with goal: {goal}")
    
    try:
        initial_state = {
            "goal": goal,
            "messages": [],
            "steps": 0,
            "max_steps": max_steps,
            "consecutive_failures": 0,
            "current_url": "",
            "page_title": "",
            "page_content": "",
            "last_error": None,
            "failure_type": None,
            "all_actions": [],
            "successful_actions": []
        }
        
        final_state = None
        async for state in app.astream(initial_state, stream_mode="values"):
            final_state = state
        
        # Report results
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total steps: {final_state['steps']}")
        logger.info(f"Successful actions: {len(final_state['successful_actions'])}")
        logger.info(f"All actions: {final_state['all_actions']}")
        logger.info(f"Final URL: {final_state.get('current_url', 'unknown')}")
        logger.info("=" * 60)
        
    except Exception:
        logger.exception("Agent execution failed")
    finally:
        try:
            browser = await get_browser()
            await browser.close()
            logger.info("Browser closed")
        except Exception:
            logger.warning("Browser cleanup failed", exc_info=True)