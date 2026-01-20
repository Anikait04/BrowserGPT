import os
import asyncio
from dotenv import load_dotenv
from typing import Any, Dict, TypedDict, Sequence, Annotated
from typing import List
from collections import Counter
from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import typing
import json
import hashlib
from browsertools import tools, get_browser
from prompt import get_prompt
from logs import logger, log_separator

load_dotenv()

class ProgressStatus(TypedDict):
    status: str              # continue | retry | finish | abort
    reason: str              # human readable
    progress_pct: float      # 0–100
    should_stop: bool

class ProgressDecision(TypedDict):
    decision: str           # continue | retry | replan | finish | abort
    reason: str
    confidence: float       # 0.0 – 1.0


class BrowserObservation(TypedDict):
    url: str
    title: str
    content_hash: str
    content_preview: str
    changed: bool
    change_reason: str

MODEL_NAME = "openai/gpt-oss-20b:free"


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


class BrowserAction(BaseModel):
    """Single browser automation action with human-readable step"""
    tool: str = Field(..., description="Tool name (navigate, type, click, read, etc.)")
    args: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Tool arguments")
    step_description: str = Field(..., description="Simple step description like 'navigate to google'")

class IntentParseResult(BaseModel):
    """Full parsing result with structured plan"""
    actions: List[BrowserAction] = Field(..., description="Sequential browser actions")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    high_level_goal: str = Field(..., description="Simplified goal summary")
    
class AgentState(TypedDict, total=False):
    """Agent state with all relevant information"""
    user_query: str
    high_level_goal: str
    messages: Annotated[Sequence[BaseMessage], "Message history"]
    steps: int
    max_steps: int
    last_action: str
    page_content: str
    all_actions: Annotated[Sequence[str], "All actions taken so far"]
    current_plan: Annotated[Sequence[str], "Current execution plan as simple steps"]

async def intent_parser(llm, goal: str, state: AgentState) -> IntentParseResult:
    """
    Parse user goal into structured browser actions and update state
    Returns Pydantic model for easy access to both structured data and simple steps
    """
    intent_prompt = get_prompt("intent_system_prompt")
    logger.info(f"Parsing intent for goal: {goal}")
    
    # Configure LLM for structured output
    structured_llm = llm.with_structured_output(IntentParseResult)
    
    messages = [
        SystemMessage(content=intent_prompt),
        HumanMessage(content=f"User goal: {goal}")
    ]
    
    try:
        # Get structured result directly
        parsed = await structured_llm.ainvoke(messages)
        
        # Update state with the parsed information
        state["high_level_goal"] = parsed.high_level_goal
        state["current_plan"] = [action.step_description for action in parsed.actions]
        
        logger.info(f"Parsed intent with {len(parsed.actions)} actions")
        return parsed
        
    except Exception:
        logger.exception("Intent parsing failed")
        # Return a valid but empty plan on failure
        return IntentParseResult(
            actions=[BrowserAction(
                tool="finish_task",
                args={"status": "failed"},
                step_description="Failed to understand user intent"
            )],
            confidence=0.0,
            high_level_goal=goal
        )


def track_progress(state: AgentState) -> dict:
    """
    Tracks agent progress and decides whether to continue or stop
    """

    steps = state["steps"]
    max_steps = state["max_steps"]
    actions: List[str] = list(state.get("all_actions", []))

    logger.debug(f"Tracking progress: step {steps}/{max_steps}")

    # -------------------------
    # Step-based progress
    # -------------------------
    progress_pct = min((steps / max_steps) * 100, 100.0)

    # -------------------------
    # Hard stop: max steps
    # -------------------------
    if steps >= max_steps:
        logger.warning("Max steps reached")
        return {
            "status": "finish",
            "reason": "Maximum steps reached",
            "progress_pct": progress_pct,
            "should_stop": True
        }

    # -------------------------
    # Loop detection
    # -------------------------
    if len(actions) >= 4:
        last_actions = actions[-4:]
        counts = Counter(last_actions)

        # Same action repeated
        if any(count >= 3 for count in counts.values()):
            logger.warning("Detected action loop")
            return {
                "status": "retry",
                "reason": "Agent is repeating the same action",
                "progress_pct": progress_pct,
                "should_stop": False
            }

    # -------------------------
    # No-op detection
    # -------------------------
    if len(actions) >= 3:
        if actions[-1] == actions[-2] == actions[-3]:
            logger.warning("Detected no progress")
            return {
                "status": "retry",
                "reason": "No visible progress detected",
                "progress_pct": progress_pct,
                "should_stop": False
            }

    # -------------------------
    # Normal continuation
    # -------------------------
    return {
        "status": "continue",
        "reason": "Agent progressing normally",
        "progress_pct": progress_pct,
        "should_stop": False
    }
def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def observe_browser_state(
    *,
    page_content: str,
    last_observation: Dict | None = None
) -> Dict:
    """
    Observe browser state and detect meaningful changes.
    This function does NOT take actions.
    """

    logger.debug("Observing browser state")

    # -------------------------
    # Extract metadata
    # -------------------------
    url = ""
    title = ""
    content = page_content or ""

    for line in content.splitlines():
        if line.startswith("URL:"):
            url = line.replace("URL:", "").strip()
        elif line.startswith("Title:"):
            title = line.replace("Title:", "").strip()

    # Fallbacks
    url = url or "unknown"
    title = title or "unknown"

    # -------------------------
    # Content fingerprinting
    # -------------------------
    normalized_content = content[:3000]
    content_hash = _hash_text(normalized_content)

    # -------------------------
    # Change detection
    # -------------------------
    changed = False
    change_reason = "initial observation"

    if last_observation:
        if content_hash != last_observation.get("content_hash"):
            changed = True
            change_reason = "page content changed"
        elif url != last_observation.get("url"):
            changed = True
            change_reason = "URL changed"
        elif title != last_observation.get("title"):
            changed = True
            change_reason = "title changed"
        else:
            change_reason = "no significant change"

    observation = {
        "url": url,
        "title": title,
        "content_hash": content_hash,
        "content_preview": normalized_content[:500],
        "changed": changed,
        "change_reason": change_reason
    }

    logger.info(
        f"Observed state | changed={changed} | reason={change_reason}"
    )

    return observation

def progress_checker(
    *,
    state: AgentState,
    observation: Dict,
    success_keywords: List[str] | None = None
) -> Dict:
    """
    Decide whether the agent is making progress toward its goal.
    """

    success_keywords = success_keywords or [
        "success",
        "completed",
        "results",
        "thank you",
        "welcome",
        "dashboard"
    ]

    actions = list(state.get("all_actions", []))
    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", 1)

    # -------------------------
    # Hard stop: step limit
    # -------------------------
    if steps >= max_steps:
        logger.warning("Progress checker: max steps exceeded")
        return {
            "decision": "finish",
            "reason": "Maximum step limit reached",
            "confidence": 0.95
        }

    # -------------------------
    # Success signal detection
    # -------------------------
    content_preview = observation.get("content_preview", "").lower()
    for keyword in success_keywords:
        if keyword in content_preview:
            logger.info(f"Progress checker: success keyword '{keyword}' found")
            return {
                "decision": "finish",
                "reason": f"Detected success signal: '{keyword}'",
                "confidence": 0.9
            }

    # -------------------------
    # Stalled page detection
    # -------------------------
    if not observation.get("changed", True):
        logger.warning("Progress checker: page not changing")
        return {
            "decision": "retry",
            "reason": "Page did not change after action",
            "confidence": 0.7
        }

    # -------------------------
    # Repeated failure detection
    # -------------------------
    if len(actions) >= 3:
        last = actions[-3:]
        if len(set(last)) == 1:
            logger.error("Progress checker: repeated identical actions")
            return {
                "decision": "replan",
                "reason": "Repeated identical actions with no success",
                "confidence": 0.85
            }

    # -------------------------
    # Low remaining budget
    # -------------------------
    remaining_ratio = (max_steps - steps) / max_steps
    if remaining_ratio < 0.15:
        logger.warning("Progress checker: low remaining step budget")
        return {
            "decision": "replan",
            "reason": "Low remaining steps, replanning required",
            "confidence": 0.6
        }

    # -------------------------
    # Default: continue
    # -------------------------
    return {
        "decision": "continue",
        "reason": "Progress appears normal",
        "confidence": 0.8
    }
class ReplanResult(TypedDict):
    new_goal: str
    strategy: str
    reason: str

def replan_goal(
    *,
    state: AgentState,
    observation: dict,
    failure_reason: str
) -> dict:
    """
    Generate a safer, more constrained goal when the agent is stuck.
    """

    original_goal = state["goal"]
    last_action = state.get("last_action", "")
    steps = state.get("steps", 0)

    logger.warning(
        f"Replanning triggered | reason={failure_reason} | "
        f"last_action={last_action}"
    )

    # -------------------------
    # Strategy selection
    # -------------------------
    strategy = "unknown"
    new_goal = original_goal

    # 1. Selector or click failure
    if "click" in failure_reason.lower() or "selector" in failure_reason.lower():
        strategy = "broaden selectors and re-read page"
        new_goal = (
            f"Read the page carefully and find an alternative clickable element "
            f"to achieve this goal: {original_goal}"
        )

    # 2. Page not changing
    elif "page did not change" in failure_reason.lower():
        strategy = "re-observe and search"
        new_goal = (
            f"Re-read the page content and decide the next best action "
            f"to continue toward: {original_goal}"
        )

    # 3. Repeated identical actions
    elif "repeated" in failure_reason.lower():
        strategy = "change approach"
        new_goal = (
            f"Try a different approach to accomplish the goal without repeating "
            f"previous actions: {original_goal}"
        )

    # 4. Low remaining steps
    elif "low remaining" in failure_reason.lower():
        strategy = "goal simplification"
        new_goal = (
            f"Perform the minimum action required to partially satisfy this goal: "
            f"{original_goal}"
        )

    # 5. Generic fallback
    else:
        strategy = "reset context"
        new_goal = (
            f"Go back to a safe starting point and reassess how to achieve: "
            f"{original_goal}"
        )

    logger.info(
        f"Replan created | strategy={strategy} | new_goal={new_goal}"
    )

    return {
        "new_goal": new_goal,
        "strategy": strategy,
        "reason": failure_reason
    }

class GeneratedAction(TypedDict):
    tool: str
    args: str
    reason: str


def generate_next_action(
    *,
    intent: Dict,
    state: AgentState,
    observation: Dict
) -> Dict:
    """
    Select the next executable browser action.
    """

    actions: List[Dict] = intent.get("actions", [])
    executed_actions = set(state.get("all_actions", []))
    last_action = state.get("last_action", "")

    logger.debug(f"Generating next action | total_plan={len(actions)}")

    # -------------------------
    # No actions left
    # -------------------------
    if not actions:
        logger.info("No actions left, finishing task")
        return {
            "tool": "finish_task",
            "args": "No remaining actions to execute",
            "reason": "Action plan exhausted"
        }

    # -------------------------
    # Find first unexecuted action
    # -------------------------
    for action in actions:
        action_key = f"{action['tool']}|{action['args']}"

        if action_key in executed_actions:
            continue

        # Prevent repeating same action when page did not change
        if (
            action_key == last_action
            and not observation.get("changed", True)
        ):
            logger.warning("Skipping repeated action due to no page change")
            continue

        logger.info(
            f"Next action selected: {action['tool']} -> {action['args']}"
        )

        return {
            "tool": action["tool"],
            "args": action["args"],
            "reason": "Next planned action"
        }

    # -------------------------
    # Fallback: re-read page
    # -------------------------
    logger.warning("All actions executed or blocked, forcing page read")
    return {
        "tool": "read_page",
        "args": "",
        "reason": "Re-observing page state"
    }


class ExecutionResult(TypedDict):
    success: bool
    tool: str
    args: str
    output: str
    error: str | None

async def execute_action(
    *,
    action: Dict,
    state: AgentState,
    tool_map: Dict[str, callable]
) -> Dict:
    """
    Execute a single browser action safely and update agent state.
    """

    tool_name = action["tool"]
    args = action.get("args", "")

    logger.info(f"Executing action: {tool_name} | args={args}")

    # -------------------------
    # Tool existence check
    # -------------------------
    if tool_name not in tool_map:
        logger.error(f"Unknown tool requested: {tool_name}")
        return {
            "success": False,
            "tool": tool_name,
            "args": args,
            "output": "",
            "error": f"Unknown tool: {tool_name}"
        }

    tool_fn = tool_map[tool_name]

    # -------------------------
    # Execute tool
    # -------------------------
    try:
        output = await tool_fn(args)

        success = not output.lower().startswith("error")
        error = None if success else output

        logger.info(
            f"Action result | success={success} | preview={output[:100]}"
        )

    except Exception as e:
        logger.exception("Tool execution failed")
        output = ""
        success = False
        error = str(e)

    # -------------------------
    # Update agent state
    # -------------------------
    action_key = f"{tool_name}|{args}"

    state["last_action"] = action_key
    state.setdefault("all_actions", []).append(action_key)
    state["steps"] += 1

    return {
        "success": success,
        "tool": tool_name,
        "args": args,
        "output": output,
        "error": error
    }
class ProgressUpdate(TypedDict):
    steps: int
    max_steps: int
    progress_pct: float
    last_action: str
    last_success: bool
    summary: str
def update_progress(
    *,
    state: AgentState,
    execution_result: dict
) -> dict:
    """
    Update agent progress after executing an action.
    """

    steps = state.get("steps", 0)
    max_steps = state.get("max_steps", 1)

    progress_pct = min((steps / max_steps) * 100, 100.0)

    last_action = state.get("last_action", "none")
    last_success = execution_result.get("success", False)

    # -------------------------
    # Human-readable summary
    # -------------------------
    if last_success:
        summary = f"Step {steps}/{max_steps} completed successfully"
    else:
        summary = f"Step {steps}/{max_steps} failed"

    progress = {
        "steps": steps,
        "max_steps": max_steps,
        "progress_pct": round(progress_pct, 2),
        "last_action": last_action,
        "last_success": last_success,
        "summary": summary
    }

    logger.info(
        f"Progress update | {progress['progress_pct']}% | "
        f"success={last_success}"
    )

    return progress