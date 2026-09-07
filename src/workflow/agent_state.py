from typing import TypedDict, Sequence, Annotated, List, Dict, Optional
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── legacy fields (kept for backward compatibility with nodes.py) ──
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
    tool_selector: str
    messages: Annotated[Sequence[BaseMessage], "node remarks messages exchanged so far"]
    chosen_element:List[str]
    task_id: Optional[str]   # SSE streaming task ID — None when not streaming

    # ── new delegated architecture ──
    current_delegated_task: str              # task string from the last DelegationDecision
    delegation_decision: Optional[dict]      # serialized DelegationDecision
    success_criteria: str                    # concrete checkable condition for verify
    navigation_result: str                   # final message reported by the deep navigation agent
    verification_result: Optional[dict]      # serialized VerificationResult
    extracted_information: Optional[str]     # output of extract_information node (skeleton)
    waiting_for_user: bool                   # True while paused in wait_for_user node
    final_response: str                      # answer returned to the user at finish
    navigation_iterations: int               # nav<->verify attempts for the current delegated task
    consecutive_failures: int                # failed verify verdicts for the current delegated task
    all_actions: Annotated[List[str], operator.add]  # execution history (append-only via reducer)
