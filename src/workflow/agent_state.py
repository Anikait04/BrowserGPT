from typing import TypedDict, Sequence, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # ── run bookkeeping ──
    goal: str
    entire_plan: List[str]
    step_count: int
    agent_decision: str
    steps: int
    max_steps: int
    progress_verification: str   # latest verify / human note (written by verify, wait_for_user)
    current_url: str
    messages: Annotated[Sequence[BaseMessage], "node remarks messages exchanged so far"]
    task_id: Optional[str]   # SSE streaming task ID — None when not streaming

    # ── new delegated architecture ──
    current_delegated_task: str              # task string from the last DelegationDecision
    delegation_decision: Optional[dict]      # serialized DelegationDecision
    success_criteria: str                    # concrete checkable condition for verify
    navigation_result: str                   # final message reported by the deep navigation agent
    verification_result: Optional[dict]      # serialized VerificationResult
    extracted_information: Optional[str]     # output of extract_information node (skeleton)
    waiting_for_user: bool                   # True while paused in wait_for_user node
    exit_requested: bool                   # True only when user explicitly asked to exit the loop
    final_response: str                      # answer returned to the user at finish
    navigation_iterations: int               # nav<->verify attempts for the current delegated task
    consecutive_failures: int                # failed verify verdicts for the current delegated task
    all_actions: Annotated[List[str], operator.add]  # execution history (append-only via reducer)
