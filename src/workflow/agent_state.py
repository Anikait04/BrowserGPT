from typing import TypedDict, Sequence, Annotated, List, Dict, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
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
    messages: Annotated[Sequence[BaseMessage], "node remarks messages exchanged so far"]
    chosen_element:List[str]
    task_id: Optional[str]   # SSE streaming task ID — None when not streaming