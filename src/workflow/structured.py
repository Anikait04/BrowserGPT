
from typing import TypedDict, List
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from typing import List



class PlanOutput(BaseModel):
    plan: List[str] = Field(description="Ordered list of high-level browser actions")
    messages: str = Field(...,description="Short description of the action taken by the agent")

class AgentDecision(TypedDict):
    route_decision: Literal["tools", "read_page", "finish", "wait"]
    tool_input: str
    tool_name:str
    element_id:int
    message: str = Field(...,description="Short description of the action taken by the agent")