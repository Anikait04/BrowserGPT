
from typing import Optional, TypedDict, List
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

class DOMElement(BaseModel):
    id: int = Field(..., description="Unique element ID")
    type: Literal["button", "input", "link"] = Field(..., description="Type of the DOM element")
    label: str = Field(..., description="Visible or accessible label of the element")
    selector: str = Field(..., description="CSS selector used to interact with the element")
    href: Optional[str] = Field(None, description="Destination URL if element is a link")
    context: Optional[str] = Field(None, description="DOM hierarchy or structural context")
    message: str = Field(..., description="Short explanation of why this element is relevant")