# schemas.py — Pydantic models for the new delegated architecture
# (planner -> delegation -> navigation <=> verify -> extract_information / wait_for_user)

from typing import List, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class Plan(BaseModel):
    """Structured output of the planner node."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    goal: str = Field(
        validation_alias=AliasChoices("goal", "messages"),
        description=(
            "Restated high-level goal. On success restate the user objective; "
            "on unplannable keep the objective plus the reason."
        ),
    )
    steps: List[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("steps", "plan"),
        description=(
            "Ordered list of high-level automation steps. "
            "Empty list means the task cannot be planned."
        ),
    )


class DelegationDecision(BaseModel):
    """What should execute next. The delegation node does NOT perform the work."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    action: Literal["planner", "navigation", "extract_information", "wait_for_user", "finish"] = Field(
        validation_alias=AliasChoices("action", "component", "route_decision"),
        description="Which component should handle the next unit of work",
    )
    task: str = Field(description="Concrete task handed to the chosen component")
    reasoning: str = Field(
        default="",
        description="Short explanation of why this delegation was chosen",
    )
    success_criteria: str = Field(
        default="",
        description=(
            "Concrete, checkable condition describing when the delegated task is complete, "
            "e.g. 'URL contains /results AND a results grid is visible'"
        ),
    )
    user_prompt: str = Field(
        default="",
        description=(
            "Exact user-facing question/message for the human, used ONLY when action "
            "is wait_for_user (ambiguous request or follow-up question). The wait "
            "node displays this verbatim. Empty for all other actions."
        ),
    )


class VerificationResult(BaseModel):
    """Did the current delegated task actually succeed?"""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    completed: bool = Field(
        validation_alias=AliasChoices("completed", "complete", "success", "verdict"),
        description="True only if the delegated task is fully satisfied",
    )
    reason: str = Field(
        default="",
        validation_alias=AliasChoices("reason", "justification", "explanation"),
        description="Short evidence-based justification for the verdict",
    )
    next_action: Literal["continue_task", "report_failure"] = Field(
        default="continue_task",
        description=(
            "continue_task: retry navigation with feedback; "
            "report_failure: caps reached or unrecoverable - give up on this task"
        ),
    )


class DepthDecision(BaseModel):
    """Overview vs detailed decision for the extract_information node."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    depth: Literal["overview", "detailed", "ask_user"] = Field(
        validation_alias=AliasChoices("depth", "detail_level", "format"),
        description=(
            "overview: short text summary; detailed: full PDF report; "
            "ask_user: ambiguous - ask the human which one they want"
        ),
    )
    question: str = Field(
        default="",
        description="Clarifying question for the human (only when depth is ask_user)",
    )
    reasoning: str = Field(
        default="",
        description="Short explanation of why this depth was chosen",
    )


class ReportSection(BaseModel):
    """One section of a detailed report."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    heading: str = Field(default="", description="Section heading")
    body: str = Field(default="", description="Section body text")


class ExtractionContent(BaseModel):
    """Structured content for an extraction (overview text or detailed report)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    title: str = Field(default="", description="Short title for the extracted content")
    summary: str = Field(default="", description="Concise summary of the extracted information")
    sections: List[ReportSection] = Field(
        default_factory=list,
        description="Detailed sections (used for the PDF report; empty for overviews)",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source URLs or page references used",
    )
