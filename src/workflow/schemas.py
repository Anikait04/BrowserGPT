# schemas.py — Pydantic models for the new delegated architecture
# (planner -> delegation -> navigation <=> verify -> extract_information / wait_for_user)

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Structured output of the planner node."""

    goal: str = Field(description="Restated high-level goal")
    steps: List[str] = Field(description="Ordered list of high-level steps")


class DelegationDecision(BaseModel):
    """What should execute next. The delegation node does NOT perform the work."""

    action: Literal["navigation", "extract_information", "wait_for_user", "finish"] = Field(
        description="Which component should handle the next unit of work"
    )
    task: str = Field(description="Concrete task handed to the chosen component")
    reasoning: str = Field(description="Short explanation of why this delegation was chosen")
    success_criteria: str = Field(
        default="",
        description=(
            "Concrete, checkable condition describing when the delegated task is complete, "
            "e.g. 'URL contains /results AND a results grid is visible'"
        ),
    )


class VerificationResult(BaseModel):
    """Did the current delegated task actually succeed?"""

    completed: bool = Field(description="True only if the delegated task is fully satisfied")
    reason: str = Field(description="Short evidence-based justification for the verdict")
    next_action: Literal["continue_task", "report_failure"] = Field(
        default="continue_task",
        description=(
            "continue_task: retry navigation with feedback; "
            "report_failure: caps reached or unrecoverable - give up on this task"
        ),
    )


class ExtractionOutput(BaseModel):
    """Skeleton interface for extract_information outputs (not implemented yet)."""

    format: Literal["text", "json", "excel", "word"] = "text"
    content: Optional[str] = None
