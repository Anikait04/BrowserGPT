import os
import asyncio
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.workflow.agent import run_agent
from logs import logger, log_separator

router = APIRouter(prefix="/nav", tags=["Navigation Agent"])
class AgentRequest(BaseModel):
    goal: str
    max_steps: int = 30

@router.post("/run-agent")
async def run_agent_endpoint(request: AgentRequest):
    """
    Trigger the browser automation agent
    """
    logger.info("Task started")

    try:
        await run_agent(
            goal=request.goal,
            max_steps=request.max_steps
        )
        logger.info("Task completed successfully")

        return {
            "status": "success",
            "message": "Agent executed successfully",
            "status_code": 200
        }

    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed"
        )


@router.get("/")
def health_check():
    return {"status": "ok"}