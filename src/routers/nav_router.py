from fastapi import APIRouter, HTTPException

from src.workflow.agent import resume_agent_server, run_agent_server
from src.logs import logger
from src.routers.common import AgentRequest, ResumeRequest

router = APIRouter(prefix="/nav", tags=["Navigation"])


# ── JSON endpoints (persistent loop, non-blocking) ────────────────────────────
@router.post("/run")
async def run_agent_endpoint(request: AgentRequest):
    """Start a persistent agent run (non-blocking, no stdin).

    Returns either:
    - {"status": "paused", "thread_id", "prompt"} — task finished or input
      needed; browser stays open. Send the next instruction via POST /nav/resume.
    - {"status": "done", ...} — only after an explicit user exit.
    """
    logger.info("Task started")

    try:
        result = await run_agent_server(
            goal=request.goal,
            max_steps=request.max_steps,
        )
        logger.info(f"Run {result.get('status')}: thread {result.get('thread_id')}")

        return {
            "status": "success",
            "message": "Agent executed successfully",
            "status_code": 200,
            **result,
        }

    except Exception as e:
        logger.exception("Agent execution failed")
        raise HTTPException(
            status_code=500,
            detail="Agent execution failed"
        )


@router.post("/resume")
async def resume_agent_endpoint(request: ResumeRequest):
    """Resume a paused persistent run with the next user instruction.

    - "exit"/"quit"/... -> {"status": "done"} (browser closed, loop ends).
    - new task after a finish -> replanned, runs until next pause.
    - otherwise -> resumes the current goal until next pause.
    """
    logger.info(f"Resuming thread {request.thread_id}")

    try:
        result = await resume_agent_server(
            thread_id=request.thread_id,
            human_input=request.human_input,
        )
        return {
            "status": "success",
            "status_code": 200,
            **result,
        }
    except Exception as e:
        logger.exception("Agent resume failed")
        raise HTTPException(status_code=500, detail="Agent resume failed")


@router.get("/")
def health_check():
    return {"status": "ok"}
