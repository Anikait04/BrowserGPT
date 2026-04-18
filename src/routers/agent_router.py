import asyncio
import json
import base64
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
 
from src.workflow.agent import run_agent
from logs import logger
 
router = APIRouter(prefix="/nav", tags=["Navigation Agent"])
 
class AgentRequest(BaseModel):
    goal: str
    max_steps: int = 30
 
 
# ── Shared screenshot queue registry ─────────────────────────────────────────
# When a stream task starts, it registers a queue here.
# tool_execution_node picks it up and pushes screenshots into it.
_screenshot_queues: dict[str, asyncio.Queue] = {}
 
def get_screenshot_queue(task_id: str) -> asyncio.Queue | None:
    return _screenshot_queues.get(task_id)
 
def register_queue(task_id: str) -> asyncio.Queue:
    q = asyncio.Queue()
    _screenshot_queues[task_id] = q
    return q
 
def unregister_queue(task_id: str):
    _screenshot_queues.pop(task_id, None)
 
 
# ── SSE helper ────────────────────────────────────────────────────────────────
def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
 
 
# ── SSE streaming endpoint ────────────────────────────────────────────────────
@router.post("/stream-agent")
async def stream_agent(request: AgentRequest):
    """
    Runs the agent and streams screenshots + step events via SSE.
    Frontend connects with fetch() and reads the event stream.
    """
    import uuid
    task_id = uuid.uuid4().hex
    queue = register_queue(task_id)
 
    async def event_generator():
        # Signal frontend: task started
        yield sse_event({"type": "start", "task_id": task_id, "goal": request.goal})
 
        # Run agent in background — it will push to queue via push_screenshot()
        agent_task = asyncio.create_task(
            run_agent(
                goal=request.goal,
                max_steps=request.max_steps,
                task_id=task_id,        # pass task_id so nodes can push screenshots
            )
        )
 
        # Stream events from queue until agent finishes
        while True:
            try:
                # Wait up to 0.5s for a new event; check if agent is done
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
 
                if event.get("type") == "done":
                    yield sse_event(event)
                    break
                if event.get("type") == "error":
                    yield sse_event(event)
                    break
 
                yield sse_event(event)
 
            except asyncio.TimeoutError:
                # Send keepalive ping so connection stays open
                yield sse_event({"type": "ping"})
 
                # Check if agent crashed
                if agent_task.done():
                    exc = agent_task.exception()
                    if exc:
                        yield sse_event({"type": "error", "message": str(exc)})
                    else:
                        yield sse_event({"type": "done", "message": "Agent finished"})
                    break
 
        unregister_queue(task_id)
        agent_task.cancel()
 
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Important for Nginx/Render — disables buffering
            "Access-Control-Allow-Origin": "*",
        }
    )
 
 
# ── Push helper — called from nodes.py after each tool execution ──────────────
async def push_screenshot(task_id: str, screenshot_b64: str, step: int, max_steps: int = 30, action: str = "", url: str = "", message: str = ""):
    """
    Call this from tool_execution_node after each browser action.
    """
    queue = get_screenshot_queue(task_id)
    if queue:
        await queue.put({
            "type": "screenshot",
            "screenshot": screenshot_b64,
            "step": step,
            "action": action,
            "max_steps": max_steps,   # ← add this
            "url": url,
            "message": message,
        })
 
 
async def push_done(task_id: str, message: str = "Task completed"):
    queue = get_screenshot_queue(task_id)
    if queue:
        await queue.put({"type": "done", "message": message})
 
 
async def push_error(task_id: str, message: str):
    queue = get_screenshot_queue(task_id)
    if queue:
        await queue.put({"type": "error", "message": message})
 
 
# ── Original blocking endpoint (keep for backwards compat) ────────────────────
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