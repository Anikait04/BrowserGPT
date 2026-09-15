import asyncio
import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from src.workflow.agent import resume_agent_server, run_agent_server
from logs import logger

router = APIRouter(prefix="/nav", tags=["Navigation Agent"])

class AgentRequest(BaseModel):
    goal: str
    max_steps: int = 30
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    human_input: str


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


async def _stream_until_agent_done(queue: asyncio.Queue, agent_task: asyncio.Task, task_id: str):
    """Yield SSE events from the queue until the agent task finishes.

    The agent task is a run_agent_server/resume_agent_server call which NEVER
    blocks on input() — it returns {"status": "paused"/"done", ...}.
    A "paused" result means the persistent loop is waiting for the user:
    emit a paused event (with thread_id + prompt) so the frontend can ask
    for the next instruction instead of treating the run as finished.
    Only an explicit user exit produces status "done".
    """
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

            if agent_task.done():
                exc = agent_task.exception()
                if exc:
                    yield sse_event({"type": "error", "message": str(exc)})
                    break
                result = agent_task.result() or {}
                if result.get("status") == "paused":
                    yield sse_event({
                        "type": "paused",
                        "task_id": task_id,
                        "thread_id": result.get("thread_id"),
                        "prompt": result.get("prompt", ""),
                        "final_response": result.get("final_response", ""),
                        "current_url": result.get("current_url", ""),
                    })
                elif result.get("status") == "done":
                    yield sse_event({
                        "type": "done",
                        "message": result.get("final_response") or "Session ended",
                        "thread_id": result.get("thread_id"),
                    })
                else:
                    yield sse_event({"type": "done", "message": "Agent finished"})
                break


# ── SSE streaming endpoints (persistent loop) ─────────────────────────────────
@router.post("/stream-agent")
async def stream_agent(request: AgentRequest):
    """
    Start a persistent agent run and stream events via SSE.
    The stream ends with either:
    - {"type": "paused", "thread_id", "prompt"} — agent waits for the next
      user instruction (browser stays open). Resume via POST /nav/stream-resume.
    - {"type": "done"} — user explicitly exited; browser closed.
    """
    task_id = uuid.uuid4().hex
    thread_id = request.thread_id or uuid.uuid4().hex
    queue = register_queue(task_id)

    async def event_generator():
        yield sse_event({"type": "start", "task_id": task_id, "thread_id": thread_id, "goal": request.goal})

        agent_task = asyncio.create_task(
            run_agent_server(
                goal=request.goal,
                max_steps=request.max_steps,
                thread_id=thread_id,
                task_id=task_id,
            )
        )

        async for evt in _stream_until_agent_done(queue, agent_task, task_id):
            yield evt

        unregister_queue(task_id)
        if not agent_task.done():
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


@router.post("/stream-resume")
async def stream_resume(request: ResumeRequest):
    """
    Resume a paused persistent run (see /stream-agent paused event) and stream
    follow-up events via SSE. Same terminal contract: paused (wait for more
    input) vs done (explicit user exit).
    """
    task_id = uuid.uuid4().hex
    queue = register_queue(task_id)

    async def event_generator():
        yield sse_event({"type": "start", "task_id": task_id, "thread_id": request.thread_id})

        agent_task = asyncio.create_task(
            resume_agent_server(
                thread_id=request.thread_id,
                human_input=request.human_input,
                task_id=task_id,
            )
        )

        async for evt in _stream_until_agent_done(queue, agent_task, task_id):
            yield evt

        unregister_queue(task_id)
        if not agent_task.done():
            agent_task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
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


# ── JSON endpoints (persistent loop, non-blocking) ────────────────────────────
@router.post("/run-agent")
async def run_agent_endpoint(request: AgentRequest):
    """Start a persistent agent run (non-blocking, no stdin).

    Returns either:
    - {"status": "paused", "thread_id", "prompt"} — task finished or input
      needed; browser stays open. Send the next instruction via POST /resume.
    - {"status": "done", ...} — only after an explicit user exit.
    """
    logger.info("Task started")

    try:
        result = await run_agent_server(
            goal=request.goal,
            max_steps=request.max_steps,
            thread_id=request.thread_id,
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
