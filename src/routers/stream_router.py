import asyncio
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.agent.agent import resume_agent_server, run_agent_server
from src.routers.common import (
    AgentRequest,
    ResumeRequest,
    _stream_until_agent_done,
    register_queue,
    sse_event,
    unregister_queue,
)

router = APIRouter(prefix="/stream", tags=["Streaming"])


# ── SSE streaming endpoints (persistent loop) ─────────────────────────────────
@router.post("/agent")
async def stream_agent(request: AgentRequest):
    """
    Start a persistent agent run and stream events via SSE.
    The stream ends with either:
    - {"type": "paused", "thread_id", "prompt"} — agent waits for the next
      user instruction (browser stays open). Resume via POST /stream/resume.
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


@router.post("/resume")
async def stream_resume(request: ResumeRequest):
    """
    Resume a paused persistent run (see /stream/agent paused event) and stream
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
