import asyncio
import json
from pydantic import BaseModel
from typing import Optional


class AgentRequest(BaseModel):
    goal: str
    max_steps: int = 30


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
                        **({k: result[k] for k in ("artifact_id", "artifact_url") if result.get(k)}),
                    })
                elif result.get("status") == "done":
                    yield sse_event({
                        "type": "done",
                        "message": result.get("final_response") or "Session ended",
                        "thread_id": result.get("thread_id"),
                        **({k: result[k] for k in ("artifact_id", "artifact_url") if result.get(k)}),
                    })
                else:
                    yield sse_event({"type": "done", "message": "Agent finished"})
                break


# ── Push helpers — call from workflow nodes to stream SSE events ────
async def push_screenshot(task_id: str, screenshot_b64: str, step: int, max_steps: int = 30, action: str = "", url: str = "", message: str = ""):
    """
    Call this after each browser action to push a screenshot event to the stream.
    """
    queue = get_screenshot_queue(task_id)
    if queue:
        await queue.put({
            "type": "screenshot",
            "screenshot": screenshot_b64,
            "step": step,
            "action": action,
            "max_steps": max_steps,
            "url": url,
            "message": message,
        })


async def push_artifact(task_id: str, artifact_id: str, url: str, title: str = "", message: str = ""):
    """Push a generated-artifact (detailed PDF) event to the SSE stream."""
    queue = get_screenshot_queue(task_id)
    if queue:
        await queue.put({
            "type": "artifact",
            "artifact_id": artifact_id,
            "url": url,
            "title": title,
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
