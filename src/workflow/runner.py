# runner.py — run orchestration shared by API servers and CLI.
#
# Extracts the duplicated pause/artifact/SSE/session logic from agent.py's
# run_agent_server / resume_agent_server / run_agent into one injectable
# AgentRunner with an EventSink protocol (implemented by src.routers.common).

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RunOutcome:
    status: str  # "paused" | "done"
    thread_id: str
    final_response: str = ""
    current_url: str = ""
    goal: str = ""
    prompt: str = ""
    artifact_id: str | None = None
    artifact_url: str | None = None

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "status": self.status,
            "thread_id": self.thread_id,
            "final_response": self.final_response,
            "current_url": self.current_url,
        }
        if self.status == "paused":
            out["prompt"] = self.prompt
            out["goal"] = self.goal
        if self.artifact_id:
            out["artifact_id"] = self.artifact_id
            out["artifact_url"] = self.artifact_url
        return out


class EventSink(Protocol):
    async def push_done(self, task_id: str, message: str) -> None: ...
    async def push_error(self, task_id: str, message: str) -> None: ...
    async def push_artifact(
        self, task_id: str, artifact_id: str, url: str, title: str, message: str
    ) -> None: ...


class RouterEventSink:
    """EventSink backed by src.routers.common (production)."""

    async def push_done(self, task_id: str, message: str) -> None:
        from src.routers.common import push_done

        await push_done(task_id, message)

    async def push_error(self, task_id: str, message: str) -> None:
        from src.routers.common import push_error

        await push_error(task_id, message)

    async def push_artifact(
        self, task_id: str, artifact_id: str, url: str, title: str, message: str
    ) -> None:
        from src.routers.common import push_artifact

        await push_artifact(task_id, artifact_id=artifact_id, url=url, title=title, message=message)


def artifact_info(values: dict) -> dict:
    """Artifact download info for API responses (empty when no detailed PDF)."""
    from src.workflow.constants import artifact_url

    artifact_id = (values or {}).get("artifact_id")
    if not artifact_id:
        return {}
    return {"artifact_id": artifact_id, "artifact_url": artifact_url(artifact_id)}


def pause_prompt(snapshot) -> str:
    """Extract the human-readable interrupt prompt from a paused snapshot."""
    try:
        for task in getattr(snapshot, "tasks", []) or []:
            for intr in getattr(task, "interrupts", []) or []:
                value = getattr(intr, "value", "")
                if value:
                    return str(value)
    except Exception:
        pass
    return "Agent is waiting for input. Type your next task or 'exit' to end."


def outcome_from_snapshot(
    *, status: str, thread_id: str, values: dict, prompt: str = "", goal: str = ""
) -> RunOutcome:
    art = artifact_info(values)
    return RunOutcome(
        status=status,  # type: ignore[arg-type]
        thread_id=thread_id,
        final_response=values.get("final_response", "") or "",
        current_url=values.get("current_url", "") or "",
        goal=values.get("goal", goal) or goal,
        prompt=prompt,
        artifact_id=art.get("artifact_id"),
        artifact_url=art.get("artifact_url"),
    )


class AgentRunner:
    """Shared start/resume loop (session binding + SSE + browser teardown)."""

    def __init__(self, app_factory, event_sink: EventSink | None = None):
        self._app_factory = app_factory
        self._sink = event_sink or RouterEventSink()

    async def start(self, goal: str, max_steps: int = 30, task_id: str | None = None) -> dict:
        """Non-interactive start (for API servers). Never blocks on input()."""
        from langgraph.types import Command  # noqa: F401 (kept for symmetry)

        from logs import log_separator, logger
        from src.workflow.browser.manager import close_browser
        from src.workflow.llm import reset_session_id, set_session_id
        from src.workflow.state_factory import initial_state

        log_separator("AGENT RUN START (server)")
        app = await self._app_factory()
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Thread ID: {thread_id}")
        session_token = set_session_id(thread_id)
        try:
            state = initial_state(goal, max_steps, task_id)
            async for _ in app.astream(state, config=config, stream_mode="values"):
                pass
            snapshot = await app.aget_state(config)
            values = snapshot.values or {}
            if snapshot.next:  # paused for user input
                outcome = outcome_from_snapshot(
                    status="paused",
                    thread_id=thread_id,
                    values=values,
                    prompt=pause_prompt(snapshot),
                    goal=goal,
                )
                return outcome.to_dict()
            if task_id:
                await self._sink.push_done(task_id, values.get("final_response") or "Task completed successfully")
            await close_browser()
            outcome = outcome_from_snapshot(status="done", thread_id=thread_id, values=values)
            return outcome.to_dict()
        except Exception as e:
            if task_id:
                await self._sink.push_error(task_id, str(e))
            raise
        finally:
            reset_session_id(session_token)
            log_separator("AGENT RUN PAUSED/END (server)")

    async def resume(self, thread_id: str, human_input: str, task_id: str | None = None) -> dict:
        """Resume a paused run with user input (for API servers)."""
        from langgraph.types import Command

        from logs import logger
        from src.workflow.browser.manager import close_browser
        from src.workflow.llm import reset_session_id, set_session_id

        app = await self._app_factory()
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"Resuming thread: {thread_id}")
        session_token = set_session_id(thread_id)
        try:
            if task_id:
                snapshot_before = await app.aget_state(config)
                before_values = snapshot_before.values or {}
                if before_values.get("task_id") != task_id and task_id:
                    try:
                        await app.aupdate_state(config, {"task_id": task_id})
                    except Exception as e:
                        logger.warning(f"[RUN] Could not rebind task_id: {e}")
            async for _ in app.astream(
                Command(resume=str(human_input or "").strip()), config=config, stream_mode="values"
            ):
                pass
            snapshot = await app.aget_state(config)
            values = snapshot.values or {}
            if snapshot.next:  # paused again
                outcome = outcome_from_snapshot(
                    status="paused",
                    thread_id=thread_id,
                    values=values,
                    prompt=pause_prompt(snapshot),
                )
                return outcome.to_dict()
            if task_id:
                await self._sink.push_done(task_id, values.get("final_response") or "Session ended")
            await close_browser()
            outcome = outcome_from_snapshot(status="done", thread_id=thread_id, values=values)
            return outcome.to_dict()
        except Exception as e:
            if task_id:
                await self._sink.push_error(task_id, str(e))
            raise
        finally:
            reset_session_id(session_token)
