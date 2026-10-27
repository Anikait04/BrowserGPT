# store.py — artifact filesystem storage + public URL building.
from __future__ import annotations

import os
import uuid

import config as config
from src.agent.artifacts.pdf import build_pdf
from src.agent.constants import artifact_url
from src.agent.schemas import ExtractionContent


class ArtifactStore:
    """Owns detailed-PDF persistence (directory layout + URL prefix)."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or config.ARTIFACTS_DIR

    def save_report(self, content: ExtractionContent, thread_id: str) -> tuple[str, str]:
        """Render + persist a report. Returns (artifact_id, dest_path)."""
        artifact_id = uuid.uuid4().hex
        scope = thread_id or "default"
        dest_dir = os.path.join(self.base_dir, scope)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{artifact_id}.pdf")
        build_pdf(content, dest_path)
        return artifact_id, dest_path

    @staticmethod
    def url_for(artifact_id: str) -> str:
        return artifact_url(artifact_id)

    @staticmethod
    def body_with_link(content: ExtractionContent) -> str:
        summary = content.summary or "Detailed report ready."
        # URL needs the id; caller fills it in — see save path in extraction node.
        return summary


# Default process-wide store (inject a custom base_dir in tests).
default_store = ArtifactStore()
