import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import src.config as config

router = APIRouter(prefix="/extract", tags=["Extraction"])


@router.get("/artifact/{artifact_id}")
async def download_artifact(artifact_id: str):
    """Download a generated artifact (detailed PDF report).

    Files live under ARTIFACTS_DIR/<thread_id>/<artifact_id>.pdf. The id is
    restricted to hex characters so no path traversal is possible.
    """
    if not artifact_id or not all(c in "0123456789abcdefABCDEF" for c in artifact_id):
        raise HTTPException(status_code=400, detail="Invalid artifact id")

    base_dir = os.path.abspath(config.ARTIFACTS_DIR)
    filename = f"{artifact_id}.pdf"
    for entry in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
        candidate = os.path.abspath(os.path.join(base_dir, entry, filename))
        if os.path.isfile(candidate) and candidate.startswith(base_dir + os.sep):
            return FileResponse(
                candidate,
                media_type="application/pdf",
                filename=f"report-{artifact_id[:8]}.pdf",
            )
    raise HTTPException(status_code=404, detail="Artifact not found")
