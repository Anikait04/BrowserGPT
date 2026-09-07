# extract_information.py — extraction node (skeleton).
#
# TODO: Implement extraction logic for text/json/excel/word outputs
#       using the ExtractionOutput schema from schemas.py.

from logs import logger
from src.workflow.agent_state import AgentState


async def extract_information_node(state: AgentState) -> dict:
    """Extract structured information from the current page.

    TODO: Implement extraction logic (text/json/excel/word outputs):
      - read the page via the LLM-free observation helpers in page_reader.py
      - ask get_llm() with a schema derived from ExtractionOutput
      - write the payload to the requested format
      - store a short summary in `extracted_information`
    """
    logger.info("[EXTRACT] extract_information node reached (skeleton, no-op)")
    return {"extracted_information": None}
