import os
from typing import Dict

from dotenv import load_dotenv
load_dotenv()
MODEL_NAME_OPENROUTER = "openai/gpt-oss-20b:free"
MODEL_NAME_OLLAMA="gpt-oss:120b-cloud"
_PAGE_CACHE: Dict[str, str] = {}
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"