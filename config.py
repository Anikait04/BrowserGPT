import os
from typing import Dict

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME_OPENCODE = "hy3-free"
# MODEL_NAME_OPENCODE = "x-preview-f-free"
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL="https://opencode.ai/zen/v1"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


MODEL_NAME_OLLAMA = "gpt-oss:120b-cloud"
OLLAMA_API_KEY=os.getenv("OLLAMA_API_KEY")


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = "https://openrouter.ai/api/v1"
PORT=int(os.getenv("PORT", 1000))
HOST=os.getenv("HOST", "0.0.0.0")
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")
LOGIN_URL = os.getenv("LOGIN_URL")
MODEL_URL = os.getenv("MODEL_URL")
thread_dir_name = "save_agentThread"

_PAGE_CACHE: Dict[str, str] = {}