import os

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME_OPENCODE = "hy3-free"
# MODEL_NAME_OPENCODE = "x-preview-f-free"
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL="https://opencode.ai/zen/v1"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")


MODEL_NAME_OLLAMA = "gpt-oss:120b-cloud"


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = "https://openrouter.ai/api/v1"
PORT=int(os.getenv("PORT", 1000))
HOST=os.getenv("HOST", "0.0.0.0")
thread_dir_name = "save_agentThread"

# ── New architecture safety limits ──
MAX_NAVIGATION_ITERATIONS = int(os.getenv("MAX_NAVIGATION_ITERATIONS", "8"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3"))
DEEP_AGENT_RECURSION_LIMIT = int(os.getenv("DEEP_AGENT_RECURSION_LIMIT", "40"))
