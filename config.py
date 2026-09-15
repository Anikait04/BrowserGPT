import os

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME_OPENCODE = "openai/gpt-oss-120b"
# MODEL_NAME_OPENCODE = "x-preview-f-free"
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL="https://opencode.ai/zen/v1"
# OPENCODE_BASE_URL="https://api.llm7.io/v1"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


MODEL_NAME_OLLAMA = "gpt-oss:120b-cloud"


# ── Groq (OpenAI-compatible endpoint, no extra dependency) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")


# ── Google Gemini (native SDK via langchain-google-genai) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")


# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# BASE_URL = "https://openrouter.ai/api/v1"
PORT=int(os.getenv("PORT", 1000))
HOST=os.getenv("HOST", "0.0.0.0")
thread_dir_name = "save_agentThread"

# ── New architecture safety limits ──
MAX_NAVIGATION_ITERATIONS = int(os.getenv("MAX_NAVIGATION_ITERATIONS", "8"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3"))
DEEP_AGENT_RECURSION_LIMIT = int(os.getenv("DEEP_AGENT_RECURSION_LIMIT", "40"))
