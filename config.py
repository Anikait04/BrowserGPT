import os

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME_OPENCODE = "mimo-v2.6-flash-free"
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL="https://opencode.ai/zen/v1"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")


MODEL_NAME_OLLAMA = "nemotron-3-super:cloud"
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_BASE_URL = "https://api.ollama.com"


# ── Groq (OpenAI-compatible endpoint, no extra dependency) ──
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")


# ── Google Gemini (native SDK via langchain-google-genai) ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")

# ── JEV (codiv.ai OpenAI-compatible endpoint, browser navigation only) ──
# API key falls back to TYPESAFE_API_KEY (same codiv.ai account).
JEV_API_KEY = os.getenv("JEV_API_KEY")
JEV_BASE_URL = os.getenv("JEV_BASE_URL", "https://api.codiv.ai/v1")
JEV_MODEL_NAME = os.getenv("JEV_MODEL_NAME", "diffusiongemma-26b")
# Navigation needs room for multi-turn tool calls; the 256-token snippet
# value is only for one-shot JSON demos, not the agent loop.
JEV_MAX_TOKENS = int(os.getenv("JEV_MAX_TOKENS", "2048"))
PORT=int(os.getenv("PORT", 1000))
HOST=os.getenv("HOST", "0.0.0.0")
thread_dir_name = "save_agentThread"

# ── New architecture safety limits ──
MAX_NAVIGATION_ITERATIONS = int(os.getenv("MAX_NAVIGATION_ITERATIONS", "8"))
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MAX_CONSECUTIVE_FAILURES", "3"))
DEEP_AGENT_RECURSION_LIMIT = int(os.getenv("DEEP_AGENT_RECURSION_LIMIT", "40"))

# ── Extraction artifacts (detailed PDF reports) ──
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
