from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from config import MODEL_NAME_OLLAMA
load_dotenv()

def llm_call():
    llm = ChatOllama(
    model=MODEL_NAME_OLLAMA,
    temperature=0,
    max_retries=2,
    model_kwargs={"format": "json"}
)
    return llm

# llm = ChatOpenAI(
#         model_name=MODEL_NAME,
#         base_url="https://openrouter.ai/api/v1",
#         temperature=0.0,
#         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#         max_retries=2,
#     )
