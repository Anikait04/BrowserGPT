from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config import MODEL_NAME_OLLAMA,MODEL_NAME_OPENROUTER,OPENROUTER_API_KEY,BASE_URL
load_dotenv()

def llm_call():
    llm = ChatOllama(
    model=MODEL_NAME_OLLAMA,
    temperature=0,
    max_retries=2,
    model_kwargs={"format": "json"}
)
    return llm

# def llm_call():
#     llm = ChatOpenAI(
#             model_name=MODEL_NAME_OPENROUTER,
#             base_url=BASE_URL,
#             temperature=0.0,
#             openai_api_key=OPENROUTER_API_KEY,
#             max_retries=2,
#         )
#     return llm
