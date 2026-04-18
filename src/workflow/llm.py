#llm.py
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import inspect
from config import MODEL_NAME_OLLAMA,MODEL_NAME_OPENROUTER,OPENROUTER_API_KEY,BASE_URL
# load_dotenv()

# def llm_call():
#     llm = ChatOllama(
#     model=MODEL_NAME_OLLAMA,
#     temperature=0,
#     max_retries=2,
#     model_kwargs={"format": "json"}
# )
#     return llm

# def llm_call():
#     llm = ChatOpenAI(
#             model_name=MODEL_NAME_OPENROUTER,
#             base_url=BASE_URL,
#             temperature=0.0,
#             openai_api_key=OPENROUTER_API_KEY,
#             max_retries=2,
#         )
#     return llm

#llm.py
import httpx
import os
from dotenv import load_dotenv
from config import USERNAME, PASSWORD, LOGIN_URL, MODEL_URL
import cloudpickle
import dill, base64
import io

load_dotenv()

class CustomLLMClient:
    def __init__(self):
        self.token = None

    async def login(self):
        async with httpx.AsyncClient() as client:
            res = await client.post(
                LOGIN_URL,
                json={
                    "username": USERNAME,
                    "password": PASSWORD
                },
                headers={"accept": "application/json"}
            )
            res.raise_for_status()
            data = res.json()
            self.token = data.get("access_token") or data.get("token")

    async def generate(self, system_prompt, user_prompt, structured=False, schema=None):
        if not self.token:
            await self.login()

        payload = {
            "model": "gpt-oss:120b-cloud",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "structured": structured,
        }
        # Only include output_schema when structured is True and schema is provided
        import inspect  # add this, but better yet, refactor away from getsource

        # Better: just send the schema as a dict
        if structured and schema:
            payload["output_schema"] = schema
        # print("PAYLOAD BEING SENT:", payload)
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(
                MODEL_URL,
                headers={
                    "accept": "application/json",
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            if res.status_code == 401:
                await self.login()
                return await self.generate(system_prompt, user_prompt, structured, schema)

            res.raise_for_status()
            return res.json()

