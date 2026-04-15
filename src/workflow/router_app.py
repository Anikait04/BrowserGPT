# app/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.agent_router import router as agent_router

app = FastAPI(title="Welcom to BrowserGPT API", version="1.0.0")
app.include_router(agent_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
