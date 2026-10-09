# app/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.nav_router import router as nav_router
from src.routers.stream_router import router as stream_router
from src.routers.extract_router import router as extract_router

app = FastAPI(title="Welcome to BrowserGPT API", version="1.0.0")
app.include_router(nav_router)
app.include_router(stream_router)
app.include_router(extract_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
