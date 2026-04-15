# app.py (at project root)
import sys
import uvicorn
import os
from dotenv import load_dotenv
import asyncio
import sys
from config import HOST, PORT

load_dotenv()
if __name__ == "__main__":
    uvicorn.run(
        "src.workflow.router_app:app",
        host=HOST,
        port=PORT,
        reload=False,
    ) 
