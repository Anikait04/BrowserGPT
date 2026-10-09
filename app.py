# app.py (at project root)
import uvicorn
from dotenv import load_dotenv
from config import HOST, PORT

load_dotenv()
if __name__ == "__main__":
    print(f"host, port {HOST}, {PORT}")
 
    uvicorn.run(
        "src.router_app:app",
        host=HOST,
        port=PORT,
        reload=False,
    ) 
