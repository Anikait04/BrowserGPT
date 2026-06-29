# BrowserGPT

AI-powered browser automation agent that executes natural language goals in a real browser using LLM-driven planning and Playwright.

## Architecture

```
User Goal
   │
   ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Planner  │───▶│  Agent   │───▶│  Tools   │───▶│ Verifier │
│ (LLM)   │    │  (LLM)   │    │(Playwright)│   │  (LLM)  │
└──────────┘    └────┬─────┘    └──────────┘    └────┬─────┘
                     │                                │
              ┌──────▼──────┐                         │
              │  Read Page  │◀────────────────────────┘
              │ (DOM Scout) │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Human Wait  │  (optional HITL pause)
              └─────────────┘
```

The agent uses a **LangGraph** state machine with the following nodes:

| Node | Purpose |
|------|---------|
| `planner` | Breaks the user goal into ordered high-level steps |
| `agent` | Decides the next single action (tool call, read page, or finish) |
| `tools` | Executes browser actions via Playwright (navigate, click, type) |
| `read_page` | Scrapes visible DOM elements and selects the best one for the next action |
| `verifier` | Checks if the last action made real progress toward the goal |
| `human_wait` | Pauses the graph and waits for human input (HITL) |

State is persisted across runs via **SQLite checkpoints** (`langgraph-checkpoint-sqlite`).

## Features

- **Natural language automation** — describe what you want; the agent plans and executes
- **Real-time browser view** — live screenshots streamed to the frontend via SSE
- **Human-in-the-loop** — the agent can pause and ask for clarification mid-task
- **Adaptive element selection** — scrapes interactive DOM elements (inputs, buttons, links) and picks the best one using an LLM
- **Progress verification** — after each action, an LLM verdict confirms whether progress was made
- **Cyberpunk frontend** — dark terminal-style UI with live viewport, chat feed, and step tracker

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenRouter API / Custom LLM endpoint |
| Agent Framework | LangGraph + LangChain |
| Browser Automation | Playwright (Chromium) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS (single file) |
| State Persistence | SQLite via `langgraph-checkpoint-sqlite` |
| Streaming | Server-Sent Events (SSE) |
| Containerization | Docker + Supervisor |

## Project Structure

```
BrowserGPT/
├── app.py                  # Uvicorn entrypoint
├── main.py                 # CLI entrypoint (runs agent directly)
├── config.py               # Environment config and constants
├── logs.py                 # Centralized logging (IST timezone, monthly folders)
├── src/
│   ├── workflow/
│   │   ├── agent.py        # LangGraph graph definition and run loop
│   │   ├── agent_state.py  # TypedDict state schema
│   │   ├── nodes.py        # All graph node implementations
│   │   ├── llm.py          # Custom LLM client (token auth, structured output)
│   │   ├── prompt.py       # System prompts for planner, navigator, DOM selector
│   │   ├── browserplugin.py# Playwright Browser wrapper
│   │   ├── browsertools.py # LangChain tools exposed to the agent
│   │   ├── structured.py   # Pydantic/TypedDict schemas for LLM responses
│   │   ├── router_app.py   # FastAPI app with CORS
│   │   └── utils.py        # Plan step helper
│   └── routers/
│       └── agent_router.py # API endpoints (/run-agent, /stream-agent)
├── frontend/
│   └── index.html          # Cyberpunk-themed SPA
├── Dockerfile
├── supervisord.conf        # Runs Xvfb + app in headless Docker
├── render.yaml             # Render.com deployment config
├── requirements.txt
└── pyproject.toml
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Playwright](https://playwright.dev/python/) browsers installed
- An LLM API key (OpenRouter or custom endpoint)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd BrowserGPT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium --with-deps
```

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM API
OPENROUTER_API_KEY=your_openrouter_key_here

# Custom LLM endpoint (if not using OpenRouter)
API_USERNAME=your_username
API_PASSWORD=your_password
LOGIN_URL=http://your-auth-endpoint/login
MODEL_URL=http://your-model-endpoint/generate

# Server
HOST=0.0.0.0
PORT=1000

# Logging
LOG_DIR=logs
DEBUG=false
```

### Running

**Option 1 — Web server (with frontend)**

```bash
python app.py
```

Then open `frontend/index.html` in your browser and set the backend URL to `http://localhost:1000`.

**Option 2 — CLI (direct agent execution)**

```bash
python main.py
```

This runs the example task: *"go on youtube.com, search for star boys song, and play the first video"*.

### Docker

```bash
docker build -t browsergpt .
docker run -p 10000:10000 --env-file .env browsergpt
```

The container uses **Xvfb** (virtual framebuffer) for headless browser rendering and **Supervisor** to manage both processes.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/nav/run-agent` | Blocking — runs the agent to completion |
| `POST` | `/nav/stream-agent` | Streaming — SSE stream of screenshots, steps, and status |
| `GET` | `/nav/` | Health check |

### POST /nav/stream-agent

```json
{
  "goal": "Go to YouTube and search for Starboy",
  "max_steps": 30
}
```

Returns an SSE stream with events:

- `start` — task initiated
- `screenshot` — base64 JPEG of current browser view + step/action info
- `done` — task completed
- `error` — agent failed
- `ping` — keepalive

## Deployment

### Render

The included `render.yaml` deploys as a Docker service on Render:

```bash
# Push to GitHub, then connect the repo in Render dashboard
# Set environment variables in the Render dashboard
```

The health check path is `/nav/`.

## License

MIT
