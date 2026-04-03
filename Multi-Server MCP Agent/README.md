# Multi-Server MCP Agent
 
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agent-purple)
![MCP](https://img.shields.io/badge/MCP-Multi--Server-orange)
 
A multi-server AI agent built with LangGraph and the Model Context Protocol (MCP) that autonomously retrieves live weather data, searches the web, and persists results to the filesystem — all orchestrated through a clean chat interface.
---
## Output
 
![Output](https://raw.githubusercontent.com/Tanmay01bhatt/New-Projects/main/Multi-Server%20MCP%20Agent/snippet.PNG)

---

## Features
 
- Real-time weather lookup for any city via OpenWeatherMap
- Google web search via Serper API
- File read/write within a sandboxed workspace
- Multi-turn conversation memory via LangGraph MemorySaver
- Clean chat UI with tool explorer sidebar
- Fully containerized with Docker Compose

---

## Tech Stack
 
| Layer | Technology |
|---|---|
| Agent Framework | LangGraph |
| Tool Protocol | Model Context Protocol (MCP) |
| LLM | Gemini 2.5 Flash |
| Backend | FastAPI |
| Frontend | Streamlit |
| Containerization | Docker + Docker Compose |
| Weather Data | OpenWeatherMap API |
| Web Search | Serper (Google Search API) |
| Filesystem | @modelcontextprotocol/server-filesystem |

---

## Getting Started

### 1. Clone the repository
 
```bash
git clone https://github.com/Tanmay01bhatt/multi-server-mcp-agent.git
cd multi-server-mcp-agent
```

### 2. Set up environment variables
 
```bash
GOOGLE_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key
OPENWEATHER_API_KEY=your_openweather_key
```

### 3. Run with Docker Compose
 
```bash
docker-compose up --build
```
 
- Streamlit UI → http://localhost:8501
- FastAPI docs → http://localhost:8000/docs
 
---
