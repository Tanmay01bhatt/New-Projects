import logging
logging.basicConfig(level=logging.ERROR)

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

WORKSPACE = os.getenv("WORKSPACE", "/app/agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],
        "transport": "stdio",
        "env": os.environ.copy()
    },
    "serper": {
        "command": "python",
        "args": ["serper_server.py"],
        "transport": "stdio",
        "env": os.environ.copy()
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE],
        "transport": "stdio",
        "env": os.environ.copy()

    },
}


agent = None
available_tools = []


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


def create_graph(tools: list):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant with three capabilities:\n"
         "1. Weather — check current weather for any city\n"
         "2. Web search — search Google for news and information\n"
         "3. File system — read and write files in the agent workspace\n\n"
         f"IMPORTANT: The workspace directory is: {WORKSPACE}\n"
         "When saving ANY file, always pass the FULL path by combining "
         "the workspace directory with the filename. "
         f"For example, to save 'weather.txt' use: '{WORKSPACE}\\weather.txt'\n"
         "NEVER save to any other location."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    def chat_node(state: State) -> State:
        response = chat_llm.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())



@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, available_tools

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()
    available_tools = [
        {"name": t.name, "description": t.description}
        for t in all_tools
    ]
    agent = create_graph(all_tools)
    print("Agent ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="MCP Agent API",
    description="LangGraph agent with MCP servers",
    version="1.0.0",
    lifespan=lifespan
)



class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default-session"


class ChatResponse(BaseModel):
    response: str
    thread_id: str



@app.get("/health")
def health():
    return {"status": "ok", "agent_ready": agent is not None}


@app.get("/tools")
def get_tools():
    return {"tools": available_tools, "count": len(available_tools)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")

    try:
        result = await agent.ainvoke(
            {"messages": [("user", req.message)]},
            config={"configurable": {"thread_id": req.thread_id}}
        )
        last_message = result["messages"][-1].content

        if isinstance(last_message, str):
            reply = last_message
        elif isinstance(last_message, list):
            texts = [block.get("text", "") for block in last_message if isinstance(block, dict) and block.get("type") == "text"]
            reply = " ".join(texts) if texts else str(last_message)
        else:
            reply = str(last_message)
        
        return ChatResponse(response=reply, thread_id=req.thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))