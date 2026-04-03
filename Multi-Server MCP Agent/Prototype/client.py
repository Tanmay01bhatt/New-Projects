import asyncio
import os
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
import logging
logging.basicConfig(level=logging.ERROR)

# Filesystem 
WORKSPACE = os.path.abspath("./agent_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

server_configs = {
    "weather": {
        "command": "python",
        "args": ["weather_server.py"],
        "transport": "stdio",
    },
    "serper": {
        "command": "python",
        "args": ["serper_server.py"],
        "transport": "stdio",
    },
    "filesystem": {                         
        "command": "npx",
        "args":["-y", "@modelcontextprotocol/server-filesystem", WORKSPACE],
        "transport": "stdio",
    },
}


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
         "3. File system — read and write files in the agent workspace\n"
          f"IMPORTANT: The workspace directory is: {WORKSPACE}\n"
         "When saving ANY file, always pass the FULL path by combining the workspace "
         "directory with the filename. For example, to save 'weather.txt' use: "
         f"'{WORKSPACE}\\weather.txt'\n"
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


async def main():
    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()

    agent = create_graph(all_tools)

    print("MCP Agent ready — weather, search, and filesystem connected.")
    print(f"Workspace: {WORKSPACE}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            break
        try:
            response = await agent.ainvoke(
                {"messages": [("user", user_input)]},
                config={"configurable": {"thread_id": "multi-server-session"}}
            )
            print("AI:", response["messages"][-1].content)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())