import requests
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("SerperSearch")

API_KEY = os.getenv("SERPER_API_KEY")


@mcp.tool()
def search_web(query: str) -> dict:
    """Search Google via Serper and return top 5 results with title, link, and snippet."""

    if not API_KEY:
        return {"error": "API key not set"}

    try:
        res = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": API_KEY, "Content-Type": "application/json"},
            json={"q": query},
            timeout=5
        )
        res.raise_for_status()
        data = res.json()

        results = []
        for item in data.get("organic", [])[:5]:
            results.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet")
            })

        return {"results": results}

    except requests.exceptions.HTTPError:
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")