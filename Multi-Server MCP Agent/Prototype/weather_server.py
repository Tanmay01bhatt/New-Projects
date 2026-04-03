# weather_server.py
import requests
from dotenv import load_dotenv
import os
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("WeatherAssistant")
API_KEY = os.getenv("OPENWEATHER_API_KEY")

@mcp.tool()
def get_weather(location: str) -> dict:
    """Get current weather for a city. Returns temp, conditions, humidity, and feels-like."""
    
    if not API_KEY:
        return {"error": "OPENWEATHER_API_KEY not set in environment"}

    try:
        res = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"q": location, "appid": API_KEY, "units": "metric"},
            timeout=5
        )
        res.raise_for_status()
        data = res.json()

        return {
            "location": data.get("name"),
            "country": data["sys"].get("country"),
            "weather": data["weather"][0]["description"],
            "temp": f"{data['main']['temp']}°C",
            "feels_like": f"{data['main']['feels_like']}°C",
            "humidity": f"{data['main']['humidity']}%",
        }

    except requests.exceptions.HTTPError:
        if res.status_code == 404:
            return {"error": f"City '{location}' not found"}
        return {"error": f"HTTP {res.status_code}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")