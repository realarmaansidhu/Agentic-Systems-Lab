# 135 - Custom MCP Server (Math + Weather)
# ==========================================
# Run directly:  python 135_MCP_Weather_Math_Server.py
# Used by:       136_MCP_Agent.py (spawns this as subprocess)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathWeather")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city}: Hot as hell 🔥"

if __name__ == "__main__":
    mcp.run(transport="stdio")