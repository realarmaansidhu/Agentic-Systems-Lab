# 136 - LangChain Agent using MCP Server

import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

async def main():
    client = MultiServerMCPClient({"math_weather": {
        "command": "python",
        "args": ["135_MCP_Weather_Math_Server.py"],
        "transport": "stdio",
    }})
    tools = await client.get_tools()
    agent = create_agent(ChatGroq(model="llama-3.3-70b-versatile"), tools)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "What is 7+3 multiplied by 4? Also whats the weather in Toronto?"}]})
    print(result["messages"][-1].content)

asyncio.run(main())