# LangGraph code to Implement a ReAct agent with Triple tool and Tavily search

import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq as Groq
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv

load_dotenv()

if os.getenv("GROQ_API_KEY") is None:
    raise ValueError("GROQ_API_KEY environment variable not set")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY environment variable not set")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define the tools
@tool
def triple(x: float) -> float:
    """Triples an input number.
    Args:       x (float): The number to be tripled.
    Returns:    float: The tripled number.
    """
    return x * 3

# Initialize the Tavily search tool
tavily_tool = TavilySearch(api_key=TAVILY_API_KEY)

# Define the tools list
tools = [tavily_tool, triple]

# Initialize the language model (don't bind tools manually; the agent handles it)
llm = Groq(model="llama-3.3-70b-versatile", temperature=0)

# Create the ReAct agent graph
agent = create_react_agent(llm, tools)

# Define the prompt
query = "What is the temperature in Los Angeles and triple the answer?"

# Invoke the agent (it will loop: reason → call tool → observe → repeat)
result = agent.invoke({"messages": [("user", query)]})

# Print the final response
for msg in result["messages"]:
    print(f"\n{msg.type}: {msg.content}")