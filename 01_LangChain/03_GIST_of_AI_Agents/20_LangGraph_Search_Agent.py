# LangGraph code to implement a search agent using the LangChain library. The code defines a search tool that takes a query and returns search results. It then creates an agent using the Groq language model and the search tool, and invokes the agent with a human message asking about the weather in Tokyo. The results of the agent's execution flow are printed in a tree format using the Rich library.

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq as Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

@tool
def search(query: str) -> str:
    """"
    A simple search tool that takes a query and searches for it.
    Arguements: query (str): The query to search for.
    Returns: str: The search results. 
    """
    print(f"Searching for: {query}")
    return "Tokyo weather is Sunny."

llm = Groq(model="llama-3.3-70b-versatile", temperature=0, verbose=True, )
tools = [search]
agent = create_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})

print(result)

# Replace print(result) with this:
# from rich.console import Console
# from rich.tree import Tree

# console = Console()

# # ... after agent.invoke ...
# tree = Tree("🤖 Agent Execution Flow")

# for msg in result["messages"]:
#     branch = tree.add(f"{type(msg).__name__}")
#     # Handle content safely
#     content = getattr(msg, "content", "No content")
#     branch.add(f"[green]{content}[/green]")

# console.print(tree)