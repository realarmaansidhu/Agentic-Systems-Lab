# LangGraph code to implement a search agent using the LangChain library. The code defines a search tool that takes a query and returns search results. It then creates an agent using the Groq language model and the search tool, and invokes the agent with a human message asking about the weather in Tokyo. The results of the agent's execution flow are printed in a tree format using the Rich library.

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq as Groq
from langchain_tavily import TavilySearch as search
load_dotenv()

llm = Groq(model="llama-3.3-70b-versatile", temperature=0, verbose=True, model_kwargs={"tool_choice": "auto"} ) # Groq' tool_choice model kwargs allows the model to automatically choose which tool to use based on the input query.
tools = [search()]
agent = create_agent(model=llm, tools=tools)

query = input("Enter your query: ")
result = agent.invoke({"messages": [HumanMessage(content=query)]})

print(result)