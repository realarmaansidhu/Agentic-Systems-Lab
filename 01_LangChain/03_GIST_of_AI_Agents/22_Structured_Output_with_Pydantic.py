# LangGraph code to implement a search agent with structured output using Pydantic.

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq as Groq
from langchain_tavily import TavilySearch as search
from pydantic import BaseModel, Field

load_dotenv()


class AgentResponse(BaseModel):
    """Agent response schema."""
    response: str = Field(description="response from agent")
    confidence: float = Field(description="confidence score of the response")

llm = Groq(model="llama-3.3-70b-versatile", temperature=0, verbose=True, model_kwargs={"tool_choice": "auto"}) # Groq' tool_choice model kwargs allows the model to automatically choose which tool to use based on the input query.
tools = [search()]
agent = create_agent(model=llm, tools=tools)
query = input("Enter your query: ")
result = agent.invoke({"messages": [HumanMessage(content=query)]})

try:
    # Ask LLM to structure the output
    structured_llm = llm.with_structured_output(AgentResponse)
    structured_response = structured_llm.invoke(
        f"Based on this answer, create a structured response: {result}"
    )
    
    print("\n" + "="*50)
    print("✅ VALIDATED RESPONSE")
    print("="*50)
    print(f"Response: {structured_response.response}")
    print(f"Confidence: {structured_response.confidence * 100:.0f}%")
    print("="*50)
    
except Exception as e:
    print(f"❌ Validation Error: {e}")