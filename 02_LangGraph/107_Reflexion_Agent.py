# LangGraph code to Implement a Reflexion Agent with a Tavily search tool using a StateGraph with separate generation and reflection nodes. The agent will generate a response, then reflect on it and generate a new response based on the reflection, looping until a stopping condition is met (in this case, a maximum number of iterations).

import os
from typing import TypedDict, Annotated, List
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ==================== CONFIGURATION ====================
MAX_ITERATIONS = 2  # How many reflection loops before stopping

# ==================== STATE SCHEMA ====================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ==================== PROMPTS ====================
actor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert researcher. Current time: {time}
    
1. {instruction}
2. Reflect on your answer: What's MISSING? What's SUPERFLUOUS?
3. Suggest 1-3 search queries to improve your answer."""),
    MessagesPlaceholder(variable_name="messages"),
])

# ==================== LLM ====================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ==================== TOOLS ====================
tavily = TavilySearch(max_results=3)

def search_tool(queries: List[str]) -> str:
    """Run search queries and return combined results."""
    all_results = []
    for q in queries:
        response = tavily.invoke(q)
        for r in response.get("results", []):
            all_results.append(f"Query: {q}\nSource: {r['url']}\nContent: {r['content']}")
    return "\n\n".join(all_results)

# ==================== NODES ====================
def draft_node(state: AgentState):
    """Generate initial response with critique + search queries."""
    prompt = actor_prompt.partial(
        time=datetime.now().isoformat(),
        instruction="Provide a detailed ~200 word answer to the user's question."
    )
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))
    return {"messages": [response]}

def extract_queries(text: str, fallback: str = "") -> List[str]:
    """Extract search queries from LLM response (looks for numbered/bulleted lists)."""
    import re
    lines = text.split("\n")
    queries = []
    for line in lines:
        # Match lines like: 1. "query" or - "query" or * query
        match = re.search(r'[\d\.\-\*]\s*["\']?(.+?)["\']?\s*$', line.strip())
        if match and len(queries) < 3:
            candidate = match.group(1).strip().strip('"\'')
            if len(candidate) > 10:  # Skip short noise
                queries.append(candidate)
    # Fallback: use provided fallback query
    if not queries and fallback:
        queries = [fallback[:200]]
    return queries[:3]

def tool_node(state: AgentState):
    """Execute search queries extracted from the last LLM message."""
    last_msg = state["messages"][-1]
    # Find original user question as fallback
    fallback = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and not msg.content.startswith("Search Results"):
            fallback = msg.content
            break
    queries = extract_queries(last_msg.content, fallback=fallback)
    print(f"  [Searching: {queries}]")
    results = search_tool(queries)
    return {"messages": [HumanMessage(content=f"Search Results:\n{results}")]}

def revise_node(state: AgentState):
    """Revise answer based on search results and critique."""
    prompt = actor_prompt.partial(
        time=datetime.now().isoformat(),
        instruction="""Revise your previous answer using the search results above.
- Add citations from search results
- Remove superfluous information
- Keep it under 250 words"""
    )
    response = llm.invoke(prompt.format_messages(messages=state["messages"]))
    return {"messages": [response]}

# ==================== GRAPH ====================
def should_continue(state: AgentState):
    """Decide whether to continue reflection loop."""
    if len(state["messages"]) > MAX_ITERATIONS * 3:  # 3 messages per iteration
        return END
    return "tools"

builder = StateGraph(AgentState)
builder.add_node("draft", draft_node)
builder.add_node("tools", tool_node)
builder.add_node("revise", revise_node)

builder.set_entry_point("draft")
builder.add_edge("draft", "tools")
builder.add_edge("tools", "revise")
builder.add_conditional_edges("revise", should_continue, {"tools": "tools", END: END})

graph = builder.compile()

# ==================== RUN ====================
if __name__ == "__main__":
    query = "What are AI-powered SOC startups that raised capital in 2025?"
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    print("\n=== FINAL ANSWER ===")
    print(result["messages"][-1].content)