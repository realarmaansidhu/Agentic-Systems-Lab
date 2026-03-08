# LangGraph code to Implement a Reflection Agent using a StateGraph with separate generation and reflection nodes. The agent will generate a response, then reflect on it and generate a new response based on the reflection, looping until a stopping condition is met (in this case, a maximum number of iterations).

import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_reflection_agent(generate_chain, reflect_chain, max_iterations=6):
    def generation_node(state):
        return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

    def reflection_node(state):
        res = reflect_chain.invoke({"messages": state["messages"]})
        return {"messages": [HumanMessage(content=res.content)]}

    def should_continue(state):
        if len(state["messages"]) > max_iterations:
            return END
        return "reflect"

    builder = StateGraph(state_schema=MessageGraph)
    builder.add_node("generate", generation_node)
    builder.add_node("reflect", reflection_node)
    builder.set_entry_point("generate")
    builder.add_conditional_edges(
        "generate", should_continue, {"reflect": "reflect", END: END}
    )
    builder.add_edge("reflect", "generate")
    return builder.compile()


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm

graph = build_reflection_agent(generate_chain, reflect_chain)
response = graph.invoke(
    {"messages": [HumanMessage(content="Write a mass appealing tweet about AI agents")]}
)

print("\n--- Final Tweet ---")
print(response["messages"][-1].content)