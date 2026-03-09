# LangGraph Agentic RAG - Routes questions, retrieves docs, grades relevance,
# generates answers, checks hallucination & answer quality, loops if needed.

import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# ==================== CONFIGURATION ====================
MAX_ITERATIONS = 3

# ==================== STATE SCHEMA ====================
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[str]
    loop_count: int

# ==================== LLM ====================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ==================== TOOLS ====================
tavily = TavilySearch(max_results=3)

# ==================== VECTOR STORE (Pinecone) ====================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = PineconeVectorStore(
    index_name="medium-blogs-embeddings-index",
    embedding=embeddings,
    text_key="text",
    namespace="default",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================== PROMPTS ====================
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at routing questions to a vectorstore or websearch.
    The vectorstore contains documents about: financial analysis, stock market, technology companies, AI industry trends.
    If the question relates to those topics → return ONLY 'vectorstore'.
    For everything else → return ONLY 'websearch'."""),
    ("human", "{question}"),
])

grade_docs_prompt = ChatPromptTemplate.from_messages([
    ("system", """Grade document relevance to the question. Return ONLY 'yes' or 'no'.
    'yes' = document contains keywords or meaning related to question."""),
    ("human", "Document: {document}\n\nQuestion: {question}"),
])

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """Is this answer grounded in the provided facts? Return ONLY 'yes' or 'no'.
    'yes' = answer is supported by the facts."""),
    ("human", "Facts:\n{documents}\n\nAnswer:\n{generation}"),
])

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """Does this answer address the question? Return ONLY 'yes' or 'no'."""),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer using ONLY the provided context. 
    Be concise (~150 words). Cite sources with URLs when available."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# ==================== CHAINS ====================
router_chain = router_prompt | llm
grade_docs_chain = grade_docs_prompt | llm
hallucination_chain = hallucination_prompt | llm
answer_chain = answer_prompt | llm
generation_chain = generation_prompt | llm

# ==================== NODES ====================
def route_node(state: GraphState):
    """Route question to vectorstore or websearch."""
    print("---ROUTE---")
    response = router_chain.invoke({"question": state["question"]})
    source = "websearch" if "web" in response.content.lower() else "vectorstore"
    print(f"  Routed to: {source}")
    return {"web_search": (source == "websearch")}

def retrieve_node(state: GraphState):
    """Retrieve documents from Pinecone vectorstore."""
    print("---RETRIEVE (Pinecone)---")
    docs = retriever.invoke(state["question"])
    doc_texts = [d.page_content for d in docs]
    print(f"  Retrieved {len(doc_texts)} documents")
    return {"documents": doc_texts}

def web_search_node(state: GraphState):
    """Search the web for relevant documents."""
    print("---WEB SEARCH---")
    results = tavily.invoke(state["question"])
    docs = [f"Source: {r['url']}\n{r['content']}" for r in results.get("results", [])]
    print(f"  Found {len(docs)} results")
    return {"documents": docs}

def grade_docs_node(state: GraphState):
    """Grade each document's relevance, keep only relevant ones."""
    print("---GRADE DOCS---")
    filtered = []
    for i, doc in enumerate(state["documents"]):
        score = grade_docs_chain.invoke({"document": doc, "question": state["question"]})
        grade = "yes" in score.content.lower()
        print(f"  Doc {i+1}: {'relevant' if grade else 'not relevant'}")
        if grade:
            filtered.append(doc)
    needs_search = len(filtered) == 0
    if needs_search:
        print("  No relevant docs found - will re-search")
    return {"documents": filtered, "web_search": needs_search}

def generate_node(state: GraphState):
    """Generate answer from retrieved documents."""
    print("---GENERATE---")
    context = "\n\n".join(state["documents"])
    response = generation_chain.invoke({"context": context, "question": state["question"]})
    return {"generation": response.content}

def check_hallucination_node(state: GraphState):
    """Check if the generated answer is grounded in the documents."""
    print("---HALLUCINATION CHECK---")
    docs_text = "\n\n".join(state["documents"])
    score = hallucination_chain.invoke({
        "documents": docs_text,
        "generation": state["generation"]
    })
    grounded = "yes" in score.content.lower()
    print(f"  Grounded: {grounded}")
    if not grounded:
        return {"web_search": True}
    # Also check if it answers the question
    score2 = answer_chain.invoke({
        "question": state["question"],
        "generation": state["generation"]
    })
    useful = "yes" in score2.content.lower()
    print(f"  Answers question: {useful}")
    return {"web_search": not useful, "loop_count": state.get("loop_count", 0) + 1}

# ==================== CONDITIONAL EDGES ====================
def after_grading(state: GraphState):
    """If no relevant docs, search again; otherwise generate."""
    if state["web_search"]:
        return "websearch"
    return "generate"

def after_checking(state: GraphState):
    """If hallucinated or doesn't answer, loop back; otherwise done."""
    if state.get("loop_count", 0) >= MAX_ITERATIONS:
        print("  Max iterations reached - returning best answer")
        return END
    if state["web_search"]:
        return "websearch"
    return END

# ==================== BUILD GRAPH ====================
builder = StateGraph(GraphState)

# Add all nodes
builder.add_node("route", route_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("websearch", web_search_node)
builder.add_node("grade_docs", grade_docs_node)
builder.add_node("generate", generate_node)
builder.add_node("check", check_hallucination_node)

# Wire the graph
builder.set_entry_point("route")
builder.add_conditional_edges("route", lambda s: "websearch" if s["web_search"] else "retrieve")
builder.add_edge("retrieve", "grade_docs")
builder.add_edge("websearch", "grade_docs")
builder.add_conditional_edges("grade_docs", after_grading, {
    "websearch": "websearch",
    "generate": "generate"
})
builder.add_edge("generate", "check")
builder.add_conditional_edges("check", after_checking, {
    "websearch": "websearch",
    END: END
})

graph = builder.compile()

# ==================== RUN ====================
if __name__ == "__main__":
    query = "What is agent memory in AI systems?"
    result = graph.invoke({
        "question": query,
        "generation": "",
        "web_search": False,
        "documents": [],
        "loop_count": 0,
    })
    print("\n=== FINAL ANSWER ===")
    print(result["generation"])