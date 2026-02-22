# LangChain Code to implement a RAG agent that can answer questions about LangChain documentation

import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()

# ============================================================================
# CONFIG: Toggle to skip re-ingestion after first run
# ============================================================================
UPLOAD_DOCS = False  # Set to True to re-crawl and re-upload docs

# ============================================================================
# PART 1: LOAD AND EMBED DOCUMENTS
# ============================================================================

if UPLOAD_DOCS:
    print("📚 Crawling LangChain documentation with Tavily...")

    # Crawl the documentation
    tavily_crawl = TavilyCrawl()
    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com/",  # String, not list
        "max_depth": 5,
        "extract_depth": "advanced"
    })

    # Convert Tavily results to Document objects
    from langchain_core.documents import Document
    documents = [Document(page_content=doc["raw_content"], metadata={"url": doc["url"]}) for doc in res["results"]]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    print(f"✅ Crawled and split {len(docs)} documents")
else:
    print("⏭️  Skipping crawl/ingestion (UPLOAD_DOCS=False). Using existing Pinecone index.")
    docs = []

# ============================================================================
# PART 2: SETUP PINECONE + RAG CHAIN
# ============================================================================

# Initialize embeddings (for query encoding)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to Pinecone index
vector_store = PineconeVectorStore(
    index_name="langchain-docs-helper-agent",
    embedding=embeddings,
    namespace="default"
)

# Add documents to Pinecone with batch processing (only if UPLOAD_DOCS=True)
if UPLOAD_DOCS and docs:
    print("📤 Adding documents to Pinecone with async batch processing...")

    # --------------------------------------------------------------------------
    # Async batch processing for Pinecone upload
    # - Uses asyncio to run batch uploads concurrently
    # - Each batch is uploaded in a thread pool for I/O efficiency
    # - Errors are caught per batch, so failed batches don't block others
    # --------------------------------------------------------------------------
    import asyncio

    BATCH_SIZE = 100  # Process 100 documents at a time
    total_docs = len(docs)
    batches = [docs[i:i + BATCH_SIZE] for i in range(0, total_docs, BATCH_SIZE)]
    total_batches = len(batches)

    async def async_add_documents(vector_store, batch):
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, vector_store.add_documents, batch)
        except Exception as e:
            # Log error for this batch
            print(f"❌ Async error: {e}")
            raise

    async def batch_upload_to_pinecone_async():
        tasks = [async_add_documents(vector_store, batch) for batch in batches]
        for i, f in enumerate(asyncio.as_completed(tasks), 1):
            try:
                await f
                print(f"✅ Batch {i}/{total_batches} complete ({len(batches[i-1])} docs)")
            except Exception as e:
                print(f"❌ Error in batch {i}: {e}")
                continue
        print(f"✅ Finished! Added {total_docs} documents to Pinecone (async)")

    # Run async upload
    asyncio.run(batch_upload_to_pinecone_async())

# Initialize LLM (keep local Ollama model)
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Create retriever (keep Pinecone + embeddings retrieval path)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Agentic RAG tool: retrieves context and returns both text + raw docs
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    retrieved_docs = retriever.invoke(query)
    # Filter out low-quality/noisy pages (short content, likely nav menus)
    filtered_docs = [
        doc for doc in retrieved_docs
        if len(doc.page_content.strip()) > 100  # Reject very short fragments
    ]
    if not filtered_docs:
        filtered_docs = retrieved_docs  # Fallback if all filtered out
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('url', 'Unknown')}\n\n"
            f"Content: {doc.page_content}"
        )
        for doc in filtered_docs
    )
    return serialized, filtered_docs


def run_llm(query: str) -> Dict[str, Any]:
    """Run agentic RAG for a user query and return answer + retrieved context docs."""
    system_prompt = (
        "You are a helpful assistant for LangChain documentation. "
        "You have access to a tool to retrieve official LangChain docs. "
        "IMPORTANT: Ground your answers primarily in the retrieved documentation and cite sources. "
        "If the docs don't contain specific information, you may provide general context, "
        "but clearly distinguish between what's in the docs vs. general knowledge. "
        "Be honest if you're uncertain or if the docs don't cover something."
    )

    agent = create_agent(llm, tools=[retrieve_context], system_prompt=system_prompt)
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})

    answer = response["messages"][-1].content
    context_docs = []
    for message in response["messages"]:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {
        "answer": answer,
        "context": context_docs
    }

# ============================================================================
# PART 3: INTERACTIVE QUERY LOOP
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 LangChain Documentation Helper - Agentic RAG")
    print("="*70)
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("❓ Ask about LangChain: ").strip()
        
        if question.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        print("\n🔍 Agent retrieving documentation...")
        result = run_llm(question)
        
        print("\n" + "-"*70)
        print(f"📝 Answer:\n{result['answer']}")
        print("-"*70)
        
        # Show source documents with snippets
        if 'context' in result and result['context']:
            print("\n📚 Sources:")
            # Deduplicate by URL while preserving document order
            seen_urls = set()
            unique_docs = []
            for doc in result['context']:
                url = doc.metadata.get('url', 'N/A')
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_docs.append(doc)
            
            for i, doc in enumerate(unique_docs, 1):
                url = doc.metadata.get('url', 'N/A')
                # Extract first 150 chars of content as snippet
                snippet = doc.page_content[:150].replace('\n', ' ').strip()
                if len(doc.page_content) > 150:
                    snippet += "..."
                print(f"   [{i}] {url}")
                print(f"       Snippet: {snippet}\n")
        print()