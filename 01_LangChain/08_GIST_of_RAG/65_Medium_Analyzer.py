import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# ============================================================================
# PART 1: LOAD AND EMBED DOCUMENTS
# ============================================================================

# Load the document
docx_path = "/Users/armaansidhu/Documents/Projects/GenAI/Agentic-Systems-Lab/01_LANGCHAIN/Deep_Research.docx"
loader = Docx2txtLoader(docx_path)
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

print(f"✅ Loaded and split {len(docs)} documents from Deep_Research.docx")

# ============================================================================
# PART 2: CREATE RAG SYSTEM WITH PINECONE RETRIEVER + OLLAMA LLM
# ============================================================================

# Initialize Pinecone vector store
# Note: LangChain needs an embedding model for query encoding, even with Inference API
# Using nomic-embed-text (768 dims) - you may need to recreate your Pinecone index 
# to use "nomic-embed-text" instead of "llama-text-embed-v2" to avoid dimension mismatch
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = PineconeVectorStore(
    index_name="medium-blogs-embeddings-index",
    embedding=embeddings,
    namespace="default"
)

# Add documents to Pinecone
print("📤 Adding documents to Pinecone...")
vector_store.add_documents(docs)
print("✅ Documents added to Pinecone with embeddings")

# Initialize Ollama LLM
llm = ChatOllama(model="llama3.2", temperature=0.7)

# Create RAG prompt template
rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert analyst. Answer the user's question based ONLY on the provided context from Deep_Research.docx.
    
Context:
{context}

Question: {question}

Answer:"""
)

# Create retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Create the RAG chain: retriever gets documents, format them as context, pass through question
def format_docs(docs):
    return "\n\n---\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================================================================
# PART 3: TEST THE RAG SYSTEM
# ============================================================================

# Example queries
test_queries = [
    "What is the strategic outlook for 2026?",
    "Explain the AI supercycle and its impact on technology equities.",
]

print("\n" + "="*80)
print("RAG SYSTEM READY - TESTING WITH SAMPLE QUERIES")
print("="*80 + "\n")

for query in test_queries:
    print(f"❓ Query: {query}")
    print("-" * 80)
    answer = rag_chain.invoke(query)
    print(f"📝 Answer: {answer}")
    print("\n") 