# 132 - MCPDocs: Using llms.txt Documentation via MCP
# ====================================================
# MCPDocs is an MCP server that exposes llms.txt documentation files
# (structured docs designed for LLMs) from LangGraph and LangChain.
#
# It provides two tools:
#   - list_doc_sources: Lists available documentation sources and their URLs
#   - fetch_docs: Fetches content from a specific documentation URL
#
# ─── VS Code Configuration (.vscode/mcp.json) ───
# VS Code uses "servers" key and auto-starts/stops the server via stdio.
#
# {
#   "servers": {
#     "langgraph-langchain-docs": {
#       "command": "uvx",
#       "args": [
#         "--from", "mcpdoc", "mcpdoc",
#         "--urls",
#         "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
#         "LangChain:https://python.langchain.com/llms.txt",
#         "--allowed-domains", "langchain-ai.github.io", "docs.langchain.com", "python.langchain.com",
#         "--transport", "stdio"
#       ]
#     }
#   }
# }
#
# ─── Claude Desktop Configuration (claude_desktop_config.json) ───
# Claude Desktop uses "mcpServers" key and needs absolute path to uvx
# (it doesn't inherit shell PATH).
#
# {
#   "mcpServers": {
#     "langgraph-langchain-docs": {
#       "command": "/opt/anaconda3/bin/uvx",
#       "args": [
#         "--from", "mcpdoc", "mcpdoc",
#         "--urls",
#         "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt",
#         "LangChain:https://python.langchain.com/llms.txt",
#         "--allowed-domains", "langchain-ai.github.io", "docs.langchain.com", "python.langchain.com",
#         "--transport", "stdio"
#       ]
#     }
#   }
# }
#
# ─── Testing with MCP Inspector ───
# Run: npx @modelcontextprotocol/inspector
# Opens a web UI at localhost:6274 to test MCP server tools interactively.
#
# ─── Running as standalone SSE server (for manual testing) ───
# uvx --from mcpdoc mcpdoc \
#     --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" \
#            "LangChain:https://python.langchain.com/llms.txt" \
#     --allowed-domains langchain-ai.github.io docs.langchain.com python.langchain.com \
#     --transport sse --port 8082 --host localhost

# ─── Programmatic Usage ───
from mcpdoc.main import create_server

server = create_server(
    [
        {
            "name": "LangGraph Python",
            "llms_txt": "https://langchain-ai.github.io/langgraph/llms.txt",
        },
        {
            "name": "LangChain Python",
            "llms_txt": "https://python.langchain.com/llms.txt",
        },
    ],
    follow_redirects=True,
    timeout=15.0,
    allowed_domains=["langchain-ai.github.io", "docs.langchain.com", "python.langchain.com"],
)

if __name__ == "__main__":
    server.run(transport="stdio")