#!/usr/bin/env python3
import os
import logging
import re
import asyncio
import json
import functools
from functools import lru_cache
from typing import List, Dict, Optional, Any, Sequence

import click
import anyio
from dotenv import load_dotenv
import numpy as np
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

# --- NLTK Import ---
import nltk
from nltk.corpus import stopwords
# --- End NLTK Import ---

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server

from starlette.applications import Starlette
from starlette.routing import Mount, Route
import uvicorn

# ── Constants ───────────────────────────────────────────────────────────────────
AZURE_OPENAI_API_VERSION = "2024-03-01-preview"
DEFAULT_SEARCH_TOP_K = 10
EMBEDDING_CACHE_SIZE = 500
SNIPPET_MAX_LENGTH = 500

# ── Configure Logging ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CodeSearchMCP_SDK")

# ── CodeSearchService (With NLTK Integration) ──────────────────────────────────
class CodeSearchService:
    def __init__(self):
        load_dotenv()
        missing = []
        env_vars = {
            "AZURE_SEARCH_ENDPOINT": "search_endpoint",
            "AZURE_SEARCH_INDEX_NAME": "index_name",
            "AZURE_SEARCH_ADMIN_KEY": "search_admin_key",
            "AZURE_OPENAI_ENDPOINT": "openai_endpoint",
            "AZURE_OPENAI_API_KEY": "openai_api_key",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "embed_model"
        }
        for env_name, attr_name in env_vars.items():
            val = os.getenv(env_name)
            setattr(self, attr_name, val)
            if not val:
                missing.append(env_name)
        if missing:
            error_msg = f"Missing required environment variables: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # --- NLTK Stopwords Setup ---
        try:
            logger.info("Checking/downloading NLTK stopwords corpus...")
            nltk.data.find('corpora/stopwords')
            logger.debug("NLTK stopwords corpus found.")
        except LookupError:
            logger.info("NLTK stopwords corpus not found, downloading...")
            try:
                nltk.download('stopwords', quiet=True) # Download quietly
                logger.info("NLTK stopwords corpus downloaded successfully.")
            except Exception as nltk_e:
                logger.error(f"Failed to download NLTK stopwords: {nltk_e}. Keyword generation may be suboptimal.", exc_info=True)
                # Decide if this is fatal or just a warning. For now, warn and continue.
                self.stop_words = set() # Use empty set if download fails
            else:
                 self.stop_words = set(stopwords.words('english'))
        else:
             self.stop_words = set(stopwords.words('english'))
        logger.info(f"Loaded {len(self.stop_words)} English stop words.")
        # --- End NLTK Setup ---

        try:
            search_credential = AzureKeyCredential(self.search_admin_key)
            self.search_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=self.index_name,
                credential=search_credential
            )
            logger.info("Azure AI Search client initialized successfully using Admin Key.")

            self.aoai = AzureOpenAI(
                api_key=self.openai_api_key,
                azure_endpoint=self.openai_endpoint,
                api_version=AZURE_OPENAI_API_VERSION
            )
            logger.info("Azure OpenAI client initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {e}", exc_info=True)
            raise RuntimeError("Azure client initialization failed.") from e

    @lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text: return None
        try:
            response = self.aoai.embeddings.create(input=[text], model=self.embed_model)
            if response.data and isinstance(response.data, list) and len(response.data) > 0 and response.data[0].embedding:
                 return np.array(response.data[0].embedding, dtype=np.float32)
            else:
                 logger.error("Received empty or unexpected embedding data from Azure OpenAI.")
                 return None
        except Exception as e:
            logger.error(f"Error generating embedding for text '{text[:50]}...': {e}", exc_info=True)
            return None

    # --- UPDATED Keyword Generation with NLTK ---
    def _generate_search_keywords(self, query: str) -> str:
        """Generates weighted keywords, removing common English stop words."""
        tokens = re.findall(r"\b\w+\b", query.lower())
        # Filter out stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        # Apply weighting to remaining tokens
        weighted_tokens = [f"{w}^{min(3, max(1, len(w)//3))}" for w in filtered_tokens if len(w) > 2]
        # Join or fallback to original query if filtering removed everything useful
        search_text = " ".join(weighted_tokens) if weighted_tokens else query
        logger.debug(f"Generated search keywords (NLTK filtered): '{search_text}' from query: '{query}'")
        return search_text
    # --- End UPDATED Keyword Generation ---

    def _escape_filepath_for_filter(self, file_path: str) -> str:
        return file_path.replace("'", "''")

    def search_code(self, query: str, top_k: int = DEFAULT_SEARCH_TOP_K) -> List[Dict]:
        """Performs the hybrid code search (synchronous implementation)."""
        logger.info(f"Performing code search for query: '{query}', top_k={top_k}")
        embedding = self.get_embedding(query)
        if embedding is None:
            logger.error("Failed to generate embedding for query.")
            return []

        search_text = self._generate_search_keywords(query) # Uses updated method
        try:
            vector_query = VectorizedQuery(vector=embedding.tolist(), k_nearest_neighbors=top_k * 2, fields="embedding")
            results = self.search_client.search(
                search_text=search_text,
                vector_queries=[vector_query],
                select=["filePath", "content", "functions", "classes"],
                top=top_k
            )
            search_results = []
            for doc in list(results):
                if "filePath" in doc and "@search.score" in doc:
                    search_results.append({
                        "file": doc["filePath"],
                        "score": doc["@search.score"],
                        "snippet": (doc.get("content") or "")[:SNIPPET_MAX_LENGTH]
                    })
                else:
                    logger.warning(f"Search result missing filePath or @search.score: {doc.get('filePath', 'N/A')}")
            logger.info(f"Found {len(search_results)} results for query '{query}'.")
            return search_results
        except Exception as e:
            logger.error(f"Error during Azure AI Search call for query '{query}': {e}", exc_info=True)
            return []

    def get_file_content(self, file_path: str) -> Optional[str]:
        """Retrieves the full content of a specific file from the index (synchronous)."""
        logger.info(f"Attempting to retrieve content for file: {file_path}")
        
        # Handle both full URLs and path-only formats
        if not file_path.startswith("https://"):
            # If it's just a path, search for documents that end with this path
            escaped_path = self._escape_filepath_for_filter(file_path)
            filter_expr = f"search.ismatch('*{escaped_path}', 'filePath')"
            logger.debug(f"Using wildcard filter for path-only: {filter_expr}")
        else:
            # If it's a full URL, do exact match
            escaped_path = self._escape_filepath_for_filter(file_path)
            filter_expr = f"filePath eq '{escaped_path}'"
            logger.debug(f"Using exact match filter for URL: {filter_expr}")
        
        try:
            results = self.search_client.search(
                search_text=None, filter=filter_expr, select=["content", "filePath"], top=1
            )
            doc = next(iter(results), None)
            if doc and "content" in doc:
                logger.info(f"Successfully retrieved content for: {file_path} (found as: {doc.get('filePath', 'N/A')})")
                return doc["content"]
            else:
                logger.warning(f"File not found or content missing in index for: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving file content for {file_path}: {e}", exc_info=True)
            return None

    def list_symbols_in_file(self, file_path: str) -> Optional[Dict[str, List[str]]]:
        """Retrieves indexed functions and classes for a specific file (synchronous)."""
        logger.info(f"Attempting to retrieve symbols for file: {file_path}")
        
        # Handle both full URLs and path-only formats
        if not file_path.startswith("https://"):
            # If it's just a path, search for documents that end with this path
            escaped_path = self._escape_filepath_for_filter(file_path)
            filter_expr = f"search.ismatch('*{escaped_path}', 'filePath')"
            logger.debug(f"Using wildcard filter for path-only: {filter_expr}")
        else:
            # If it's a full URL, do exact match
            escaped_path = self._escape_filepath_for_filter(file_path)
            filter_expr = f"filePath eq '{escaped_path}'"
            logger.debug(f"Using exact match filter for URL: {filter_expr}")
        
        try:
            results = self.search_client.search(
                search_text=None, filter=filter_expr, select=["functions", "classes", "filePath"], top=1
            )
            doc = next(iter(results), None)
            if doc:
                symbols = {
                    "functions": doc.get("functions", []),
                    "classes": doc.get("classes", [])
                }
                logger.info(f"Successfully retrieved symbols for: {file_path} (found as: {doc.get('filePath', 'N/A')})")
                return symbols
            else:
                logger.warning(f"File not found or symbols missing in index for: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving symbols for {file_path}: {e}", exc_info=True)
            return None

# --- Main Application Setup ---

@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="sse",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    """Runs the CodeSearch MCP Server."""
    try:
        code_search_service = CodeSearchService()
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Failed to initialize CodeSearchService: {e}")
        return 1

    app = Server("mcp-code-searcher")

    @app.call_tool()
    async def handle_tool_call(
        name: str, arguments: dict
    ) -> Sequence[types.TextContent]:
        """Handles requests to call any of the registered tools."""
        logger.debug(f"handle_tool_call received: name='{name}', args={arguments}")
        loop = asyncio.get_running_loop()
        user_friendly_error_message = f"An internal error occurred while trying to execute the '{name}' tool. Please try again later."

        if name == "code_search":
            query = arguments.get("query")
            if not query: raise ValueError("Missing required argument 'query'")
            top_k = arguments.get("top_k", DEFAULT_SEARCH_TOP_K)
            if not isinstance(top_k, int):
                try: top_k = int(top_k)
                except (ValueError, TypeError): top_k = DEFAULT_SEARCH_TOP_K

            try:
                search_func = functools.partial(code_search_service.search_code, query, top_k)
                results = await loop.run_in_executor(None, search_func)
                if not results: return [types.TextContent(type="text", text="No results found.")]
                formatted_results = []
                for i, res in enumerate(results):
                     formatted_results.append(
                        f"**Result {i+1}:**\n"
                        f"*File:* `{res.get('file', 'N/A')}`\n"
                        f"*Score:* {res.get('score', 'N/A'):.4f}\n"
                        f"*Snippet:*\n```\n{res.get('snippet', '')}\n```\n" + "-"*20
                    )
                return [types.TextContent(type="text", text="\n".join(formatted_results))]
            except Exception as e:
                logger.error(f"Error executing code_search tool for query '{query}': {e}", exc_info=True)
                return [types.TextContent(type="text", text=user_friendly_error_message)]

        elif name == "get_file_content":
            file_path = arguments.get("filePath")
            if not file_path: raise ValueError("Missing required argument 'filePath'")
            try:
                get_content_func = functools.partial(code_search_service.get_file_content, file_path)
                content = await loop.run_in_executor(None, get_content_func)
                if content is not None:
                    return [types.TextContent(type="text", text=f"--- Content for `{file_path}` ---\n\n```\n{content}\n```")]
                else:
                    return [types.TextContent(type="text", text=f"Could not retrieve content for file `{file_path}`. It might not exist in the index.")]
            except Exception as e:
                logger.error(f"Error executing get_file_content tool for {file_path}: {e}", exc_info=True)
                return [types.TextContent(type="text", text=user_friendly_error_message)]

        elif name == "list_symbols_in_file":
            file_path = arguments.get("filePath")
            if not file_path: raise ValueError("Missing required argument 'filePath'")
            try:
                list_symbols_func = functools.partial(code_search_service.list_symbols_in_file, file_path)
                symbols = await loop.run_in_executor(None, list_symbols_func)
                if symbols is not None:
                    output_lines = [f"--- Symbols in `{file_path}` ---"]
                    classes = symbols.get("classes", [])
                    functions = symbols.get("functions", [])
                    if classes:
                        output_lines.append("\n**Classes:**")
                        output_lines.extend([f"- `{c}`" for c in classes])
                    else: output_lines.append("\n**Classes:** None found in index.")
                    if functions:
                        output_lines.append("\n**Functions:**")
                        output_lines.extend([f"- `{f}()`" for f in functions])
                    else: output_lines.append("\n**Functions:** None found in index.")
                    return [types.TextContent(type="text", text="\n".join(output_lines))]
                else:
                    return [types.TextContent(type="text", text=f"Could not retrieve symbols for file `{file_path}`. It might not exist or have symbols indexed.")]
            except Exception as e:
                logger.error(f"Error executing list_symbols_in_file tool for {file_path}: {e}", exc_info=True)
                return [types.TextContent(type="text", text=user_friendly_error_message)]
        else:
            raise ValueError(f"Unknown tool requested: {name}")

    @app.list_tools()
    async def list_tools_handler() -> list[types.Tool]:
        """Handles requests to list available tools."""
        logger.debug("list_tools called")
        # Tool definitions remain the same as the previous version with detailed descriptions
        code_search_tool = types.Tool(
            name="code_search",
            description=(
                "Performs a hybrid (keyword + semantic vector) search across a comprehensive index of the project's "
                "**entire, large codebase** using Azure AI Search. **Crucially, the local VS Code workspace often contains "
                "only a small subset of this full codebase.** Therefore, standard file searches (like grep/glob) "
                "will likely miss relevant code or definitions. **This tool searches the complete index, making it the "
                "necessary and preferred method for reliable code discovery, including locating specific symbol definitions "
                "(classes, functions, variables) that might exist outside your current workspace.** Use it for finding code "
                "based on natural language descriptions, concepts, error messages, or specific symbols. "
                "Examples: 'where is the EnlistmentCleaner class defined?', 'how is user authentication handled?', "
                "'find code related to database pooling', 'search for CentralSecrets usage'."
            ),
            inputSchema={
                "type": "object", "required": ["query"], "properties": {
                    "query": { "type": "string", "description": (
                            "A natural language query OR a specific symbol name (class, function, variable) to find "
                            "within the **entire codebase index** (necessary due to potentially incomplete local workspace). "
                            "Examples: 'function to handle user login', 'database connection setup', "
                            "'usage of the CentralSecrets class', 'EnlistmentCleaner', 'fix for timeout error'."
                        )
                    },
                    "top_k": { "type": "integer", "description": (
                            f"Optional. The maximum number of code snippets to return from the comprehensive index. Defaults to {DEFAULT_SEARCH_TOP_K} if not specified."
                        )
                    }
                },
            },
        )
        get_content_tool = types.Tool(
            name="get_file_content",
            description=(
                "Retrieves the full content of a specific file *from the comprehensive codebase index*. "
                "Use this tool AFTER a 'code_search' when the user needs to see the complete file for a given search result snippet, "
                "especially if the file might not be present in their local workspace, or if the snippet is insufficient."
            ),
            inputSchema={
                "type": "object", "required": ["filePath"], "properties": {
                    "filePath": { "type": "string", "description": (
                            "The exact file path obtained from a previous 'code_search' result or user input, representing a file within the indexed codebase."
                        )
                    }
                },
            },
        )
        list_symbols_tool = types.Tool(
            name="list_symbols_in_file",
            description=(
                "Lists the functions and classes defined within a specific file, based on information stored *in the comprehensive codebase index*. "
                "Useful for quickly understanding the structure and main components of a file found via 'code_search' or specified by the user, "
                "particularly if the file isn't fully available in the local workspace or if the user wants a summary before viewing the full content."
            ),
            inputSchema={
                "type": "object", "required": ["filePath"], "properties": {
                    "filePath": { "type": "string", "description": (
                            "The exact file path obtained from a previous 'code_search' result or user input, representing a file within the indexed codebase."
                        )
                    }
                },
            },
        )
        return [code_search_tool, get_content_tool, list_symbols_tool]

    # --- Transport Setup (Remains the same) ---
    if transport == "sse":
        logger.info(f"Starting SSE server on port {port}")
        sse_transport = SseServerTransport("/messages/")
        async def handle_sse_connection(request):
            async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())
        starlette_app = Starlette(debug=True, routes=[
                Route("/sse", endpoint=handle_sse_connection),
                Mount("/messages/", app=sse_transport.handle_post_message),
            ])
        uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_config=None)
    elif transport == "stdio":
        logger.info("Starting stdio server")
        async def arun_stdio():
            async with stdio_server() as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())
        try: anyio.run(arun_stdio)
        except Exception as e: logger.error(f"Error running stdio server: {e}", exc_info=True); return 1
    else: logger.error(f"Invalid transport type specified: {transport}"); return 1
    return 0

# --- Entrypoint ---
if __name__ == "__main__":
    main()