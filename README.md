# Azure AI Code Search MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a Python-based server implementing the Model Context Protocol (MCP). It acts as an intelligent agent tool provider, exposing code search capabilities powered by Azure AI Search and Azure OpenAI embeddings to compatible clients like Visual Studio Code Copilot Chat.

The server is designed specifically for scenarios where the project codebase is very large and developers typically work with only a small subset locally. This server searches a comprehensive index of the *entire* codebase, providing more reliable results than local file searches.

## Features

*   **Hybrid Code Search:** Combines keyword search with semantic vector search using Azure OpenAI embeddings for natural language queries (`code_search` tool).
*   **Full File Content Retrieval:** Fetches the complete content of a specific file from the Azure AI Search index (`get_file_content` tool).
*   **Symbol Listing:** Lists functions and classes defined within a specific file, as extracted during indexing (`list_symbols_in_file` tool).
*   **Optimized for Large Codebases:** Explicitly designed to overcome limitations of local file searching when the workspace is incomplete.
*   **MCP SDK Implementation:** Uses the `model-context-protocol` Python SDK for clean handling of the MCP protocol.
*   **SSE Transport:** Supports Server-Sent Events (SSE) for robust integration with clients like VS Code.
*   **STDIO Transport:** Also supports standard I/O for alternative integrations or testing.
*   **NLTK Stop Word Filtering:** Enhances keyword search relevance by filtering common English stop words.

## Prerequisites

*   **Python:** Version 3.9 or higher recommended.
*   **pip & venv:** Standard Python package management tools.
*   **Git:** For cloning the repository.
*   **Azure Subscription:** Access to Azure services.
*   **Azure AI Search Service:** An active Azure AI Search instance.
*   **Azure OpenAI Service:** An active Azure OpenAI instance with access to an embedding model (e.g., `text-embedding-3-large`, `text-embedding-ada-002`). You'll need a deployment of this model.
*   **Pre-existing Azure AI Search Index:** This server *consumes* an existing index. You must have already indexed your codebase into Azure AI Search. The index schema **must** include at least the following fields:
    *   `filePath` (string, filterable, retrievable) - The unique path identifier for the file.
    *   `content` (string, retrievable) - The full content of the file.
    *   `embedding` (vector, searchable, retrievable) - The vector embedding of the file content (or relevant chunks).
    *   `functions` (Collection(Edm.String), retrievable) - Optional, but needed for `list_symbols_in_file`. A list of function names defined in the file.
    *   `classes` (Collection(Edm.String), retrievable) - Optional, but needed for `list_symbols_in_file`. A list of class names defined in the file.
    *(Creating and populating this index is outside the scope of this server - see Azure AI Search documentation for indexing strategies).*

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate Virtual Environment:**
    *   **Linux/macOS:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install Requirements:**
    Create a file named `requirements.txt` in the root of the repository with the following content:

    ```txt
    # requirements.txt
    model-context-protocol
    click
    anyio
    python-dotenv
    openai
    azure-core
    azure-search-documents
    numpy
    starlette
    uvicorn
    httpx
    nltk
    ```

    Then, install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data:** The first time you run the server, it will attempt to download the necessary 'stopwords' corpus from NLTK automatically. Ensure you have an internet connection during the first run.

## Configuration

The server requires Azure credentials and endpoints, which should be provided via a `.env` file in the root directory of the project.

1.  **Create `.env` file:** Create a file named `.env`.
2.  **Add Environment Variables:** Copy the following template into your `.env` file and replace the placeholder values with your actual Azure service details:

    ```dotenv
    # .env file
    AZURE_SEARCH_ENDPOINT="https://YOUR_SEARCH_SERVICE_NAME.search.windows.net"
    AZURE_SEARCH_INDEX_NAME="your-code-index-name"
    AZURE_SEARCH_ADMIN_KEY="YOUR_AZURE_AI_SEARCH_ADMIN_API_KEY"
    AZURE_OPENAI_ENDPOINT="https://YOUR_AOAI_RESOURCE_NAME.openai.azure.com/"
    AZURE_OPENAI_API_KEY="YOUR_AZURE_OPENAI_API_KEY"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT="your-embedding-deployment-name" # e.g., text-embedding-3-large

    # Optional: Override default server settings
    # PORT=8000
    # HOST="0.0.0.0"
    # LOG_LEVEL="info"
    ```

    *   `AZURE_SEARCH_ENDPOINT`: Full URL of your Azure AI Search service.
    *   `AZURE_SEARCH_INDEX_NAME`: The name of the index containing your codebase.
    *   `AZURE_SEARCH_ADMIN_KEY`: An Admin API key for your Azure AI Search service (required for querying with `AzureKeyCredential`). *Consider using Managed Identity and `DefaultAzureCredential` in production environments for better security.*
    *   `AZURE_OPENAI_ENDPOINT`: Full URL of your Azure OpenAI service.
    *   `AZURE_OPENAI_API_KEY`: API key for your Azure OpenAI service.
    *   `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`: The specific deployment name of your embedding model within Azure OpenAI.

## Running the Server

Ensure your virtual environment is activated and the `.env` file is configured correctly.

*   **Start with SSE Transport (Default for VS Code):**
    ```bash
    python mcp_sdk_server.py # Assuming your file is named mcp_sdk_server.py
    ```
    The server will start listening on `http://0.0.0.0:8000` by default.

*   **Start on a Different Port:**
    ```bash
    python mcp_sdk_server.py --port 8080
    ```

*   **Start with STDIO Transport:**
    ```bash
    python mcp_sdk_server.py --transport stdio
    ```

Keep the terminal window open while you intend to use the server. Monitor the logs for connection information, tool calls, and potential errors.

## VS Code Integration (e.g., Copilot Chat)

To allow VS Code features like Copilot Chat to use this server as an agent tool:

1.  **Ensure the Server is Running:** Start the server using the `sse` transport as described above. Note the host and port (e.g., `http://localhost:8000`).

2.  **Open VS Code User Settings (JSON):**
    *   Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS).
    *   Type `Preferences: Open User Settings (JSON)` and select it.

3.  **Add MCP Server Configuration:** Add the following JSON object to your `settings.json`. The exact location might vary depending on the VS Code extension providing the MCP client:

    **Option A (Directly at Root):** Try this structure first.
    ```json
    // In settings.json
    {
        // ... other settings ...

        "mcp": {
            "servers": {
                "code_Search": { // This name becomes @code_search in chat
                    "type": "sse", // MUST be "sse"
                    "url": "http://localhost:8000/sse" // URL to the GET /sse endpoint
                                                      // Adjust port if you changed it!
                }
                // You could add other custom MCP servers here
            }
        },

        // ... more settings ...
    }
    ```

    **Option B (Nested under Copilot):** If Option A doesn't work, try nesting it under `github.copilot.advanced` (this was common previously).
    ```json
    // In settings.json
    {
        // ... other settings ...

        "github.copilot.advanced": {
            // ... potentially other advanced settings ...
            "mcp": {
                "servers": {
                    "code_search": { // This name becomes @code_search in chat
                        "type": "sse",
                        "url": "http://localhost:8000/sse"
                    }
                }
            }
        },

        // ... more settings ...
    }
    ```

    *   Replace `"code_search"` with your preferred name for invoking the agent (e.g., `@AzureSearch`, `@MyCode`).
    *   Ensure `"type": "sse"` matches the server's transport.
    *   Verify the `"url"` points to the correct `host:port/sse` where your server is running. Use `localhost` or `127.0.0.1` for the host if running locally.
    *   **Note:** If neither structure works, consult the documentation for the specific VS Code extension (e.g., Copilot Chat, specific Agent extensions) you are using to see where it expects MCP server configurations.

4.  **Save `settings.json`**.

5.  **Restart VS Code:** Completely close and reopen VS Code to ensure it picks up the new settings.

6.  **Test in Chat:**
    *   Open the Copilot ChaAgent view (or equivalent).
    *   Type `#code_search` (or the name you chose). It might appear in autocompletion.
    *   Ask questions that should trigger the tools:
        *   `#code_search where is the EnlistmentCleaner class defined?`
        *   `#code_search show me how database connections are configured`
        *   (After a search) `#code_search get file content for src/database/connection.py`
        *   (After a search) `#code_search list symbols in file src/api/auth.py`
    *   Observe the chat response and the server logs in your terminal.

## Troubleshooting

*   **Server Fails to Start:** Check the terminal logs. Common issues include missing environment variables in `.env` or incorrect Azure credentials/endpoints. Ensure NLTK data download succeeds on first run.
*   **Connection Refused (VS Code):** Ensure the server is running and accessible. Verify the `url` in `settings.json` matches the host and port shown in the server startup logs. Check firewalls if not running locally.
*   **404 Not Found Errors:** Double-check the `url` path in `settings.json` is exactly `/sse`.
*   **Tool Errors in Chat:** Check the server logs for Python exceptions during tool execution (e.g., Azure API errors, embedding failures, search failures). Improve error handling in the server code if needed.
*   **Tool Not Used as Expected:** Refine the tool descriptions in `list_tools_handler` to be clearer about the tool's purpose and when it should be used (as demonstrated in the code). Restart VS Code after changing descriptions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming you add an MIT license file).