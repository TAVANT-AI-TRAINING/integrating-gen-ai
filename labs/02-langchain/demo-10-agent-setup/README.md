# LangChain Multi-Provider Agent Setup

A FastAPI application demonstrating how to set up and configure a LangChain Agent with multi-provider LLM support and external tools.

## Features

- **Multi-Provider LLM Support**: Works with OpenAI, Gemini, and other OpenAI-compatible providers
- **Modern Agent Creation**: Uses LangChain's `create_agent` function for streamlined setup
- **Multi-Tool Integration**: Combines multiple tools (order status and geolocation)
- **Pydantic Response Models**: Structured, validated API responses
- **FastAPI Web Interface**: RESTful API endpoints with automatic documentation
- **Root Endpoint**: API information and metadata
- **Tool Integration**: Easy to extend with additional tools

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- LLM API keys (OpenAI, Gemini, or compatible provider)
- API key from [apiip.net](https://apiip.net/) (for geolocation tool)

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-10-agent-setup
   ```

   **For Windows:**

   ```cmd
   cd demo-10-agent-setup
   ```

2. Install dependencies using UV:

   **For Linux/macOS/Windows (Same command):**

   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project environment

3. Activate the virtual environment:

   **For Linux/macOS:**

   ```bash
   source .venv/bin/activate
   ```

   **For Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **For Windows (CMD):**

   ```cmd
   .venv\Scripts\activate.bat
   ```

   **Note**: If using `uv run` command (as shown in Running section), activation is optional as `uv run` automatically uses the virtual environment.

## Configuration

1. Create or update your `.env` file in the project root:

### For OpenAI

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
APIIP_API_KEY=your_apiip_key_here
```

### For Google Gemini

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL_NAME=gemini-2.5-flash
APIIP_API_KEY=your_apiip_key_here
```

**How to get API keys:**

- **OpenAI**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
- **Gemini**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
- **apiip.net**: Visit [apiip.net](https://apiip.net/) and sign in with Google

## Running the Application

**For Linux/macOS/Windows:**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The service will be available at `http://localhost:8000`

## API Usage

### Root Endpoint

**URL:** `GET /`

Returns API information including the currently configured provider and model.

**Response:**

```json
{
  "title": "LangChain Multi-Provider Agent",
  "description": "Demonstrates LangChain agent setup with multiple tools",
  "version": "1.0.0",
  "provider": "OPENAI",
  "model": "gpt-4o-mini",
  "endpoints": ["/", "/ask_agent", "/docs", "/openapi.json"]
}
```

### Agent Query Endpoint

**URL:** `GET /ask_agent`

Query the agent with a natural language question. The agent will use available tools to provide a comprehensive answer.

**Parameters:**

- `query` (string, required): Your question or request

**Response:**

```json
{
  "query": "What's the status of order ABC-123 and where am I located?",
  "response": "Based on the information I gathered:\n\n**Your Location:** You are currently located in Mountain View, California, United States\n\n**Order Status:** Your order ABC-123 is currently 'Shipped' and is expected to be delivered on 2024-10-25 via FedEx.",
  "model": "gpt-4o-mini",
  "provider": "OPENAI"
}
```

**Example Queries:**

```bash
# Check order status
curl "http://localhost:8000/ask_agent?query=Check%20my%20order%20status%20for%20order%20ID%20ABC-123"

# Get location
curl "http://localhost:8000/ask_agent?query=Where%20am%20I%20located%20right%20now%3F"

# Multi-tool query
curl "http://localhost:8000/ask_agent?query=What%27s%20the%20status%20of%20order%20DEF-456%20and%20where%20am%20I%20located%3F"
```

## API Documentation

Once the server is running, visit:

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Key Concepts

### Multi-Provider LLM Support

The `initialize_llm_client()` function provides:

- Dynamic provider selection via `LLM_PROVIDER` environment variable
- Provider-specific configuration loading
- Flexible base URL support for custom endpoints
- Consistent interface across different providers

### Agent Creation Process

1. **Model Initialization**: Create a chat model with multi-provider support
2. **Tool Definition**: Define tools using `@tool` decorator in `agent_tools.py`
3. **Tool Collection**: Collect tools into a list for the agent
4. **Agent Creation**: Use `create_agent(model, tools=tools, system_prompt="...")` setup
5. **Query Handling**: Invoke agent with message-based API

### Tool Types

- **Internal Tools**: `get_order_status` - Simulates an order database lookup
- **External Tools**: `get_user_location` - Calls external geolocation API via IP

## Project Structure

```
demo-10-agent-setup/
├── main.py                # Main FastAPI application
├── agent_tools.py         # Tool definitions (order status, location)
├── pyproject.toml         # Project dependencies
├── .env                   # Environment variables (create this file)
├── .python-version        # Python version specification
├── .gitignore            # Git ignore rules
└── README.md             # This file
```
