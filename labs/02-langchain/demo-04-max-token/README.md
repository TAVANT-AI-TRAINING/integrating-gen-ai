# Max Tokens Limit Demo

A FastAPI application demonstrating how to control response length using the max_tokens parameter.

## Objective

Understand how token limits affect response length:

- Low max_tokens: Short, concise responses
- High max_tokens: Longer, detailed responses
- Useful for cost control and response time optimization

## Project Structure

```
demo-04-max-token/
├── .env                    # Environment variables (API key)
├── main.py                 # FastAPI application
├── pyproject.toml          # Project dependencies
├── README.md               # This file
└── .python-version         # Python version specification
```

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- LLM API key (OpenAI, Gemini, or other supported provider)

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-04-max-token
   ```

   **For Windows:**

   ```cmd
   cd demo-04-max-token
   ```

2. Install dependencies using UV:

   **For Linux/Windows (Same command):**

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

1. Create a `.env` file in the project root:

   **For Linux:**

   ```bash
   touch .env
   ```

   **For Windows (PowerShell):**

   ```powershell
   New-Item -Path .env -ItemType File
   ```

   **For Windows (CMD):**

   ```cmd
   type nul > .env
   ```

2. Add your LLM provider configuration to the `.env` file:

   **For OpenAI:**

   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL_NAME=gpt-4o-mini
   OPENAI_BASE_URL=https://api.openai.com/v1
   ```

   **For Gemini:**

   ```
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_MODEL_NAME=gemini-2.5-flash
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   ```

   **Note**:
   - To get an API key, visit your provider's documentation
   - The `LLM_PROVIDER` variable determines which configuration is used
   - Model names may change over time; refer to your provider's latest documentation

## Running the Application

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## Testing the API

- Open your browser to `http://localhost:8000/docs` for interactive API documentation
- Or send a POST request to `http://localhost:8000/chat` with JSON body:
  ```json
  {
    "message": "Explain the concept of machine learning in detail"
  }
  ```

**Example Response:**

```json
{
  "response": "Machine learning is a subset of artificial intelligence where...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "max_tokens": 100
}
```

## How Max Tokens Works

The `max_tokens` parameter controls response length:

1. **Token Count**: One token ≈ 4 characters on average
2. **Hard Limit**: Responses are truncated when reaching max_tokens limit
3. **Cost Control**: Limits API costs by preventing excessive token usage
4. **Performance**: Shorter limits mean faster response times
5. **Examples**:
   - 50 tokens: Brief response (≈200 characters)
   - 100 tokens: Standard response (≈400 characters)
   - 200+ tokens: Detailed response (≈800+ characters)

## Features

- ✅ Environment variable loading with `python-dotenv`
- ✅ **Token limiting with `max_output_tokens=10`** for strict response truncation
- ✅ FastAPI web service with automatic API documentation
- ✅ Pydantic models for request/response validation
- ✅ Token limiting functionality
- ✅ Interactive API testing via Swagger UI

```

```
