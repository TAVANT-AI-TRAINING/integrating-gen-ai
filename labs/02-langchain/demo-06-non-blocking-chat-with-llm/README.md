# Non-Blocking Chat with Async LLM

A FastAPI application demonstrating asynchronous LLM calls using async/await pattern for non-blocking operations.

## Objective

Learn how to use async/await for non-blocking LLM calls:

- Define coroutines with async def
- Use .ainvoke() with await keyword
- Handle concurrent requests efficiently
- Improve application responsiveness

## Project Structure

```
demo-06-non-blocking-chat-with-llm/
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
   cd demo-06-non-blocking-chat-with-llm
   ```

   **For Windows:**

   ```cmd
   cd demo-06-non-blocking-chat-with-llm
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
- Or send a POST request to `http://localhost:8000/async-chat`:

**Example Request:**

```json
{
  "message": "Write a short poem about learning async programming"
}
```

**Example Response:**

```json
{
  "response": "In the realm where tasks dance free...",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "execution_type": "async"
}
```

## Async Programming Pattern

The application demonstrates the async/await pattern:

1. **Define Coroutine**: Use `async def` to create an async function
2. **Instantiate LLM**: Create ChatOpenAI instance (works with both sync and async)
3. **Await ainvoke**: Call `.ainvoke()` with `await` keyword inside async function
4. **Execute with asyncio**: Use `asyncio.run()` to execute the coroutine

Benefits:

- **Non-blocking**: Doesn't freeze the application during API calls
- **Concurrent**: Can handle multiple requests simultaneously
- **Efficient**: Better resource utilization with high-traffic scenarios
- **Scalable**: Foundation for production-grade applications

```json
{
  "response": "Generated response from async call...",
  "model": "gemini-2.0-flash",
  "execution_type": "async"
}
```
