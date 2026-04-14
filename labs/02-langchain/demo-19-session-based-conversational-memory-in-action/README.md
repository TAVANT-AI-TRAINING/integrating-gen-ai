# Simple Memory Agent Demo

A FastAPI application demonstrating how to build a stateful conversational agent using LangChain Expression Language (LCEL) with session-based memory. This demo shows how to make an agent remember previous user inputs across multiple turns of conversation using `MessagesPlaceholder`, `InMemoryChatMessageHistory`, and `RunnableWithMessageHistory`.

## Overview

This application implements a friendly greeter agent that maintains conversation history per session. Each session has its own isolated memory, allowing the agent to remember context (like the user's name) within that session while keeping different sessions completely separate.

## Features

- **Memory-Aware LCEL Chain**: Adds a memory slot for previous conversation history using `MessagesPlaceholder`
- **Session-Based Conversation**: Each conversation is identified by a unique `session_id`, maintaining isolated memory per session
- **In-Memory Session Store**: Uses `InMemoryChatMessageHistory` to store conversation history for each session
- **LCEL Pipe Syntax**: Demonstrates chaining `Prompt → LLM` using the `|` operator
- **FastAPI Integration**: Exposes RESTful endpoints for multi-turn conversations
- **Google Gemini Integration**: Uses Google Gemini API via LangChain's `ChatOpenAI` adapter

## Architecture

The application uses the following key components:

1. **ChatPromptTemplate**: Defines the system prompt and includes a `MessagesPlaceholder` for chat history
2. **InMemoryChatMessageHistory**: Stores conversation messages for each session
3. **RunnableWithMessageHistory**: Wraps the core chain to automatically manage message history per session
4. **Session Store**: A dictionary-based in-memory store that maps `session_id` to `ChatMessageHistory` instances

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-19-session-based-conversational-memory-in-action
   ```

   **For Windows:**

   ```cmd
   cd demo-19-session-based-conversational-memory-in-action
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

   **For Linux/macOS:**

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

2. Add your API configuration to the `.env` file:

   **For OpenAI:**

   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL_NAME=gpt-4o-mini
   ```

   **For Google Gemini:**

   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   GEMINI_MODEL_NAME=gemini-2.5-flash
   ```

   **How to get API keys:**
   - **OpenAI**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)

   **Note**: The model names can be updated to any supported model. Model names may change over time, so always refer to the latest options in the provider's documentation.

   All three environment variables are required. The application will raise an error if any are missing.

## Running the Application

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or run directly with Python:

```bash
uv run python main.py
```

The app will start at `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## API Endpoints

### GET /new-session

Generate a new unique chat session ID for starting a new conversation.

**Response Example:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Note**: Do not copy-paste the example session ID. Generate a new one using this endpoint.

### POST /chat

Send a message to the stateful greeter agent. The agent remembers context within the same session.

**Request Body:**

```json
{
  "input": "User message (string)",
  "session_id": "Unique session identifier (string)"
}
```

**Request Example:**

```bash
POST /chat
Content-Type: application/json

{
  "input": "Hi! My name is Alex.",
  "session_id": "user_abc_123"
}
```

**Response Example:**

```json
{
  "session_id": "user_abc_123",
  "user_input": "Hi! My name is Alex.",
  "agent_response": "Hello Alex! It's great to meet you. How can I help you today?"
}
```

### Follow-up Message (Same Session)

**Request:**

```bash
POST /chat
Content-Type: application/json

{
  "input": "What is my name?",
  "session_id": "user_abc_123"
}
```

**Response:**

```json
{
  "session_id": "user_abc_123",
  "user_input": "What is my name?",
  "agent_response": "Your name is Alex."
}
```

The agent remembers the user's name from the previous message in the same session.

### New Session Example

**Request:**

```bash
POST /chat
Content-Type: application/json

{
  "input": "Do you remember me?",
  "session_id": "user_xyz_789"
}
```

**Response:**

```json
{
  "session_id": "user_xyz_789",
  "user_input": "Do you remember me?",
  "agent_response": "I don't have any record. What's your name?"
}
```

The agent has no memory of previous conversations because this is a new session with a different `session_id`.

## Testing the API

### Using FastAPI Interactive Docs

Visit `http://localhost:8000/docs` for FastAPI's interactive Swagger UI where you can test all endpoints directly.

## How It Works

1. **Session Creation**: When a new `session_id` is used, `get_session_history()` creates a new `InMemoryChatMessageHistory` instance for that session.

2. **Message Storage**: `RunnableWithMessageHistory` automatically:
   - Retrieves the session's chat history
   - Injects it into the prompt via `MessagesPlaceholder`
   - Adds the new user message
   - Sends everything to the LLM
   - Stores both the user message and LLM response in the session history

3. **Memory Isolation**: Each `session_id` maintains its own separate conversation history, ensuring conversations don't interfere with each other.

## Project Structure

```
demo-3-session-based-conversational-memory-in-action/
├── main.py              # Main application code
├── pyproject.toml       # Project dependencies
├── README.md            # This file
├── .env                 # Environment variables (create this)
└── .gitignore          # Git ignore rules
```

## Dependencies

- `fastapi`: Web framework for building the API
- `uvicorn`: ASGI server for running FastAPI
- `langchain`: Core LangChain library
- `langchain-openai`: LangChain integration for OpenAI-compatible APIs (used for Gemini)
- `python-dotenv`: Environment variable management
- `pydantic`: Data validation

## Notes

- The session store is in-memory, so all conversation history is lost when the server restarts
- For production use, consider implementing a persistent session store (e.g., Redis, database)
- The application uses Google Gemini API but through LangChain's OpenAI-compatible interface
