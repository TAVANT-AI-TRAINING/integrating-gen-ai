# Intelligent Customer Support Agent Demo

A FastAPI application demonstrating how to build a **stateful, classification-based routing LCEL workflow** using **LangChain Expression Language (LCEL)**. This demo shows how to create an intelligent customer support agent that automatically classifies queries, routes them to appropriate handlers, and maintains conversation context using **InMemoryChatMessageHistory** and **RunnableWithMessageHistory**.

## Overview

This application implements an intelligent customer support agent that:

- **Classifies** incoming customer queries as either TECHNICAL or GENERAL
- **Routes** queries to specialized handlers based on classification
- **Maintains** session-based conversation memory for context-aware responses
- **Escalates** technical issues to the technical support team
- **Handles** general inquiries directly

## Features

- **Smart Classification**: Automatically classifies queries as TECHNICAL or GENERAL using LLM
- **Intelligent Routing**: Uses `RunnableBranch` to route queries to appropriate handlers
- **Session-Based Memory**: Each conversation is tracked via a unique `session_id` with isolated memory
- **Stateful LCEL Chain**: Combines classification, routing, and memory in a single chain
- **RunnableWithMessageHistory**: Demonstrates automatic memory handling with minimal code changes
- **FastAPI Integration**: Exposes `/route_query` and `/new-session` endpoints for testing

## Architecture

The application uses a multi-stage processing pipeline:

1. **Classification Chain**: Classifies queries as TECHNICAL or GENERAL
2. **Router Function**: `is_technical()` determines routing based on classification
3. **Branch Chains**: Separate handlers for technical and general queries
4. **Memory Integration**: `RunnableWithMessageHistory` maintains conversation context per session
5. **Session Store**: In-memory dictionary stores `ChatMessageHistory` objects per session

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-20-intelligent-customer-support-agent
   ```

   **For Windows:**

   ```cmd
   cd demo-20-intelligent-customer-support-agent
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

   **Note**:
   To get a Gemini API key:
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Create a new API key
   - The GEMINI_MODEL_NAME value can be updated to any supported model. Model names may change over time, so always refer to the latest options in Google’s documentation.

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

### POST /route_query

Send a customer query to the intelligent support agent. The agent classifies the query, routes it to the appropriate handler, and maintains conversation context within the same session.
Use the session Id you have got from /new-session endpoint.

**Request Body:**

```json
{
  "query": "User message (string)",
  "session_id": "Unique session identifier (string)"
}
```

**Request Example:**

```http
POST /route_query
Content-Type: application/json

{
  "query": "I'm having trouble logging into my account",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response Example:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "I'm having trouble logging into my account",
  "response": "I understand you're experiencing login issues. This technical problem has been forwarded to our technical support team who will investigate and resolve the issue promptly."
}
```

### Follow-up Message (Same Session)

**Request:**

```http
POST /route_query
Content-Type: application/json

{
  "query": "Can you also help me with my billing question?",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "Can you also help me with my billing question?",
  "response": "I'd be happy to help you with your billing question. What specific billing information do you need assistance with?"
}
```

The agent remembers the context from the previous message and routes the new query appropriately.

### New Session Example

**Request:**

```http
POST /route_query
Content-Type: application/json

{
  "query": "What's your return policy?",
  "session_id": "654e8400-e29b-41d4-a716-446655465348"
}
```

**Response:**

```json
{
  "session_id": "654e8400-e29b-41d4-a716-446655465348",
  "query": "What's your return policy?",
  "response": "Our return policy allows returns within 30 days of purchase with original receipt. Items must be in original condition. For more details, please visit our returns page or contact customer service."
}
```

Each `session_id` has isolated memory. The agent has no memory of previous sessions.

## How It Works

The application uses a **multi-stage processing pipeline**:

1. **Classification Stage**:
   - The LLM classifies incoming queries as either:
     - **TECHNICAL**: Issues with product functionality, bugs, errors, technical problems
     - **GENERAL**: Questions about policies, billing, shipping, general information
   - Classification is performed using a dedicated prompt and chain

2. **Routing Stage**:
   - Based on the classification, `RunnableBranch` routes queries:
     - **Technical queries** → Escalated to technical support team with professional messaging
     - **General queries** → Handled directly by the customer support assistant
   - The `is_technical()` function determines which branch to take

3. **Memory Integration**:
   - Each session maintains conversation history using `RunnableWithMessageHistory`
   - The `get_session_history()` function retrieves or creates a `ChatMessageHistory` for each session
   - Conversation context is automatically injected into the chain, allowing the agent to remember previous interactions

4. **Chain Composition**:
   - `RunnableLambda(prepare_router_input)` prepares the input with classification
   - `RunnableBranch` routes to appropriate handler
   - `RunnableWithMessageHistory` wraps the chain to add memory capabilities

## Testing the API

### Using FastAPI Interactive Docs

Visit `http://localhost:8000/docs` for FastAPI's interactive Swagger UI where you can test all endpoints directly.

### Testing Classification and Routing

1. **Technical Query Test**:
   - Send: "I'm getting an error when trying to save my file"
   - Expected: Query classified as TECHNICAL, routed to technical support handler

2. **General Query Test**:
   - Send: "What is your refund policy?"
   - Expected: Query classified as GENERAL, handled directly by support assistant

3. **Session Memory Test**:
   - Use the same `session_id` for multiple queries
   - Verify the agent remembers context from previous messages

## Project Structure

```
demo-4-intelligent-customer-support-agent/
├── main.py              # Main FastAPI application with LCEL chain
├── pyproject.toml       # Project dependencies and configuration
├── README.md            # This file
├── .env                 # Environment variables (create this)
├── .gitignore          # Git ignore rules
└── uv.lock              # Dependency lock file
```

## Dependencies

- `fastapi`: Web framework for building the API
- `uvicorn`: ASGI server for running FastAPI
- `langchain`: Core LangChain library
- `langchain-openai`: LangChain integration for OpenAI-compatible APIs (used for Gemini)
- `python-dotenv`: Environment variable management
- `pydantic`: Data validation

## Key Implementation Details

- **Classification Chain**: Uses `ChatPromptTemplate` with a system prompt to classify queries as TECHNICAL or GENERAL
- **Router Function**: `is_technical()` function checks classification result and returns boolean for routing
- **Branch Chains**: Separate `technical_chain` and `general_chain` with specialized prompts
- **Memory Store**: In-memory dictionary (`store`) maps `session_id` to `InMemoryChatMessageHistory` objects
- **Stateful Wrapper**: `RunnableWithMessageHistory` automatically handles conversation context injection
- **Chain Composition**: Combines `RunnableLambda`, `RunnableBranch`, and `RunnableWithMessageHistory` for complete workflow

## Notes

- The session store is in-memory, so all conversation history is lost when the server restarts
- For production use, consider implementing a persistent session store (e.g., Redis, database)
- The application uses Google Gemini API but through LangChain's OpenAI-compatible interface
- Classification happens on every query, allowing dynamic routing based on query content
