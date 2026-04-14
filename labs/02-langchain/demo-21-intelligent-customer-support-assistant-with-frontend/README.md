# Intelligent Customer Support Assistant with Frontend

A full-stack conversational AI application demonstrating stateful chat sessions using LangChain, FastAPI, and Streamlit. The application features an intelligent customer support agent that automatically classifies queries, routes them to appropriate handlers, and maintains conversation history across sessions.

## Overview

This application implements an intelligent customer support agent that:

- **Classifies** incoming customer queries as either TECHNICAL or GENERAL
- **Routes** queries to specialized handlers based on classification
- **Maintains** session-based conversation memory for context-aware responses
- **Escalates** technical issues to the technical support team
- **Handles** general inquiries directly
- **Provides** interactive web-based chat interface using Streamlit

## Features

- **Smart Classification**: Automatically classifies queries as TECHNICAL or GENERAL using LLM
- **Intelligent Routing**: Uses `RunnableBranch` to route queries to appropriate handlers
- **Session-Based Memory**: Each conversation is tracked via a unique `session_id` with isolated memory
- **Stateful LCEL Chain**: Combines classification, routing, and memory in a single chain
- **RunnableWithMessageHistory**: Demonstrates automatic memory handling with minimal code changes
- **Multi-Provider Support**: Supports both OpenAI and Google Gemini LLM providers
- **FastAPI Backend**: RESTful API with `/route_query` and `/new-session` endpoints
- **Streamlit Frontend**: Interactive chat UI with message history display
- **Real-time Communication**: Frontend communicates with backend via HTTP requests

## Architecture

### Backend (FastAPI)

- **Classification Chain**: Uses LLM to classify queries as TECHNICAL or GENERAL
- **Router Chain**: Uses `RunnableBranch` to route queries to appropriate handlers
- **Stateful Memory**: Maintains conversation history per session using `RunnableWithMessageHistory`
- **Session Management**: Generates and manages unique session IDs
- **API Endpoints**: RESTful endpoints for session creation and query routing

### Frontend (Streamlit)

- **Chat Interface**: Interactive chat UI with message history display
- **Session Management**: Creates and manages chat sessions with "New Session" button
- **Real-time Communication**: Communicates with backend via HTTP requests
- **Error Handling**: Robust error handling for network and session issues

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- OpenAI API key OR Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-21-intelligent-customer-support-assistant-with-frontend
   ```

   **For Windows:**

   ```cmd
   cd demo-21-intelligent-customer-support-assistant-with-frontend
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

## Running the Application

The application consists of two components that need to be run separately: the backend (FastAPI) and the frontend (Streamlit).

### Step 1: Start the Backend

Open a terminal and run:

**For Linux/Windows (Same commands):**

```bash
cd backend
uv run python stateful_router_chain.py
```

Or using uvicorn directly:

```bash
cd backend
uv run uvicorn stateful_router_chain:app --host 0.0.0.0 --port 8000 --reload
```

The backend will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

### Step 2: Start the Frontend

Open a **new terminal** (keep the backend running) and run:

**For Linux/Windows (Same commands):**

```bash
cd frontend
uv run streamlit run streamlit_stateful_agent_ui.py
```

The frontend will start on `http://localhost:8501` and automatically open in your browser.

**Note**: Make sure the backend is running before starting the frontend, as the frontend needs to communicate with the backend API.

## How It Works

### Application Flow

1. **Session Creation**:
   - Frontend requests a new session ID from the backend `/new-session` endpoint
   - Backend generates a unique UUID and stores it in session state

2. **Query Processing**:
   - User sends a message through the Streamlit chat interface
   - Frontend sends the query and session ID to the backend `/route_query` endpoint

3. **Classification**:
   - Backend uses a classification chain to determine if the query is TECHNICAL or GENERAL
   - Classification is performed using a dedicated LLM prompt

4. **Routing**:
   - Based on classification, `RunnableBranch` routes the query:
     - **TECHNICAL** → Technical support escalation handler
     - **GENERAL** → Direct customer support handler

5. **Memory Integration**:
   - `RunnableWithMessageHistory` retrieves conversation history for the session
   - Conversation context is automatically injected into the chain

6. **Response**:
   - Backend returns the response with session context
   - Frontend displays the response in the chat interface

7. **Memory Update**:
   - Conversation history is automatically updated for the session
   - Subsequent queries in the same session will have access to previous context

### Backend Components

#### Classification Chain

The backend uses a classification chain to categorize queries:

- **TECHNICAL**: Issues with product functionality, bugs, errors, technical problems
- **GENERAL**: Questions about policies, billing, shipping, general information

#### Router Chain

Uses `RunnableBranch` to route queries based on classification:

- Technical queries are escalated with a professional message
- General queries are handled directly by the support assistant

#### Session Management

- In-memory dictionary stores `InMemoryChatMessageHistory` objects per session
- `get_session_history()` function retrieves or creates history for each session
- Each session maintains isolated conversation memory

### Frontend Components

#### Session Initialization

- Automatically creates a new session on page load
- Displays session ID in the UI
- Handles session creation errors gracefully

#### Chat Interface

- Real-time message display with role-based styling (user/assistant)
- Session ID display for reference
- "New Session" button to start fresh conversations
- Loading indicators during API calls
- Error handling for network and backend issues

## API Endpoints

### GET `/new-session`

Creates a new chat session and returns a unique session ID.

**Response:**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Note**: Do not copy-paste the example session ID. Generate a new one using this endpoint.

### POST `/route_query`

Processes a user query, classifies it, routes it to the appropriate handler, and returns a response with session context.

**Request Body:**

```json
{
  "session_id": "Unique session identifier (string)",
  "query": "User's question or message (string)"
}
```

**Request Example:**

```http
POST /route_query
Content-Type: application/json

{
    "session_id": "support_ticket_456",
    "query": "I'm having trouble logging into my account"
}
```

**Response:**

```json
{
  "session_id": "support_ticket_456",
  "query": "I'm having trouble logging into my account",
  "response": "I understand you're experiencing login issues. This technical problem has been forwarded to our technical support team who will investigate and resolve the issue promptly."
}
```

## Query Classification

The system automatically classifies queries into two categories:

### TECHNICAL Queries

- Product functionality issues
- Bugs and errors
- Technical problems
- System errors
- **Response**: Professional escalation message to technical team

### GENERAL Queries

- Policy questions
- Billing inquiries
- Shipping information
- General information
- Account questions
- **Response**: Direct assistance from customer support assistant

## User Interface

### Chat Interface Features

- **Message History**: Displays conversation with user and assistant messages in a chat-like format
- **Session ID Display**: Shows current session identifier for reference
- **New Session Button**: Starts a fresh conversation with a new session ID
- **Input Field**: Text input for user messages with placeholder text
- **Loading Indicator**: Shows "Thinking..." spinner during API processing
- **Role-based Styling**: Different styling for user and assistant messages

### Error Handling

- **Session Creation Errors**: Clear error messages displayed if session creation fails
- **Network Errors**: Timeout and connection error handling with user-friendly messages
- **Backend Errors**: Graceful handling of API failures with error display
- **Automatic Retry**: Frontend handles transient errors and provides retry options

## Testing the Application

### Using the Streamlit Interface

1. Start both backend and frontend servers (see Running the Application section)
2. Open `http://localhost:8501` in your browser
3. The chat interface will automatically create a new session
4. Type a message and press Enter to send
5. Observe the classification and routing behavior:
   - Technical queries will receive escalation messages
   - General queries will receive direct assistance
6. Use the "New Session" button to start a fresh conversation

### Testing Classification

1. **Technical Query Test**:
   - Send: "I'm getting an error when trying to save my file"
   - Expected: Query classified as TECHNICAL, routed to technical support handler

2. **General Query Test**:
   - Send: "What is your refund policy?"
   - Expected: Query classified as GENERAL, handled directly by support assistant

3. **Session Memory Test**:
   - Send multiple messages in the same session
   - Verify the assistant remembers context from previous messages

### Using FastAPI Interactive Docs

You can also test the backend API directly:

- Visit `http://localhost:8000/docs` for FastAPI's interactive Swagger UI
- Test the `/new-session` and `/route_query` endpoints directly

## Project Structure

```
demo-21-intelligent-customer-support-assistant-with-frontend/
├── backend/
│   └── stateful_router_chain.py    # FastAPI backend with LangChain routing
├── frontend/
│   └── streamlit_stateful_agent_ui.py  # Streamlit frontend UI
├── .env                            # Environment variables (create this)
├── pyproject.toml                  # Python dependencies
├── uv.lock                         # Lock file for dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## Dependencies

- `streamlit`: Web framework for building the chat UI
- `fastapi`: Web framework for building the backend API
- `uvicorn`: ASGI server for running FastAPI
- `langchain`: Core LangChain library
- `langchain-openai`: LangChain integration for OpenAI and Gemini
- `python-dotenv`: Environment variable management
- `requests`: HTTP library for frontend-backend communication
- `pydantic`: Data validation

## Key Implementation Details

- **Classification Chain**: Uses `ChatPromptTemplate` with a system prompt to classify queries
- **Router Function**: `is_technical()` function checks classification result for routing
- **Branch Chains**: Separate `technical_chain` and `general_chain` with specialized prompts
- **Memory Store**: In-memory dictionary stores `InMemoryChatMessageHistory` objects per session
- **Stateful Wrapper**: `RunnableWithMessageHistory` automatically handles conversation context
- **Chain Composition**: Combines `RunnableLambda`, `RunnableBranch`, and `RunnableWithMessageHistory`
- **Multi-Provider Support**: Backend automatically detects LLM provider from environment variable

## Notes

- The session store is in-memory, so all conversation history is lost when the backend server restarts
- For production use, consider implementing a persistent session store (e.g., Redis, database)
- The application supports both OpenAI and Google Gemini through LangChain's unified interface
- Both backend and frontend must be running for the application to work
- The frontend expects the backend to be running on `http://localhost:8000` by default
- Session IDs are generated using UUID4 for uniqueness

## Troubleshooting

### Backend won't start

- Verify your `.env` file is configured correctly with valid API keys
- Check that the `LLM_PROVIDER` is set to either `openai` or `gemini`
- Ensure port 8000 is not already in use

### Frontend can't connect to backend

- Verify the backend is running on `http://localhost:8000`
- Check for firewall issues blocking local connections
- Ensure both services are running in the same network context

### API key errors

- Double-check your API key is valid and active
- Ensure there are no extra spaces or quotes in the `.env` file
- Verify you have credits/quota available for your LLM provider

## License

This project is for educational purposes.

## Contributing

Feel free to submit issues and enhancement requests!
