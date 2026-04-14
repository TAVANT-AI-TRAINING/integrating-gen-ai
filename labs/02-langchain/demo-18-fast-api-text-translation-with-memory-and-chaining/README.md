# Memory-Aware LCEL Chain Demo

A FastAPI application demonstrating how to make a LangChain Expression Language (LCEL) chain memory-aware by adding a MessagesPlaceholder.
This exercise shows how to build a conversational assistant that maintains conversation history across multiple interactions.

## Features

- **MessagesPlaceholder Integration**: Enables prompts to accept past conversation history
- **LCEL Pipe Syntax**: Demonstrates chaining Prompt → LLM → Parser using the | operator
- **Conversation Memory**: Maintains conversation history with HumanMessage and AIMessage objects
- **FastAPI Endpoint**: Exposes a `/ask` API for conversational interactions
- **Stateful Conversations**: Each interaction builds upon previous conversation context

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-18-fast-api-text-translation-with-memory-and-chaining
   ```

   **For Windows:**

   ```cmd
   cd demo-18-fast-api-text-translation-with-memory-and-chaining
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

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The app will start at `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## API Endpoints

### **POST /ask**

Send a message to the conversational assistant. The assistant maintains conversation history and can reference previous interactions.

#### Request Body:

| Parameter | Type   | Description        |
| --------- | ------ | ------------------ |
| **input** | string | User message/query |

#### Request Example:

```http
POST /ask
Content-Type: application/json

{
  "input": "What is my name?"
}
```

#### Response Example:

```json
{
  "input": "What is my name?",
  "response": "Your name is Bob! Nice to meet you, Bob!"
}
```

---

## Example Conversation Flow

### First Interaction

**Request:**

```json
{
  "input": "What is my name?"
}
```

**Response:**

```json
{
  "input": "What is my name?",
  "response": "Your name is Bob! Nice to meet you, Bob!"
}
```

### Follow-up Interaction

**Request:**

```json
{
  "input": "What did I tell you earlier?"
}
```

**Response:**

```json
{
  "input": "What did I tell you earlier?",
  "response": "You told me that your name is Bob, and I said 'Nice to meet you, Bob!'"
}
```

## How It Works

1. **Memory Integration**: The prompt template includes `MessagesPlaceholder(variable_name="chat_history")` to accept conversation history
2. **History Management**: Each user input and AI response is stored in the `conversation_history` list
3. **Context Awareness**: The AI can reference previous parts of the conversation
4. **LCEL Chain**: Uses `prompt | llm | parser` to process inputs with memory context

## Key Implementation Details

- **MessagesPlaceholder**: Creates a slot in the prompt for conversation history
- **Message Objects**: Uses `HumanMessage` and `AIMessage` to represent conversation turns
- **StrOutputParser**: Converts AI responses to strings for easier handling
- **Global Memory**: Maintains conversation history across all interactions (shared between users)

---

## Testing the API

Visit [http://localhost:8000/docs](http://localhost:8000/docs)  
to open **FastAPI's Swagger UI** and test the `/ask` endpoint interactively.

### Testing Memory Functionality

1. **First Request**: Ask "What is my name?" - The AI should respond with "Your name is Bob!"
2. **Second Request**: Ask "What did I tell you earlier?" - The AI should reference the previous conversation
3. **Third Request**: Ask "Tell me a joke" - The AI should respond with a joke while maintaining context

### Expected Behavior

- The assistant remembers the initial context (Bob's name)
- Follow-up questions reference previous interactions
- Each new interaction builds upon the conversation history
- The conversation history grows with each exchange

---

## Project Structure

```
practice_lo2/
├── main.py              # Main FastAPI application with memory-aware LCEL chain
├── pyproject.toml        # Project dependencies and configuration
├── .envbackup           # Environment variables template
├── .env                 # Your environment variables (create from .envbackup)
├── README.md            # This file
└── uv.lock              # Dependency lock file
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **LangChain**: Framework for building LLM applications
- **LangChain OpenAI**: OpenAI integration for LangChain
- **Uvicorn**: ASGI server for running FastAPI
- **Python-dotenv**: Environment variable management
