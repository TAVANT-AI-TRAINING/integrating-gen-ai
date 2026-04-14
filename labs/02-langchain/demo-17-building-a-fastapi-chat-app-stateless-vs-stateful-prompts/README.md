# Memory-Aware Prompt Demo

A simple Python example demonstrating how to make a LangChain prompt memory-aware using the MessagesPlaceholder class.
This example shows the difference between a stateless prompt and a stateful prompt that can include past conversation history.

## Features

- Demonstrates MessagesPlaceholder: Adds a slot for inserting conversation history into a prompt

- Before vs After Comparison: Shows how adding memory changes prompt behavior

- Dynamic Context Insertion: Past Human and AI messages can now be added automatically

- Foundation for Stateful Agents: Prepares the prompt for use with RunnableWithMessageHistory

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-17-building-a-fastapi-chat-app-stateless-vs-stateful-prompts
   ```

   **For Windows:**

   ```cmd
   cd demo-17-building-a-fastapi-chat-app-stateless-vs-stateful-prompts
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

### FastAPI Interactive Docs

Visit `http://localhost:8000/docs` for interactive API documentation where you can test both endpoints directly.

## API Endpoints

The application provides two endpoints to demonstrate the difference between stateless and stateful prompts:

### 1. Stateless Endpoint (`/stateless`)

**Purpose**: Demonstrates a prompt without memory - each request is independent.

**Method**: `GET`

**URL**: `http://localhost:8000/stateless`

**Query Parameters**:

- `input` (string, required): The user's question or message

**Example Request**:

```
GET http://localhost:8000/stateless?input=What is my name?
```

**Example Response**:

```json
{
  "response": "I don't have any information about your name. Could you please tell me what your name is?"
}
```

### 2. Stateful Endpoint (`/stateful`)

**Purpose**: Demonstrates a prompt with memory - includes conversation history that grows with each interaction.

**Method**: `GET`

**URL**: `http://localhost:8000/stateful`

**Query Parameters**:

- `input` (string, required): The user's question or message

**How it works**:

- Each user input and AI response is automatically added to the conversation history
- The AI can reference previous parts of the conversation
- Memory persists across multiple requests

**Example Request**:

```
GET http://localhost:8000/stateful?input=What is my name?
```

**Example Response**:

```json
{
  "response": "Your name is Bob! Nice to meet you, Bob!"
}
```

**Follow-up Example**:

```
GET http://localhost:8000/stateful?input=What did I tell you earlier?
```

**Follow-up Response**:

```json
{
  "response": "You told me that your name is Bob, and I said 'Nice to meet you, Bob!'"
}
```

## Expected Outputs

| Input                  | Stateless Response                                       | Stateful Response                                                        |
| ---------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| "What is my name?"     | "I don't have any information about your name..."        | "Your name is Bob! Nice to meet you, Bob!"                               |
| "What did I tell you?" | "I don't have any information about what you told me..." | "You told me that your name is Bob, and I said 'Nice to meet you, Bob!'" |
| "Tell me a joke"       | "Here's a joke: [joke]"                                  | "Here's a joke: [joke]" (remembers context from previous interactions)   |

## Key Differences

- **Stateless**: Each request is independent, no memory of previous conversations
- **Stateful**: Maintains conversation history, can reference past interactions, memory grows with each exchange
