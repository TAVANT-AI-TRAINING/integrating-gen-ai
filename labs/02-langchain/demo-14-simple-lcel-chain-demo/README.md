# Simple LCEL Chain Demo

A FastAPI application that demonstrates how to build a **basic sequential AI workflow** using the **LangChain Expression Language (LCEL)**.  
This demo translates English text into French using a simple chain of components: `Prompt → LLM → Parser`.

## Features

- **LCEL Pipe Syntax**: Demonstrates chaining components using the `|` operator
- **Prompt-LLM-Parser Pipeline**: Shows how data flows automatically between steps
- **FastAPI Integration**: Exposes a REST API endpoint to test the LCEL chain
- **Declarative and Readable Workflow**: Simple and transparent data flow

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-14-simple-lcel-chain-demo
   ```

   **For Windows:**

   ```cmd
   cd demo-14-simple-lcel-chain-demo
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

### POST /translate

Translates the provided English text into French using the LCEL chain.

**Request Example:**
text= Hello, how are you today?

**Response Example:**

```json
{
  "original": "Hello, how are you today?",
  "translated": "Bonjour, comment allez-vous aujourd'hui?"
}
```

## Testing the API

Visit http://localhost:8000/docs for FastAPI’s interactive Swagger UI.
