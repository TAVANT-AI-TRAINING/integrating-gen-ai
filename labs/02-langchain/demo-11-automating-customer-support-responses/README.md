# Customer Support Assistant

A FastAPI application that uses LLM providers (OpenAI or Gemini) to generate customer support responses based on customer emails and issue categories.

## Features

- **Multi-Provider LLM Support**: Works with OpenAI, Gemini, and other OpenAI-compatible providers
- **Few-shot Examples**: Includes example interactions to guide the AI responses
- **Issue Categorization**: Tailors responses based on the type of customer issue
- **Professional Tone**: Generates empathetic and clear customer support responses
- **FastAPI Integration**: RESTful API with automatic documentation

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- LLM API keys (OpenAI, Gemini, or compatible provider)

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-11-automating-customer-support-responses
   ```

   **For Windows:**

   ```cmd
   cd demo-11-automating-customer-support-responses
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

**For Linux/macOS/Windows:**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

## Features

- **Few-shot Examples**: Includes example interactions to guide the AI responses
- **Issue Categorization**: Tailors responses based on the type of customer issue
- **Professional Tone**: Generates empathetic and clear customer support responses
- **Stateless Processing**: Each request is processed independently without conversation memory
- **Simple API**: Clean interface requiring only customer email and issue category

If you encounter authentication errors, ensure:

1. Your API key is correctly set in the `.env` file
2. The API key is valid and has the necessary permissions
3. You have an active internet connection
