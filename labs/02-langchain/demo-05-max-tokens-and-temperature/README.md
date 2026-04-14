# Temperature and Token Experimentation

A FastAPI application demonstrating how temperature and max_tokens parameters work together to control LLM response characteristics.

## Objective

Understand how combining temperature and token limits creates different response outcomes:

- Low temperature + high tokens: Focused and detailed responses
- High temperature + high tokens: Creative and lengthy responses
- Medium temperature + low tokens: Balanced but brief responses

## Project Structure

```
demo-05-max-tokens-and-temperature/
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
   cd demo-05-max-tokens-and-temperature
   ```

   **For Windows:**

   ```cmd
   cd demo-05-max-tokens-and-temperature
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
- Send POST requests to `http://localhost:8000/experiment`

## Features

- ✅ Environment variable loading with `python-dotenv`
- ✅ **Three separate LLM initialization functions** with different parameter combinations
- ✅ **Temperature experimentation** (0.0 to 1.0) for creativity control
- ✅ **Token limit experimentation** for response length control
- ✅ FastAPI web service with automatic API documentation
- ✅ **Single endpoint** showing all three experiments
- ✅ Interactive API testing via Swagger UI

## Three LLM Initialization Functions

### 1. `initialize_low_temp_high_tokens_model()`

- **Temperature**: 0.2 (focused, deterministic)
- **Max Tokens**: 100 (detailed response)
- **Purpose**: Produces focused, detailed, and consistent responses

### 2. `initialize_high_temp_high_tokens_model()`

- **Temperature**: 1.0 (creative, random)
- **Max Tokens**: 100 (detailed response)
- **Purpose**: Produces creative, varied, and unpredictable responses

### 3. `initialize_medium_temp_low_tokens_model()`

- **Temperature**: 0.5 (balanced)
- **Max Tokens**: 100 (very short response)
- **Purpose**: Produces truncated, concise responses that end abruptly

## Use Cases

This parameter experimentation is useful for:

- **Content Generation**: Finding the right balance of creativity vs consistency
- **Response Length Control**: Matching output to UI constraints
- **A/B Testing**: Comparing different parameter combinations
- **Model Tuning**: Optimizing parameters for specific use cases
