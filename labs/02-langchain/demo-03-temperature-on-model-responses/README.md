# Temperature Experiment: LLM Output Comparison

A FastAPI application demonstrating how different temperature values affect LLM response creativity and consistency.

## Objective

Understand how temperature controls LLM behavior:

1. **Low Temperature (0.1)**: Produces factual, consistent, focused responses
2. **High Temperature (0.9)**: Produces creative, varied, exploratory responses
3. **Comparison**: Send the same prompt to both models and observe the differences

## Project Structure

```
demo-03-temperature-on-model-responses/
├── .env                    # Environment variables (API key)
├── main.py                 # FastAPI application with temperature comparison
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
   cd demo-03-temperature-on-model-responses
   ```

   **For Windows:**

   ```cmd
   cd demo-3-temperature-on-model-responses
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
- Or send POST requests to the comparison endpoints

## Features

- ✅ **Dual Model Instantiation**: Factual model (temp=0.1) and Creative model (temp=0.9)
- ✅ **Parallel Invocation**: Both models called with identical prompts
- ✅ **Response Comparison**: Side-by-side analysis of outputs
- ✅ **Environment variable loading** with `python-dotenv`
- ✅ **Retry mechanism** with `max_retries=2` for reliability
- ✅ **FastAPI web service** with automatic API documentation
- ✅ **Pydantic models** for request/response validation
- ✅ **Comprehensive error handling** with HTTP status codes

## API Endpoints

### POST /compare-temperatures

Temperature comparison endpoint.

**Request Body:**

```json
{
  "message": "Write a creative story about a robot learning to paint"
}
```

**Response:**

```json
{
  "factual_response": "A robot learning to paint would involve...",
  "creative_response": "In a world where metallic fingers first touched canvas...",
  "factual_temperature": 0.1,
  "creative_temperature": 0.9,
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

## Temperature Experiment Workflow

### Conceptual Flow:

1. **Environment Setup**: Load the GEMINI_API_KEY from a .env file
2. **Dual Instantiation**:
   - Create a "Factual" model with temperature=0.1 (low temperature for consistent, conventional responses)
   - Create a "Creative" model with temperature=0.9 (high temperature for varied, imaginative responses)
3. **Parallel Invocation**: Call `.invoke()` on both models with the exact same creative prompt
4. **Response Comparison**: Return both responses for side-by-side analysis
5. **Output Analysis**: Observe how temperature affects response characteristics

### Expected Behavior:

- **Low Temperature (0.1)**: More deterministic, conventional, and consistent responses
- **High Temperature (0.9)**: More creative, varied, and unpredictable responses
