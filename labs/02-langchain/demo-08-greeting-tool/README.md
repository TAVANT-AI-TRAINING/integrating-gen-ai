# LangChain Greeting Tool

A FastAPI application demonstrating LangChain tool integration with structured response models.

## Features

- Custom LangChain tool using the `@tool` decorator
- Structured Pydantic response models
- FastAPI web interface with documentation
- Tool metadata inspection capabilities
- RESTful API design with root endpoint

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-08-greeting-tool
   ```

   **For Windows:**

   ```cmd
   cd demo-08-greeting-tool
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

## Running the Application

**For Linux/macOS/Windows:**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

## API Usage

### Root Endpoint

**URL:** `GET /`

Returns API information and available endpoints.

**Response:**

```json
{
  "title": "LangChain Greeting Tool",
  "description": "Demonstrates how to define and invoke LangChain tools",
  "version": "1.0.0",
  "endpoints": ["/", "/test_greeting", "/docs", "/openapi.json"]
}
```

### Greeting Tool Endpoint

**URL:** `GET /test_greeting`

**Parameters:**

- `name` (string, required): Name to generate greeting for

**Response:**

```json
{
  "tool_name": "generate_greeting",
  "description": "Generates a personalized greeting for a given name. Use this when a user wants a simple greeting.",
  "args_schema": "...",
  "result": "Hello, John! Welcome to our system."
}
```

**Example:**

```bash
curl "http://localhost:8000/test_greeting?name=John"
```

## API Documentation

Once the server is running, visit:

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Key Concepts

### LangChain Tools

Tools in LangChain are reusable functions that agents and chains can call. The `@tool` decorator:

- Defines the function as a tool
- Extracts docstring as tool description
- Creates schema from function signature
- Enables tool invocation via `.invoke()`

### Pydantic Response Models

Response models ensure consistent, validated output:

- `GreetingResponse`: Structured greeting tool result
- `APIInfo`: Root endpoint information

## Project Structure

```
demo-08-greeting-tool/
├── main.py              # Main application
├── pyproject.toml       # Project dependencies
├── .env                 # Environment variables (if needed)
├── .python-version      # Python version specification
├── .gitignore          # Git ignore rules
└── README.md           # This file
```
