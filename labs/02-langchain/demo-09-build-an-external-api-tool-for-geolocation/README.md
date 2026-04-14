# LangChain Geolocation Tool

A FastAPI application demonstrating how to create and integrate external API tools with LangChain for geolocation functionality.

## Features

- **External Tool Integration**: Demonstrates LangChain tools that call external APIs
- **IP-based Geolocation**: Automatically detects user location by IP address
- **Structured Response Models**: Pydantic models for validated responses
- **FastAPI Web Interface**: REST API endpoint to test the location tool
- **Error Handling**: Proper error handling for network requests and API failures
- **Root Endpoint**: API information and available endpoints

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Free API key from [apiip.net](https://apiip.net/)

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-09-build-an-external-api-tool-for-geolocation
   ```

   **For Windows:**

   ```cmd
   cd demo-09-build-an-external-api-tool-for-geolocation
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

1. Create or update your `.env` file in the project root with your apiip.net API key:

   ```
   APIIP_API_KEY=your_api_key_here
   ```

   **To get an API key:**
   - Visit [apiip.net](https://apiip.net/)
   - Click the Login button, then select "Sign in with Google"
   - Once signed in, your API key will be created automatically

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
  "title": "LangChain Geolocation Tool",
  "description": "Demonstrates LangChain tool integration with external APIs",
  "version": "1.0.0",
  "endpoints": ["/", "/get_location", "/docs", "/openapi.json"]
}
```

### Get Location Endpoint

**URL:** `GET /get_location`

Fetches the current user's location information based on their IP address.

**Response:**

```json
{
  "tool_name": "get_user_location",
  "description": "Fetches the user's geographical location based on their IP address. Use this tool when a user's country, city, or region is needed to answer a question. This tool does not require any input.",
  "result": {
    "ip": "203.0.113.42",
    "city": "Mountain View",
    "region": "California",
    "country": "United States"
  }
}
```

**Example:**

```bash
curl "http://localhost:8000/get_location"
```

## API Documentation

Once the server is running, visit:

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Key Concepts

### External Tools in LangChain

External tools are functions decorated with `@tool` that:

- Call external APIs or services
- Return structured data for agents to use
- Include comprehensive docstrings describing their purpose
- Handle errors gracefully

### Error Handling

The demo includes error handling for:

- Missing API credentials
- Network failures
- API timeouts (5 second timeout configured)
- Invalid responses

## Project Structure

```
demo-09-build-an-external-api-tool-for-geolocation/
├── main.py              # Main application
├── pyproject.toml       # Project dependencies
├── .env                 # Environment variables (create this file)
├── .python-version      # Python version specification
├── .gitignore          # Git ignore rules
└── README.md           # This file
```
