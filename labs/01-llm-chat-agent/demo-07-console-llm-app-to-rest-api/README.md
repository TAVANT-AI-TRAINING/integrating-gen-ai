# Converting Console LLM App to REST API

This project demonstrates how to **transform a console-based AI application into a REST API** using **FastAPI**.
It converts the Sprint 6 Study Buddy console app into a production-ready web service with HTTP endpoints.
It is implemented using **uv** for environment management.

---

## Features

- **POST Endpoint:** `/query` endpoint accepts questions and returns AI responses
- **Health Check:** Root `/` endpoint for service monitoring
- **CORS Support:** Enable browser-based clients to access the API
- **Pydantic Validation:** Automatic request/response validation
- **Error Handling:** Graceful handling of LLM errors with HTTP status codes
- **OpenAI SDK Integration:** Uses Gemini via OpenAI-compatible interface
- **Auto-Generated Documentation:** Interactive Swagger UI at `/docs`

---

## Project Structure

```bash
demo-07-console-llm-app-to-rest-api/
├── main.py                      # FastAPI application
├── .env                        # Environment variables
├── .gitignore                  # Ignore sensitive/config files
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Lock file
└── README.md                   # Documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-07-console-llm-app-to-rest-api
cd demo-07-console-llm-app-to-rest-api
```

---

### 2. Create Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
uv add fastapi uvicorn openai python-dotenv
```

---

### 4. Configure Environment Variables

Change the filename from `.envbackup` to `.env`, and include the appropriate keys within it.

```bash
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
GEMINI_MODEL_NAME=gemini-2.5-flash
```

## **Note**: The GEMINI_MODEL value can be updated to any supported model. Model names may change over time, so always refer to the latest options in Google’s documentation.

## Usage

Run the script using `uv`:

```bash
uv run uvicorn main:app --reload --port 8000
```

The server will start at: http://localhost:8000

---

## Example Output

### Success Response (200):

```json
{
  "model": "gemini-2.5-flash",
  "answer": "A list in Python is an ordered, mutable collection of items enclosed in square brackets []. Lists can contain elements of different data types and are indexed starting from 0. Example: my_list = [1, 2, 'hello', True]"
}
```

---

## Key Concepts

| Step | Concept            | Description                          |
| ---- | ------------------ | ------------------------------------ |
| 1    | FastAPI Setup      | Creating web application instance    |
| 2    | Pydantic Models    | Defining request/response schemas    |
| 3    | POST Endpoint      | Handling HTTP POST requests          |
| 4    | CORS Middleware    | Enabling cross-origin requests       |
| 5    | Error Handling     | Converting exceptions to HTTP errors |
| 6    | Health Check       | Adding monitoring endpoint           |
| 7    | Auto Documentation | Using FastAPI's built-in Swagger UI  |

---

## Summary

This project demonstrates how to convert a **console-based LLM application into a REST API** using the **OpenAI-compatible SDK** and **Gemini API**.
It shows the transformation from terminal input/output to HTTP endpoints, making the application **accessible to web, mobile, and any HTTP client**.
Perfect foundation for building **production-ready AI web services**.
