# LCEL Simple Router Chain API

A **FastAPI** application that demonstrates how to build a **router chain** using the **LangChain Expression Language (LCEL)**.  
This example classifies customer queries as either **TECHNICAL** or **GENERAL** and routes them to different response chains using **RunnableBranch**.

---

## Features

- **Conditional Routing with LCEL** – Implements a router chain using `RunnableBranch`
- **Automatic Query Classification** – Uses LLM to decide whether a query is _technical_ or _general_
- **Dynamic Response Generation** – Routes each query to the correct LLM prompt for a suitable answer
- **FastAPI Interface** – Provides a REST API endpoint for interacting with the router chain
- **Declarative Chain Composition** – Demonstrates `RunnableLambda` + `RunnableBranch` with the pipe (`|`) syntax

---

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) or `pip` for dependency installation
- A valid **Gemini API key** from Google AI Studio

---

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-16-simple-router-chain-api
   ```

   **For Windows:**

   ```cmd
   cd demo-16-simple-router-chain-api
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

---

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
   GEMINI_MODEL_NAME=gemini-2.5-flash
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   ```

   **How to get API keys:**
   - **OpenAI**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)

   **Note**: The model names can be updated to any supported model. Model names may change over time, so always refer to the latest options in the provider's documentation.

---

## How It Works

1. The user sends a query to the API.
2. The system uses `ChatOpenAI` + `ChatPromptTemplate` to classify the query as **TECHNICAL** or **GENERAL**.
3. The `RunnableBranch` directs the flow:
   - **TECHNICAL** → Escalation message for support
   - **GENERAL** → Direct customer answer
4. The LLM generates a final response.

---

## Running the Application

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

---

## API Endpoint

### **POST** `/route_query`

#### Request Body:

```json
{
  "query": "My app crashes every time I try to upload a file."
}
```

#### Example Response:

```json
{
  "query": "My app crashes every time I try to upload a file.",
  "response": "Thank you for reporting this issue. I'm escalating your case to our technical support team for further investigation."
}
```

---

## Testing the Router

Try different inputs in the API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

**Examples:**
| Query | Expected Classification | Response Type |
|--------|--------------------------|----------------|
| “My app keeps crashing.” | TECHNICAL | Escalation message |
| “What’s your refund policy?” | GENERAL | Informational answer |

---

## Project Structure

```
simple_router_chain/
│
├── main.py                # FastAPI + LCEL Router code
├── .env                   # Gemini API configuration
├── README.md              # Project documentation
└── requirements.txt       # Optional dependency file
```

---
