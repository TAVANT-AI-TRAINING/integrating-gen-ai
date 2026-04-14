# LangChain Streaming Responses

A FastAPI application demonstrating real-time streaming of LLM responses using LangChain's streaming capabilities.

## Objective

Learn how to implement streaming responses for better user experience:

- Real-time feedback as responses are generated
- Reduced perceived latency for long responses
- Progressive content rendering
- Server-Sent Events (SSE) implementation

## Project Structure

```
demo-07-langchain-streaming/
├── .env                    # Environment variables (API key)
├── main.py                 # FastAPI application with streaming
├── pyproject.toml          # Project dependencies
├── README.md               # This file
├── test_client.html        # HTML test client for streaming demo
├── .gitignore              # Git ignore patterns
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
   cd demo-07-langchain-streaming
   ```

   **For Windows:**

   ```cmd
   cd demo-07-langchain-streaming
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

## API Endpoints

### GET /

Root endpoint with API information.

**Response:**

```json
{
  "message": "LangChain Streaming Demo",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "endpoints": {
    "chat": "/chat (POST) - Non-streaming response",
    "stream": "/chat/stream (POST) - Streaming response"
  }
}
```

### POST /chat

Non-streaming endpoint (for comparison).

**Request:**

```json
{
  "message": "Write a short story about a robot"
}
```

**Response:**

```json
{
  "response": "Once upon a time, there was a robot named...",
  "model": "gpt-4o-mini",
  "provider": "openai"
}
```

### POST /chat/stream

Streaming endpoint with real-time response generation.

**Request:**

```json
{
  "message": "Write a short story about a robot"
}
```

**Response (Server-Sent Events):**

```
data: {"chunk": "Once", "done": false}

data: {"chunk": " upon", "done": false}

data: {"chunk": " a", "done": false}

...

data: {"chunk": "", "done": true}
```

## Testing the Streaming API

### Using the HTML Test Client (Easiest):

1. Start the FastAPI server (see "Running the Application" section)
2. Open `test_client.html` in your web browser
3. Enter a message and click "⚡ Stream Response" to see real-time streaming
4. Compare with "📨 Normal Response" to see the difference

The test client provides:

- ✅ Visual comparison between streaming and non-streaming
- ✅ Real-time statistics (time, chunks, character count)
- ✅ Beautiful UI with status indicators
- ✅ Easy-to-use interface

### Using cURL (Linux/macOS/Windows with Git Bash):

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a joke"}'
```

### Using Python:

```python
import requests
import json

url = "http://localhost:8000/chat/stream"
data = {"message": "Write a short poem about AI"}

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        # Decode SSE format
        if line.startswith(b'data: '):
            chunk_data = json.loads(line[6:])
            if not chunk_data.get('done'):
                print(chunk_data['chunk'], end='', flush=True)

print()  # New line at end
```

### Using JavaScript (Browser):

```javascript
const response = await fetch("http://localhost:8000/chat/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "Tell me about AI" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split("\n");

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const data = JSON.parse(line.substring(6));
      if (!data.done) {
        console.log(data.chunk);
      }
    }
  }
}
```

## How Streaming Works

The streaming implementation follows this pattern:

1. **Enable Streaming**: Set `streaming=True` when initializing ChatOpenAI
2. **Use .stream()**: Call `.stream()` method instead of `.invoke()`
3. **Iterate Chunks**: Loop through chunks as they're generated
4. **Send via SSE**: Use FastAPI's StreamingResponse with Server-Sent Events format
5. **Signal Completion**: Send a final message with `done: true`

### Benefits of Streaming:

- **Real-time Feedback**: Users see responses as they're generated
- **Better UX**: Reduces perceived latency, especially for long responses
- **Progressive Rendering**: Display content incrementally
- **Responsive**: Users know the system is working
- **Lower Memory**: Process chunks without storing entire response
- **Cancellable**: Can stop generation mid-stream if needed

### Streaming vs Non-Streaming:

| Aspect          | Non-Streaming                     | Streaming                        |
| --------------- | --------------------------------- | -------------------------------- |
| Response Time   | Wait for complete response        | Immediate feedback               |
| User Experience | May feel slow for long responses  | Feels faster and responsive      |
| Memory Usage    | Buffers entire response           | Processes chunks incrementally   |
| Use Case        | Short responses, batch processing | Long responses, interactive chat |

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI.

**Note**: The Swagger UI doesn't fully support streaming responses. Use cURL, Python, or JavaScript for testing streaming endpoints.

## Troubleshooting

### Streaming not working?

- Ensure `streaming=True` is set in ChatOpenAI initialization
- Check that your client supports Server-Sent Events (SSE)
- Verify network proxies aren't buffering the response

### Chunks arriving all at once?

- Some proxies/CDNs buffer responses; check `X-Accel-Buffering` header
- Try using `curl -N` flag to disable buffering

### Connection timeout?

- Increase timeout settings in your HTTP client
- Consider implementing keep-alive mechanisms for long-running streams
