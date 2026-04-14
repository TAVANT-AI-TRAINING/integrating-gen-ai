import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm() -> ChatOpenAI:
    """Initialize and return a ChatOpenAI model instance for async operations.
    
    Supports multiple LLM providers via environment variables.
    
    This instance supports both synchronous .invoke() and asynchronous .ainvoke() methods.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # Generic ChatOpenAI initialization
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": 2,
    }
    
    return ChatOpenAI(**config)


async def async_generate_response(prompt: str) -> str:
    """Generate a response asynchronously using the LLM.
    
    Demonstrates async/await pattern for non-blocking LLM calls:
    1. Define coroutine: Use async def
    2. Instantiate LLM: Create ChatOpenAI instance
    3. Await ainvoke: Call .ainvoke() with await keyword
    4. Return result: Extract and return content
    
    This pattern is useful for:
    - Handling multiple requests concurrently
    - Improving response times in high-traffic scenarios
    - Preventing blocking of other operations
    
    Args:
        prompt: The input prompt for the LLM
        
    Returns:
        The generated response content
    """
    try:
        # Initialize LLM
        llm = initialize_llm()
        
        # Await ainvoke - Asynchronous invocation
        result = await llm.ainvoke(prompt)
        
        return result.content if hasattr(result, "content") else str(result)
    except Exception as e:
        raise Exception(f"Error generating async response: {str(e)}")


# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM with Async Support",
    version="2.0.0",
    description=f"Non-blocking chat using async/await pattern. Currently using {provider.upper()} provider"
)


class ChatRequest(BaseModel):
    message: str


class AsyncResponse(BaseModel):
    response: str
    model: str
    provider: str
    execution_type: str


@app.post("/async-chat", response_model=AsyncResponse)
def async_chat(request: ChatRequest) -> AsyncResponse:
    """
    Non-blocking chat endpoint using async/await pattern.
    
    Workflow:
    1. Environment Setup: Load API key and model configuration from .env
    2. Define Coroutine: Create async function with async def
    3. Instantiate LLM: Create ChatOpenAI instance
    4. Await ainvoke: Call .ainvoke() with await keyword inside coroutine
    5. Execute: Use asyncio.run() to run the top-level coroutine
    
    Benefits:
    - Non-blocking: Doesn't freeze the application
    - Scalable: Handles multiple requests efficiently
    - Responsive: Better user experience with faster perceived response times
    """
    try:
        prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Execute async function using asyncio.run()
        response = asyncio.run(async_generate_response(prompt))
        
        # Get current provider and model name
        current_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if current_provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        else:
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        
        return AsyncResponse(
            response=response,
            model=model_name,
            provider=current_provider,
            execution_type="async"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
