import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm() -> ChatOpenAI:
    """Initialize and return a ChatOpenAI model instance with retry mechanism.
    
    Supports multiple LLM providers via environment variables.
    
    The retry mechanism provides resilience against transient failures:
    1. max_retries: Automatically retries failed requests
    2. Handles transient errors (HTTP 500, timeouts, etc.)
    3. Returns successful response after retry attempt
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # Generic ChatOpenAI initialization with retry support
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": 2,  # Retry up to 2 times on failure
    }
    
    return ChatOpenAI(**config)

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM with Retry",
    version="2.0.0",
    description=f"LLM invocation with automatic retry mechanism. Currently using {provider.upper()} provider"
)
# Initialize model
llm = initialize_llm()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Demonstrate LLM invocation with automatic retry mechanism.
    
    Workflow:
    1. Environment Setup: Load API key and model configuration from .env
    2. Instantiation with Retries: ChatOpenAI created with max_retries=2
    3. Invocation: Call the .invoke() method with prompt and message
    4. Automatic Retries: LangChain automatically retries on transient failures
    5. Response Handling: Extract and return content from successful attempt
    
    The retry mechanism helps handle temporary service disruptions without user intervention.
    """
    try:
        prompt = "You are a helpful assistant. Please respond to the user's message."
        full_prompt = f"{prompt}\n\nUser: {request.message}\n"
        
        # Invoke the LLM (retries are automatic)
        result = llm.invoke(full_prompt)
        
        # Extract content from the response
        content = result.content if hasattr(result, "content") else str(result)
        
        # Get current provider and model name
        current_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if current_provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        else:
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        
        return ChatResponse(
            response=content,
            model=model_name,
            provider=current_provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
