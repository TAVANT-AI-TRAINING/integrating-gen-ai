import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm(max_tokens: int) -> ChatOpenAI:
    """Initialize and return a ChatOpenAI model instance with token limit.
    
    Supports multiple LLM providers via environment variables.
    
    Max tokens controls response length:
    - Low max_tokens: Short, concise responses
    - High max_tokens: Longer, detailed responses
    - Token = roughly 4 characters on average
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # Generic ChatOpenAI initialization with token limit
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "max_tokens": max_tokens,
        "max_retries": 2,
    }
    
    return ChatOpenAI(**config)

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM with Token Limit",
    version="2.0.0",
    description=f"Demonstrates how max_tokens affects response length. Currently using {provider.upper()} provider"
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str
    provider: str
    max_tokens: int


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Demonstrate LLM invocation with token limit control.
    
    Workflow:
    1. Environment Setup: Load API key and model configuration from .env
    2. Instantiation with Token Limit: ChatOpenAI created with max_tokens=100
    3. Invocation: Call the .invoke() method with the user message
    4. Response Handling: Extract content from the response (will be truncated to max_tokens)
    5. Output: Return the truncated response
    
    The token limit ensures responses don't exceed specified length,
    useful for cost control and response time optimization.
    """
    try:
        # Initialize LLM with token limit
        llm = initialize_llm(max_tokens=100)
        
        system_prompt = "You are a helpful assistant. Please provide a detailed explanation."
        full_prompt = f"{system_prompt}\n\nUser: {request.message}\n"
        
        # Invoke the LLM
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
            provider=current_provider,
            max_tokens=100
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
