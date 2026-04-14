import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm(temperature: float, max_tokens: int) -> ChatOpenAI:
    """Initialize and return a ChatOpenAI model instance with custom parameters.
    
    Supports multiple LLM providers via environment variables.
    
    Combines temperature and max_tokens for fine-grained control:
    - Low temperature + high tokens: Focused and detailed responses
    - High temperature + high tokens: Creative and lengthy responses
    - Medium temperature + low tokens: Balanced but brief responses
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # Generic ChatOpenAI initialization with temperature and token control
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": 2,
    }
    
    return ChatOpenAI(**config)

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM with Temperature and Token Control",
    version="2.0.0",
    description=f"Demonstrates combined temperature and token limit effects. Currently using {provider.upper()} provider"
)


class ChatRequest(BaseModel):
    message: str


class ExperimentResponse(BaseModel):
    low_temp_high_tokens_response: str
    high_temp_high_tokens_response: str
    medium_temp_low_tokens_response: str
    model: str
    provider: str


@app.post("/experiment", response_model=ExperimentResponse)
def experiment(request: ChatRequest) -> ExperimentResponse:
    """
    Demonstrate combined temperature and token limit effects on LLM responses.
    
    Workflow:
    1. Environment Setup: Load API key and model configuration from .env
    2. Triple Instantiation: Create models with different temperature/token combinations:
       - Low temperature (0.2) + high tokens (100): Focused and detailed
       - High temperature (1.0) + high tokens (150): Creative and lengthy
       - Medium temperature (0.5) + low tokens (50): Balanced but brief
    3. Parallel Invocation: Call all three with the same prompt
    4. Response Comparison: Return all responses for side-by-side analysis
    5. Observation: Compare how combinations affect both creativity and length
    """
    try:
        # Initialize models with different parameter combinations
        low_temp_high_tokens = initialize_llm(temperature=0.2, max_tokens=100)
        high_temp_high_tokens = initialize_llm(temperature=1.0, max_tokens=150)
        medium_temp_low_tokens = initialize_llm(temperature=0.5, max_tokens=50)
        
        prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Parallel invocation with same prompt
        response_1 = low_temp_high_tokens.invoke(prompt)
        response_2 = high_temp_high_tokens.invoke(prompt)
        response_3 = medium_temp_low_tokens.invoke(prompt)
        
        # Extract content from responses
        content_1 = response_1.content if hasattr(response_1, "content") else str(response_1)
        content_2 = response_2.content if hasattr(response_2, "content") else str(response_2)
        content_3 = response_3.content if hasattr(response_3, "content") else str(response_3)
        
        # Get current provider and model name
        current_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if current_provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        else:
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        
        return ExperimentResponse(
            low_temp_high_tokens_response=content_1,
            high_temp_high_tokens_response=content_2,
            medium_temp_low_tokens_response=content_3,
            model=model_name,
            provider=current_provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running experiments: {str(e)}")
