import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm(temperature: float) -> ChatOpenAI:
    """Initialize and return a ChatOpenAI model instance with specified temperature.
    
    Supports multiple LLM providers via environment variables.
    
    Temperature controls the randomness of responses:
    - Low temperature (0.0-0.3): Factual, consistent, focused responses
    - Medium temperature (0.4-0.6): Balanced responses
    - High temperature (0.7-1.0): Creative, varied, exploratory responses
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # Generic ChatOpenAI initialization with temperature control
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "max_retries": 2,
    }
    
    return ChatOpenAI(**config)

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM with Temperature Control",
    version="2.0.0",
    description=f"Demonstrates how temperature affects LLM responses. Currently using {provider.upper()} provider"
)


class ChatRequest(BaseModel):
    message: str


class TemperatureResponse(BaseModel):
    factual_response: str
    creative_response: str
    factual_temperature: float
    creative_temperature: float
    model: str
    provider: str


@app.post("/compare-temperatures", response_model=TemperatureResponse)
def compare_temperatures(request: ChatRequest) -> TemperatureResponse:
    """
    Compare LLM responses with different temperature settings.
    
    Workflow:
    1. Environment Setup: Load API key and model configuration from .env
    2. Dual Instantiation: Create models with temperature=0.1 (factual) and temperature=0.9 (creative)
    3. Parallel Invocation: Call .invoke() on both models with the same prompt
    4. Response Comparison: Return both responses for side-by-side analysis
    5. Observation: Low-temperature response is consistent and factual, while high-temperature response is creative and varied
    
    This demonstrates how temperature is the primary control for response creativity and diversity.
    """
    try:
        # Initialize models with different temperatures
        factual_llm = initialize_llm(temperature=0.1)
        creative_llm = initialize_llm(temperature=0.9)
        
        prompt = f"You are a helpful assistant. Please respond to the user's message: {request.message}"
        
        # Parallel invocation with same prompt
        factual_result = factual_llm.invoke(prompt)
        creative_result = creative_llm.invoke(prompt)
        
        # Extract content from responses
        factual_content = factual_result.content if hasattr(factual_result, "content") else str(factual_result)
        creative_content = creative_result.content if hasattr(creative_result, "content") else str(creative_result)
        
        # Get current provider and model name
        current_provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if current_provider == "openai":
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        else:
            model_name = os.getenv("GEMINI_MODEL_NAME", "unknown")
        
        return TemperatureResponse(
            factual_response=factual_content,
            creative_response=creative_content,
            factual_temperature=0.1,
            creative_temperature=0.9,
            model=model_name,
            provider=current_provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")



if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
