import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import uvicorn


# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    """Initialize and return a ChatOpenAI model instance.
    
    Supports multiple LLM providers via environment variables.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    # base url is optional and only needed for certain providers, so we won't raise an error if it's missing. 
    # Instead, we'll just not include it in the config.

    # Generic ChatOpenAI initialization
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url  # This will be None if not set, which is fine for providers that don't require it
    }
    
    # Add base_url for providers that require it
    # base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    # if base_url:
    #     config["base_url"] = base_url
    
    return ChatOpenAI(**config)

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LangChain + Multi-Provider LLM",
    version="2.0.0",
    description=f"Currently using {provider.upper()} provider"
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
    Basic LLM invocation endpoint demonstrating the fundamental workflow:
    
    1. Environment Setup: GEMINI_API_KEY, GEMINI_MODEL_NAME, and GEMINI_BASE_URL loaded from .env file
    2. Instantiation: ChatOpenAI instance created with model from GEMINI_MODEL_NAME environment variable
    3. Invocation: Call the .invoke() method with the user's message
    4. Response Handling: LangChain sends request to Gemini API and parses JSON response into AIMessage
    5. Output: Return the content of the AIMessage object
    """
    try:
        prompt = "You are a helpful assistant. Please respond to the user's message."
        full_prompt = f"{prompt}\n\nUser: {request.message}\n"
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
            provider=current_provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


