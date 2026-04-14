"""
LangChain Agent Setup

A FastAPI application demonstrating how to set up and configure a LangChain Agent
with multi-provider LLM support and external tools.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import uvicorn

# Import your tools
from agent_tools import get_order_status, get_user_location

# --- Load environment variables ---
load_dotenv()


# --- Step 1: Define Response Models ---
class AgentResponse(BaseModel):
    """Response model for agent query"""
    query: str
    response: str
    model: str
    provider: str


class APIInfo(BaseModel):
    """Response model for the root endpoint"""
    title: str
    description: str
    version: str
    provider: str
    model: str
    endpoints: list[str]


# --- Step 2: Initialize LLM with Multi-Provider Support ---
def initialize_llm_client():
    """
    Initialize and return a ChatOpenAI model instance with multi-provider support.
    
    Supports providers:
    - openai: Uses OpenAI API directly
    - gemini: Uses Google Gemini via OpenAI-compatible endpoint
    
    Environment variables required:
    - LLM_PROVIDER: Provider name (default: 'openai')
    - {PROVIDER}_API_KEY: API key for the provider
    - {PROVIDER}_MODEL_NAME: Model name to use
    - {PROVIDER}_BASE_URL: Base URL for the provider (optional, mainly for Gemini)
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
    }
    
    # Add base_url for providers that require it (e.g., Gemini)
    if base_url:
        config["base_url"] = base_url
    
    return ChatOpenAI(**config), model_name, provider


# --- Step 3: Initialize Model and Agent ---
try:
    model, model_name, provider = initialize_llm_client()
except ValueError as e:
    raise ValueError(f"LLM initialization failed: {str(e)}")

# Initialize FastAPI App
app = FastAPI(
    title="LangChain Multi-Provider Agent",
    version="1.0.0",
    description=f"Demonstrates LangChain agent setup with multiple tools. Currently using {provider.upper()} ({model_name})"
)

# --- Step 4: Define Tools
tools = [get_order_status, get_user_location]

# --- Step 5: Create Agent
agent = create_agent(
    model,
    tools=tools,
    system_prompt="You are a helpful assistant. Use the available tools to answer user queries accurately.",
    debug=True
)


# --- Step 6: Root Endpoint ---
@app.get("/", response_model=APIInfo)
async def root():
    """API information and available endpoints"""
    return APIInfo(
        title="LangChain Multi-Provider Agent",
        description="Demonstrates LangChain agent setup with multiple tools",
        version="1.0.0",
        provider=provider.upper(),
        model=model_name,
        endpoints=["/", "/ask_agent", "/docs", "/openapi.json"]
    )


# --- Step 7: Agent Query Endpoint ---
@app.get("/ask_agent", response_model=AgentResponse)
async def ask_agent(query: str):
    """
    Query the agent with a natural language question.
    
    The agent will use available tools to provide a comprehensive answer.
    
    Examples:
    - /ask_agent?query=Check my order status for order ID ABC-123
    - /ask_agent?query=Where am I located right now?
    - /ask_agent?query=What's the status of order DEF-456 and where am I located?
    
    Args:
        query: Natural language question for the agent
    
    Returns:
        AgentResponse with the agent's answer, model name, and provider
    """
    try:
        inputs = {"messages": [{"role": "user", "content": query}]}
        response = agent.invoke(inputs)
        answer = response["messages"][-1].content if response.get("messages") else str(response)
        
        return AgentResponse(
            query=query,
            response=answer,
            model=model_name,
            provider=provider.upper()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


# --- Step 8: Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
