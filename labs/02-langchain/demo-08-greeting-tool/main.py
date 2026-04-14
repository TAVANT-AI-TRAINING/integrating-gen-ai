"""
Greeting Tool Demo

A FastAPI application demonstrating LangChain tool integration.
Shows how to define custom tools using the @tool decorator and invoke them via FastAPI endpoints.
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.tools import tool


# --- Step 1: Define Response Models ---
class GreetingResponse(BaseModel):
    """Response model for the greeting endpoint"""
    tool_name: str
    description: str
    args_schema: str
    result: str


class APIInfo(BaseModel):
    """Response model for the root endpoint"""
    title: str
    description: str
    version: str
    endpoints: list[str]


# --- Step 2: Initialize FastAPI App ---
app = FastAPI(
    title="LangChain Greeting Tool",
    version="1.0.0",
    description="Demonstrates LangChain tool definition and invocation via FastAPI"
)


# --- Step 3: Define the Tool ---
@tool
def generate_greeting(name: str) -> str:
    """
    Generates a personalized greeting for a given name.
    Use this when a user wants a simple greeting.
    """
    return f"Hello, {name}! Welcome to our system."

# --- Step 4: Root Endpoint ---
@app.get("/", response_model=APIInfo)
async def root():
    """API information and available endpoints"""
    return APIInfo(
        title="LangChain Greeting Tool",
        description="Demonstrates how to define and invoke LangChain tools",
        version="1.0.0",
        endpoints=["/", "/test_greeting", "/docs", "/openapi.json"]
    )


# --- Step 5: Tool Invocation Endpoint ---
@app.get("/test_greeting", response_model=GreetingResponse)
async def test_greeting(name: str):
    """
    Invoke the generate_greeting tool and return result.
    
    Args:
        name: The name to generate a greeting for
    
    Returns:
        GreetingResponse containing tool metadata and result
    """
    result = generate_greeting.invoke({"name": name})
    return GreetingResponse(
        tool_name=generate_greeting.name,
        description=generate_greeting.description,
        args_schema=str(generate_greeting.args),
        result=result
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)