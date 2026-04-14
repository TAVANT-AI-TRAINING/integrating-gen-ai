"""
Demo 2: Converting Console LLM App to REST API

This project demonstrates how to **transform a console-based AI application into a REST API** using **FastAPI**.
It converts the Sprint 6 Study Buddy console app into a production-ready web service with HTTP endpoints.
"""

# 1. Import dependencies: FastAPI, HTTPException, CORS middleware, Pydantic, OpenAI, os, dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from dotenv import load_dotenv

# 2. Load environment variables from .env file api key, base url, and model name and initialize OpenAI client
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables!")

base_url = os.getenv("GEMINI_BASE_URL")
model = os.getenv("GEMINI_MODEL_NAME")

# Initialize OpenAI client for Gemini API
client = OpenAI(api_key=api_key, base_url=base_url)

# 3. Initialize FastAPI application with metadata
app = FastAPI(
    title="Console to REST API",
    description="A REST API version of the Study Buddy console app using FastAPI and OpenAI SDK.",
    version="1.0.0"
)

# 4. Configure CORS middleware
# This allows browser-based clients from any origin ("*") to access the API
# WARNING: For production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Define Request and Response Models
class QueryRequest(BaseModel):
    """Request model for user prompts"""
    prompt: str = Field(..., description="Explain the theory of relativity in simple terms.")    

class AnswerResponse(BaseModel):
    """Response model for AI answers"""
    model: str
    answer: str
    
# 6. Define Health Check Endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API is running and accessible.
    
    Returns:
        dict: Service status information
    """
    return {
        "status": "healthy",
        "service": "llm-query-api",
        "version": "1.0.0"
    }

# 7. Define Query Endpoint
@app.post("/query", response_model=AnswerResponse, summary="Ask your AI")
def ask_study_buddy(request: QueryRequest):
    """
    Accepts a study prompt, calls the LLM, and returns the generated answer.
    
    Args:
        request: QueryRequest object containing user prompt
        
    Returns:
        AnswerResponse: AI-generated answer with model information
    """
    try:
        # 7. Call the llm api to generate the answer
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        answer = completion.choices[0].message.content
        return AnswerResponse(model=model, answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM processing failed: {str(e)}"
        )