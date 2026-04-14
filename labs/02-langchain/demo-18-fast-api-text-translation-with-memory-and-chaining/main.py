import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
import uvicorn

# -------------------------------
# Step 0: Load environment variables
# -------------------------------
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
    
    # Generic ChatOpenAI initialization
    config = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url
    }
    
    return ChatOpenAI(**config)

# -------------------------------
# Step 1: Initialize FastAPI and LLM
# -------------------------------
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LCEL Memory Chain",
    description=f"Currently using {provider.upper()} provider"
)

llm = initialize_llm()

# -------------------------------
# Step 2: Define the Prompt (Memory-Aware)
# -------------------------------
# Add MessagesPlaceholder so the prompt can accept conversation history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),   # Added placeholder for memory
    ("user", "{input}")
])

# -------------------------------
# Step 3: Create Output Parser and Chain
# -------------------------------
parser = StrOutputParser()
output_chain = prompt | llm | parser

# Global conversation history - in production, this should be stored in a database
# and managed per user/session
conversation_history: List = [
    HumanMessage(content="My name is Bob."),
    AIMessage(content="Nice to meet you, Bob!")
]

# -------------------------------
# Step 4: Define FastAPI Endpoint
# -------------------------------
@app.post("/ask")
def ask_endpoint(input: str):
   
    # Add the current user input to history
    conversation_history.append(HumanMessage(content=input))
    # IMPORTANT: Now chain expects 'chat_history' key as well.
    # For now, we pass an empty list (memory will be added in LO3).
    result = output_chain.invoke({
        "input": input,
        "chat_history": conversation_history  # Empty history for now
    })
    conversation_history.append(AIMessage(content=result))
    return {"input": input, "response": result}

# -------------------------------
# Step 5: Run FastAPI
# -------------------------------
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
