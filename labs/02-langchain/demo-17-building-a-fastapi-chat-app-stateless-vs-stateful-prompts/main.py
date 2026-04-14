from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from typing import List
import uvicorn

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

# Create FastAPI app
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="Memory Prompt Example",
    description=f"Currently using {provider.upper()} provider"
)

# Initialize model
llm = initialize_llm()
# -----------------------------
# BEFORE (Stateless Prompt)
# -----------------------------

# Stateless prompt — no memory slot
stateless_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])
# endpoint for stateless prompt
# ask any question first it will answer but after that if you ask him what you asked earlier it will not remember
@app.get("/stateless")
async def stateless_endpoint(input: str):
    formatted_messages = stateless_prompt.format_messages(input=input)
    response = llm.invoke(formatted_messages)
    return {"response": response.content}

# -----------------------------
# AFTER (With Memory)
# -----------------------------

# Create the new prompt template
stateful_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # memory slot added
    ("user", "{input}")
])

# Global conversation history - in production, this should be stored in a database
# and managed per user/session
conversation_history: List = [
    HumanMessage(content="My name is Bob."),
    AIMessage(content="Nice to meet you, Bob!")
]

# endpoint for stateful prompt
# ask what is my name it will answer bob because of memory
@app.get("/stateful")
async def stateful_endpoint(input: str):
    # Add the current user input to history
    conversation_history.append(HumanMessage(content=input))
    
    # Format messages with the updated history
    formatted_messages = stateful_prompt.format_messages(
        input=input,
        chat_history=conversation_history  # provide the updated history
    )
    
    # Get AI response
    response = llm.invoke(formatted_messages)
    
    # Add the AI response to history
    conversation_history.append(AIMessage(content=response.content))
    
    return {"response": response.content}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
