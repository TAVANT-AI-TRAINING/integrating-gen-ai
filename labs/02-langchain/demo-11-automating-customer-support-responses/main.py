from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import uvicorn

# Load environment variables
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
    title="Customer Support Assistant",
    description=f"Currently using {provider.upper()} provider"
)

# Initialize model
llm = initialize_llm()

# ---------- Step 1: Define Input Schema ----------
class SupportRequest(BaseModel):
    customer_email: str
    issue_category: str

# ---------- Step 2: Build ChatPromptTemplate ----------
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional customer support agent. Respond to customer emails with empathy and clarity."),

    ("human", "Customer Email: I received a damaged phone. | Issue Category: Product Replacement"),
    ("ai", "I'm really sorry to hear about the damaged phone. Please provide your order number so we can arrange a replacement immediately."),

    ("human", "Customer Email: I haven’t received my order yet. | Issue Category: Shipping Delay"),
    ("ai", "I apologize for the delay. Could you please share your order ID so we can check the shipment status right away?"),

    # Current query
    ("human", "Customer Email: {customer_email} | Issue Category: {issue_category}")
])

# ---------- Step 3: Define Endpoint ----------
@app.post("/generate-reply")
async def generate_reply(request: SupportRequest):
    # Format messages for the model
    formatted_messages = chat_prompt.format_messages(
        customer_email=request.customer_email,
        issue_category=request.issue_category
    )

    # Invoke LLM model
    response = llm.invoke(formatted_messages)
    
    return {"reply": response.content}

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
