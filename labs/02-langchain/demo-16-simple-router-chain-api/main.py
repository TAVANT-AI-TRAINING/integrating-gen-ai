import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
import uvicorn

# Load env
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

# Initialize LLM
llm = initialize_llm()

# --- Step 1: Classification Chain ---
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify this customer query as either TECHNICAL or GENERAL.\n"
               "TECHNICAL: Issues with product functionality, bugs, errors, technical problems\n"
               "GENERAL: Questions about policies, billing, shipping, general information\n"
               "Respond with ONLY one word: TECHNICAL or GENERAL"),
    ("user", "{query}")
])
classification_chain = classification_prompt | llm | StrOutputParser()

# --- Step 2: Condition Function ---
def is_technical(input_data: dict) -> bool:
    classification = input_data.get("classification", "").strip().upper()
    print(f"[Router] Classification: {classification}")
    return classification == "TECHNICAL"

# --- Step 3: Branch Chains ---
technical_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. Generate a professional escalation reply that their issue will be forwarded to the technical team for a technical support query. \n"
               "Explain that the issue is being forwarded to the technical team."),
    ("user", "Original query: {original_query}")
])
technical_chain = technical_prompt | llm | StrOutputParser()

general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer support assistant. "
               "Answer this general question directly and professionally."),
    ("user", "{original_query}")
])
general_chain = general_prompt | llm | StrOutputParser()

# --- Step 4: Router ---
router = RunnableBranch(
    (is_technical, technical_chain),
    general_chain
)

# --- Step 5: Orchestration ---
def prepare_router_input(query:str)-> dict:
    classification = classification_chain.invoke({"query": query})
    return {"classification": classification, "original_query": query}

full_chain = RunnableLambda(prepare_router_input) | router

# --- FastAPI Setup ---
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="LCEL Simple Router Chain API",
    description=f"Currently using {provider.upper()} provider"
)

class QueryInput(BaseModel):
    query: str

@app.post("/route_query")
def route_query(data: QueryInput):
    result = full_chain.invoke({"query": data.query})
    return {"query": data.query, "response": result}



if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
