import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uvicorn

# -------------------------------
# Environment Setup
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
# Model Initialization
# -------------------------------
llm = initialize_llm()

# -------------------------------
# Step 1: Classification Chain
# -------------------------------
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify this customer query as either TECHNICAL or GENERAL.\n"
               "TECHNICAL: Issues with product functionality, bugs, errors, technical problems\n"
               "GENERAL: Questions about policies, billing, shipping, general information\n"
               "Respond with ONLY one word: TECHNICAL or GENERAL."),
    ("user", "{query}")
])
classification_chain = classification_prompt | llm | StrOutputParser()

# -------------------------------
# Step 2: Router Condition Function
# -------------------------------
def is_technical(input_data: dict) -> bool:
    classification = input_data.get("classification", "").strip().upper()
    print(f"[Router] Classification: {classification}")
    return classification == "TECHNICAL"

# -------------------------------
# Step 3: Branch Chains
# -------------------------------
technical_prompt = ChatPromptTemplate.from_messages([
    ("system", "Generate a professional escalation message for a technical support query.\n"
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

# -------------------------------
# Step 4: Router Definition
# -------------------------------
router = RunnableBranch(
    (is_technical, technical_chain),
    general_chain
)

# -------------------------------
# Step 5: Combine all with preparation
# -------------------------------
def prepare_router_input(query: str) -> dict:
    classification = classification_chain.invoke({"query": query})
    return {"classification": classification, "original_query": query}

core_chain = RunnableLambda(prepare_router_input) | router

# -------------------------------
# Step 6: Make it Stateful
# -------------------------------

# Session memory storage
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with message history
stateful_chain = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="query")

# -------------------------------
# Step 7: FastAPI Setup
# -------------------------------
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="Stateful Router Chain",
    description=f"Currently using {provider.upper()} provider"
)

class QueryInput(BaseModel):
    session_id: str
    query: str
@app.get("/new-session")
def new_session():
    """Generate a new chat session ID."""
    return {"session_id": str(uuid.uuid4())}
@app.post("/route_query")
def route_query(data: QueryInput):
    config = {"configurable": {"session_id": data.session_id}}
    result = stateful_chain.invoke({"query": data.query}, config=config)
    return {
        "session_id": data.session_id,
        "query": data.query,
        "response": result
    }

# -------------------------------
# Local Run
# -------------------------------
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
