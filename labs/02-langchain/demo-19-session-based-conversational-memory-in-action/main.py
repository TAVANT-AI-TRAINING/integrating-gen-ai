import os
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import uvicorn

# -------------------------------
# Step 1: Load Environment Variables
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
# Step 2: Initialize FastAPI and LLM
# -------------------------------
provider = os.getenv("LLM_PROVIDER", "openai").lower()
app = FastAPI(
    title="Stateful Greeter agent API",
    description=f"Currently using {provider.upper()} provider"
)

llm = initialize_llm()

# -------------------------------
# Step 3: Create Prompt with MessagesPlaceholder
# -------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly greeter agent. You remember the user's name."),
    MessagesPlaceholder(variable_name="chat_history"),  # Memory placeholder
    ("user", "{input}")
])

# Base LCEL chain: prompt → LLM
core_chain = prompt | llm

# -------------------------------
# Step 4: Create In-Memory Session Store
# -------------------------------
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return ChatMessageHistory for a given session_id."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# -------------------------------
# Step 5: Wrap Chain with Message History
# -------------------------------
stateful_chain = RunnableWithMessageHistory(
    runnable=core_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# -------------------------------
# Step 6: Request Model
# -------------------------------
class ChatInput(BaseModel):
    input: str
    session_id: str

# -------------------------------
# Step 7: API Routes
# -------------------------------
@app.get("/new-session")
def new_session():
    """Generate a new chat session ID."""
    return {"session_id": str(uuid.uuid4())}


@app.post("/chat")
def chat_with_agent(payload: ChatInput):
    """Chat endpoint maintaining conversation memory per session."""
    response = stateful_chain.invoke(
        {"input": payload.input},
        config={"configurable": {"session_id": payload.session_id}}
    )

    return {
        "session_id": payload.session_id,
        "user_input": payload.input,
        "agent_response": response.content
    }

# -------------------------------
# Step 8: Run Server
# -------------------------------
if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
