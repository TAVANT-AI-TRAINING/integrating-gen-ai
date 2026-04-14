import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
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
    title="LCEL Text Translation API",
    description=f"Currently using {provider.upper()} provider"
)

# Initialize the LLM
llm = initialize_llm()

# Step 1: Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate the user's text to French."),
    ("user", "{text}")
])

# Step 2: Create output parser
parser = StrOutputParser()

# Step 3: Build the LCEL chain
translation_chain = prompt | llm | parser


@app.post("/translate")
def translate_text(text:str):
    """
    Accepts JSON input: {"text": "Hello, how are you?"}
    Returns translated text.
    """
       
    result = translation_chain.invoke({"text": text})
    return {"original": text, "translated": result}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
