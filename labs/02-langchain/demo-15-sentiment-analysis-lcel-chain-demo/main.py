import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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
    title="LCEL Sentiment Analysis API",
    description=f"Currently using {provider.upper()} provider"
)

# Initialize model
llm = initialize_llm()

# --- Step 1: Sentiment Analysis Prompt ---
sentiment_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the sentiment of the following customer feedback. "
               "Respond with ONLY one word: POSITIVE, NEGATIVE, or NEUTRAL."),
    ("user", "{feedback}")
])

# --- Step 2: Custom Urgency Classification Function ---
def classify_urgency(sentiment_result: str) -> dict:
    sentiment = sentiment_result.strip().upper()

    if sentiment == "NEGATIVE":
        urgency = "HIGH"
        priority = "Immediate response required"
    elif sentiment == "NEUTRAL":
        urgency = "MEDIUM"
        priority = "Standard response timeline"
    else:
        urgency = "LOW"
        priority = "Acknowledge and thank"

    return {
        "sentiment": sentiment,
        "urgency": urgency,
        "priority": priority
    }

urgency_classifier = RunnableLambda(classify_urgency)

# --- Step 3: Strategy Generation Prompt ---
strategy_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a customer service manager. Based on the sentiment analysis, "
               "provide a specific response strategy.\n\n"
               "Sentiment: {sentiment}\n"
               "Urgency: {urgency}\n"
               "Priority: {priority}"),
    ("user", "Generate a 2-3 sentence response strategy for this customer.")
])

# --- Step 4: Full Multi-Step Chain ---
analysis_chain = (
    sentiment_prompt
    | llm
    | StrOutputParser()
    | urgency_classifier
    | strategy_prompt
    | llm
    | StrOutputParser()
)


@app.post("/analyze-feedback")
def analyze_feedback(customer_feedback:str):
    """
    Accepts JSON input: {"feedback": "..."}
    Returns sentiment, urgency, and response strategy.
    """
    strategy = analysis_chain.invoke({"feedback": customer_feedback})
    return {"feedback": customer_feedback, "strategy": strategy}



if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
