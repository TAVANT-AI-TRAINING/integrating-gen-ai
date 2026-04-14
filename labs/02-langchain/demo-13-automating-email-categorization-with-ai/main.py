from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
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
    title="PydanticOutputParser Demo",
    description=f"Currently using {provider.upper()} provider"
)

# ---------------- Pydantic Schema Definition ----------------
class EmailClassification(BaseModel):
    category: Literal["urgent", "normal", "spam"] = Field(
        description="Urgency category of the email"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in classification (0-1)"
    )
    action_required: bool = Field(
        description="Whether human action is needed"
    )
    summary: str = Field(
        max_length=100,
        description="One-sentence summary of email content"
    )

# ---------------- Parser Setup ----------------
parser = PydanticOutputParser(pydantic_object=EmailClassification)
format_instructions = parser.get_format_instructions()

# ---------------- Prompt Template ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert email classifier. Analyze emails and provide structured output."),
    ("human", "Classify this email:\n\n{email_text}\n\n{format_instructions}")
])

# ---------------- LLM Setup ----------------
llm = initialize_llm()

# ---------------- FastAPI Schema ----------------
class EmailInput(BaseModel):
    email_text: str

@app.post("/classify_email")
async def classify_email(data: EmailInput):
    """Classify an email and return structured response."""
    try:
        # Format prompt dynamically
        formatted_prompt = prompt.format(
            email_text=data.email_text,
            format_instructions=format_instructions
        )

        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Parse and validate using PydanticOutputParser
        parsed_output = parser.parse(response.content)

        return parsed_output.dict()

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation Error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
