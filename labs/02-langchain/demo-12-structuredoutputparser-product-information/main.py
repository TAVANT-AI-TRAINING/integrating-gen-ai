from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os
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
    title="StructuredOutputParser Demo",
    description=f"Currently using {provider.upper()} provider"
)

# ---------------- Response Schema Definition (Using Pydantic) ----------------
class ProductInfo(BaseModel):
    """Structured product information extracted from text."""
    product_name: str = Field(description="Name of the product")
    price: float = Field(description="Price in USD")
    in_stock: bool = Field(description="Whether product is available")

# ---------------- Parser Setup ----------------
parser = PydanticOutputParser(pydantic_object=ProductInfo)
format_instructions = parser.get_format_instructions()

# ---------------- Prompt Template ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract product information from the provided text. You must respond in the exact JSON format specified.\n{format_instructions}"),
    ("human", "{product_page_text}")
])

# ---------------- LLM Setup ----------------
llm = initialize_llm()

# ---------------- FastAPI Schema ----------------
class ProductInput(BaseModel):
    product_page_text: str

@app.post("/extract_product_info")
async def extract_product_info(data: ProductInput):
    """Extract product details from product page text."""
    try:
        # Format prompt dynamically with format instructions
        formatted_prompt = prompt.format_messages(
            product_page_text=data.product_page_text,
            format_instructions=format_instructions
        )

        # Invoke LLM
        response = llm.invoke(formatted_prompt)

        # Parse structured output using PydanticOutputParser
        parsed_output = parser.parse(response.content)

        return parsed_output.model_dump()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing product info: {str(e)}")


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
