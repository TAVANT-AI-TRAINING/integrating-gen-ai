# StructuredOutputParser Demo: LangChain + Gemini

A FastAPI application demonstrating how to use LangChain's `StructuredOutputParser` to extract structured data from AI model responses using Google's Gemini AI model.

## Objective

To demonstrate the StructuredOutputParser workflow:

1. **Define Response Schemas**: Create structured data schemas using ResponseSchema
2. **Initialize Parser**: Use StructuredOutputParser with the response schemas
3. **Generate Format Instructions**: Get structured prompt instructions for the LLM
4. **Invoke LLM**: Call the model with format instructions
5. **Parse Structured Output**: Extract structured data from the response
6. **Return Structured Response**: Demonstrate structured JSON output

## Project Structure

```
structured_output_parser_demo/
├── .env                    # Environment variables (API key)
├── main.py                # FastAPI application with StructuredOutputParser demo
├── pyproject.toml         # Project dependencies
├── README.md              # This file
└── .python-version        # Python version specification
```

## Setup Instructions

1. **Install Dependencies**

   ```bash
   uv sync
   ```

2. **Activate the virtual environment:**

   **For Linux/macOS:**

   ```bash
   source .venv/bin/activate
   ```

   **For Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **For Windows (CMD):**

   ```cmd
   .venv\Scripts\activate.bat
   ```

   **Note**: If using `uv run` command (as shown in Running section), activation is optional as `uv run` automatically uses the virtual environment.

3. **Environment Configuration**
   Create a `.env` file in the project root:

   **For OpenAI:**

   ```env
   LLM_PROVIDER=openai
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL_NAME=gpt-4o-mini
   ```

   **For Google Gemini:**

   ```env
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   GEMINI_MODEL_NAME=gemini-2.5-flash
   ```

   **How to get API keys:**
   - **OpenAI**: Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)

   **Note:**
   - The model names can be updated to any supported model. Model names may change over time, so always refer to the latest options in the provider's documentation.

4. **Run the FastAPI Server**

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Test the API**
   - Open your browser to `http://localhost:8000/docs` for interactive API documentation
   - Or send POST requests to the extract_product_info endpoint

## Features

- ✅ **StructuredOutputParser Integration**: Extract structured data from AI responses
- ✅ **ResponseSchema Definition**: Define structured data schemas with types
- ✅ **Product Information Extraction**: Real-world example of structured data extraction
- ✅ **Format Instructions**: Automatic generation of LLM prompt instructions
- ✅ **Google Gemini Integration**: Uses Gemini 2.0 Flash model through LangChain
- ✅ **Environment variable loading** with `python-dotenv`
- ✅ **FastAPI web service** with automatic API documentation
- ✅ **Comprehensive error handling** with HTTP status codes

## API Endpoints

### POST /extract_product_info

Extract product information from product page text.

**Request Body:**

```json
{
  "product_page_text": "iPhone 15 Pro Max - $1199 - Available in stock with free shipping"
}
```

**Response:**

```json
{
  "product_name": "iPhone 15 Pro Max",
  "price": 1199,
  "in_stock": true
}
```

## Implementation notes

Implementation-specific code and detailed internal examples have been removed from this README for brevity. If you want to see the actual implementation, please check the `main.py` source in this folder.

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with:

- Try-it-out functionality
- Request/response schemas
- Example requests and responses
- Error code documentation

## Use Cases

This StructuredOutputParser functionality is useful for:

- **Data Extraction**: Converting unstructured text to structured data
- **E-commerce**: Extracting product information from descriptions
- **Content Processing**: Parsing structured information from text
- **API Integration**: Ensuring consistent data formats for downstream systems
- **Automation**: Automating data extraction from various sources

## Expected Behavior

When you test the extract_product_info endpoint, you'll observe:

- ✅ **Structured Extraction**: Consistent JSON format with defined fields
- ✅ **Type Conversion**: Proper data types (strings, numbers, booleans)
- ✅ **Schema Compliance**: All defined fields are extracted
- ✅ **Real-world Application**: Practical product information extraction

## Expected Output

When you run the server and test the `/extract_product_info` endpoint, you'll receive:

- A JSON response with structured product information
- Extracted fields matching the defined schema
- Interactive API documentation at `/docs`
