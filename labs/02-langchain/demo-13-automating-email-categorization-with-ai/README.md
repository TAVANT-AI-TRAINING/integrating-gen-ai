# PydanticOutputParser Demo: LangChain + Gemini

A FastAPI application demonstrating how to use LangChain's `PydanticOutputParser` to extract structured data from AI model responses using Google's Gemini AI model.

## Objective

To demonstrate the PydanticOutputParser workflow:

1. **Define Pydantic Schema**: Create a structured data model with validation rules
2. **Initialize Parser**: Use PydanticOutputParser with the schema
3. **Generate Format Instructions**: Get structured prompt instructions for the LLM
4. **Invoke LLM**: Call the model with format instructions
5. **Parse and Validate**: Extract structured data and validate against schema
6. **Return Structured Response**: Demonstrate structured JSON output

## Project Structure

```
pydantic_output_parser_demo/
├── .env                    # Environment variables (API key)
├── main.py                # FastAPI application with PydanticOutputParser demo
├── pyproject.toml         # Project dependencies
├── README.md              # This file
└── .python-version        # Python version specification
```

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

## Installation

1. Navigate to the project directory:

   **For Linux/macOS:**

   ```bash
   cd demo-13-automating-email-categorization-with-ai
   ```

   **For Windows:**

   ```cmd
   cd demo-13-automating-email-categorization-with-ai
   ```

2. Install dependencies using UV:

   **For Linux/Windows (Same command):**

   ```bash
   uv sync
   ```

   This will automatically:
   - Create a virtual environment
   - Install all dependencies from `pyproject.toml`
   - Set up the project environment

3. Activate the virtual environment:

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

## Configuration

1. Create a `.env` file in the project root:

   **For Linux/macOS:**

   ```bash
   touch .env
   ```

   **For Windows (PowerShell):**

   ```powershell
   New-Item -Path .env -ItemType File
   ```

   **For Windows (CMD):**

   ```cmd
   type nul > .env
   ```

2. Add your API configuration to the `.env` file:

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

   **Note**: The model names can be updated to any supported model. Model names may change over time, so always refer to the latest options in the provider's documentation.

## Running the Application

**For Linux/Windows (Same commands):**

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The application will start on `http://localhost:8000`

**Note**: On Windows, you can use either PowerShell or CMD for these commands.

## Testing the API

- Open your browser to `http://localhost:8000/docs` for interactive API documentation
- Or send POST requests to the classify_email endpoint

## Features

- ✅ **PydanticOutputParser Integration**: Extract structured data from AI responses
- ✅ **Pydantic Schema Validation**: Type-safe data models with validation rules
- ✅ **Email Classification**: Real-world example of structured data extraction
- ✅ **Format Instructions**: Automatic generation of LLM prompt instructions
- ✅ **Google Gemini Integration**: Uses Gemini 2.0 Flash model through LangChain
- ✅ **Environment variable loading** with `python-dotenv`
- ✅ **FastAPI web service** with automatic API documentation
- ✅ **Comprehensive error handling** with validation and HTTP status codes

## API Endpoints

### POST /classify_email

Classify an email and return structured response.

**Request Body:**

```json
{
  "email_text": "URGENT: Server is down! Please fix immediately!"
}
```

**Response:**

```json
{
  "category": "urgent",
  "confidence_score": 0.95,
  "action_required": true,
  "summary": "Server down notification requiring immediate attention"
}
```

**Request Body:**

```json
{
  "email_text": "I hope this message finds you well. I wanted to take a moment to share a comprehensive update regarding the ongoing global rollout of our next-generation product suite, the migration of our internal systems to a unified infrastructure, and a preview of the upcoming strategic realignments that will shape our roadmap for the next three quarters.As you’re aware, over the past eighteen months, our cross-functional teams across engineering, marketing, sales, and customer success have been deeply involved in planning, testing, and executing multiple phases of our product expansion initiative. This project, codenamed “Aurora,” aims to unify our customer experience under a single, intelligent interface that integrates analytics, automation, and personalized support.The first wave of this rollout was launched in North America and Western Europe in July, with initial adoption metrics surpassing projections by 28%. Customer feedback has been overwhelmingly positive, particularly around the AI-driven insights dashboard and the simplified subscription management system. However, as with any large-scale deployment, we’ve also encountered several issues requiring our immediate attention. Chief among these challenges has been the migration of legacy data from region-specific servers to the centralized cloud environment. While the transition has been largely smooth, a few clients in the enterprise segment reported latency and synchronization errors in the early stages. Our DevOps and Infrastructure teams have been working around the clock to patch these issues, introduce improved caching mechanisms, and implement real-time monitoring to ensure reliability and uptime. Parallel to this, our customer success teams are focusing on client education and onboarding. We’ve launched a new self-service resource center with video tutorials, interactive walkthroughs, and guided troubleshooting. Early engagement metrics show that customers who use the resource center are resolving queries 40% faster than those relying solely on email support.From an organizational standpoint, the most significant change on the horizon is the migration of our internal CRM and analytics systems to the new unified data platform. This move will consolidate reporting, forecasting, and performance tracking, allowing teams to work with a single source of truth. The migration window is scheduled for the second week of next month, and all department heads are requested to ensure data backups are completed before that date.In terms of marketing and external communications, our next major push will coincide with the Asia-Pacific launch scheduled for mid-November. The regional marketing team has been coordinating with our global brand strategy group to adapt messaging that resonates with local audiences while maintaining global brand consistency. Expect to see new campaigns centered on sustainability, efficiency, and digital transformation — themes that have tested exceptionally well with early focus groups. On the technical front, our AI research division has been experimenting with several model optimizations that should reduce inference latency by nearly 20% while lowering cloud compute costs. These improvements, combined with the rollout of our hybrid edge architecture, will make our systems more responsive and scalable. A pilot test is planned for select partners before general availability. I’d also like to recognize the efforts of our cybersecurity team, who recently completed a full audit of our new authentication pipeline. Their proactive approach identified several potential vulnerabilities early in development, all of which have been addressed through code refactoring, token lifecycle improvements, and enhanced encryption standards. As we move forward, the leadership team will be placing greater emphasis on transparent communication and inter-departmental collaboration. We are introducing bi-weekly “open sprint reviews,” where representatives from each major function can present progress, highlight blockers, and suggest cross-team solutions. The goal is to foster shared accountability and reduce the communication silos that have occasionally slowed us down in the past. Lastly, I want to acknowledge the immense dedication everyone has shown throughout this process. Balancing the dual responsibilities of ongoing client work and transformative internal projects is no small feat. Your resilience, creativity, and commitment continue to be the driving force behind our success. Looking ahead, the next six months will be crucial as we scale Aurora globally, refine our automation systems, and solidify our market leadership position. Please review the attached rollout timeline and ensure your teams are aligned on upcoming deliverables. Department leads will receive follow-up meeting invites for detailed planning sessions early next week. Thank you all for your continued hard work, focus, and collaboration. Let’s keep pushing forward with the same energy and innovation that brought us this far."
}
```

**Response:**

```json
{
  "detail": "Failed to parse EmailClassification from completion {\"category\": \"normal\", \"confidence_score\": 0.95, \"action_required\": true, \"summary\": \"Comprehensive update on product rollout, system migration, and strategic realignments, requiring review of the attached timeline and alignment on deliverables.\"}. Got: 1 validation error for EmailClassification\nsummary\n  String should have at most 100 characters [type=string_too_long, input_value='Comprehensive update on ...gnment on deliverables.', input_type=str]\n    For further information visit https://errors.pydantic.dev/2.10/v/string_too_long\nFor troubleshooting, visit: https://python.langchain.com/docs/troubleshooting/errors/OUTPUT_PARSING_FAILURE "
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with:

- Try-it-out functionality
- Request/response schemas
- Example requests and responses
- Error code documentation

## Use Cases

This PydanticOutputParser functionality is useful for:

- **Data Extraction**: Converting unstructured text to structured data
- **API Integration**: Ensuring consistent data formats for downstream systems
- **Validation**: Automatic type checking and constraint validation
- **Real-world Applications**: Email classification, document processing, data parsing
- **Type Safety**: Ensuring data integrity with Pydantic models

## Expected Behavior

When you test the classify_email endpoint, you'll observe:

- ✅ **Structured Classification**: Consistent JSON format with validated fields
- ✅ **Type Safety**: Proper data types (strings, floats, booleans)
- ✅ **Validation**: Automatic constraint checking (confidence_score 0-1, summary max 100 chars)
- ✅ **Real-world Application**: Practical email classification use case

## Expected Output

When you run the server and test the `/classify_email` endpoint, you'll receive:

- A JSON response with structured email classification data
- Validated fields with proper types and constraints
- Interactive API documentation at `/docs`
