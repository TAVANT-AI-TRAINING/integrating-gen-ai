# LangChain Comprehensive Guide

## Table of Contents

1. [Introduction & Background](#introduction--background)
2. [Why LangChain?](#why-langchain)
3. [Core Concepts](#core-concepts)
4. [Key Components](#key-components)
5. [LangChain vs Provider-Specific Implementations](#langchain-vs-provider-specific-implementations)
6. [FastAPI Integration](#fastapi-integration)
7. [LangChain Expression Language (LCEL)](#langchain-expression-language-lcel)
8. [Advanced Concepts](#advanced-concepts)
9. [Best Practices](#best-practices)
10. [Common Patterns](#common-patterns)

---

## Introduction & Background

### What is LangChain?

**LangChain** is an open-source framework designed to simplify the development of applications powered by Large Language Models (LLMs). Created by Harrison Chase in October 2022, it has rapidly become one of the most popular frameworks for building LLM-powered applications.

### History & Evolution

- **October 2022**: Initial release as an open-source Python library
- **Early 2023**: JavaScript/TypeScript version released (LangChain.js)
- **Mid 2023**: Introduction of LangChain Expression Language (LCEL)
- **2024-Present**: Ecosystem expansion with LangSmith, LangServe, and LangGraph

### The Problem LangChain Solves

Before LangChain, developers faced several challenges:

- **Provider Lock-in**: Code tightly coupled to specific LLM providers (OpenAI, Anthropic, etc.)
- **Boilerplate Code**: Repetitive code for prompt management, memory, and chains
- **Complex Workflows**: Difficulty orchestrating multi-step LLM interactions
- **No Standardization**: Each provider had different APIs and response formats

---

## Why LangChain?

### Key Benefits

1. **🔄 Provider Agnostic**
   - Switch between LLM providers without rewriting application logic
   - Unified interface for OpenAI, Anthropic, Google, Hugging Face, etc.

2. **🧩 Modular Architecture**
   - Composable components that work together seamlessly
   - Build complex applications from simple building blocks

3. **📚 Rich Ecosystem**
   - Pre-built integrations with 100+ services
   - Document loaders, vector stores, retrievers, and more

4. **🚀 Production Ready**
   - Built-in observability with LangSmith
   - Easy deployment with LangServe
   - Streaming support for real-time responses

5. **👥 Community & Support**
   - Large, active community
   - Extensive documentation and examples
   - Regular updates and improvements

---

## Core Concepts

### 1. Models

LangChain provides unified interfaces for different types of models:

#### **LLMs (Large Language Models)**

- Text-in, text-out models
- Examples: GPT-3, Claude, PaLM

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke("What is the capital of France?")
```

#### **Chat Models**

- Message-in, message-out models
- Structured conversation format

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat_model = ChatOpenAI(model="gpt-4")
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of France?")
]
response = chat_model.invoke(messages)
```

#### **Embeddings**

- Convert text to numerical vectors
- Used for semantic search and similarity

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(["Hello world", "Goodbye world"])
```

### 2. Prompts

Templates for generating prompts dynamically:

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Simple template
prompt = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}"
)
formatted = prompt.format(adjective="funny", content="cats")

# Chat template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Tell me about {topic}")
])
```

### 3. Output Parsers

Convert LLM output into structured data:

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# String parser
parser = StrOutputParser()

# Structured output
class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline")

json_parser = JsonOutputParser(pydantic_object=Joke)
```

### 4. Retrievers

Fetch relevant documents from a data source:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Vector store retriever
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Retrieve relevant docs
docs = retriever.invoke("What is machine learning?")
```

### 5. Memory

Store and retrieve conversation context:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "Hello"},
    {"output": "Hi there!"}
)
print(memory.load_memory_variables({}))
```

---

## Key Components

### Models Layer

| Component       | Purpose                | Examples                    |
| --------------- | ---------------------- | --------------------------- |
| **LLMs**        | Text completion        | GPT-3, Claude, Llama        |
| **Chat Models** | Conversational AI      | GPT-4, Claude, Gemini       |
| **Embeddings**  | Vector representations | OpenAI, Cohere, HuggingFace |

### Prompts Layer

| Component                  | Purpose           | Use Case                    |
| -------------------------- | ----------------- | --------------------------- |
| **PromptTemplate**         | String templates  | Simple text prompts         |
| **ChatPromptTemplate**     | Message templates | Chat conversations          |
| **FewShotPromptTemplate**  | Example-based     | Learning from examples      |
| **PipelinePromptTemplate** | Composite prompts | Complex multi-stage prompts |

### Data Connection Layer

| Component            | Purpose             | Examples                    |
| -------------------- | ------------------- | --------------------------- |
| **Document Loaders** | Load data           | PDF, CSV, Web, Database     |
| **Text Splitters**   | Chunk documents     | Recursive, Character, Token |
| **Vector Stores**    | Store embeddings    | Chroma, Pinecone, FAISS     |
| **Retrievers**       | Fetch relevant docs | Vector, BM25, MultiQuery    |

### Chains Layer

| Component           | Purpose             | When to Use          |
| ------------------- | ------------------- | -------------------- |
| **LLMChain**        | Basic LLM call      | Simple prompting     |
| **SequentialChain** | Sequential steps    | Multi-step workflows |
| **RouterChain**     | Conditional routing | Dynamic flow control |
| **TransformChain**  | Data transformation | Pre/post processing  |

### Agents Layer

| Component         | Purpose                   | Capability                   |
| ----------------- | ------------------------- | ---------------------------- |
| **Agent**         | Autonomous decision-maker | Tool selection & execution   |
| **Tools**         | Actions agent can take    | Search, Calculate, API calls |
| **AgentExecutor** | Run agent loop            | Orchestrate agent execution  |

---

## LangChain vs Provider-Specific Implementations

### Provider-Specific Implementation (e.g., OpenAI SDK)

```python
# Direct OpenAI SDK usage
import openai

openai.api_key = "your-api-key"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

**Pros:**

- ✅ Direct control over API
- ✅ Minimal dependencies
- ✅ Latest features immediately available
- ✅ Provider-specific optimizations

**Cons:**

- ❌ Tightly coupled to one provider
- ❌ No standardization across providers
- ❌ Manual prompt management
- ❌ No built-in retry logic
- ❌ Complex chaining requires custom code
- ❌ No observability tools

### LangChain Implementation

```python
# LangChain implementation
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create reusable components
llm = ChatOpenAI(model="gpt-4", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}")
])
parser = StrOutputParser()

# Build chain
chain = prompt | llm | parser

# Execute
response = chain.invoke({"input": "Hello!"})
print(response)
```

**Pros:**

- ✅ Provider agnostic (switch easily)
- ✅ Composable components
- ✅ Built-in retry & error handling
- ✅ Powerful chaining capabilities
- ✅ Observability with LangSmith
- ✅ Rich ecosystem of integrations
- ✅ Memory management built-in
- ✅ Streaming support

**Cons:**

- ❌ Additional abstraction layer
- ❌ Slight performance overhead
- ❌ Learning curve for framework
- ❌ May lag behind provider updates

### When to Use Each

| Scenario                   | Best Choice  | Reason                |
| -------------------------- | ------------ | --------------------- |
| **Simple API calls**       | Provider SDK | Less overhead         |
| **Multi-provider support** | LangChain    | Unified interface     |
| **Complex workflows**      | LangChain    | Chain composition     |
| **RAG applications**       | LangChain    | Built-in retrievers   |
| **Agent-based systems**    | LangChain    | Agent framework       |
| **Production monitoring**  | LangChain    | LangSmith integration |
| **Quick prototypes**       | Provider SDK | Faster setup          |
| **Long-term maintenance**  | LangChain    | Better abstraction    |

### Migration Example

**From OpenAI SDK to LangChain:**

```python
# BEFORE: OpenAI SDK
import openai

def chat_with_openai(message):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

# AFTER: LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    ("user", "{message}")
])

def chat_with_langchain(message):
    chain = prompt | llm
    return chain.invoke({"message": message}).content
```

**Switching Providers:**

```python
# With provider-specific SDK - requires code rewrite
# OpenAI → Anthropic requires changing entire implementation

# With LangChain - just change model initialization
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# OpenAI
llm = ChatOpenAI(model="gpt-4")

# Switch to Anthropic - same chain works!
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Use same chain regardless of provider
chain = prompt | llm | parser
```

---

## FastAPI Integration

### Why Integrate LangChain with FastAPI?

FastAPI + LangChain provides:

- 🚀 **Fast, async-ready** REST APIs
- 📝 **Auto-generated documentation** (OpenAPI/Swagger)
- ✅ **Type safety** with Pydantic
- 🔄 **Streaming support** for real-time responses
- 🛡️ **Built-in validation** and error handling

### Basic Integration Pattern

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize FastAPI
app = FastAPI(title="LangChain API", version="1.0.0")

# Initialize LangChain components
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{question}")
])
chain = prompt | llm | StrOutputParser()

# Request/Response models
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    model: str

# Endpoints
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        answer = chain.invoke({"question": request.question})
        return AnswerResponse(answer=answer, model="gpt-4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### Advanced: Streaming Responses

```python
from fastapi.responses import StreamingResponse
from langchain_core.callbacks import StreamingStdOutCallbackHandler

class StreamingCallback(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens.append(token)

@app.post("/stream")
async def stream_response(request: QuestionRequest):
    async def generate():
        callback = StreamingCallback()
        llm_stream = ChatOpenAI(
            model="gpt-4",
            streaming=True,
            callbacks=[callback]
        )
        chain_stream = prompt | llm_stream | StrOutputParser()

        for chunk in chain_stream.stream({"question": request.question}):
            yield f"data: {chunk}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Production-Ready Example

```python
import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

# Load environment
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LangChain Production API",
    version="1.0.0",
    description="Production-ready LangChain + FastAPI integration"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency: Get LLM chain
def get_chain():
    try:
        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4"),
            temperature=0.7,
            request_timeout=30
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant"),
            ("user", "{input}")
        ])
        return prompt | llm | StrOutputParser()
    except Exception as e:
        logger.error(f"Failed to initialize chain: {e}")
        raise

# Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: int = 0

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chain=Depends(get_chain)):
    """Send a message and get AI response"""
    try:
        logger.info(f"Processing request: {request.message[:50]}...")
        response = chain.invoke({"input": request.message})

        return ChatResponse(
            response=response,
            model=os.getenv("MODEL_NAME", "gpt-4"),
            tokens_used=len(response.split())  # Simplified
        )
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": os.getenv("MODEL_NAME", "gpt-4")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### LangServe Integration

LangServe makes it even easier to deploy LangChain applications:

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI()

# Define chain
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
chain = prompt | llm

# Automatically add routes
add_routes(
    app,
    chain,
    path="/joke",
    enabled_endpoints=["invoke", "batch", "stream"]
)

# Provides:
# POST /joke/invoke
# POST /joke/batch
# POST /joke/stream
# GET /joke/playground (interactive UI)
```

---

## LangChain Expression Language (LCEL)

### What is LCEL?

**LCEL** (LangChain Expression Language) is a declarative way to compose chains using the pipe operator (`|`). It's the modern, recommended way to build LangChain applications.

### Why LCEL?

1. **Intuitive Syntax**: Read left-to-right like Unix pipes
2. **Automatic Features**: Streaming, async, batching built-in
3. **Type Safety**: Better IDE support and error detection
4. **Optimized**: Parallel execution when possible
5. **Composability**: Easy to build and modify chains

### Basic LCEL Syntax

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Components
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

# Compose with pipe operator
chain = prompt | model | parser

# Execute
result = chain.invoke({"topic": "Python"})
```

### LCEL vs Legacy Chains

**Legacy (LLMChain):**

```python
from langchain.chains import LLMChain

# Old way
chain = LLMChain(llm=model, prompt=prompt)
result = chain.run(topic="Python")
```

**Modern (LCEL):**

```python
# New way
chain = prompt | model | parser
result = chain.invoke({"topic": "Python"})
```

### LCEL Features

#### 1. Automatic Streaming

```python
# Stream output token by token
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

#### 2. Automatic Async Support

```python
# Async execution
result = await chain.ainvoke({"topic": "AI"})

# Async streaming
async for chunk in chain.astream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

#### 3. Batch Processing

```python
# Process multiple inputs in parallel
results = chain.batch([
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"}
])
```

#### 4. Parallel Execution

```python
from langchain_core.runnables import RunnableParallel

# Execute multiple chains in parallel
analyze_chain = RunnableParallel(
    sentiment=prompt_sentiment | model | parser,
    summary=prompt_summary | model | parser,
    keywords=prompt_keywords | model | parser
)

result = analyze_chain.invoke({"text": "Your text here"})
# Returns: {"sentiment": "...", "summary": "...", "keywords": "..."}
```

#### 5. Fallbacks

```python
# Try primary, fall back to secondary
chain_with_fallback = primary_chain.with_fallbacks([backup_chain])
```

#### 6. Retry Logic

```python
# Automatic retry on failure
chain_with_retry = chain.with_retry(
    retry_if_exception_type=(ConnectionError,),
    stop_after_attempt=3,
    wait_exponential_multiplier=1
)
```

### Complex LCEL Example

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from operator import itemgetter

# Define multiple processing steps
def extract_keywords(text: str) -> list:
    return text.lower().split()

def filter_keywords(keywords: list) -> list:
    stopwords = {"the", "a", "an", "in", "on", "at"}
    return [k for k in keywords if k not in stopwords]

# Complex chain with branching
chain = (
    # Input passthrough and transformation
    RunnablePassthrough.assign(
        keywords=lambda x: extract_keywords(x["text"])
    )
    # Parallel operations
    | RunnableParallel(
        original=itemgetter("text"),
        filtered=lambda x: filter_keywords(x["keywords"]),
        summary=itemgetter("text") | prompt | model | parser,
    )
    # Final processing
    | RunnableLambda(lambda x: {
        "summary": x["summary"],
        "keywords": x["filtered"],
        "original_length": len(x["original"])
    })
)

result = chain.invoke({"text": "Your document text here"})
```

### LCEL Best Practices

1. **Use Type Hints**: Helps with debugging and IDE support

```python
from typing import Dict, Any

def process_input(data: Dict[str, Any]) -> str:
    return data["text"].upper()

chain = RunnableLambda(process_input) | model | parser
```

2. **Break Complex Chains**: Create sub-chains for clarity

```python
# Instead of one long chain
preprocessing_chain = step1 | step2 | step3
analysis_chain = step4 | step5
postprocessing_chain = step6 | step7

# Combine
full_chain = preprocessing_chain | analysis_chain | postprocessing_chain
```

3. **Use RunnablePassthrough**: Preserve data through chain

```python
chain = (
    RunnablePassthrough.assign(processed=lambda x: process(x["input"]))
    | RunnablePassthrough.assign(analyzed=lambda x: analyze(x["processed"]))
    | format_output
)
```

4. **Test Components Individually**

```python
# Test each component
assert prompt.invoke({"topic": "AI"}).messages[0].content
assert model.invoke("Test").content
assert parser.parse("Test output") == "Test output"

# Then combine
chain = prompt | model | parser
```

---

## Advanced Concepts

### 1. Agents & Tools

Agents can autonomously decide which tools to use:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# Define custom tools
@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    # Simulated - would call real API
    return f"Weather in {location}: Sunny, 72°F"

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [calculate, get_weather]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Execute
result = agent_executor.invoke({
    "input": "What's 25 * 17 and what's the weather in Paris?"
})
```

### 2. Retrieval-Augmented Generation (RAG)

Combine LLMs with external knowledge:

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. Create RAG chain
template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI(model="gpt-4")

rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Query
answer = rag_chain.invoke("What is the main topic?")
```

### 3. Conversational Memory

Maintain context across interactions:

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# Setup memory
memory = ConversationBufferMemory(return_messages=True)

# Create prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Create chain with memory
llm = ChatOpenAI(model="gpt-4")

def get_chat_history():
    return memory.load_memory_variables({})["history"]

chain = (
    RunnablePassthrough.assign(history=lambda _: get_chat_history())
    | prompt
    | llm
    | StrOutputParser()
)

# Use in conversation
def chat(message: str) -> str:
    response = chain.invoke({"input": message})
    memory.save_context({"input": message}, {"output": response})
    return response

# Conversation
print(chat("My name is Alice"))
print(chat("What's my name?"))  # Remembers context
```

### 4. Custom Chains & Runnables

Create reusable custom components:

```python
from langchain_core.runnables import Runnable
from typing import Any, Dict

class CustomProcessor(Runnable):
    """Custom processing step"""

    def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        # Custom logic
        processed = input["text"].upper()
        return {"processed": processed, "length": len(processed)}

    async def ainvoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
        return self.invoke(input, config)

# Use in chain
custom_chain = CustomProcessor() | prompt | llm | parser
```

### 5. Structured Output

Force LLM to return structured data:

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class MovieReview(BaseModel):
    """Movie review with structured fields"""
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 0-10")
    pros: list[str] = Field(description="Positive aspects")
    cons: list[str] = Field(description="Negative aspects")
    verdict: str = Field(description="Final recommendation")

# Create structured LLM
llm = ChatOpenAI(model="gpt-4")
structured_llm = llm.with_structured_output(MovieReview)

# Get structured response
review = structured_llm.invoke("Review the movie Inception")
print(f"Title: {review.title}")
print(f"Rating: {review.rating}/10")
```

### 6. Multi-Modal Inputs

Process images and text together:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4-vision-preview")

message = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
    ]
)

response = llm.invoke([message])
print(response.content)
```

### 7. Callbacks & Observability

Monitor and debug chain execution:

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class CustomCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        print(f"LLM finished with: {response.generations[0][0].text[:100]}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"Chain finished with: {outputs}")

# Use callback
callback = CustomCallbackHandler()
chain = prompt | llm | parser
result = chain.invoke({"input": "Hello"}, config={"callbacks": [callback]})
```

---

## Best Practices

### 1. Configuration Management

```python
# ✅ Good: Use environment variables
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=float(os.getenv("TEMPERATURE", "0.7"))
)

# ❌ Bad: Hard-coded values
llm = ChatOpenAI(
    model="gpt-4",
    api_key="sk-hardcoded-key",
    temperature=0.7
)
```

### 2. Error Handling

```python
# ✅ Good: Comprehensive error handling
from langchain_core.exceptions import OutputParserException

try:
    result = chain.invoke({"input": user_input})
except OutputParserException as e:
    logger.error(f"Parsing error: {e}")
    result = "Sorry, couldn't process that."
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    result = "An error occurred."

# With retry
chain_with_retry = chain.with_retry(
    retry_if_exception_type=(ConnectionError, TimeoutError),
    stop_after_attempt=3
)
```

### 3. Prompt Engineering

```python
# ✅ Good: Clear, specific prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Python expert.
    - Provide concise, working code
    - Include comments for complex logic
    - Follow PEP 8 style guide"""),
    ("user", "{question}")
])

# ❌ Bad: Vague prompts
prompt = ChatPromptTemplate.from_template("Answer this: {question}")
```

### 4. Testing

```python
import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

@pytest.fixture
def llm_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")
    return prompt | llm

def test_chain_output(llm_chain):
    result = llm_chain.invoke({"name": "Alice"})
    assert "Alice" in result.content
    assert len(result.content) > 0

def test_chain_batch(llm_chain):
    results = llm_chain.batch([
        {"name": "Alice"},
        {"name": "Bob"}
    ])
    assert len(results) == 2
    assert all(r.content for r in results)
```

### 5. Performance Optimization

```python
# Use batch processing
inputs = [{"topic": t} for t in topics]
results = chain.batch(inputs, config={"max_concurrency": 5})

# Use streaming for long responses
for chunk in chain.stream({"input": question}):
    display(chunk, end="", flush=True)

# Cache expensive operations
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())
```

### 6. Security

```python
# ✅ Input validation
from pydantic import BaseModel, validator

class UserInput(BaseModel):
    message: str

    @validator('message')
    def validate_length(cls, v):
        if len(v) > 2000:
            raise ValueError('Message too long')
        return v

# ✅ Sanitize outputs
def sanitize_output(text: str) -> str:
    # Remove sensitive patterns
    import re
    # Remove API keys, emails, etc.
    text = re.sub(r'sk-[a-zA-Z0-9]{32,}', '[REDACTED]', text)
    return text
```

---

## Common Patterns

### 1. Question-Answering Pattern

```python
qa_chain = (
    {"question": RunnablePassthrough()}
    | ChatPromptTemplate.from_template("Answer: {question}")
    | llm
    | StrOutputParser()
)
```

### 2. Summarization Pattern

```python
summarize_chain = (
    {"text": RunnablePassthrough()}
    | ChatPromptTemplate.from_template(
        "Summarize the following in 3 sentences:\n\n{text}"
    )
    | llm
    | StrOutputParser()
)
```

### 3. Translation Pattern

```python
translate_chain = (
    {
        "text": itemgetter("text"),
        "target_language": itemgetter("language")
    }
    | ChatPromptTemplate.from_template(
        "Translate to {target_language}:\n\n{text}"
    )
    | llm
    | StrOutputParser()
)
```

### 4. Classification Pattern

```python
from langchain_core.pydantic_v1 import BaseModel

class Classification(BaseModel):
    category: str
    confidence: float

classify_chain = (
    ChatPromptTemplate.from_template(
        "Classify this text into one of: tech, sports, politics\n\n{text}"
    )
    | llm.with_structured_output(Classification)
)
```

### 5. Multi-Step Analysis Pattern

```python
analysis_chain = RunnableParallel(
    sentiment=sentiment_prompt | llm | parser,
    entities=entity_prompt | llm | parser,
    summary=summary_prompt | llm | parser
) | RunnableLambda(lambda x: {
    "analysis": {
        "sentiment": x["sentiment"],
        "entities": x["entities"],
        "summary": x["summary"]
    }
})
```

---

## Summary

### Quick Reference

| Feature               | LangChain | Provider SDK |
| --------------------- | --------- | ------------ |
| **Learning Curve**    | Medium    | Easy         |
| **Provider Lock-in**  | No        | Yes          |
| **Complex Workflows** | Easy      | Hard         |
| **Streaming**         | Built-in  | Manual       |
| **Memory**            | Built-in  | Manual       |
| **Agents**            | Built-in  | Custom code  |
| **Observability**     | LangSmith | Custom       |
| **Production Ready**  | Yes       | Yes          |

### When to Choose LangChain

Choose LangChain when you need:

- ✅ Multi-provider support
- ✅ Complex chains and workflows
- ✅ Built-in RAG capabilities
- ✅ Agent-based systems
- ✅ Production observability
- ✅ Team collaboration
- ✅ Long-term maintenance

### Key Takeaways

1. **LangChain is a framework** - Not just an API wrapper
2. **LCEL is the modern way** - Use pipe operators for chains
3. **Composability is key** - Build complex from simple
4. **FastAPI integration** - Easy API deployment
5. **Provider agnostic** - Switch models easily
6. **Production ready** - With proper error handling and monitoring

---

## Additional Resources

- **Official Docs**: https://python.langchain.com/
- **API Reference**: https://api.python.langchain.com/
- **LangSmith**: https://smith.langchain.com/
- **GitHub**: https://github.com/langchain-ai/langchain
- **Discord Community**: https://discord.gg/langchain

---

_Last Updated: February 2026_
