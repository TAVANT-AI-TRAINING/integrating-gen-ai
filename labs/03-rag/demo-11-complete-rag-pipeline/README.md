# Demo 11: Complete RAG Pipeline

This demo demonstrates the **complete RAG (Retrieval-Augmented Generation) pipeline** from end to end, including both retrieval AND generation phases.

## What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by retrieving relevant information from a knowledge base before generating answers.

### RAG Workflow

```
1. INGESTION PHASE
   ┌─────────────┐
   │  Documents  │ → Load documents (PDF, text, web)
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Chunks    │ → Split into smaller pieces
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Embeddings  │ → Generate vector embeddings
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Vector DB   │ → Store in vector database
   └─────────────┘

2. RETRIEVAL-GENERATION PHASE (Query Time)
   ┌─────────────┐
   │   Query     │ → User's question
   └─────────────┘
          ↓
   ┌─────────────┐
   │ Embeddings  │ → Convert query to vector
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Retrieve   │ → Find similar chunks
   └─────────────┘
          ↓
   ┌─────────────┐
   │  Generate   │ → LLM generates answer using context
   └─────────────┘
          ↓
   ┌─────────────┐
   │   Answer    │ → Final response to user
   └─────────────┘
```

## What Makes This Demo Complete?

This demo shows the **FULL RAG cycle**:

1. **Ingestion**: Load → Chunk → Embed → Store
2. **Retrieval**: Search for relevant information
3. **Generation**: Use LLM to generate answer with retrieved context

Previous demos showed only parts of this flow. This demo combines everything into one complete pipeline.

## Key Features

✅ **Complete RAG Pipeline** - Full ingestion, retrieval, and generation flow  
✅ **Multi-Source Loading** - Load from PDF, text files, and web pages  
✅ **Config-Driven Vector DB** - Switch between ChromaDB and Pinecone via .env  
✅ **Standard OpenAI API** - No Azure required  
✅ **Single File** - All code in main.py (~500 lines)  
✅ **Clear Output** - Step-by-step console output showing each phase  
✅ **Retrieval Scenarios** - Demonstrates different k values and configurations  
✅ **LCEL Chain** - Uses LangChain Expression Language for RAG chain

## Prerequisites

- Python 3.12+
- OpenAI API key

## Installation

1. Set up Python environment:

```bash
cd demo-10-complete-rag-pipeline
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
uv pip install -e .
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Configuration

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-...

# Vector Database Selection (chromadb or pinecone)
VECTOR_DB=chromadb

# Optional LLM Model
OPENAI_MODEL=gpt-4o-mini

# ChromaDB Settings (if using VECTOR_DB=chromadb)
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies

# Pinecone Settings (if using VECTOR_DB=pinecone)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### Vector Database Options

**ChromaDB (Default)**

- Local, file-based storage
- No external services required
- Perfect for learning and development
- Set `VECTOR_DB=chromadb`

**Pinecone**

- Cloud-based vector database
- Requires Pinecone API key
- Scalable for production
- Set `VECTOR_DB=pinecone`

## Usage

Run the complete demo:

```bash
uv run python main.py
```

## What the Demo Does

### Step 1: Load Documents

Loads documents from multiple sources:

- PDF: `Documents/company_policy.pdf`
- Text files: `Documents/*.txt`
- Web page: https://www.python.org/

### Step 2: Chunk Documents

Splits documents into smaller chunks:

- Chunk size: 1000 characters
- Overlap: 200 characters
- Uses `RecursiveCharacterTextSplitter`

### Step 3: Store with Embeddings

Generates embeddings and stores in vector database:

- Embeddings: OpenAI `text-embedding-3-small` (1536 dimensions)
- Storage: ChromaDB or Pinecone (config-driven)

### Step 4: Retrieve Relevant Documents

Demonstrates different retrieval scenarios:

- Similarity search with different k values (k=2, k=4)
- Retriever interface configuration
- Shows retrieved document metadata and content

### Step 5: Generate Answers

Uses LLM to generate answers with retrieved context:

- LLM: `gpt-4o-mini` (configurable)
- RAG chain built with LCEL
- Answers based only on retrieved context

### Test Queries

The demo automatically tests these queries:

1. "What is the remote work policy?"
2. "What are the code review guidelines?"
3. "Tell me about Python programming"

## Understanding the Output

The demo produces detailed console output showing each step:

```
======================================================================
COMPLETE RAG PIPELINE CONFIGURATION
======================================================================
Vector Database: CHROMADB
LLM Model: gpt-4o-mini
Chunk Size: 1000 characters
Chunk Overlap: 200 characters
======================================================================

✓ OpenAI embeddings initialized: text-embedding-3-small
✓ OpenAI LLM initialized: gpt-4o-mini
✓ ChromaDB initialized
  - Storage: ./chroma_db
  - Collection: company_policies
✓ Vector store ready!

======================================================================
STEP 1: LOADING DOCUMENTS
======================================================================
[1.1] Loading PDF...
  ✓ Loaded 5 page(s) from PDF
[1.2] Loading text files...
  ✓ Loaded: guidelines.txt
  ✓ Loaded: policies.txt
[1.3] Loading web page...
  ✓ Loaded web page
✓ Total documents loaded: 8

======================================================================
STEP 2: CHUNKING DOCUMENTS
======================================================================
✓ Created 127 chunks
  - Average length: 847 characters
  - Min length: 234 characters
  - Max length: 1000 characters

======================================================================
STEP 3: STORE CHUNKS WITH EMBEDDINGS
======================================================================
🔄 Processing 127 chunks...
  - Generating embeddings with OpenAI
  - Storing in CHROMADB
✓ Successfully stored 127 chunks with embeddings!

======================================================================
STEP 4: RETRIEVE RELEVANT DOCUMENTS
======================================================================
Query: "What is the remote work policy?"
Retrieving top 3 most relevant chunks...

✓ Retrieved 3 relevant chunks:
[1] Source: Documents/company_policy.pdf
    Page: 2
    Length: 892 characters
    Preview: Remote Work Policy  All employees are eligible for remote work...

======================================================================
STEP 5: GENERATE ANSWER WITH LLM
======================================================================
🤖 Generating answer with LLM...
   Using 3 retrieved chunks as context
✓ Answer generated!

======================================================================
RAG PIPELINE RESULT
======================================================================
❓ Question: What is the remote work policy?

💡 Answer:
All employees are eligible for remote work arrangements. They must
submit a remote work request form and maintain regular communication
with their team during remote work days...
======================================================================
```

## Code Structure

```python
# Configuration
- Load environment variables
- Initialize embeddings (OpenAI)
- Initialize LLM (OpenAI)
- Initialize vector store (ChromaDB or Pinecone)

# Step 1: load_documents()
- Load PDF files
- Load text files
- Load web pages

# Step 2: chunk_documents()
- Split documents into chunks
- Calculate statistics

# Step 3: store_chunks()
- Generate embeddings
- Store in vector database

# Step 4: retrieve_documents()
- Perform similarity search
- Return relevant chunks

# Step 5: generate_answer()
- Format retrieved docs as context
- Create RAG prompt
- Build LCEL chain
- Generate answer with LLM

# Pipeline: run_rag_pipeline()
- Orchestrates Steps 4-5
- Complete query-to-answer flow

# Scenarios: demonstrate_retrieval_scenarios()
- Different k values
- Retriever interface
```

## Key Concepts Explained

### 1. Document Loading

LangChain provides specialized loaders for different formats:

- `PyPDFLoader` - PDF files
- `TextLoader` - Text files
- `WebBaseLoader` - Web pages

### 2. Text Splitting

`RecursiveCharacterTextSplitter`:

- Splits text recursively by separators: `["\n\n", "\n", " ", ""]`
- Tries to keep related content together
- Maintains overlaps for context continuity

### 3. Embeddings

Vector representations of text:

- Converts text to 1536-dimensional vectors
- Similar content has similar vectors
- Enables semantic search (meaning-based, not keyword-based)

### 4. Vector Database

Stores embeddings for efficient similarity search:

- **ChromaDB**: Local, file-based, no setup
- **Pinecone**: Cloud-based, scalable

### 5. Similarity Search

Finds most relevant chunks:

- Compares query embedding with stored embeddings
- Returns top-k most similar chunks
- Uses cosine similarity metric

### 6. RAG Chain (LCEL)

LangChain Expression Language chain:

```python
rag_chain = (
    {
        "context": lambda x: format_docs(retrieved_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

Components:

- **Input**: Prepares context and question
- **Prompt**: Formats input for LLM
- **LLM**: Generates answer
- **Parser**: Extracts string output

### 7. RAG vs Standard LLM

**Standard LLM Query:**

```
User: "What is the remote work policy?"
LLM: "I don't have information about your specific remote work policy..."
```

**RAG Query:**

```
User: "What is the remote work policy?"
System:
1. Retrieves relevant company policy chunks
2. Provides chunks as context to LLM
LLM: "According to the policy, all employees are eligible for remote
      work arrangements. They must submit a remote work request form..."
```

## Switching Vector Databases

### Switch to ChromaDB:

```bash
# .env
VECTOR_DB=chromadb
CHROMA_DB_DIR=./chroma_db
COLLECTION_NAME=company_policies
```

### Switch to Pinecone:

```bash
# .env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-api-key
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Common Issues

### "OPENAI_API_KEY not found"

- Copy `.env.example` to `.env`
- Add your OpenAI API key
- Ensure `.env` is in the same directory as main.py

### "PINECONE_API_KEY not found" (when VECTOR_DB=pinecone)

- Get API key from [Pinecone](https://www.pinecone.io/)
- Add to `.env` file
- Or switch to ChromaDB: `VECTOR_DB=chromadb`

### "No documents loaded"

- Ensure `Documents/` folder exists
- Add sample documents (PDF, TXT files)
- Check file permissions

### "No results found"

- Vector database might be empty
- Check if documents were loaded and stored successfully
- Try different queries

## Learning Objectives

After running this demo, you should understand:

1. ✓ Complete RAG pipeline flow (ingestion + retrieval + generation)
2. ✓ How to load documents from multiple sources
3. ✓ How chunking affects retrieval quality
4. ✓ How embeddings enable semantic search
5. ✓ How vector databases store and retrieve embeddings
6. ✓ How LLMs use retrieved context to generate answers
7. ✓ How LCEL chains compose RAG workflows
8. ✓ How configuration enables vector database flexibility

## Next Steps

- **Customize chunking**: Experiment with different chunk sizes and overlaps
- **Add metadata filtering**: Filter retrieval by document type, date, etc.
- **Try different embeddings**: Compare OpenAI vs other embedding models
- **Experiment with prompts**: Modify the RAG prompt template
- **Add streaming**: Stream LLM responses in real-time
- **Build API**: Wrap RAG pipeline in FastAPI endpoint
- **Add memory**: Implement conversation history for multi-turn chat

## Related Demos

- **demo-07**: CRUD operations with vector databases
- **demo-08**: Loading documents from multiple sources
- **demo-09**: RAG ingestion pipeline (load, chunk, store)
- **demo-11**: Multi-query retrieval strategies (if available)
- **demo-12**: Advanced RAG techniques (if available)

## References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)

## License

Educational use only.
