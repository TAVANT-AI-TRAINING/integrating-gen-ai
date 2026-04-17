RoboDesk — Complete RAG Pipeline: Ingestion, Retrieval & Answer Generation

Project Context
You're building RoboDesk, an AI-powered Customer Service Desk assistant for Axiom Robotics. It helps customers get instant, accurate answers about robot models, product specifications, installation, troubleshooting, warranties, and support policies.

A fully functional RAG system has two equally important halves:

- Ingestion Pipeline: Load raw documents → chunk them → generate embeddings → index into a vector store.
- Retrieval Pipeline: Accept a user query → retrieve relevant chunks → assemble context → generate a grounded LLM answer with citations.

You will build both halves end-to-end and expose them as a FastAPI service. The dataset under capstone/dataset/ (product catalog, FAQ, troubleshooting guide, warranty policy) provides the knowledge base.

Problem Statement
Build a production-ready RAG system that ingests the RoboDesk knowledge base from plain-text documents and answers customer support queries by retrieving relevant context and generating grounded responses. Without a working ingestion pipeline there is nothing to retrieve; without a working retrieval pipeline the indexed data is useless. Both must be correct for the system to function.

You will complete four tasks:

Task 1 — Document Ingestion Pipeline

Goal
Build the ingestion pipeline that loads, chunks, embeds, and indexes the knowledge base into a vector store.

Requirements

Document Loading

- Load all .txt files from the capstone/dataset/ directory tree (products/, faq/, support/, policies/)
- Use LangChain document loaders (e.g., DirectoryLoader with TextLoader)
- Preserve the source file path as document metadata

Text Chunking

- Split documents using RecursiveCharacterTextSplitter
- Chunk size: 800 characters, overlap: 150 characters
- Retain source metadata on every chunk
- Log the total number of chunks produced per source file

Embedding Generation

- Use HuggingFace sentence-transformers (e.g., all-MiniLM-L6-v2) or OpenAI text-embedding-ada-002
- Generate embeddings for all chunks
- Log embedding dimensions and sample similarity scores to validate quality

Vector Store Indexing

- Store embeddings in Chroma (local persistent collection) or Pinecone (cloud index)
- Collection/index name: roboDesk-kb
- Include metadata fields: source, chunk_index, document_type
- Implement upsert logic so re-running ingestion does not create duplicate entries
- Verify the indexed count matches the number of chunks produced

Ingestion Runner

- Create an ingest.py script that runs the full pipeline end-to-end
- Print a summary report: files loaded, chunks created, chunks indexed, time taken

Task 2 — Semantic Retrieval and Vector Search

Goal
Implement a retrieval service that connects to the indexed vector store and fetches the most relevant chunks for a given query.

Requirements

Retrieval Service

- Connect to the same Chroma or Pinecone collection populated in Task 1
- Accept a natural-language query string and an optional top_k parameter (default: 5)
- Perform similarity search using cosine distance
- Return a list of results, each containing: chunk text, source file name, chunk_index, and similarity score
- Handle empty results gracefully with a clear message

Retrieval Validation

- Write a simple test script (test_retrieval.py) with at least five representative customer queries
- For each query print the top-3 retrieved chunks and their source files
- Confirm that results are topically relevant (e.g., a warranty question retrieves from warranty_and_support_policy.txt)

Task 3 — RAG Chain and LLM Integration

Goal
Build a RAG chain that assembles retrieved context into a prompt and uses an LLM to generate a grounded, cited answer.

Requirements

Prompt Templates

- Create a system prompt that instructs the LLM to answer only from the provided context
- Include explicit instructions to say "I don't know" when context is insufficient (hallucination prevention)
- Format retrieved chunks into a numbered context block within the prompt
- Include a citations instruction so the LLM lists the source file names it used

LLM Service

- Initialize Azure OpenAI (gpt-4o-mini or similar) or a local OpenAI-compatible model
- Configure temperature ≤ 0.2 for factual consistency and set an appropriate max_tokens limit
- Handle LLM API errors and timeouts gracefully

RAG Chain (LCEL)

- Build the chain using LangChain Expression Language (LCEL):
  user_query → retriever → document_formatter → prompt_template → LLM → output_parser
- The document_formatter step must combine chunk texts and inject source metadata
- The output_parser must extract the answer text and the cited sources as separate fields
- Ensure every step has error handling; a failure in retrieval must not crash the LLM step

Task 4 — REST API: Ingestion & Query Endpoints

Goal
Expose both the ingestion pipeline and the query pipeline as a FastAPI application so the system can be used programmatically.

Requirements

Ingestion Endpoint
POST /ingest - Triggers the full ingestion pipeline (Task 1) on demand - Accepts an optional request body with a dataset_path override - Returns: files_loaded, chunks_created, chunks_indexed, duration_seconds

Query Endpoints
POST /query - Accepts: { "question": "...", "top_k": 5 } - Runs the full RAG chain (retrieval + LLM generation) - Returns: question, answer, sources (list of {source_file, chunk_index}), model_used, timestamp

POST /retrieve - Accepts: { "query": "...", "top_k": 5 } - Runs retrieval only (no LLM), returns raw chunks with scores - Useful for debugging relevance without LLM cost

GET /health - Returns service status, vector store connection status, and indexed document count

Response Format (POST /query)
{
"question": "What is the payload capacity of the AX-200?",
"answer": "The AX-200 cobot has a payload capacity of 5 kg...",
"sources": [
{ "source_file": "products/robot_catalog.txt", "chunk_index": 3 }
],
"model_used": "gpt-4o-mini",
"timestamp": "2026-04-17T10:30:00Z"
}

Application Structure
Organize the project with the following layout:
capstone/
ingest.py ← standalone ingestion runner (Task 1)
test_retrieval.py ← retrieval validation script (Task 2)
app/
main.py ← FastAPI app entry point
ingestion/
loader.py ← document loading & chunking
embedder.py ← embedding model wrapper
indexer.py ← vector store upsert logic
retrieval/
retriever.py ← vector similarity search
rag/
chain.py ← LCEL RAG chain
prompt.py ← prompt templates
api/
routes.py ← all FastAPI route handlers
dataset/ ← knowledge base documents

Evaluation Criteria
Ensure you evaluate your solution against the below criteria:

Ingestion Pipeline

- All four document categories (products, faq, support, policies) are loaded and chunked correctly
- Chunk metadata (source file, chunk_index, document_type) is preserved through to the vector store
- Re-running ingest.py does not create duplicate entries in the vector store
- Ingestion summary report is printed with correct counts and timing

Retrieval

- Retrieval service connects to Chroma or Pinecone and performs cosine similarity search
- test_retrieval.py demonstrates topically correct results for at least 5 diverse queries
- Empty-result edge cases are handled gracefully

RAG Chain

- LCEL chain is correctly composed: retriever → formatter → prompt → LLM → parser
- Prompt template prevents hallucination and instructs the LLM to cite sources
- LLM is configured with low temperature for factual accuracy
- Chain returns both the answer text and a structured list of cited sources

REST API

- POST /ingest triggers the full pipeline and returns a summary
- POST /query returns a well-formed response with answer and source citations
- POST /retrieve returns raw chunks without invoking the LLM
- GET /health reflects actual vector store connection and document count

End-to-End Flow

- Running ingest.py followed by querying POST /query produces correct, grounded answers
- Answers for out-of-scope questions include an "I don't know" or "insufficient context" response
- The complete pipeline — ingestion → indexing → retrieval → generation — works without manual intervention
