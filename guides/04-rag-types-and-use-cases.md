# RAG Types and Use Cases

A comprehensive reference guide covering every major RAG (Retrieval-Augmented Generation) architecture pattern, when to use each, and practical implementation guidance.

---

## Table of Contents

1. [What Is RAG — Foundational Recap](#1-what-is-rag--foundational-recap)
2. [Naive RAG](#2-naive-rag)
3. [Advanced RAG](#3-advanced-rag)
4. [Modular RAG](#4-modular-rag)
5. [Agentic RAG](#5-agentic-rag)
6. [Self-RAG](#6-self-rag)
7. [Corrective RAG (CRAG)](#7-corrective-rag-crag)
8. [Graph RAG](#8-graph-rag)
9. [Hybrid RAG](#9-hybrid-rag)
10. [Multi-Modal RAG](#10-multi-modal-rag)
11. [Multi-Hop RAG](#11-multi-hop-rag)
12. [Structured Data RAG (Text-to-SQL)](#12-structured-data-rag-text-to-sql)
13. [Streaming RAG](#13-streaming-rag)
14. [RAG Type Selection Guide](#14-rag-type-selection-guide)
15. [Architecture Comparison Matrix](#15-architecture-comparison-matrix)

---

## 1. What Is RAG — Foundational Recap

Retrieval-Augmented Generation solves the fundamental problem that LLMs have **static, bounded knowledge**: their parameters capture only what appeared in training data, nothing more.

### The Core RAG Equation

```
Answer = LLM( Prompt + Retrieved_Context )
```

Instead of the LLM relying solely on memorised patterns, you inject **live, relevant, accurate context** into the prompt at inference time. The LLM's job shifts from "recall facts from memory" to "reason over the provided evidence."

### The Three Phases Common to All RAG Variants

```
Phase 1 — INDEXING (offline)
  Raw Documents
       ↓
  Load → Split → Embed → Store in Vector DB

Phase 2 — RETRIEVAL (online, per query)
  User Question
       ↓
  Embed Query → Search Vector DB → Retrieve top-k chunks

Phase 3 — GENERATION (online, per query)
  Retrieved Chunks + User Question
       ↓
  Prompt LLM → Generate grounded answer
```

Every RAG type either extends, replaces, or adds intelligence to one or more of these three phases.

---

## 2. Naive RAG

### What It Is

The simplest, most direct implementation of RAG. Documents are chunked, embedded, and stored. At query time, the top-k most similar chunks are retrieved via cosine similarity and injected verbatim into the LLM prompt. No pre-processing of the query, no post-processing of retrieved chunks.

### Architecture

```
[Documents] → chunk → embed → [Vector DB]

Query → embed → similarity search → top-k chunks
                                          ↓
                              [System Prompt + chunks + Query]
                                          ↓
                                        [LLM]
                                          ↓
                                       Answer
```

### Implementation (Python / LangChain)

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# INDEXING
loader = DirectoryLoader("./docs", glob="**/*.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# RETRIEVAL + GENERATION
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)

answer = qa_chain.invoke("What is the refund policy?")
```

### Strengths

| Strength          | Detail                                             |
| ----------------- | -------------------------------------------------- |
| Simple to build   | Minimal code, well-supported by all RAG frameworks |
| Fast to prototype | Can go from idea to working demo in hours          |
| Low cost          | No complex pre-processing or re-ranking            |
| Transparent       | Easy to debug — straightforward data flow          |

### Weaknesses

| Weakness                               | Impact                                                          |
| -------------------------------------- | --------------------------------------------------------------- |
| Chunking quality determines everything | Poor chunk boundaries break context                             |
| No query understanding                 | Ambiguous queries return irrelevant chunks                      |
| No iterative reasoning                 | Cannot answer questions requiring multiple lookups              |
| Fixed chunk retrieval                  | May miss context that spans multiple chunks                     |
| Keyword-embedding gap                  | Exact keyword matches can outscore semantically relevant chunks |

### Best Use Cases

- **Internal Q&A chatbots** — employees querying HR policies, IT documentation, onboarding guides
- **Simple document search** — legal contract lookup, product manual navigation
- **Proof-of-concept / prototyping** — validating that RAG is viable for a domain
- **Low-complexity knowledge bases** — FAQs, help centre articles, single-domain datasets

### When NOT to Use

- Multi-step reasoning required
- Documents have complex cross-references
- Queries require synthesising information from many documents simultaneously

---

## 3. Advanced RAG

### What It Is

Advanced RAG systematically improves on Naive RAG at every phase — pre-retrieval (query refinement), retrieval (better search), and post-retrieval (re-ranking, filtering). It treats each bottleneck individually.

### Architecture

```
Query
  ↓
[PRE-RETRIEVAL]
  Query expansion / rewriting / HyDE
  ↓
[RETRIEVAL]
  Hybrid search (semantic + BM25)
  Parent-child chunking
  Multiple vector stores
  ↓
[POST-RETRIEVAL]
  Re-ranking (cross-encoder)
  Redundancy filtering
  Context compression
  ↓
[LLM]
  ↓
Answer
```

### Pre-Retrieval Techniques

#### Query Rewriting

The user's natural language query may be ambiguous or poorly phrased for retrieval. Rewrite it to maximise retrieval quality.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

rewrite_prompt = ChatPromptTemplate.from_template("""
You are a search query optimiser. Rewrite the following user question
into a concise, keyword-rich search query optimised for document retrieval.
Return only the rewritten query, nothing else.

User question: {question}
Optimised query:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewriter = rewrite_prompt | llm

optimised_query = rewriter.invoke({"question": "what happens if i'm late on my payment?"})
# → "late payment policy consequences fees"
```

#### HyDE — Hypothetical Document Embeddings

Instead of embedding the question (which may not match the language of the answer), generate a **hypothetical answer document** and embed that. This narrows the semantic distance between query and stored chunks.

```python
hyde_prompt = ChatPromptTemplate.from_template("""
Write a short paragraph that directly answers the following question
as if it were extracted from an authoritative document.
Be factual and specific.

Question: {question}
Hypothetical answer:""")

hypothetical_doc = (hyde_prompt | llm).invoke({"question": "What is the maternity leave policy?"})
# → "Employees are entitled to 16 weeks of paid maternity leave..."

# Embed the hypothetical doc, not the question
query_embedding = embeddings.embed_query(hypothetical_doc.content)
results = vectorstore.similarity_search_by_vector(query_embedding, k=4)
```

#### Multi-Query Expansion

Generate multiple reformulations of the question and retrieve for each. Union the results.

```python
multi_query_prompt = ChatPromptTemplate.from_template("""
Generate 3 different search queries to retrieve documents relevant to this question.
Return each query on a new line.

Question: {question}
Queries:""")

queries_text = (multi_query_prompt | llm).invoke({"question": question})
queries = [q.strip() for q in queries_text.content.split("\n") if q.strip()]

all_results = []
for q in queries:
    results = vectorstore.similarity_search(q, k=3)
    all_results.extend(results)

# Deduplicate by document content
unique_results = list({doc.page_content: doc for doc in all_results}.values())
```

### Post-Retrieval Techniques

#### Re-Ranking with a Cross-Encoder

Embedding models (bi-encoders) are fast but approximate. A cross-encoder takes the query AND each candidate chunk together and scores their joint relevance. Slower but far more accurate.

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "What is the refund policy for digital products?"
candidates = vectorstore.similarity_search(query, k=20)  # retrieve more initially

# Score query-document pairs
pairs = [(query, doc.page_content) for doc in candidates]
scores = reranker.predict(pairs)

# Sort by score and keep top 4
ranked = sorted(zip(scores, candidates), reverse=True)
top_chunks = [doc for _, doc in ranked[:4]]
```

#### Context Compression

Strip irrelevant sentences from retrieved chunks before sending to the LLM (reduces token cost and noise).

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

compressor = LLMChainExtractor.from_llm(llm)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 6})
)

compressed_docs = compressed_retriever.invoke("What is the refund policy?")
```

### Best Use Cases

- **Customer support automation** — complex support queries requiring precise policy answers
- **Legal research assistants** — where retrieval accuracy directly affects advice quality
- **Medical knowledge bases** — clinical guideline lookup requiring high-precision retrieval
- **Enterprise search** — large, heterogeneous document corpora where naive retrieval fails
- **Compliance Q&A** — regulatory lookup where a wrong answer has risk implications

---

## 4. Modular RAG

### What It Is

Modular RAG treats each component of the RAG pipeline as an **independently swappable module**. Rather than a fixed pipeline, you compose the system from interchangeable parts — different retrievers, memory stores, re-rankers, and generators — and route queries to different pipeline configurations based on type.

### Architecture

```
Query
  ↓
[ROUTER] ─────────────────────────────────────
   │                    │                    │
   ▼                    ▼                    ▼
[Pipeline A]       [Pipeline B]        [Pipeline C]
Simple Q&A         Complex analysis    Real-time data
  │                    │                    │
[Retriever A]    [Retriever B + C]    [API Tool]
[Prompt A]       [Re-ranker]          [Prompt C]
[LLM A]          [CoT Prompt]         [LLM C]
   │                    │                    │
   └────────────────────┴────────────────────┘
                        ↓
                     Answer
```

### Key Modules

| Module Type   | Options                                                      | Selection Criteria              |
| ------------- | ------------------------------------------------------------ | ------------------------------- |
| **Retriever** | Dense (embedding), Sparse (BM25), Hybrid, Graph, SQL         | Query type, data format         |
| **Memory**    | Short-term (conversation), Long-term (episodic vector store) | Session vs persistent           |
| **Re-ranker** | Cross-encoder, LLM-based, None                               | Accuracy requirement vs latency |
| **Generator** | GPT-4o, Claude, Llama, domain fine-tuned                     | Cost, accuracy, domain          |
| **Prompt**    | Zero-shot, few-shot, CoT, structured output                  | Task complexity                 |

### Router Implementation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from enum import Enum

class QueryType(str, Enum):
    FACTUAL = "factual"          # Direct document lookup
    ANALYTICAL = "analytical"   # Multi-step reasoning over docs
    CONVERSATIONAL = "conversational"  # Chitchat / clarification

router_prompt = ChatPromptTemplate.from_template("""
Classify the following query into exactly one category:
- factual: direct lookup question answerable from documents
- analytical: requires combining and reasoning over multiple documents
- conversational: chitchat, clarification, or off-topic

Query: {query}
Category (one word only):""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
router_chain = router_prompt | llm

def route_query(query: str) -> QueryType:
    result = router_chain.invoke({"query": query})
    return QueryType(result.content.strip().lower())

def answer_query(query: str) -> str:
    query_type = route_query(query)

    if query_type == QueryType.FACTUAL:
        return factual_pipeline(query)
    elif query_type == QueryType.ANALYTICAL:
        return analytical_pipeline(query)
    else:
        return conversational_pipeline(query)
```

### Best Use Cases

- **Multi-domain enterprise assistants** — handles HR, IT, Finance, and Legal in one system with specialised pipelines per domain
- **Mixed-data products** — some queries need vector search, others need SQL, others need web search
- **Tiered-cost systems** — route simple queries to cheap models, complex ones to expensive models
- **Regulated industries** — different compliance rules apply depending on what data is retrieved

---

## 5. Agentic RAG

### What It Is

In Agentic RAG, the LLM acts as an **autonomous agent** that decides when to retrieve, what to retrieve, how many times, and from which sources. Instead of a fixed retrieve-once pipeline, the agent loops: reason → act (retrieve) → observe → reason → act...

### Architecture

```
User Query
    ↓
[AGENT / ReAct Loop]
    ↓
  Thought: "I need to find the refund policy first"
    ↓
  Action: retrieval_tool(query="refund policy digital goods")
    ↓
  Observation: [retrieved chunks]
    ↓
  Thought: "I need more info about the 30-day window mentioned"
    ↓
  Action: retrieval_tool(query="30 day return window exceptions")
    ↓
  Observation: [additional chunks]
    ↓
  Thought: "I now have enough context to answer"
    ↓
  Final Answer: "..."
```

### Tools an Agentic RAG System Can Use

| Tool              | Function                          |
| ----------------- | --------------------------------- |
| `vector_search`   | Semantic lookup in document store |
| `web_search`      | Real-time web retrieval           |
| `sql_query`       | Query structured databases        |
| `code_executor`   | Run calculations or data analysis |
| `api_caller`      | External REST/GraphQL APIs        |
| `calendar_lookup` | Scheduling and date resolution    |
| `email_sender`    | Compose and send emails           |

### Implementation with LangChain

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define tools
def search_documents(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

def search_web(query: str) -> str:
    # Integration with Tavily, SerpAPI, etc.
    ...

tools = [
    Tool(
        name="document_search",
        func=search_documents,
        description="Search internal company documents for policies, procedures, and knowledge base articles."
    ),
    Tool(
        name="web_search",
        func=search_web,
        description="Search the web for current events, prices, and information not in internal documents."
    ),
]

# ReAct agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Compare our refund policy with industry standard practices and identify any gaps."
})
```

### Best Use Cases

- **Research assistants** — autonomously gathers, cross-references, and synthesises from multiple sources
- **IT helpdesk automation** — diagnoses issues by querying kb articles, runbooks, and live system logs
- **Financial analysis bots** — fetches internal reports + live market data to produce investment summaries
- **Complex customer support** — resolves multi-step issues (check account → find policy → apply exception → log ticket)
- **Competitive intelligence** — retrieves internal positioning docs + web competitor data + analyst reports

### Considerations

| Concern     | Mitigation                                                |
| ----------- | --------------------------------------------------------- |
| Cost        | Agent loops multiply LLM calls — set `max_iterations`     |
| Latency     | Multiple retrieval rounds add latency; use async tools    |
| Reliability | Agents can get stuck in loops; add termination conditions |
| Security    | Tool access must be tightly scoped; apply least privilege |

---

## 6. Self-RAG

### What It Is

Self-RAG introduces **self-reflection and critique** into the RAG loop. After retrieval and generation, the model scores its own output: Was retrieval necessary? Were the retrieved chunks relevant? Is the generated answer grounded in the evidence? If not, it retrieves again or regenerates.

### Architecture

```
Query
  ↓
[Retrieve?] ← reflection token: [Retrieve] / [No Retrieve]
  ↓ (if retrieve)
[Relevance Check] ← reflection token: [Relevant] / [Irrelevant] per chunk
  ↓ (keep relevant chunks only)
[Generate Answer]
  ↓
[Groundedness Check] ← is each claim in the answer supported by retrieved chunks?
  reflection token: [Fully supported] / [Partially supported] / [No support]
  ↓
[Utility Check] ← is this answer actually useful for the query?
  reflection token: [Utility: 1–5]
  ↓
If score too low → loop back to retrieval with refined query
  ↓ (if score acceptable)
Final Answer
```

### Reflection Tokens

Self-RAG trains the model to generate special **reflection tokens** inline with its output:

| Token                        | Meaning                                     |
| ---------------------------- | ------------------------------------------- |
| `[Retrieve]`                 | Retrieval is needed for this query          |
| `[No Retrieve]`              | The model can answer from its own knowledge |
| `[Relevant]`                 | This retrieved chunk is relevant            |
| `[Irrelevant]`               | This chunk should be discarded              |
| `[Fully supported]`          | The claim is grounded in retrieved evidence |
| `[Partially supported]`      | Partially grounded — treat with caution     |
| `[No support / Contradicts]` | Hallucination alert — regenerate            |
| `[Utility: 1–5]`             | Self-assessed usefulness of the response    |

### Simplified Self-Critique Implementation (without fine-tuning)

You can approximate Self-RAG behaviour using prompting:

```python
critique_prompt = """You are a factual accuracy reviewer.

Retrieved context:
{context}

Generated answer:
{answer}

Evaluate the answer on two dimensions:
1. GROUNDEDNESS: Is every factual claim in the answer directly supported by the retrieved context?
   Score: fully_supported / partially_supported / not_supported
2. COMPLETENESS: Does the answer fully address the question?
   Score: complete / incomplete

Respond as JSON:
{{"groundedness": "...", "completeness": "...", "issues": "..."}}"""

def self_evaluate(context: str, answer: str, question: str) -> dict:
    result = llm.invoke(critique_prompt.format(context=context, answer=answer))
    return json.loads(result.content)

def self_rag_answer(question: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        chunks = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([c.page_content for c in chunks])
        answer = generate_answer(question, context)

        critique = self_evaluate(context, answer, question)

        if critique["groundedness"] == "fully_supported":
            return answer

        # Refine query based on identified issues
        question = refine_query(question, critique["issues"])

    return answer  # Return best attempt after max retries
```

### Best Use Cases

- **High-stakes factual Q&A** — medical, legal, financial domains where hallucination has real consequences
- **Compliance verification** — generating audit reports that must be fully grounded in source documents
- **Automated content quality assurance** — self-checking generated summaries before publishing
- **Research paper assistants** — ensuring every claim cites a retrieved source

---

## 7. Corrective RAG (CRAG)

### What It Is

CRAG adds a **retrieval quality evaluator** after the initial retrieval step. If retrieved documents score below a confidence threshold, CRAG automatically falls back to alternative sources (typically a web search) before generating. It has three modes: **Correct** (docs are good), **Incorrect** (docs are bad → web search), **Ambiguous** (mix retrieved + web).

### Architecture

```
Query
  ↓
[Initial Retrieval] from Vector DB
  ↓
[Retrieval Evaluator] — LLM scores: Correct / Ambiguous / Incorrect
  │
  ├── CORRECT ──────────────────→ Generate from retrieved docs
  │
  ├── AMBIGUOUS ─────────────── → Combine retrieved docs + web search → Generate
  │
  └── INCORRECT ──────────────→ Web search only → Generate
```

### Retrieval Evaluator

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

evaluator_prompt = ChatPromptTemplate.from_template("""
Given the following question and retrieved document chunks,
assess whether the retrieved documents are relevant and sufficient
to answer the question accurately.

Question: {question}

Retrieved documents:
{documents}

Respond with exactly one word:
- "correct" if the documents are clearly relevant and sufficient
- "incorrect" if the documents are clearly irrelevant or insufficient
- "ambiguous" if the documents are partially relevant

Assessment:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator = evaluator_prompt | llm

def evaluate_retrieval(question: str, docs: list) -> str:
    doc_text = "\n\n---\n\n".join([d.page_content for d in docs])
    result = evaluator.invoke({"question": question, "documents": doc_text})
    return result.content.strip().lower()

def crag_answer(question: str) -> str:
    # Initial retrieval
    docs = vectorstore.similarity_search(question, k=4)

    # Evaluate quality
    quality = evaluate_retrieval(question, docs)

    if quality == "correct":
        context = format_docs(docs)
    elif quality == "incorrect":
        # Fall back to web search
        web_results = web_search_tool(question)
        context = web_results
    else:  # ambiguous
        web_results = web_search_tool(question)
        context = format_docs(docs) + "\n\nAdditional web sources:\n" + web_results

    return generate_answer(question, context)
```

### Best Use Cases

- **Time-sensitive knowledge bases** — internal docs may be outdated; web search fills gaps
- **General-purpose assistants** — need to gracefully handle out-of-scope questions
- **Hybrid internal/external knowledge** — answer from internal docs when possible, web otherwise
- **Support bots with product + market knowledge** — product docs internally, pricing/competitors from web

---

## 8. Graph RAG

### What It Is

Graph RAG structures the knowledge base as a **knowledge graph** (nodes = entities, edges = relationships) rather than (or in addition to) a flat vector store. This enables retrieval that follows semantic relationships — essential for questions requiring multi-hop reasoning across connected entities.

### Architecture

```
INDEXING:
Documents → Named Entity Recognition (NER)
          → Relation Extraction
          → Build Knowledge Graph (e.g., Neo4j, NetworkX)
          → Also embed document chunks for hybrid retrieval

RETRIEVAL:
Query → Extract entities
     → Graph traversal (find related entities, relationships)
     → Hybrid: graph nodes + embedding search
     → Assemble subgraph as context
          ↓
        [LLM]
          ↓
        Answer
```

### Example Knowledge Graph Structure

```
                    ┌──────────────┐
                    │   Aspirin    │
                    │  (Drug)      │
                    └──────┬───────┘
                           │ treats
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         [Headache]   [Heart Attack] [Fever]
              │                         │
        contraindicated              interacts_with
              │                         │
         [Warfarin]               [Ibuprofen]
```

Flat vector search: "Does Aspirin interact with Ibuprofen?" — might retrieve isolated chunks without connecting the relationship.

Graph RAG: Traverses `Aspirin → interacts_with → Ibuprofen` directly.

### Implementation with LangChain + Neo4j

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# The chain translates natural language → Cypher query → executes → generates answer
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
)

result = chain.invoke({
    "query": "Which drugs does Aspirin interact with and what are the risks?"
})
```

### Hybrid Graph + Vector RAG

```python
def hybrid_graph_rag(query: str) -> str:
    # 1. Graph traversal for entity relationships
    graph_context = graph_retriever(query)

    # 2. Vector search for relevant text chunks
    vector_context = vectorstore.similarity_search(query, k=3)

    # 3. Combine both contexts
    combined_context = f"""
    Relationship context (from knowledge graph):
    {graph_context}

    Document context (from vector search):
    {format_docs(vector_context)}
    """

    return generate_answer(query, combined_context)
```

### Best Use Cases

- **Biomedical/pharmaceutical knowledge** — drug-disease-gene relationships, clinical trial networks
- **Legal knowledge bases** — legislation → regulation → case law chains
- **Cyber security** — CVE → affected systems → patches → mitigation chains
- **Financial compliance** — entity → ownership → jurisdictions → regulations
- **Supply chain intelligence** — supplier → component → product → regulation chains
- **Recommendation engines** — user → purchases → related products → restock suggestions

---

## 9. Hybrid RAG

### What It Is

Hybrid RAG combines **dense retrieval** (semantic vector search) with **sparse retrieval** (keyword-based BM25), merging results using a score fusion algorithm. This captures both semantic similarity and exact keyword relevance, covering gaps that each method has alone.

### Why Each Method Has Gaps

| Scenario                             | Dense (Embedding)                            | Sparse (BM25)                       | Hybrid            |
| ------------------------------------ | -------------------------------------------- | ----------------------------------- | ----------------- |
| "What is photosynthesis?"            | ✅ Finds conceptually related content        | ✅ Matches "photosynthesis" exactly | ✅ Both           |
| "Error code ERR_SSL_PROTOCOL_ERROR"  | ❌ Embedding may not encode error codes well | ✅ Exact keyword match              | ✅ BM25 saves it  |
| "Something about the sun and plants" | ✅ Semantic match to photosynthesis          | ❌ No keyword overlap               | ✅ Dense saves it |
| Medical acronyms: "CABG procedure"   | ❌ Ambiguous embedding                       | ✅ Exact match                      | ✅ BM25 saves it  |

### Reciprocal Rank Fusion (RRF)

The standard algorithm for merging ranked lists from different retrievers:

$$RRF\_score(d) = \sum_{r \in \text{rankers}} \frac{1}{k + rank_r(d)}$$

Where $k = 60$ is a constant that controls the penalty for lower-ranked documents.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Dense retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Sparse retriever (BM25)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# Ensemble with RRF (weights control relative contribution)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Favour semantic slightly
)

results = hybrid_retriever.invoke("SSL handshake failure ERR_SSL_PROTOCOL_ERROR")
```

### Best Use Cases

- **Technical documentation search** — code error messages, API names, version numbers (BM25 wins on exact terms)
- **Medical records** — ICD codes, drug names, procedure codes + semantic clinical context
- **E-commerce product search** — SKU/model numbers (BM25) + semantic product descriptions (dense)
- **Legal search** — statute numbers, case citations (BM25) + legal concepts (dense)
- **General enterprise search** — the default choice for production systems with diverse query types

---

## 10. Multi-Modal RAG

### What It Is

Multi-modal RAG extends the retrieval pipeline to handle diverse data types beyond plain text: **images, tables, charts, audio, video, diagrams, and presentations**. The system can retrieve and reason over all modalities simultaneously.

### Architecture

```
INDEXING:
PDF / PPTX / HTML
  ├── Text chunks → text embeddings → vector store
  ├── Tables → structured extraction → text representation → embedded
  └── Images/Charts → image embeddings (CLIP) or LLM captioning → embedded

RETRIEVAL:
Text query → embed → search across all modality stores
                └── top-k text chunks
                └── top-k table representations
                └── top-k image matches

GENERATION:
Multi-modal LLM (GPT-4V, Claude 3.x) receives text + images
  ↓
Answer that reasons over all modalities
```

### Handling Tables

```python
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI

# Extract tables from PDF as structured elements
elements = partition_pdf(
    filename="annual_report.pdf",
    strategy="hi_res",
    infer_table_structure=True,
    extract_images_in_pdf=True,
)

tables = [e for e in elements if e.category == "Table"]
images = [e for e in elements if e.category == "Image"]
texts = [e for e in elements if e.category in ("NarrativeText", "Title")]

# Convert tables to text for embedding (preserve structure)
def table_to_text(table_element) -> str:
    return f"TABLE:\n{table_element.text}\nHTML:\n{table_element.metadata.text_as_html}"
```

### Image Understanding with Vision Models

```python
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

def describe_image(image_path: str, context_question: str) -> str:
    """Use GPT-4V to generate a description of an image in the context of a question."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        },
        {
            "type": "text",
            "text": f"Describe this image in detail, focusing on information relevant to: {context_question}"
        }
    ])

    return llm.invoke([message]).content
```

### Best Use Cases

- **Technical manuals and engineering docs** — diagrams, circuit schematics, exploded-view drawings
- **Financial report analysis** — earnings tables, trend charts, balance sheets
- **Medical imaging Q&A** — radiology reports with accompanying scan images
- **Slide deck search** — query a library of PowerPoint presentations
- **Manufacturing defect detection** — query quality control images alongside spec documents
- **Scientific literature** — papers containing figures, graphs, and tables essential to understanding results

---

## 11. Multi-Hop RAG

### What It Is

Multi-hop RAG handles questions that require **chaining multiple retrieval steps** to reach an answer — each retrieval result informs the next query. Simple vector search retrieves once; multi-hop retrieves iteratively, with each step guided by what was retrieved before.

### The Problem It Solves

```
Question: "What programming languages are used in the project
           that won the 2024 innovation award mentioned in the company news?"

Single retrieval: Searches for "2024 innovation award programming languages" → likely misses
Multi-hop:
  Hop 1: Search "2024 innovation award" → finds "Project Atlas won the award"
  Hop 2: Search "Project Atlas programming languages" → finds "Python and Go"
  Answer: "Python and Go"
```

### Implementation

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

hop_prompt = ChatPromptTemplate.from_template("""
You are solving a complex question that may require multiple information lookups.

Original question: {original_question}
Information gathered so far: {gathered_info}

Based on what you know so far, what specific sub-question should be searched next
to make progress toward answering the original question?
If you have enough information to answer the original question, say "ANSWER: [your answer]".

Next sub-question or ANSWER:""")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
hop_chain = hop_prompt | llm

def multi_hop_rag(question: str, max_hops: int = 4) -> str:
    gathered_info = ""

    for hop in range(max_hops):
        # Determine next search query
        response = hop_chain.invoke({
            "original_question": question,
            "gathered_info": gathered_info or "Nothing gathered yet."
        })

        response_text = response.content.strip()

        # Check if we have enough to answer
        if response_text.startswith("ANSWER:"):
            return response_text.replace("ANSWER:", "").strip()

        # Execute retrieval for the sub-question
        sub_question = response_text
        docs = vectorstore.similarity_search(sub_question, k=3)
        retrieved_text = "\n".join([d.page_content for d in docs])

        # Accumulate gathered information
        gathered_info += f"\n\nHop {hop + 1} — Query: {sub_question}\n{retrieved_text}"

    # Final answer generation after max hops
    return generate_answer(question, gathered_info)
```

### Best Use Cases

- **Organisational knowledge graphs** — "Who is the manager of the team that owns Product X, and what is their email?"
- **Supply chain tracing** — "Which supplier provides the component used in the product recalled last month?"
- **Regulatory compliance chains** — "What is the fine for violating regulation X, which is referenced in policy Y?"
- **Academic research** — "What experiments validated the theory cited by the paper that introduced concept Z?"
- **IT root cause analysis** — "What service failure triggered the alert that caused the customer-facing outage?"

---

## 12. Structured Data RAG (Text-to-SQL)

### What It Is

Structured Data RAG enables natural language querying of **relational databases, data warehouses, and spreadsheets**. The LLM translates a natural language question into SQL (or another query language), executes it, and summarises the results.

### Architecture

```
User Question
    ↓
[Schema Context] — table names, columns, sample data, relationships
    ↓
[LLM → SQL Generation]
    ↓
[SQL Executor] → Database
    ↓
[Query Results]
    ↓
[LLM → Natural Language Summary]
    ↓
Answer
```

### Implementation

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

# Connect to database
db = SQLDatabase.from_uri(
    "postgresql://user:password@localhost/company_db",
    include_tables=["orders", "customers", "products"],  # limit scope
    sample_rows_in_table_info=3,  # include sample rows for context
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Create SQL agent
agent = create_sql_agent(
    llm=llm,
    db=db,
    verbose=True,
    agent_type="openai-tools",
)

# Natural language queries
result = agent.invoke({
    "input": "Which product category had the highest revenue last quarter, "
             "and how does it compare to the same quarter last year?"
})
```

### Combining Text-to-SQL with Vector RAG

For systems where knowledge lives in both databases and documents:

```python
def hybrid_structured_unstructured(question: str) -> str:
    """
    Route to SQL for quantitative/relational questions,
    vector store for policy/procedure questions,
    or combine both.
    """
    question_lower = question.lower()

    needs_sql = any(kw in question_lower for kw in [
        "how many", "total", "average", "count", "revenue",
        "sales", "last quarter", "year over year", "rank"
    ])
    needs_docs = any(kw in question_lower for kw in [
        "policy", "procedure", "how to", "what is", "explain", "guideline"
    ])

    if needs_sql and needs_docs:
        sql_result = sql_agent.invoke({"input": question})
        doc_result = vector_retriever.invoke(question)
        return synthesise([sql_result, doc_result], question)
    elif needs_sql:
        return sql_agent.invoke({"input": question})
    else:
        return vector_qa_chain.invoke(question)
```

### Security Considerations for Text-to-SQL

```
⚠️ CRITICAL: SQL injection via prompt injection is a real risk.

Mitigations:
- Use read-only database credentials (SELECT only, no INSERT/UPDATE/DELETE)
- Allowlist tables the agent can access (never expose ALL tables)
- Validate generated SQL before execution (regex or AST parser)
- Never concatenate raw LLM output directly into SQL strings
- Log all generated queries for audit
- Use parameterised query wrappers where possible
```

### Best Use Cases

- **Business intelligence chatbots** — "Show me sales by region for Q3" without a BI analyst
- **Customer 360 dashboards** — "What is the lifetime value of customers who purchased in Q1 2024?"
- **HR self-service** — "How many employees in the London office took more than 10 sick days last year?"
- **Inventory management** — "Which SKUs are below reorder threshold right now?"
- **Financial reporting assistants** — "Compare OPEX this quarter vs last quarter by department"

---

## 13. Streaming RAG

### What It Is

Streaming RAG delivers **token-by-token output** to the client as the LLM generates, rather than waiting for the complete response. For RAG systems where retrieval adds latency, streaming dramatically improves perceived responsiveness.

### Architecture

```
User Query
    ↓
[Retrieval] ← runs first, blocks briefly (100–500ms typical)
    ↓
[LLM Generation] → token 1 → token 2 → token 3 → ...
                      ↓         ↓         ↓
                  [SSE/WebSocket stream to client]
```

### FastAPI Streaming RAG Endpoint

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import asyncio

app = FastAPI()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}
Answer:""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/stream")
async def stream_rag(query: str):
    async def generate():
        async for chunk in rag_chain.astream(query):
            # Server-Sent Events format
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

### Client-Side Consumption (JavaScript)

```javascript
const eventSource = new EventSource("/stream?query=What+is+the+refund+policy");
let fullResponse = "";

eventSource.onmessage = (event) => {
  if (event.data === "[DONE]") {
    eventSource.close();
    return;
  }
  fullResponse += event.data;
  document.getElementById("response").textContent = fullResponse;
};

eventSource.onerror = () => eventSource.close();
```

### Best Use Cases

- **Any user-facing RAG chatbot** — streaming is the expected UX for modern AI chat
- **Long document summarisation** — users see progress instead of waiting 15–30 seconds
- **Real-time report generation** — finance, legal, or HR reports generated section by section
- **Code generation assistants** — code streams in as it is generated, user can abort early

---

## 14. RAG Type Selection Guide

Use this decision tree to determine the right RAG architecture for your use case.

```
START: What is the primary challenge you're solving?
│
├── "My documents are too old / I need current info"
│       → Corrective RAG (CRAG) with web search fallback
│
├── "Users ask questions requiring multiple lookups"
│   ├── Lookups follow entity relationships → Graph RAG
│   └── Lookups are sequential reasoning steps → Multi-Hop RAG
│
├── "My data includes images, tables, charts"
│       → Multi-Modal RAG
│
├── "My knowledge is in a relational database"
│       → Structured RAG (Text-to-SQL)
│       → OR Hybrid if docs + database both needed
│
├── "I need high accuracy / this is high-stakes"
│   ├── With self-checking → Self-RAG
│   └── With external verification → Corrective RAG
│
├── "I'm building a complex task-performing assistant"
│       → Agentic RAG
│
├── "I have multiple document types / query types"
│   ├── With routing logic → Modular RAG
│   └── For better retrieval accuracy → Hybrid RAG
│
├── "Keyword search misses relevant docs"
│       → Hybrid RAG (Dense + BM25)
│
├── "Users expect instant responses"
│       → Streaming RAG (add to any pattern above)
│
└── "Starting fresh / proof of concept"
        → Naive RAG → then advance as needed
```

### Complexity vs Reward Matrix

```
HIGH REWARD
     │
     │   Graph RAG        Agentic RAG
     │   (relationships)  (autonomous)
     │
     │   Self-RAG         Multi-Hop RAG
     │   (accuracy)       (complex Q)
     │
     │   Hybrid RAG       Advanced RAG
     │   (better recall)  (better precision)
     │
     │                    Naive RAG
LOW  │                    (simple Q&A)
     └─────────────────────────────────
     LOW complexity              HIGH complexity
```

---

## 15. Architecture Comparison Matrix

| RAG Type        | Retrieval                | Reasoning          | Modalities           | Accuracy     | Latency       | Complexity  | Best For                    |
| --------------- | ------------------------ | ------------------ | -------------------- | ------------ | ------------- | ----------- | --------------------------- |
| **Naive**       | Single vector search     | None               | Text                 | Medium       | Low           | Low         | Prototypes, simple Q&A      |
| **Advanced**    | Hybrid + rerank          | Query expansion    | Text                 | High         | Medium        | Medium      | Production Q&A              |
| **Modular**     | Configurable             | Router             | Multi                | High         | Variable      | Medium-High | Multi-domain products       |
| **Agentic**     | Multi-source, repeated   | ReAct loop         | Multi                | High         | High          | High        | Complex task completion     |
| **Self-RAG**    | Iterative                | Self-critique      | Text                 | Very High    | High          | High        | High-stakes factual Q&A     |
| **CRAG**        | Primary + fallback       | Evaluator          | Text                 | High         | Medium        | Medium      | Mixed internal/external     |
| **Graph**       | Graph traversal + vector | Relationship       | Text + graph         | Very High    | Medium        | High        | Entity-relationship queries |
| **Hybrid**      | Dense + sparse (BM25)    | None               | Text                 | High         | Low-Medium    | Low         | General production search   |
| **Multi-Modal** | Cross-modal              | Vision LLM         | Text + image + table | High         | Medium-High   | High        | Docs with images/tables     |
| **Multi-Hop**   | Sequential retrieval     | Chain-of-retrieval | Text                 | High         | High          | High        | Multi-step reasoning        |
| **Text-to-SQL** | SQL execution            | NL→SQL             | Structured data      | High         | Low           | Medium      | Database Q&A                |
| **Streaming**   | Any (modifier)           | Any (modifier)     | Any                  | Same as base | Perceived low | Low add-on  | All user-facing RAG         |

---

## Summary

Every RAG variant is an evolution designed to address a specific failure mode of the step before it:

```
Naive RAG
  └── Bad retrieval? → Advanced RAG (query rewriting, re-ranking)
        └── Mixed data types? → Modular RAG (pipeline routing)
              └── Need autonomous behaviour? → Agentic RAG
                    └── Hallucination still an issue? → Self-RAG / CRAG
                          └── Relationships matter? → Graph RAG
                                └── Keywords missed? → Hybrid RAG
                                      └── Images/tables? → Multi-Modal RAG
                                            └── Multi-step Q? → Multi-Hop RAG
                                                  └── Database? → Text-to-SQL
                                                        └── UX matters? → Streaming
```

**Production recommendation:** Start with **Hybrid RAG** (dense + BM25, with re-ranking) as your baseline over Naive RAG. It handles the widest class of real queries without the complexity overhead of agentic systems. Layer on additional patterns only when that baseline hits a specific, measured limitation.

---

_Guide maintained as part of the **Integrating Generative AI** training programme._
