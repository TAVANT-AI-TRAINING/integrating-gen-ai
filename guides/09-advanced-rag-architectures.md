# Module-5: Advanced RAG Architectures

A comprehensive guide to advanced RAG design patterns — covering the architectural evolution from Naive to Advanced RAG, Hybrid RAG, Graph RAG, Corrective RAG, Adaptive RAG, Multi-hop retrieval, and Agent-assisted retrieval, with production implementation code and a business scenario selection framework.

---

## Table of Contents

1. [Why Advanced Architectures Are Needed](#1-why-advanced-architectures-are-needed)
2. [Naive RAG vs Advanced RAG — Deep Comparison](#2-naive-rag-vs-advanced-rag--deep-comparison)
3. [Hybrid RAG — Production-Ready Patterns](#3-hybrid-rag--production-ready-patterns)
4. [Graph RAG — Knowledge Graph Retrieval](#4-graph-rag--knowledge-graph-retrieval)
5. [Corrective RAG (CRAG) — Self-Correcting Pipelines](#5-corrective-rag-crag--self-correcting-pipelines)
6. [Adaptive RAG — Query-Driven Pipeline Routing](#6-adaptive-rag--query-driven-pipeline-routing)
7. [Multi-Hop Retrieval Patterns](#7-multi-hop-retrieval-patterns)
8. [Agent-Assisted Retrieval Concepts](#8-agent-assisted-retrieval-concepts)
9. [Architecture Selection Framework](#9-architecture-selection-framework)
10. [Lab: Choose the Right RAG Architecture for Business Scenarios](#10-lab-choose-the-right-rag-architecture-for-business-scenarios)

---

## 1. Why Advanced Architectures Are Needed

### The Ceiling of Naive RAG

Naive RAG — embed, store, retrieve top-K, generate — solves a specific problem: answering questions when the answer is contained in a single, clearly worded document chunk. It hits a hard ceiling when queries require:

```
QUERY TYPE                        NAIVE RAG FAILURE MODE
──────────────────────────────────────────────────────────────────────
Multi-concept synthesis           Retrieves for one concept, misses others
Relationship traversal            Can't follow entity–entity connections
Temporal reasoning                No awareness of document freshness/order
Stale corpus                      Returns outdated answer confidently
Implicit query intent             Embeds surface words, not underlying need
Multi-document reasoning          Finds individual facts, fails to synthesise
High-confidence threshold needed  No self-checking; delivers guesses as facts
Mixed structured/unstructured     Can't combine SQL results with text retrieval
```

### The Architecture Evolution Map

```
Naive RAG
    │ Problem: retrieval noise, no query understanding
    ▼
Advanced RAG ─────────────────────── query rewriting, re-ranking, HyDE
    │ Problem: single-pass retrieval still misses complex queries
    ▼
Hybrid RAG ──────────────────────── dense + sparse, covers keyword gaps
    │ Problem: graph relationships invisible to vector search
    ▼
Graph RAG ───────────────────────── entity/relationship-aware retrieval
    │ Problem: no quality check on retrieved docs
    ▼
Corrective RAG (CRAG) ───────────── evaluates retrieval, falls back to web
    │ Problem: one-size-fits-all pipeline wastes resources on simple queries
    ▼
Adaptive RAG ────────────────────── routes query to appropriate strategy
    │ Problem: single retrieval round insufficient for chained reasoning
    ▼
Multi-Hop RAG ───────────────────── iterative retrieval following evidence
    │ Problem: static tool set, no autonomous decision-making
    ▼
Agentic / Agent-Assisted RAG ─────── LLM decides what to retrieve and when
```

### Choosing the Right Level

```
Start at Naive RAG. Advance one level ONLY when you have:
  (a) a specific, measured failure at the current level, AND
  (b) evidence that the next level fixes it on your evaluation dataset.

Do NOT pre-emptively implement Graph RAG for a simple FAQ bot.
Do NOT add agentic loops to a single-source Q&A system.
```

---

## 2. Naive RAG vs Advanced RAG — Deep Comparison

### Architecture Side-by-Side

```
NAIVE RAG:                          ADVANCED RAG:
─────────────────────────────────   ────────────────────────────────────────────
User Query                          User Query
    │                                   │
    ▼                                   ▼
[Embed Query]                       [PRE-RETRIEVAL]
    │                                 Query rewriting / HyDE / multi-query
    ▼                                   │
[Vector Search → top-K]             [RETRIEVAL]
    │                                 Hybrid dense + BM25 search
    ▼                                 Parent-child chunk expansion
[Inject chunks into prompt]           │
    │                               [POST-RETRIEVAL]
    ▼                                 Cross-encoder re-ranking
[LLM generates answer]               Context compression
    │                                   │
    ▼                               [LLM generates answer]
Answer                                  │
                                    Answer
```

### Pre-Retrieval: Query Transformation Techniques

#### HyDE — Hypothetical Document Embeddings

The user's query is often phrased differently from how the answer is written in the corpus. HyDE generates a hypothetical answer document and embeds *that* — closing the semantic gap.

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma

llm        = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

HYDE_PROMPT = ChatPromptTemplate.from_template("""Write a concise paragraph that
directly and factually answers the following question, as if extracted from an
authoritative policy document. Be specific and complete.

Question: {question}

Hypothetical answer paragraph:""")

def hyde_retrieval(question: str, k: int = 5) -> list:
    # Generate hypothetical document
    hypothetical_doc = (HYDE_PROMPT | llm).invoke({"question": question}).content
    
    # Embed the hypothetical doc (not the raw question)
    hyp_embedding = embeddings.embed_query(hypothetical_doc)
    
    # Retrieve using the hypothetical doc's embedding
    results = vectorstore.similarity_search_by_vector(hyp_embedding, k=k)
    return results

# Example
# Query:            "what happens if I miss a payment?"
# Hypothetical doc: "If a payment is missed, a late fee of £25 is charged..."
# → Retrieval finds the actual payment penalty clause
```

#### Step-Back Prompting

Abstract the specific query to a more general "step-back" question, retrieve for the general concept, then use both general and specific context.

```python
STEP_BACK_PROMPT = ChatPromptTemplate.from_template("""You are an expert at
abstracting specific questions into broader conceptual questions.

Given a specific question, generate a more general "step-back" question that
captures the underlying concept or domain needed to answer the original question.

Specific question: {question}

Step-back question (broader, conceptual):""")

def step_back_retrieval(question: str, k: int = 5) -> list:
    step_back_q = (STEP_BACK_PROMPT | llm).invoke({"question": question}).content.strip()
    
    # Retrieve for both specific and step-back question
    specific_docs  = vectorstore.similarity_search(question,     k=k)
    stepback_docs  = vectorstore.similarity_search(step_back_q,  k=k)
    
    # Deduplicate and combine
    seen, combined = set(), []
    for doc in specific_docs + stepback_docs:
        key = doc.page_content[:100]
        if key not in seen:
            combined.append(doc)
            seen.add(key)
    
    return combined[:k]

# Example
# Specific:   "What is the penalty for late invoice submission?"
# Step-back:  "What are the rules and consequences for financial compliance?"
# → Retrieves both the specific penalty clause AND the broader compliance framework
```

#### Multi-Query Expansion

Generate N reformulations of the question, retrieve for each, union the results.

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""Generate {n} different
search queries to retrieve documents that would help answer this question.
Each query should approach the topic from a different angle or use different
terminology. Return as a comma-separated list.

Question: {question}
Queries:""")

def multi_query_retrieval(question: str, n: int = 3, k: int = 3) -> list:
    parser = CommaSeparatedListOutputParser()
    queries_text = (MULTI_QUERY_PROMPT | llm | parser).invoke(
        {"question": question, "n": n}
    )
    
    all_docs, seen = [], set()
    for query in queries_text:
        docs = vectorstore.similarity_search(query.strip(), k=k)
        for doc in docs:
            key = doc.page_content[:80]
            if key not in seen:
                all_docs.append(doc)
                seen.add(key)
    
    return all_docs
```

### Post-Retrieval: Context Window Management

```python
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever

# Filter: remove chunks below a relevance threshold (not just re-rank)
filter_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
doc_filter  = LLMChainFilter.from_llm(filter_llm)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
filtered_retriever = ContextualCompressionRetriever(
    base_compressor=doc_filter,
    base_retriever=base_retriever,
)

# Sentence Window Retrieval: index single sentences, return surrounding window
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_community.document_transformers import LongContextReorder

def reorder_for_lost_in_the_middle(docs: list) -> list:
    """
    Research shows LLMs recall beginning/end of context better than middle.
    Place most relevant docs at start and end, less relevant in the middle.
    """
    reorderer = LongContextReorder()
    return reorderer.transform_documents(docs)
```

### Advanced RAG Performance Benchmarks

```
Technique               Recall@5   Faithfulness   Latency Add   Cost Add
───────────────────────────────────────────────────────────────────────
Baseline (naive)          0.51       0.71           0 ms          $0
+ HyDE                    0.65       0.79          +300 ms        +$0.001
+ Multi-query (3×)        0.71       0.77          +200 ms        +$0.001
+ Step-back               0.67       0.80          +150 ms        +$0.001
+ Cross-encoder rerank    0.83       0.91          +150 ms        $0 (local)
+ Context compression     0.81       0.93          +400 ms        +$0.002
+ Lost-in-middle reorder  0.83       0.94           +10 ms        $0
───────────────────────────────────────────────────────────────────────
Best single addition:     cross-encoder rerank (highest gain, lowest cost)
Best combined:            HyDE + rerank + reorder (+63% recall, +32% faith)
```

---

## 3. Hybrid RAG — Production-Ready Patterns

### Why Vector Search Alone Fails in Production

```
PRODUCTION QUERY TYPES WHERE PURE VECTOR SEARCH FAILS:

  Error codes:    "ERR_SSL_PROTOCOL_ERROR" → embedding dilutes into generic SSL concepts
  Product IDs:    "SKU-449821-B" → no semantic signal to embed
  Person names:   "Priya Ramaswamy's onboarding checklist" → name not in training
  Legal refs:     "GDPR Article 17(1)(b)" → exact citation required
  Version nums:   "LangChain 0.3.x breaking changes" → semantic drift from 0.2.x
  Regex / code:   "[A-Z]{2}\d{6}" → meaningless as an embedding
```

### BM25 + Dense: The Production Default

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk
loader = DirectoryLoader("./docs", glob="**/*.{pdf,txt,md}")
docs   = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)\
           .split_documents(docs)

# Dense store
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
dense_ret   = vectorstore.as_retriever(search_kwargs={"k": 10})

# Sparse store (BM25)
bm25_ret    = BM25Retriever.from_documents(chunks)
bm25_ret.k  = 10

# Hybrid (Reciprocal Rank Fusion)
hybrid = EnsembleRetriever(
    retrievers=[dense_ret, bm25_ret],
    weights=[0.6, 0.4],
)

results = hybrid.invoke("ERR_SSL_PROTOCOL_ERROR tls handshake nginx")
```

### SPLADE — Learned Sparse Retrieval

SPLADE is a neural model that produces *learned* sparse representations — more expressive than BM25 but still exact-match capable. Available via HuggingFace.

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import scipy.sparse as sp

class SPLADEEncoder:
    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
    
    def encode(self, text: str) -> dict[str, float]:
        """Returns token → weight sparse vector."""
        inputs  = self.tokenizer(text, return_tensors="pt",
                                  truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # SPLADE aggregation: log(1 + ReLU(logits)).max(dim=1)
        logits   = outputs.logits.squeeze(0)
        weights  = torch.log1p(torch.relu(logits)).max(dim=0).values
        
        # Return only non-zero terms
        indices  = weights.nonzero().squeeze()
        return {
            self.tokenizer.convert_ids_to_tokens(i.item()): weights[i].item()
            for i in indices
        }

splade = SPLADEEncoder()
query_sparse = splade.encode("SSL handshake failure")
# → {'ssl': 2.1, 'handshake': 1.8, 'tls': 1.4, 'certificate': 0.9, ...}
```

### Hybrid Search with Metadata Filtering

Combine hybrid retrieval with metadata pre-filtering for department- or date-scoped queries.

```python
from datetime import datetime, timedelta

def hybrid_filtered_retrieval(
    query:      str,
    department: str  = None,
    after_date: str  = None,  # ISO format "2025-01-01"
    k:          int  = 5,
) -> list:
    """Hybrid search with optional metadata pre-filters."""
    
    filter_dict = {}
    if department:
        filter_dict["department"] = department
    if after_date:
        filter_dict["document_date"] = {"$gte": after_date}
    
    dense_k   = vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": filter_dict or None}
    )
    
    # BM25 with post-filter (BM25Retriever doesn't support metadata filters natively)
    bm25_docs = [c for c in chunks
                 if (not department or c.metadata.get("department") == department)
                 and (not after_date or c.metadata.get("document_date", "") >= after_date)]
    bm25_k = BM25Retriever.from_documents(bm25_docs, k=10) if bm25_docs else None
    
    if bm25_k:
        hybrid = EnsembleRetriever(retrievers=[dense_k, bm25_k], weights=[0.6, 0.4])
        return hybrid.invoke(query)[:k]
    else:
        return dense_k.invoke(query)[:k]

# Example: only search IT documents published after Jan 2025
results = hybrid_filtered_retrieval(
    "SSL certificate renewal process",
    department="IT",
    after_date="2025-01-01",
)
```

---

## 4. Graph RAG — Knowledge Graph Retrieval

### When Vector Search Misses Relationships

```
KNOWLEDGE BASE:
  Doc A: "Alice is the Head of Engineering."
  Doc B: "The Engineering team owns Project Atlas."
  Doc C: "Project Atlas uses the payment processing microservice."
  Doc D: "The payment processing microservice has a known CVE-2024-1234 vulnerability."

QUERY: "Who is responsible for the system with CVE-2024-1234?"

VECTOR SEARCH: retrieves Doc D (CVE) but misses the chain:
  CVE → payment service → Project Atlas → Engineering → Alice

GRAPH RAG traversal:
  CVE-2024-1234 → affects → payment_service
                          → owned_by → Project Atlas
                                     → owned_by → Engineering
                                                → managed_by → Alice
  Answer: Alice, Head of Engineering
```

### Building a Knowledge Graph from Documents

```python
# pip install networkx spacy
# python -m spacy download en_core_web_trf

import networkx as nx
import spacy
from langchain_openai import ChatOpenAI
import json, re

nlp = spacy.load("en_core_web_trf")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Text: {text}

Return JSON with this structure:
{{
  "entities": [
    {{"id": "unique_id", "name": "entity name", "type": "PERSON|ORG|SYSTEM|POLICY|CONCEPT|LOCATION"}}
  ],
  "relationships": [
    {{"source": "entity_id", "relation": "verb phrase", "target": "entity_id"}}
  ]
}}

Only include clear, factual relationships stated in the text."""

def extract_graph_elements(text: str) -> dict:
    result = llm.invoke(ENTITY_EXTRACTION_PROMPT.format(text=text[:2000]))
    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}

def build_knowledge_graph(chunks: list) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for chunk in chunks:
        elements = extract_graph_elements(chunk.page_content)
        
        for entity in elements.get("entities", []):
            G.add_node(
                entity["id"],
                name=entity["name"],
                type=entity["type"],
                source=chunk.metadata.get("source", ""),
            )
        
        for rel in elements.get("relationships", []):
            if rel["source"] in G and rel["target"] in G:
                G.add_edge(
                    rel["source"],
                    rel["target"],
                    relation=rel["relation"],
                    source_doc=chunk.metadata.get("source", ""),
                )
    
    return G

# Build graph
G = build_knowledge_graph(chunks)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

### Graph Traversal for Retrieval

```python
from sentence_transformers import SentenceTransformer, util
import torch

node_encoder = SentenceTransformer("all-MiniLM-L6-v2")

def find_seed_entities(query: str, G: nx.DiGraph, top_k: int = 3) -> list[str]:
    """Find the graph nodes most semantically similar to the query."""
    node_names   = [(nid, G.nodes[nid]["name"]) for nid in G.nodes]
    if not node_names:
        return []
    
    query_emb    = node_encoder.encode(query, convert_to_tensor=True)
    entity_embs  = node_encoder.encode([n for _, n in node_names], convert_to_tensor=True)
    sims         = util.cos_sim(query_emb, entity_embs)[0]
    
    top_indices  = torch.topk(sims, min(top_k, len(node_names))).indices
    return [node_names[i][0] for i in top_indices]

def graph_traversal_retrieve(
    query:      str,
    G:          nx.DiGraph,
    vectorstore,
    max_hops:   int = 2,
    k:          int = 5,
) -> dict:
    """
    1. Find seed entities via embedding similarity.
    2. Traverse the graph (BFS) up to max_hops.
    3. Collect all node names and relationship triples.
    4. Retrieve document chunks for each traversed entity.
    """
    
    seeds = find_seed_entities(query, G, top_k=3)
    
    # BFS traversal
    visited   = set(seeds)
    frontier  = set(seeds)
    subgraph_triples = []
    
    for hop in range(max_hops):
        next_frontier = set()
        for node_id in frontier:
            # Outgoing edges
            for _, target, data in G.out_edges(node_id, data=True):
                src_name = G.nodes[node_id]["name"]
                tgt_name = G.nodes[target]["name"]
                relation = data.get("relation", "relates to")
                subgraph_triples.append(f"{src_name} → {relation} → {tgt_name}")
                if target not in visited:
                    next_frontier.add(target)
                    visited.add(target)
            # Incoming edges
            for source, _, data in G.in_edges(node_id, data=True):
                src_name = G.nodes[source]["name"]
                tgt_name = G.nodes[node_id]["name"]
                relation = data.get("relation", "relates to")
                subgraph_triples.append(f"{src_name} → {relation} → {tgt_name}")
                if source not in visited:
                    next_frontier.add(source)
                    visited.add(source)
        frontier = next_frontier
    
    # Retrieve document chunks for all traversed entities
    entity_names = [G.nodes[nid]["name"] for nid in visited if nid in G.nodes]
    vector_docs  = []
    for name in entity_names[:5]:  # limit to top 5 to control context size
        docs = vectorstore.similarity_search(name, k=2)
        vector_docs.extend(docs)
    
    # Deduplicate
    seen, unique_docs = set(), []
    for doc in vector_docs:
        key = doc.page_content[:80]
        if key not in seen:
            unique_docs.append(doc)
            seen.add(key)
    
    return {
        "graph_triples":  subgraph_triples[:20],
        "document_chunks": unique_docs[:k],
        "traversed_entities": entity_names,
    }

def graph_rag_answer(query: str, G: nx.DiGraph, vectorstore, llm) -> str:
    retrieval = graph_traversal_retrieve(query, G, vectorstore)
    
    graph_context = "\n".join(retrieval["graph_triples"])
    doc_context   = "\n\n".join(
        f"[Doc {i+1}]: {d.page_content}"
        for i, d in enumerate(retrieval["document_chunks"])
    )
    
    GRAPH_RAG_PROMPT = f"""You are an assistant with access to both a knowledge graph
and document excerpts. Use both to answer the question.

Knowledge Graph Relationships:
{graph_context}

Document Excerpts:
{doc_context}

Question: {query}

Answer (cite both graph relationships and document sources where relevant):"""
    
    return llm.invoke(GRAPH_RAG_PROMPT).content
```

### LangChain + Neo4j Graph RAG

For production, use a managed graph database instead of in-memory NetworkX.

```python
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

llm   = ChatOpenAI(model="gpt-4o", temperature=0)
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
)

# Extract graph from documents (automatic via LLMGraphTransformer)
transformer = LLMGraphTransformer(llm=llm)
graph_docs  = transformer.convert_to_graph_documents(chunks)
graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)

# Natural language → Cypher → Graph query → LLM answer
chain = GraphCypherQAChain.from_llm(
    llm=llm, graph=graph, verbose=True,
    allow_dangerous_requests=True,
)

result = chain.invoke({"query": "Who is responsible for systems with CVE vulnerabilities?"})
print(result["result"])
```

### Hybrid Graph + Vector RAG

```python
def hybrid_graph_vector_rag(query: str, G: nx.DiGraph, vectorstore, llm) -> str:
    """Combine graph traversal with semantic vector retrieval."""
    
    # Path 1: Graph-guided retrieval
    graph_result = graph_traversal_retrieve(query, G, vectorstore, max_hops=2)
    
    # Path 2: Pure semantic retrieval
    semantic_docs = vectorstore.similarity_search(query, k=5)
    
    # Merge: prioritise graph-guided docs, fill with semantic docs
    all_docs = graph_result["document_chunks"].copy()
    seen     = {d.page_content[:80] for d in all_docs}
    for doc in semantic_docs:
        if doc.page_content[:80] not in seen:
            all_docs.append(doc)
            seen.add(doc.page_content[:80])
    
    combined_context = (
        "=== Knowledge Graph Context ===\n"
        + "\n".join(graph_result["graph_triples"][:15])
        + "\n\n=== Document Context ===\n"
        + "\n\n".join(d.page_content for d in all_docs[:6])
    )
    
    PROMPT = f"""Answer using the knowledge graph relationships and document context below.

{combined_context}

Question: {query}
Answer:"""
    
    return llm.invoke(PROMPT).content
```

---

## 5. Corrective RAG (CRAG) — Self-Correcting Pipelines

### The Problem CRAG Solves

Naive and Advanced RAG retrieve from the knowledge base unconditionally. If the knowledge base doesn't contain the answer, they either hallucinate or fail silently. CRAG adds a **retrieval evaluator** that grades retrieved documents and routes the pipeline accordingly.

```
CRAG DECISION STATES:

  CORRECT   → retrieved docs are relevant and sufficient → generate from them
  AMBIGUOUS → partially relevant → combine internal + web search → generate
  INCORRECT → irrelevant → discard internal → web search only → generate
```

### CRAG State Machine (LangGraph)

```python
# pip install langgraph langchain-openai tavily-python

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from typing import TypedDict, Annotated
import operator, json

llm           = ChatOpenAI(model="gpt-4o-mini", temperature=0)
web_search    = TavilySearchResults(max_results=3)

class CRAGState(TypedDict):
    question:           str
    retrieved_docs:     list
    web_results:        list
    retrieval_grade:    str   # "correct" | "ambiguous" | "incorrect"
    context:            str
    answer:             str

# ── Node 1: Retrieve from vector store ──────────────────────────────────────
def retrieve(state: CRAGState) -> CRAGState:
    docs = vectorstore.similarity_search(state["question"], k=5)
    return {**state, "retrieved_docs": docs}

# ── Node 2: Grade retrieved documents ───────────────────────────────────────
GRADER_PROMPT = ChatPromptTemplate.from_template("""You are a retrieval quality evaluator.

Question: {question}

Retrieved Documents:
{documents}

Assess whether the retrieved documents are sufficient to answer the question.
Return exactly one word: "correct", "ambiguous", or "incorrect".

- correct:   documents clearly contain the answer
- ambiguous: documents are partially relevant or insufficient
- incorrect: documents are irrelevant to the question

Grade:""")

def grade_documents(state: CRAGState) -> CRAGState:
    doc_text = "\n\n---\n\n".join(d.page_content for d in state["retrieved_docs"])
    grade    = (GRADER_PROMPT | llm).invoke({
        "question":  state["question"],
        "documents": doc_text,
    }).content.strip().lower()
    
    if grade not in ("correct", "ambiguous", "incorrect"):
        grade = "ambiguous"
    
    return {**state, "retrieval_grade": grade}

# ── Node 3: Web search fallback ──────────────────────────────────────────────
def web_search_node(state: CRAGState) -> CRAGState:
    results = web_search.invoke(state["question"])
    return {**state, "web_results": results}

# ── Node 4: Build context ────────────────────────────────────────────────────
def build_context(state: CRAGState) -> CRAGState:
    grade = state["retrieval_grade"]
    parts = []
    
    if grade in ("correct", "ambiguous") and state.get("retrieved_docs"):
        parts.append("=== Internal Documents ===")
        parts.append("\n\n".join(d.page_content for d in state["retrieved_docs"]))
    
    if grade in ("ambiguous", "incorrect") and state.get("web_results"):
        parts.append("=== Web Search Results ===")
        for r in state["web_results"]:
            parts.append(f"[{r['url']}]\n{r['content']}")
    
    return {**state, "context": "\n\n".join(parts)}

# ── Node 5: Generate answer ──────────────────────────────────────────────────
GENERATION_PROMPT = ChatPromptTemplate.from_template("""Answer the question using
ONLY the provided context. If you cannot answer from the context, say so.

Context:
{context}

Question: {question}
Answer:""")

def generate(state: CRAGState) -> CRAGState:
    answer = (GENERATION_PROMPT | llm).invoke({
        "context":  state["context"],
        "question": state["question"],
    }).content
    return {**state, "answer": answer}

# ── Routing function ─────────────────────────────────────────────────────────
def route_after_grading(state: CRAGState) -> str:
    grade = state["retrieval_grade"]
    if grade == "correct":
        return "build_context"       # skip web search
    else:
        return "web_search_node"     # ambiguous or incorrect → search web

# ── Build the graph ──────────────────────────────────────────────────────────
workflow = StateGraph(CRAGState)
workflow.add_node("retrieve",       retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search_node", web_search_node)
workflow.add_node("build_context",  build_context)
workflow.add_node("generate",       generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve",       "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "build_context":  "build_context",
        "web_search_node": "web_search_node",
    }
)
workflow.add_edge("web_search_node", "build_context")
workflow.add_edge("build_context",  "generate")
workflow.add_edge("generate",       END)

crag_app = workflow.compile()

def run_crag(question: str) -> dict:
    result = crag_app.invoke({"question": question, "retrieved_docs": [],
                               "web_results": [], "retrieval_grade": "",
                               "context": "", "answer": ""})
    print(f"Retrieval grade: {result['retrieval_grade']}")
    return result

# Example
result = run_crag("What is the latest interest rate set by the Bank of England?")
# → grade: "incorrect" (internal docs don't have live rates)
# → falls back to web search
# → answers from Tavily results
```

### CRAG with Knowledge Refinement

Before using retrieved documents, transform them to keep only the relevant sentences (knowledge refinement).

```python
REFINEMENT_PROMPT = ChatPromptTemplate.from_template("""Given a question and a
retrieved document, extract ONLY the sentences that are directly relevant to
answering the question. Remove all irrelevant content.

Question: {question}

Document:
{document}

Relevant sentences only (preserve exact wording):""")

def refine_documents(question: str, docs: list) -> list:
    """Keep only the relevant sentences from each retrieved document."""
    from langchain.schema import Document
    refined = []
    for doc in docs:
        refined_text = (REFINEMENT_PROMPT | llm).invoke({
            "question": question,
            "document": doc.page_content,
        }).content.strip()
        if refined_text and len(refined_text) > 30:
            refined.append(Document(
                page_content=refined_text,
                metadata=doc.metadata,
            ))
    return refined
```

---

## 6. Adaptive RAG — Query-Driven Pipeline Routing

### The Core Idea

Adaptive RAG analyses the incoming query and dynamically selects the retrieval strategy most appropriate for that query type — avoiding the "one-size-fits-all" overhead.

```
Query type          → Optimal strategy
────────────────────────────────────────────────────────────────────
Factual / specific  → Single-pass vector search (fastest, cheapest)
Ambiguous / broad   → HyDE or Multi-query expansion
Multi-concept       → Multi-hop retrieval
Relationship-based  → Graph RAG traversal
Out-of-date corpus  → CRAG with web fallback
Structured data     → Text-to-SQL
Simple greetings    → No retrieval needed
```

### Query Classifier

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
import json

class QueryStrategy(str, Enum):
    NO_RETRIEVAL   = "no_retrieval"    # chitchat, greetings
    SIMPLE_VECTOR  = "simple_vector"   # direct factual lookup
    HYDE           = "hyde"            # ambiguous, needs expansion
    MULTI_QUERY    = "multi_query"     # broad topic
    MULTI_HOP      = "multi_hop"       # chained reasoning
    GRAPH          = "graph"           # entity relationships
    CRAG           = "crag"            # potentially stale/out-of-scope
    SQL            = "sql"             # quantitative / structured data

classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""You are a RAG query router.
Classify the query into one of the following retrieval strategies:

- no_retrieval:  Greeting, chitchat, or question answerable without documents.
- simple_vector: Specific factual question, single document, direct lookup.
- hyde:          Vague or ambiguous question needing semantic expansion.
- multi_query:   Broad topic where multiple angles improve coverage.
- multi_hop:     Requires chaining facts across multiple documents.
- graph:         Asks about relationships between entities.
- crag:          Potentially time-sensitive or out-of-scope for internal docs.
- sql:           Asks for counts, totals, averages, or structured data.

Query: "{query}"

Return JSON: {{"strategy": "<strategy>", "reasoning": "<one sentence>"}}""")

def classify_query(query: str) -> tuple[QueryStrategy, str]:
    result   = classifier_llm.invoke(CLASSIFIER_PROMPT.format(query=query))
    parsed   = json.loads(result.content)
    strategy = QueryStrategy(parsed["strategy"])
    return strategy, parsed["reasoning"]
```

### Adaptive Router

```python
def adaptive_rag(query: str) -> dict:
    strategy, reasoning = classify_query(query)
    
    print(f"Strategy: {strategy.value} | Reasoning: {reasoning}")
    
    if strategy == QueryStrategy.NO_RETRIEVAL:
        answer = llm.invoke(f"Answer conversationally: {query}").content
        return {"answer": answer, "strategy": strategy, "sources": []}
    
    elif strategy == QueryStrategy.SIMPLE_VECTOR:
        docs   = vectorstore.similarity_search(query, k=4)
        answer = generate_answer(query, docs)
        return {"answer": answer, "strategy": strategy, "sources": docs}
    
    elif strategy == QueryStrategy.HYDE:
        docs   = hyde_retrieval(query, k=5)
        answer = generate_answer(query, docs)
        return {"answer": answer, "strategy": strategy, "sources": docs}
    
    elif strategy == QueryStrategy.MULTI_QUERY:
        docs   = multi_query_retrieval(query, n=3, k=3)
        answer = generate_answer(query, docs)
        return {"answer": answer, "strategy": strategy, "sources": docs}
    
    elif strategy == QueryStrategy.MULTI_HOP:
        return multi_hop_rag(query)   # see Section 7
    
    elif strategy == QueryStrategy.GRAPH:
        answer = graph_rag_answer(query, G, vectorstore, llm)
        return {"answer": answer, "strategy": strategy, "sources": []}
    
    elif strategy == QueryStrategy.CRAG:
        result = run_crag(query)
        return {"answer": result["answer"], "strategy": strategy,
                "retrieval_grade": result["retrieval_grade"]}
    
    elif strategy == QueryStrategy.SQL:
        answer = sql_agent.invoke({"input": query})
        return {"answer": answer, "strategy": strategy, "sources": []}
    
    # Fallback
    docs   = vectorstore.similarity_search(query, k=5)
    answer = generate_answer(query, docs)
    return {"answer": answer, "strategy": QueryStrategy.SIMPLE_VECTOR, "sources": docs}
```

### Self-Reflective Adaptive RAG (with LangGraph)

Add a self-evaluation loop: if the generated answer is graded as insufficient, retry with a different strategy.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AdaptiveRAGState(TypedDict):
    question:       str
    strategy:       str
    answer:         str
    answer_grade:   str   # "sufficient" | "insufficient"
    retry_count:    int
    sources:        list

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_template("""Is this answer sufficient
to fully address the question? Consider completeness and accuracy.

Question: {question}
Answer:   {answer}

Return exactly one word: "sufficient" or "insufficient".""")

def grade_answer(state: AdaptiveRAGState) -> AdaptiveRAGState:
    grade = (ANSWER_GRADER_PROMPT | llm).invoke({
        "question": state["question"],
        "answer":   state["answer"],
    }).content.strip().lower()
    return {**state, "answer_grade": grade}

def should_retry(state: AdaptiveRAGState) -> str:
    if state["answer_grade"] == "sufficient" or state["retry_count"] >= 2:
        return END
    return "escalate_strategy"

def escalate_strategy(state: AdaptiveRAGState) -> AdaptiveRAGState:
    """Upgrade to a more powerful strategy on retry."""
    current = state["strategy"]
    escalation_map = {
        "simple_vector": "multi_query",
        "multi_query":   "multi_hop",
        "multi_hop":     "crag",
        "hyde":          "multi_hop",
    }
    new_strategy = escalation_map.get(current, "crag")
    result       = adaptive_rag_with_strategy(state["question"], new_strategy)
    return {
        **state,
        "strategy":    new_strategy,
        "answer":      result["answer"],
        "retry_count": state["retry_count"] + 1,
    }

workflow = StateGraph(AdaptiveRAGState)
workflow.add_node("retrieve_and_generate", lambda s: {**s, **adaptive_rag(s["question"])})
workflow.add_node("grade_answer",          grade_answer)
workflow.add_node("escalate_strategy",     escalate_strategy)
workflow.set_entry_point("retrieve_and_generate")
workflow.add_edge("retrieve_and_generate", "grade_answer")
workflow.add_conditional_edges("grade_answer", should_retry,
                                {"escalate_strategy": "escalate_strategy", END: END})
workflow.add_edge("escalate_strategy", "grade_answer")
adaptive_rag_app = workflow.compile()
```

---

## 7. Multi-Hop Retrieval Patterns

### When Single-Pass Retrieval Fails

```
MULTI-HOP QUERY EXAMPLE:

"What programming language is used by the team that maintains
 the service that was cited in the Q3 security audit?"

Single pass: embeds the full sentence → retrieves security audit section
             → Q3 audit mentions "payment-service was flagged"
             → Answer: "Python" (if lucky) — more likely misses the connection

Multi-hop:
  Hop 1: "Q3 security audit flagged services"
         → finds: "payment-service cited for CVE-2024-1234"

  Hop 2: "payment-service team owner"
         → finds: "payment-service owned by Platform Engineering team"

  Hop 3: "Platform Engineering team tech stack"
         → finds: "Platform Engineering uses Go and Kubernetes"

  Answer: "Go (used by the Platform Engineering team)"
```

### Iterative Decomposition (Chain-of-Thought Retrieval)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o", temperature=0)

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template("""You are solving a complex
question that requires multiple information lookups.

Original question: {original_question}

Information gathered so far:
{gathered_info}

Based on what you know so far, what is the single most important sub-question
to search for next to make progress toward answering the original question?

If you now have enough information to answer the original question fully,
write: FINAL_ANSWER: <your complete answer>

Otherwise write the next sub-question only (no explanation):""")

def multi_hop_rag(
    question:  str,
    max_hops:  int = 5,
    k:         int = 3,
) -> dict:
    gathered_info = ""
    hop_trace     = []
    
    for hop in range(max_hops):
        response = (DECOMPOSE_PROMPT | llm).invoke({
            "original_question": question,
            "gathered_info":     gathered_info or "Nothing gathered yet.",
        }).content.strip()
        
        # Check for final answer
        if response.startswith("FINAL_ANSWER:"):
            return {
                "answer":    response.replace("FINAL_ANSWER:", "").strip(),
                "hops":      hop_trace,
                "hop_count": hop,
            }
        
        # Execute sub-question retrieval
        sub_question = response
        docs         = vectorstore.similarity_search(sub_question, k=k)
        retrieved    = "\n".join(d.page_content for d in docs)
        
        hop_trace.append({
            "hop":          hop + 1,
            "sub_question": sub_question,
            "retrieved":    retrieved[:300] + "...",
        })
        
        gathered_info += (
            f"\n\n[Hop {hop + 1}] Sub-question: {sub_question}\n"
            f"Retrieved:\n{retrieved}"
        )
    
    # Max hops reached — generate from what we have
    final_prompt = f"""Answer the following question using the information gathered
through multiple retrieval steps.

Original question: {question}

Gathered information:
{gathered_info}

Final answer:"""
    
    answer = llm.invoke(final_prompt).content
    return {"answer": answer, "hops": hop_trace, "hop_count": max_hops}
```

### IRCoT — Interleaved Retrieval with Chain-of-Thought

IRCoT interleaves reasoning and retrieval: the LLM reasons one sentence at a time, and each new sentence triggers a retrieval if it implies a knowledge need.

```python
IRCOT_PROMPT = ChatPromptTemplate.from_template("""Solve this step by step.
After each reasoning step, if you need more information, write:
RETRIEVE: <what to search for>
If you have enough information, write:
ANSWER: <final answer>

Question: {question}

Retrieved context so far:
{context}

Continue reasoning from where you left off:
{reasoning_so_far}""")

def ircot(question: str, max_iterations: int = 6) -> str:
    context         = ""
    reasoning_so_far = ""
    
    for iteration in range(max_iterations):
        response = (IRCOT_PROMPT | llm).invoke({
            "question":        question,
            "context":         context or "None yet.",
            "reasoning_so_far": reasoning_so_far,
        }).content.strip()
        
        # Check for final answer
        if "ANSWER:" in response:
            return response.split("ANSWER:")[-1].strip()
        
        # Extract retrieval request
        if "RETRIEVE:" in response:
            search_query = response.split("RETRIEVE:")[-1].split("\n")[0].strip()
            new_docs     = vectorstore.similarity_search(search_query, k=3)
            new_context  = "\n\n".join(d.page_content for d in new_docs)
            context     += f"\n\n[Retrieval {iteration + 1}: '{search_query}']\n{new_context}"
        
        reasoning_so_far += "\n" + response
    
    # Fallback
    return reasoning_so_far.split("ANSWER:")[-1].strip() if "ANSWER:" in reasoning_so_far \
        else "Could not reach a definitive answer within the allowed steps."
```

### Beam Search Multi-Hop

Explore multiple retrieval paths in parallel and pick the best final answer.

```python
import asyncio

async def beam_hop(question: str, beam: list[str], k: int = 3) -> list[list[str]]:
    """Expand each beam candidate with a new retrieval step."""
    new_beams = []
    for accumulated_context in beam:
        sub_q  = (DECOMPOSE_PROMPT | llm).invoke({
            "original_question": question,
            "gathered_info":     accumulated_context,
        }).content.strip()
        
        if sub_q.startswith("FINAL_ANSWER:"):
            new_beams.append([accumulated_context, sub_q])
            continue
        
        docs    = vectorstore.similarity_search(sub_q, k=k)
        snippet = "\n".join(d.page_content[:200] for d in docs)
        new_beams.append([accumulated_context, f"Q: {sub_q}\nA: {snippet}"])
    
    return new_beams

async def beam_search_multi_hop(question: str, beam_width: int = 3, max_hops: int = 4) -> str:
    beams = [""]  # start with empty context
    
    for hop in range(max_hops):
        tasks    = [beam_hop(question, [b], k=3) for b in beams]
        expanded = await asyncio.gather(*tasks)
        
        # Flatten and score
        all_beams  = [b for sublist in expanded for b in sublist]
        
        # Score each beam by relevance to original question
        scored = []
        for beam_ctx in all_beams:
            context_str = "\n".join(beam_ctx)
            score_resp  = llm.invoke(
                f"Score 0-10 how useful this context is for answering: '{question}'\n\n"
                f"Context:\n{context_str[:500]}\n\nScore (integer only):"
            ).content.strip()
            try:
                scored.append((int(score_resp), beam_ctx))
            except ValueError:
                scored.append((5, beam_ctx))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        beams = ["\n".join(b) for _, b in scored[:beam_width]]
        
        # Check for final answer in top beam
        if "FINAL_ANSWER:" in beams[0]:
            return beams[0].split("FINAL_ANSWER:")[-1].strip()
    
    return llm.invoke(f"Answer based on:\n{beams[0]}\n\nQuestion: {question}").content
```

---

## 8. Agent-Assisted Retrieval Concepts

### From Static to Dynamic Retrieval

The key difference between pipeline-based RAG and Agentic RAG is **who decides what to retrieve**:

```
PIPELINE RAG:                       AGENTIC RAG:
────────────────────────────────    ──────────────────────────────────
Retrieval step is fixed             LLM decides whether to retrieve
Always retrieves from vector store  LLM chooses which tool to call
Always retrieves once               LLM retrieves multiple times
Developer controls flow             LLM controls flow within guardrails
```

### ReAct Agent with RAG Tools

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, StructuredTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Tool definitions ─────────────────────────────────────────────────────────

class DocumentSearchInput(BaseModel):
    query: str = Field(description="The search query to find relevant documents")
    k:     int = Field(default=4, description="Number of documents to retrieve")

def search_internal_docs(query: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'Internal Doc')}]\n{d.page_content}"
        for d in docs
    )

def search_knowledge_graph(query: str) -> str:
    result = graph_traversal_retrieve(query, G, vectorstore)
    triples = "\n".join(result["graph_triples"][:10])
    if not triples:
        return "No graph relationships found."
    return f"Knowledge graph relationships:\n{triples}"

def run_sql_query(question: str) -> str:
    result = sql_agent.invoke({"input": question})
    return str(result.get("output", "No SQL result."))

# Define tools
tools = [
    StructuredTool(
        name="search_internal_documents",
        func=search_internal_docs,
        args_schema=DocumentSearchInput,
        description=(
            "Search the internal document knowledge base. Use for questions about "
            "company policies, procedures, HR rules, IT documentation, or any "
            "topic covered by internal knowledge base documents."
        ),
    ),
    Tool(
        name="search_knowledge_graph",
        func=search_knowledge_graph,
        description=(
            "Query the knowledge graph to find relationships between entities "
            "(people, systems, teams, policies). Use when the question asks "
            "WHO is responsible for something, or HOW entities are connected."
        ),
    ),
    Tool(
        name="search_web",
        func=TavilySearchResults(max_results=3).run,
        description=(
            "Search the web for current events, live data, or information "
            "that may not be in internal documents. Use sparingly — only when "
            "internal docs are insufficient and the information may be time-sensitive."
        ),
    ),
    Tool(
        name="query_database",
        func=run_sql_query,
        description=(
            "Query the structured data database. Use for numerical questions: "
            "counts, totals, averages, trends, comparisons, 'how many', 'what is the "
            "total', 'which department has the most', etc."
        ),
    ),
]

# ── ReAct agent ──────────────────────────────────────────────────────────────
prompt  = hub.pull("hwchase17/react")
agent   = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

def agentic_rag(question: str) -> dict:
    result = executor.invoke({"input": question})
    return {
        "answer": result["output"],
        "steps":  [(s[0].tool, s[0].tool_input) for s in result["intermediate_steps"]],
    }

# Example
result = agentic_rag(
    "Which team owns the service with CVE-2024-1234, "
    "how many employees are on that team, and what is the current CVE severity score?"
)
# Steps: search_knowledge_graph → search_internal_documents → query_database → search_web
```

### Tool Selection Strategy

Design tool descriptions to be maximally discriminative — the LLM chooses based on description quality.

```python
# POOR tool description (too generic):
Tool(name="search", description="Search for information.")

# GOOD tool description (discriminative):
Tool(
    name="search_internal_documents",
    description=(
        "Search the internal HR, IT, legal, and policy document knowledge base. "
        "ALWAYS use this tool FIRST for questions about company policies, employee "
        "benefits, IT procedures, or internal processes. Returns text from official "
        "company documents. Do NOT use for live data, prices, or current events."
    ),
)
```

### Guardrails for Agentic RAG

```python
from langchain_core.callbacks import BaseCallbackHandler

class AgentGuardrailCallback(BaseCallbackHandler):
    MAX_TOOL_CALLS   = 8
    FORBIDDEN_TOOLS  = []  # e.g., ["send_email"] in read-only mode
    
    def __init__(self):
        self.tool_call_count = 0
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_call_count += 1
        tool_name = serialized.get("name", "")
        
        if self.tool_call_count > self.MAX_TOOL_CALLS:
            raise RuntimeError(
                f"Agent exceeded maximum tool calls ({self.MAX_TOOL_CALLS}). "
                "Stopping to prevent runaway loops."
            )
        
        if tool_name in self.FORBIDDEN_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not permitted in this context."
            )
    
    def on_agent_finish(self, finish, **kwargs):
        print(f"Agent completed in {self.tool_call_count} tool call(s).")

guardrail = AgentGuardrailCallback()
safe_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[guardrail],
    max_iterations=6,
    handle_parsing_errors=True,
)
```

---

## 9. Architecture Selection Framework

### Decision Tree

```
START: What is the primary challenge with your current RAG?
│
├── "It doesn't work at all / starting fresh"
│       → Naive RAG (prototype) → measure → improve
│
├── "Retrieval misses relevant docs"
│   ├── On exact terms (codes, names, IDs)    → Hybrid RAG (BM25 + dense)
│   ├── Query is vague or ambiguous           → Advanced RAG (HyDE / multi-query)
│   └── Answer spans multiple documents       → Multi-hop RAG
│
├── "Retrieval finds docs but answer is wrong"
│   ├── Knowledge base is outdated            → CRAG (web fallback)
│   ├── Hallucination despite good retrieval  → Advanced RAG (prompt + self-check)
│   └── Relationships between entities missed → Graph RAG
│
├── "Different query types need different treatment"
│       → Adaptive RAG (query classifier + strategy router)
│
├── "Complex tasks requiring tools + retrieval"
│       → Agentic RAG (ReAct / LangGraph)
│
├── "My data is in tables/databases"
│       → Structured Data RAG (Text-to-SQL) + optional hybrid with docs
│
└── "All of the above at enterprise scale"
        → Modular / Adaptive RAG with pluggable pipeline modules
```

### Architecture Comparison Matrix

| Architecture     | Retrieval       | Self-Correction | Multi-step | Structured Data | Latency  | Complexity | Best For                              |
| ---------------- | --------------- | --------------- | ---------- | --------------- | -------- | ---------- | ------------------------------------- |
| Naive RAG        | Single vector   | No              | No         | No              | Low      | Minimal    | Prototypes, simple Q&A                |
| Advanced RAG     | Hybrid + rerank | Partial         | No         | No              | Medium   | Medium     | Production Q&A, most use cases        |
| Hybrid RAG       | Dense + BM25    | No              | No         | No              | Low-Med  | Low        | Technical docs, mixed query types     |
| Graph RAG        | Graph traversal | No              | Implicit   | No              | Medium   | High       | Relationship queries, knowledge graphs|
| CRAG             | Internal + web  | Yes             | No         | No              | Medium   | Medium     | Stale corpora, mixed internal/web     |
| Adaptive RAG     | Dynamic         | Optional        | Optional   | Optional        | Variable | High       | Multi-domain systems, efficiency      |
| Multi-hop RAG    | Iterative       | No              | Yes        | No              | High     | Medium     | Chained reasoning, complex questions  |
| Agentic RAG      | Tool-based      | Implicit        | Yes        | Yes             | High     | Very High  | Complex task completion, mixed data   |

### Business Scenario Mapping

| Scenario                           | Recommended Architecture              | Why                                             |
| ---------------------------------- | ------------------------------------- | ----------------------------------------------- |
| HR policy chatbot (simple Q&A)     | Advanced RAG                          | Direct policy lookup, low complexity            |
| IT support (error codes + manuals) | Hybrid RAG + re-ranking               | Exact error codes + semantic context            |
| Legal contract analysis            | Multi-hop + Graph RAG                 | Clause cross-references, entity relationships   |
| Customer support (mixed internal/external) | CRAG                         | Internal FAQs + live product/price info         |
| Financial analytics bot            | Adaptive RAG (SQL + vector + web)     | Numbers from DB, context from docs, news from web|
| Compliance audit assistant         | Graph RAG + Multi-hop                 | Regulation → policy → evidence chains           |
| R&D research assistant             | Agentic RAG                           | Autonomous multi-source exploration             |
| Enterprise universal assistant     | Adaptive + Modular RAG                | Different domains need different strategies     |
| Sales enablement bot               | Hybrid RAG + CRAG                     | Internal playbooks + live competitor intel      |
| Medical knowledge assistant        | Graph RAG + Advanced RAG              | Drug-disease-symptom relationships              |

---

## 10. Lab: Choose the Right RAG Architecture for Business Scenarios

### Lab Objective

Analyse five business scenarios, justify the architecture choice, implement a minimal working version of each, and measure the performance difference vs a Naive RAG baseline.

### Setup

```bash
pip install langchain langchain-openai langchain-community langchain-experimental \
            langgraph chromadb sentence-transformers ragas networkx \
            tavily-python rank_bm25 datasets python-dotenv
```

### Scenario 1 — IT Helpdesk (Hybrid RAG)

**Scenario:** An IT support bot handles queries mixing error codes, system names, and conceptual questions about network configuration.

```python
# lab/scenario_1_it_helpdesk.py

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

IT_QUERIES = [
    "ERR_CONNECTION_REFUSED port 8080",           # exact code → BM25 wins
    "How do I configure nginx reverse proxy?",    # semantic → dense wins
    "TLS 1.2 handshake failed certificate chain", # mixed → hybrid wins
    "What is the VPN policy for remote workers?", # semantic
]

# Build baseline (dense only) and hybrid retrievers
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db_it", embedding_function=embeddings)

dense_ret = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_ret  = BM25Retriever.from_documents(it_chunks, k=5)
hybrid_ret = EnsembleRetriever(retrievers=[dense_ret, bm25_ret], weights=[0.6, 0.4])

# Compare retrieval recall
def evaluate_retriever(retriever, queries, golden):
    total, hits = 0, 0
    for q, relevant_ids in zip(queries, golden):
        docs = retriever.invoke(q)
        ids  = {d.metadata.get("doc_id","") for d in docs}
        hits += len(ids & set(relevant_ids))
        total += len(relevant_ids)
    return hits / total if total else 0

baseline_recall = evaluate_retriever(dense_ret,  IT_QUERIES, it_golden_ids)
hybrid_recall   = evaluate_retriever(hybrid_ret, IT_QUERIES, it_golden_ids)
print(f"Dense only recall: {baseline_recall:.2f}")
print(f"Hybrid recall:     {hybrid_recall:.2f}")
```

### Scenario 2 — Legal Contract Assistant (Multi-Hop)

**Scenario:** A legal assistant must answer questions that chain across multiple contract clauses, referencing definitions in one clause and obligations in another.

```python
# lab/scenario_2_legal_multihop.py

LEGAL_QUERIES = [
    "What are the penalty terms that apply to a Force Majeure event under this contract?",
    "Who bears the liability if the defined 'Confidential Information' is disclosed by a subcontractor?",
    "What notice period is required before invoking the termination clause defined in Schedule B?",
]

# Compare single-hop vs multi-hop
for query in LEGAL_QUERIES:
    print(f"\nQuery: {query}")
    
    # Naive single-pass
    naive_docs = vectorstore.similarity_search(query, k=4)
    naive_ans  = generate_answer(query, naive_docs)
    
    # Multi-hop
    multihop_result = multi_hop_rag(query, max_hops=4, k=3)
    
    print(f"  Single-hop ({len(naive_docs)} docs): {naive_ans[:150]}...")
    print(f"  Multi-hop  ({multihop_result['hop_count']} hops): {multihop_result['answer'][:150]}...")
    print(f"  Hops taken: {[h['sub_question'] for h in multihop_result['hops']]}")
```

### Scenario 3 — Customer Support (CRAG)

**Scenario:** A customer support bot answers questions about products, policies, and current promotions. The product catalogue and pricing change frequently; internal docs are always 1–2 weeks behind.

```python
# lab/scenario_3_customer_support_crag.py

SUPPORT_QUERIES = [
    "What is the current price of the Pro subscription plan?",   # likely stale → web
    "How do I cancel my subscription?",                          # policy → internal
    "Is there a Black Friday discount running right now?",       # time-sensitive → web
    "What is the refund process for annual plans?",             # policy → internal
]

for query in SUPPORT_QUERIES:
    result = run_crag(query)
    print(f"\nQuery: {query}")
    print(f"  Grade:  {result['retrieval_grade']}")
    print(f"  Answer: {result['answer'][:200]}...")
```

### Scenario 4 — HR Knowledge Bot (Adaptive RAG)

**Scenario:** An enterprise HR assistant handles everything from simple leave balance questions to complex multi-department policy chains to live salary benchmarking.

```python
# lab/scenario_4_hr_adaptive.py

HR_QUERIES = [
    "Hi, how are you?",                                          # no_retrieval
    "How many days of sick leave do I have?",                    # simple_vector
    "Explain the full parental leave policy including all eligibility criteria",  # multi_query
    "Who is the HR contact for the London Engineering team?",    # graph
    "What is the market rate for a Senior Data Engineer in London?",  # crag (web)
    "How many employees took parental leave last year?",         # sql
]

for query in HR_QUERIES:
    strategy, reasoning = classify_query(query)
    print(f"\nQuery:    {query}")
    print(f"Strategy: {strategy.value}")
    print(f"Reason:   {reasoning}")
    
    result = adaptive_rag(query)
    print(f"Answer:   {result['answer'][:150]}...")
```

### Scenario 5 — Research Intelligence Assistant (Agentic RAG)

**Scenario:** A research assistant must autonomously explore multiple sources to answer complex, multi-step research questions combining internal IP, public research, and structured data.

```python
# lab/scenario_5_research_agentic.py

RESEARCH_QUERIES = [
    "What internal patents do we hold related to transformer attention mechanisms, "
    "how many citations do those patents have externally, and which competitors "
    "have published similar work in the last 6 months?",
]

for query in RESEARCH_QUERIES:
    print(f"\nResearch query: {query}\n")
    result = agentic_rag(query)
    
    print("Tool calls made:")
    for step_tool, step_input in result["steps"]:
        print(f"  → {step_tool}: {str(step_input)[:80]}...")
    
    print(f"\nFinal answer: {result['answer']}")
```

### Lab Results Template

```
╔═══════════════════════════════════════════════════════════════════════╗
║              ARCHITECTURE SELECTION LAB RESULTS                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Scenario 1 — IT Helpdesk                                            ║
║  Architecture: Hybrid RAG                                            ║
║  Baseline Recall@5:  ____      Hybrid Recall@5: ____                 ║
║  Key insight: _______________________________________________         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Scenario 2 — Legal Contract                                         ║
║  Architecture: Multi-hop RAG                                         ║
║  Single-hop correctness: ____  Multi-hop correctness: ____           ║
║  Avg hops needed: ____                                               ║
║  Key insight: _______________________________________________         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Scenario 3 — Customer Support                                       ║
║  Architecture: CRAG                                                  ║
║  % queries needing web fallback: ____                                ║
║  Faithfulness with CRAG: ____   Without: ____                        ║
║  Key insight: _______________________________________________         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Scenario 4 — HR Bot                                                 ║
║  Architecture: Adaptive RAG                                          ║
║  Strategy distribution: no_retrieval:__% simple:__% graph:__%        ║
║  Cost saving vs always-complex: ____%                                ║
║  Key insight: _______________________________________________         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Scenario 5 — Research Assistant                                     ║
║  Architecture: Agentic RAG                                           ║
║  Avg tool calls per query: ____                                      ║
║  Tool distribution: docs:__% graph:__% web:__% sql:__%               ║
║  Key insight: _______________________________________________         ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Summary

The advanced RAG landscape is a spectrum from simple, cheap, and fast to complex, powerful, and expensive. The art of RAG architecture is matching complexity to the actual problem.

```
ARCHITECTURE SELECTION CHEAT SHEET:

  Got error codes / exact terms in queries?
      → Add Hybrid RAG (BM25 + dense)

  Got vague or ambiguous queries?
      → Add HyDE or Multi-query expansion (Advanced RAG)

  Got chained reasoning questions?
      → Multi-Hop RAG (IRCoT or iterative decomposition)

  Got entity relationship questions?
      → Graph RAG (NetworkX for PoC, Neo4j for production)

  Got stale corpus or out-of-scope questions?
      → CRAG with web search fallback

  Got multiple query types needing different strategies?
      → Adaptive RAG with query classifier + router

  Got complex tasks needing tool orchestration?
      → Agentic RAG with guardrails (max iterations, permitted tools)

  General production default (covers 80% of enterprise needs):
      → Advanced RAG + Hybrid Search + Cross-encoder Re-ranking
        (optimise this first before adding any of the above)
```

**Key takeaways:**

1. **Start simple, measure, then advance** — every architecture above Naive RAG adds cost and latency; only add it when you have a measured failure at the current level.
2. **Hybrid RAG is the universal retrieval baseline** — it covers keyword AND semantic gaps with minimal complexity overhead; make it your default over pure vector search.
3. **CRAG and Adaptive RAG are about resource efficiency** — CRAG avoids hallucination from stale docs; Adaptive RAG avoids over-engineering simple queries.
4. **Multi-hop and Graph RAG solve fundamentally different problems** — Multi-hop is for sequential reasoning; Graph RAG is for relationship traversal. Choose based on the query type.
5. **Agentic RAG is powerful but expensive and hard to test** — reserve it for genuinely complex, multi-tool scenarios; for most enterprise knowledge Q&A, Advanced RAG + re-ranking is sufficient and more reliable.

---

_Guide maintained as part of the **Integrating Generative AI — Advanced RAG & Enterprise Patterns (Level 2)** training programme._
