# Module-2: RAG Optimization Techniques

A comprehensive guide to maximising retrieval relevance, generation quality, and production efficiency in RAG systems — covering chunking strategies, embedding selection, hybrid search, re-ranking, prompt engineering, and cost/latency optimization.

---

## Table of Contents

1. [Why Optimization Matters](#1-why-optimization-matters)
2. [Chunking Strategies](#2-chunking-strategies)
3. [Embedding Model Selection](#3-embedding-model-selection)
4. [Hybrid Search: Keyword + Vector](#4-hybrid-search-keyword--vector)
5. [Re-Ranking Models](#5-re-ranking-models)
6. [Prompt Optimization for RAG](#6-prompt-optimization-for-rag)
7. [Cost and Latency Optimization](#7-cost-and-latency-optimization)
8. [Optimization Decision Framework](#8-optimization-decision-framework)
9. [Lab: Improve Retrieval Relevance Using Chunking and Re-Ranking](#9-lab-improve-retrieval-relevance-using-chunking-and-re-ranking)
10. [Optimization Benchmarking Template](#10-optimization-benchmarking-template)

---

## 1. Why Optimization Matters

### The Performance Gap Between Demo and Production

A naive RAG system built in a day works acceptably on hand-picked queries. In production it encounters:

- **Diverse query phrasings** — the same intent expressed dozens of different ways
- **Domain-specific terminology** — acronyms, product names, code identifiers that generic embeddings handle poorly
- **Scale** — millions of document chunks where approximate retrieval degrades
- **Cost pressure** — GPT-4o on every query becomes unaffordable at scale
- **Latency requirements** — users expect < 2 s; naive pipelines often take 4–8 s

```
Naive RAG (demo quality):                Optimised RAG (production quality):

  P@5  = 0.42 (42% of retrieved = relevant)   P@5  = 0.79
  R@5  = 0.51 (51% of relevant found)         R@5  = 0.88
  Faithfulness = 0.71                          Faithfulness = 0.93
  Latency p95  = 6.2 s                         Latency p95  = 1.8 s
  Cost / 1k Q  = $4.20                         Cost / 1k Q  = $0.95
```

### The RAG Optimization Stack

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 6 — Cost & Latency          caching, smaller models   │
├──────────────────────────────────────────────────────────────┤
│  LAYER 5 — Prompt Optimization     grounding, few-shot, CoT  │
├──────────────────────────────────────────────────────────────┤
│  LAYER 4 — Re-Ranking              cross-encoder, LLM-judge  │
├──────────────────────────────────────────────────────────────┤
│  LAYER 3 — Hybrid Search           dense + sparse (BM25)     │
├──────────────────────────────────────────────────────────────┤
│  LAYER 2 — Embedding Model         domain fit, dimensionality │
├──────────────────────────────────────────────────────────────┤
│  LAYER 1 — Chunking Strategy       size, overlap, semantic   │
└──────────────────────────────────────────────────────────────┘
         ▲ Each layer builds on the one below.
         Start at Layer 1; only add higher layers when lower ones are solid.
```

### Optimization Principle

> Fix retrieval before fixing generation. A hallucinating LLM given perfect context is a prompt problem (cheap to fix). A correct LLM given wrong context produces wrong answers no matter what prompt you use.

---

## 2. Chunking Strategies

Chunking is the single highest-leverage optimization in most RAG systems. The chunk is the unit of retrieval — its size and boundaries directly determine what context the LLM sees.

### The Chunking Problem

```
DOCUMENT:

"Section 3.2 — Refund Policy
All physical products may be returned within 30 days of purchase for a full
refund, provided the item is unused and in its original packaging. Digital
downloads are non-refundable once accessed. For defective items, the return
window extends to 90 days.

Section 3.3 — Exchange Policy
Exchanges are accepted within 60 days. The customer is responsible for
return shipping costs unless the item is defective."

QUERY: "Can I exchange a defective item and who pays for shipping?"

The answer spans both Section 3.2 (defective = 90-day window) and
Section 3.3 (defective → we pay shipping). A naive fixed-size chunk
that cuts at 200 tokens may split these sections, losing the connection.
```

### Strategy 1 — Fixed-Size Chunking

Split documents into chunks of exactly N tokens (or characters), with an optional overlap to preserve context across boundaries.

```
Document:  [████████████████████████████████████████████]
           [  chunk 1  ][  chunk 2  ][  chunk 3  ][chunk4]
           [    overlap ][    overlap][    overlap ]
```

```python
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter

# Character-based fixed split
char_splitter = CharacterTextSplitter(
    chunk_size=1000,       # characters
    chunk_overlap=200,     # overlap between consecutive chunks
    separator="\n\n",      # prefer to split at paragraph breaks
)

# Token-based fixed split (more accurate for LLM context windows)
token_splitter = TokenTextSplitter(
    chunk_size=256,        # tokens
    chunk_overlap=32,
)

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("policy_document.pdf")
pages = loader.load()

chunks = char_splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
```

**Chunk size guide:**

| Document type           | Recommended chunk size | Overlap  |
| ----------------------- | ---------------------- | -------- |
| Short FAQs / policies   | 256–512 tokens         | 32–64    |
| Technical documentation | 512–1024 tokens        | 64–128   |
| Long-form reports       | 1024–2048 tokens       | 128–256  |
| Code files              | 512 tokens (by function) | minimal |

**Strengths:** Simple, predictable, fast to index.  
**Weaknesses:** Ignores document structure; chunks may split mid-sentence or mid-concept.

---

### Strategy 2 — Recursive Character Splitting

Tries a hierarchy of separators in order (`\n\n`, `\n`, `. `, ` `, `""`) and only falls back to a finer separator when a chunk still exceeds the target size. This preserves natural language boundaries as much as possible.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=[
        "\n\n",    # paragraph break — try first
        "\n",      # line break
        ". ",      # sentence end
        ", ",      # clause
        " ",       # word
        "",        # character (last resort)
    ],
)

chunks = splitter.split_documents(pages)
```

This is the **default recommended strategy** for most general text. It respects paragraph and sentence structure without requiring any document parsing.

---

### Strategy 3 — Semantic Chunking

Groups sentences into chunks based on **semantic similarity** rather than character count. A new chunk begins when the topic shifts (measured by a jump in embedding distance between consecutive sentences).

```
Document sentences:

  s1: "Refunds are processed within 5 business days."
  s2: "The amount is returned to the original payment method."
  s3: "For bank transfers, allow up to 10 business days."
  s4: "Exchange requests must be submitted via the portal."  ← topic shift
  s5: "Attach the original invoice to the exchange request."

Semantic chunks:
  Chunk A: [s1, s2, s3]  — all about refunds (high cosine similarity)
  Chunk B: [s4, s5]      — topic shifts to exchanges
```

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",   # split where similarity drops below Nth percentile
    breakpoint_threshold_amount=90,            # split at the bottom 10% similarity jumps
)

chunks = semantic_splitter.split_documents(pages)

# Inspect where topic shifts were detected
for i, chunk in enumerate(chunks[:5]):
    print(f"Chunk {i}: {len(chunk.page_content)} chars | Preview: {chunk.page_content[:80]}...")
```

**Threshold tuning:**

| `breakpoint_threshold_amount` | Effect                                     |
| ----------------------------- | ------------------------------------------ |
| 70–80                         | Many splits — small, highly focused chunks |
| 90 (recommended)              | Balanced — splits at clear topic changes   |
| 95+                           | Few splits — large, multi-topic chunks     |

**Strengths:** Chunks are topically coherent; retrieval precision improves significantly.  
**Weaknesses:** Requires an embedding call per sentence during indexing (slower, costlier to index); needs `langchain_experimental`.

---

### Strategy 4 — Parent-Child Retrieval (Small-to-Big)

Store **small child chunks** in the vector store for precise retrieval, but return the **larger parent chunk** to the LLM for richer context.

```
INDEXING:

  Parent chunk (1000 tokens):
  "Section 4: Data Retention Policy. Personal data is retained for..."
     │
     ├── Child chunk A (200 tokens): "Personal data is retained for..."
     ├── Child chunk B (200 tokens): "Backups are deleted within 30 days..."
     └── Child chunk C (200 tokens): "Audit logs are retained for 7 years..."

RETRIEVAL:

  Query: "How long are audit logs kept?"
  → Vector search matches Child chunk C (precise semantic match)
  → System fetches Parent chunk (full section 4 context)
  → LLM sees 1000-token parent — full policy context

  Result: precise retrieval + rich generation context
```

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define parent and child splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
child_splitter  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(collection_name="child_chunks", embedding_function=embeddings)
docstore    = InMemoryStore()  # stores parent chunks (use Redis/MongoDB for production)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Index — stores child embeddings in vectorstore, parents in docstore
retriever.add_documents(pages)

# Retrieve — queries child chunks, returns parent chunks
results = retriever.invoke("How long are audit logs kept?")
print(f"Retrieved {len(results)} parent chunks")
print(f"Parent chunk size: {len(results[0].page_content)} chars")
```

**Strengths:** Best of both worlds — precise retrieval + comprehensive context.  
**Weaknesses:** Requires two storage layers; docstore must persist between restarts (use Redis or a database in production).

---

### Chunking Strategy Comparison

| Strategy         | Precision | Recall | Indexing Cost | Complexity | Best For                              |
| ---------------- | --------- | ------ | ------------- | ---------- | ------------------------------------- |
| Fixed-size       | Low       | Medium | Very low      | Minimal    | Prototyping, uniform docs             |
| Recursive char   | Medium    | Medium | Low           | Low        | General text (default starting point) |
| Semantic         | High      | High   | Medium        | Medium     | Long-form, topic-diverse docs         |
| Parent-child     | High      | High   | Medium        | High       | Rich docs needing context + precision |

### Chunking Best Practices

```
1. Always add metadata to chunks:
   - source filename, page number, section title, doc_id
   - Used for attribution and filtering

2. Measure chunk quality with Recall@K before and after changes.
   Don't guess — A/B test with your evaluation dataset.

3. Overlap helps at boundaries but costs tokens:
   - overlap = 10–20% of chunk size is a good default
   - Too much overlap → duplicate context, higher cost

4. Pre-clean your documents:
   - Remove headers/footers (page 1 of 43)
   - Remove table-of-contents pages
   - Normalise whitespace

5. Consider document type:
   - PDFs with columns → use unstructured.io for layout-aware parsing
   - Code → split by function/class, not by character count
   - Markdown → split by heading (##, ###)
```

---

## 3. Embedding Model Selection

The embedding model determines the quality of semantic similarity scores. A mismatch between the model's training domain and your documents directly degrades retrieval.

### How Embeddings Affect Retrieval

```
Query: "myocardial infarction treatment protocol"

Generic model (trained on web text):
  → Retrieves "heart problems" docs (semantic similarity: 0.72)
  → Misses "STEMI management pathway" doc (similarity: 0.61) ← wrong

Medical embedding model (PubMedBERT-based):
  → Retrieves "STEMI management pathway" (similarity: 0.91) ← correct
  → Also retrieves "MI thrombolysis guidelines" (similarity: 0.88)
```

### Embedding Model Landscape

| Model                              | Provider   | Dimensions | Context    | Domain    | Relative Cost |
| ---------------------------------- | ---------- | ---------- | ---------- | --------- | ------------- |
| `text-embedding-3-small`           | OpenAI     | 1536       | 8192 tok   | General   | Low           |
| `text-embedding-3-large`           | OpenAI     | 3072       | 8192 tok   | General   | Medium        |
| `embed-english-v3.0`               | Cohere     | 1024       | 512 tok    | General   | Low           |
| `embed-multilingual-v3.0`          | Cohere     | 1024       | 512 tok    | Multi-lang| Low           |
| `all-MiniLM-L6-v2`                 | HuggingFace| 384        | 256 tok    | General   | Free (local)  |
| `all-mpnet-base-v2`                | HuggingFace| 768        | 384 tok    | General   | Free (local)  |
| `BAAI/bge-large-en-v1.5`           | BAAI       | 1024       | 512 tok    | General   | Free (local)  |
| `pritamdeka/PubMedBERT-mnli-snli`  | HuggingFace| 768        | 512 tok    | Medical   | Free (local)  |
| `nlpaueb/legal-bert-base-uncased`  | HuggingFace| 768        | 512 tok    | Legal     | Free (local)  |

### Benchmarking Embedding Models on Your Data

Do not choose an embedding model based on general benchmarks (MTEB). Benchmark on **your documents with your queries**.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np

# Load your golden dataset
import json
with open("evaluation/golden_dataset_v1.0.json") as f:
    golden = json.load(f)["samples"]

queries = [s["question"] for s in golden]
relevant_ids = [s["relevant_doc_ids"] for s in golden]

def recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant) / len(relevant)

def benchmark_embedding_model(embedding_model, model_name: str, documents, k: int = 5):
    # Build a fresh vectorstore with this model
    vectorstore = Chroma.from_documents(
        documents,
        embedding_model,
        collection_name=f"bench_{model_name}",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    recalls = []
    for query, rel_ids in zip(queries, relevant_ids):
        results = retriever.invoke(query)
        retrieved_ids = [doc.metadata.get("doc_id", "") for doc in results]
        recalls.append(recall_at_k(retrieved_ids, set(rel_ids), k))

    mean_recall = np.mean(recalls)
    print(f"{model_name:40s}  Recall@{k}: {mean_recall:.3f}")
    return mean_recall

# Compare models
models = {
    "text-embedding-3-small":  OpenAIEmbeddings(model="text-embedding-3-small"),
    "text-embedding-3-large":  OpenAIEmbeddings(model="text-embedding-3-large"),
    "all-MiniLM-L6-v2":        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    "BAAI/bge-large-en-v1.5":  HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5"),
}

print("=== Embedding Model Benchmark ===")
results = {name: benchmark_embedding_model(model, name, documents) for name, model in models.items()}
best = max(results, key=results.get)
print(f"\nBest model for your data: {best} (Recall@5={results[best]:.3f})")
```

### Dimension Reduction (Matryoshka Embeddings)

OpenAI's `text-embedding-3-*` models support **Matryoshka Representation Learning** — you can truncate the embedding to fewer dimensions and trade accuracy for storage/speed.

```python
from langchain_openai import OpenAIEmbeddings

# Full 1536 dimensions — highest accuracy, highest storage
full_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=1536
)

# Reduced to 512 dimensions — 3x smaller, ~2% accuracy loss
reduced_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

# Reduced to 256 dimensions — 6x smaller, ~5% accuracy loss
compact_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=256
)
```

**When to reduce dimensions:**
- Vector store storage is expensive (millions of chunks)
- Retrieval latency is critical
- You've verified < 5% recall drop on your benchmark

### Embedding Model Selection Guide

```
START: What is your primary constraint?
│
├── "Cost is critical, latency is critical"
│       → all-MiniLM-L6-v2 (free, local, fast)
│       → or text-embedding-3-small with 512 dimensions
│
├── "I need the best possible accuracy"
│       → BAAI/bge-large-en-v1.5 (free, strong on MTEB)
│       → or text-embedding-3-large
│
├── "My documents are in a specific domain"
│   ├── Medical → pritamdeka/PubMedBERT-mnli-snli
│   ├── Legal   → nlpaueb/legal-bert-base-uncased
│   ├── Code    → microsoft/codebert-base
│   └── Finance → ProsusAI/finbert (+ fine-tune on your corpus)
│
├── "Multiple languages"
│       → Cohere embed-multilingual-v3.0
│       → or intfloat/multilingual-e5-large
│
└── "I don't know yet"
        → Benchmark text-embedding-3-small vs BAAI/bge-large-en-v1.5
          on your golden dataset. Pick the winner.
```

---

## 4. Hybrid Search: Keyword + Vector

### Why Pure Vector Search Falls Short

Dense vector search captures semantic meaning but fails on:

- **Exact terminology** — product codes, error messages, regulatory identifiers, names
- **Rare words** — newly coined terms not well-represented in training data
- **Short queries** — "GDPR Article 17" has very little semantic signal to embed

BM25 (sparse/keyword search) captures exact matches but fails on:

- **Paraphrasing** — "late payment consequences" vs "penalty for missed payment"
- **Synonyms** — "myocardial infarction" vs "heart attack"
- **Long queries** — diluted keyword signal

Hybrid search combines both, using **Reciprocal Rank Fusion (RRF)** to merge the ranked lists.

### Reciprocal Rank Fusion (RRF)

```
RRF_score(document d) = Σ 1 / (k + rank_r(d))
                         r ∈ rankers

Where k = 60 (constant smoothing factor)
      rank_r(d) = position of d in ranker r's result list

Higher RRF score = document ranks high in more rankers
```

**Example:**

```
Query: "SSL handshake error TLS 1.2"

BM25 ranking:   doc_A(rank 1), doc_C(rank 2), doc_B(rank 7)
Dense ranking:  doc_B(rank 1), doc_A(rank 2), doc_D(rank 3)

RRF scores (k=60):
  doc_A: 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252  ← winner
  doc_B: 1/(60+7) + 1/(60+1) = 0.01493 + 0.01639 = 0.03132
  doc_C: 1/(60+2)             = 0.01613
  doc_D: 1/(60+3)             = 0.01587
```

### Implementation with LangChain

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk documents
loader = DirectoryLoader("./docs", glob="**/*.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Dense retriever (semantic)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Sparse retriever (BM25 / keyword)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 10

# Hybrid retriever (RRF fusion)
hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],   # favour semantic slightly; tune on your evaluation dataset
)

# Query
results = hybrid_retriever.invoke("SSL handshake failure ERR_SSL_PROTOCOL_ERROR")
print(f"Retrieved {len(results)} chunks via hybrid search")
```

### Tuning Hybrid Weights

```python
import numpy as np

# Grid search over weight combinations using your golden dataset
weight_grid = [(0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]

best_recall = 0
best_weights = None

for w_dense, w_sparse in weight_grid:
    retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[w_dense, w_sparse],
    )
    
    recalls = []
    for sample in golden_dataset:
        results = retriever.invoke(sample["question"])
        retrieved_ids = [doc.metadata.get("doc_id", "") for doc in results]
        recalls.append(recall_at_k(retrieved_ids, set(sample["relevant_doc_ids"]), k=5))
    
    mean_recall = np.mean(recalls)
    print(f"Dense={w_dense}, Sparse={w_sparse}  →  Recall@5={mean_recall:.3f}")
    
    if mean_recall > best_recall:
        best_recall = mean_recall
        best_weights = (w_dense, w_sparse)

print(f"\nBest weights: dense={best_weights[0]}, sparse={best_weights[1]}")
```

### Hybrid Search with Pinecone (Production)

For production at scale, vector databases with native hybrid support eliminate the need for a separate BM25 index.

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key="your_key")

# Create an index with dotproduct metric (required for hybrid)
pc.create_index(
    name="rag-hybrid",
    dimension=1536,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

index = pc.Index("rag-hybrid")

# Upsert with sparse values (BM25 computed separately)
from pinecone_text.sparse import BM25Encoder

bm25 = BM25Encoder().default()
bm25.fit([chunk.page_content for chunk in chunks])

def upsert_hybrid(chunks, embeddings_model, bm25_encoder, index):
    dense_vecs = embeddings_model.embed_documents([c.page_content for c in chunks])
    sparse_vecs = bm25_encoder.encode_documents([c.page_content for c in chunks])
    
    vectors = []
    for i, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        vectors.append({
            "id": chunk.metadata.get("doc_id", str(i)),
            "values": dense,
            "sparse_values": sparse,
            "metadata": {"text": chunk.page_content, **chunk.metadata},
        })
    
    index.upsert(vectors=vectors)

# Query with hybrid
def hybrid_query(query: str, alpha: float = 0.7, top_k: int = 5):
    dense_vec = embeddings.embed_query(query)
    sparse_vec = bm25.encode_queries(query)
    
    # alpha controls dense vs sparse balance
    # alpha=1.0: pure dense, alpha=0.0: pure sparse
    results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True,
        alpha=alpha,
    )
    return results
```

### When Hybrid Search Helps Most

| Query type                        | Dense alone | BM25 alone | Hybrid |
| --------------------------------- | ----------- | ---------- | ------ |
| Semantic paraphrase               | ✅          | ❌         | ✅     |
| Exact error code / product SKU    | ❌          | ✅         | ✅     |
| Person / company name             | ❌          | ✅         | ✅     |
| Conceptual question               | ✅          | ❌         | ✅     |
| Mixed (acronym + concept)         | ❌          | ❌         | ✅     |

---

## 5. Re-Ranking Models

Retrieval returns candidates — re-ranking reorders them by **joint query-document relevance**. It operates as a post-retrieval filter: retrieve more (K=20), re-rank, keep top (k=4).

### Why Retrieval Alone Is Not Enough

```
Retrieval (bi-encoder):
  Encodes query and document SEPARATELY, then computes similarity.
  Fast (pre-computed document embeddings), but approximate.

  query vector ────────────────┐
                               ├── cosine_sim() → score
  document vector (cached) ───┘

Re-ranking (cross-encoder):
  Encodes query AND document TOGETHER — sees their interaction.
  Slower (no pre-computation), but much more accurate.

  [query + document] → CrossEncoder → relevance score 0-1
```

### Cross-Encoder Re-Ranking

```python
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load a cross-encoder (runs locally, no API cost)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def retrieve_and_rerank(query: str, initial_k: int = 20, final_k: int = 4) -> list:
    # Step 1: retrieve broadly
    candidates = vectorstore.similarity_search(query, k=initial_k)
    
    # Step 2: re-rank with cross-encoder
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = cross_encoder.predict(pairs)
    
    # Step 3: sort by cross-encoder score, keep top final_k
    ranked = sorted(zip(scores, candidates), reverse=True)
    top_docs = [doc for _, doc in ranked[:final_k]]
    
    # Debug: show score improvement
    for score, doc in ranked[:final_k]:
        print(f"Score: {score:.3f} | {doc.page_content[:80]}...")
    
    return top_docs

# Usage
results = retrieve_and_rerank("What is the 30-day return policy for electronics?")
```

### Available Cross-Encoder Models

| Model                                     | Speed   | Accuracy | Use Case                     |
| ----------------------------------------- | ------- | -------- | ---------------------------- |
| `cross-encoder/ms-marco-MiniLM-L-6-v2`   | Fast    | Good     | General English, production  |
| `cross-encoder/ms-marco-MiniLM-L-12-v2`  | Medium  | Better   | General English, quality     |
| `cross-encoder/ms-marco-electra-base`    | Slow    | Best     | Highest accuracy requirement |
| `BAAI/bge-reranker-large`                | Medium  | Very good| Multilingual + English       |
| `mixedbread-ai/mxbai-rerank-large-v1`    | Medium  | Very good| General-purpose              |

### LLM-Based Re-Ranking

For domains where cross-encoders were not trained on similar text, use an LLM to score relevance.

```python
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rerank_prompt = """You are a relevance evaluator. Given a query and a document excerpt,
score how relevant the document is to answering the query.

Query: {query}

Document excerpt:
{document}

Return JSON with a single key "score" (integer 1-10, where 10 = perfectly relevant).
Only return the JSON, nothing else."""

def llm_rerank(query: str, candidates: list, final_k: int = 4) -> list:
    scored = []
    for doc in candidates:
        result = llm.invoke(rerank_prompt.format(
            query=query,
            document=doc.page_content[:500]  # truncate for cost control
        ))
        score = json.loads(result.content)["score"]
        scored.append((score, doc))
    
    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:final_k]]
```

### LangChain Contextual Compression Re-Ranker

LangChain's `ContextualCompressionRetriever` chains retrieval and re-ranking in one call.

```python
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder as a LangChain compressor
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=4)

# Wrap base retriever with re-ranking
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Use exactly like any retriever
results = reranking_retriever.invoke("Return policy for damaged goods")
print(f"Retrieved and re-ranked: {len(results)} documents")
```

### Re-Ranking Impact on MRR

```
Without re-ranking (K=5 from vector search):
  Query 1: relevant doc at rank 3 → MRR contribution: 0.333
  Query 2: relevant doc at rank 5 → MRR contribution: 0.200
  Query 3: relevant doc at rank 1 → MRR contribution: 1.000
  Overall MRR: 0.511

With cross-encoder re-ranking (retrieve 20, return top 5):
  Query 1: relevant doc moved to rank 1 → 1.000
  Query 2: relevant doc moved to rank 2 → 0.500
  Query 3: relevant doc stays at rank 1 → 1.000
  Overall MRR: 0.833   ← +63% improvement
```

### Cost-Accuracy Trade-off for Re-Ranking

```
Option A — No re-ranking
  Cost: 0 extra API calls
  Latency: +0 ms
  MRR: 0.51 (baseline)

Option B — Local cross-encoder (ms-marco-MiniLM-L-6-v2)
  Cost: 0 API calls (runs on CPU)
  Latency: +80–200 ms for 20 docs
  MRR: 0.83 (+63%)
  ✅ Best cost-performance ratio for most production systems

Option C — LLM re-ranker (gpt-4o-mini, 20 docs)
  Cost: ~20 LLM calls per query
  Latency: +500–1500 ms
  MRR: 0.87 (+70%)
  Use only when domain is highly specialised and cross-encoders underperform
```

---

## 6. Prompt Optimization for RAG

The prompt is the interface between your retrieved context and the LLM's output. A poorly designed RAG prompt wastes good retrieval and produces hallucinated, irrelevant, or poorly structured answers.

### RAG Prompt Anatomy

```
┌──────────────────────────────────────────────────────────┐
│  SYSTEM PROMPT                                           │
│  Role definition + behavioural constraints               │
│  "You are a helpful HR assistant..."                     │
│  "Answer ONLY using the context below."                  │
├──────────────────────────────────────────────────────────┤
│  CONTEXT BLOCK                                           │
│  Retrieved document chunks                               │
│  Formatted clearly, with source attribution              │
├──────────────────────────────────────────────────────────┤
│  FEW-SHOT EXAMPLES  (optional)                           │
│  1-3 examples of correct Q&A behaviour                   │
├──────────────────────────────────────────────────────────┤
│  USER QUESTION                                           │
│  The actual query                                        │
├──────────────────────────────────────────────────────────┤
│  OUTPUT FORMAT INSTRUCTION  (optional)                   │
│  "Answer in bullet points." / "Return JSON."             │
└──────────────────────────────────────────────────────────┘
```

### Baseline RAG Prompt (Anti-Pattern)

```python
# DON'T do this — no grounding instruction, no structure
bad_prompt = ChatPromptTemplate.from_template("""
{context}
{question}""")
```

Problems:
- LLM doesn't know it must use only the context
- No separator between context and question
- No instruction on how to handle cases where context is insufficient

### Optimised RAG Prompt

```python
from langchain_core.prompts import ChatPromptTemplate

rag_system_prompt = """You are a precise, factual assistant. Your task is to answer
the user's question using ONLY the information provided in the CONTEXT section below.

Rules:
1. If the answer is in the context, answer directly and cite the relevant section.
2. If the context does not contain sufficient information, say:
   "I don't have enough information in the provided documents to answer this accurately."
   Do NOT guess or use general knowledge.
3. Never contradict the context.
4. Be concise. Do not repeat the question back.

CONTEXT:
{context}"""

rag_human_prompt = "Question: {question}"

prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("human", rag_human_prompt),
])
```

### Context Formatting

How you format the retrieved chunks inside the prompt matters for LLM comprehension.

```python
def format_docs_basic(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def format_docs_with_sources(docs: list) -> str:
    """Numbered sections with source attribution — best for citation."""
    sections = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "")
        header = f"[Source {i}: {source}" + (f", page {page}" if page else "") + "]"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

def format_docs_xml(docs: list) -> str:
    """XML tags — helps models like Claude distinguish document boundaries."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "doc")
        parts.append(f'<document index="{i}" source="{source}">\n{doc.page_content}\n</document>')
    return "\n".join(parts)

# Example output of format_docs_with_sources:
"""
[Source 1: hr_policy.pdf, page 12]
Annual leave entitlement is 25 days for full-time employees...

---

[Source 2: hr_policy.pdf, page 13]
Carryover of unused leave is permitted up to a maximum of 5 days...
"""
```

### Handling "I Don't Know" Correctly

```python
no_answer_prompt = """You are a factual assistant. Answer based strictly on the CONTEXT below.

CONTEXT:
{context}

QUESTION: {question}

Instructions:
- If the context answers the question: provide the answer, referencing the relevant context.
- If the context is partially relevant: answer what you can and explicitly state what is missing.
- If the context is completely irrelevant: respond with exactly:
  "The provided documents do not contain information about [topic]. 
   Please consult [appropriate resource]."

Do not fabricate information. Do not use knowledge outside the provided context."""
```

### Few-Shot Prompting for RAG

Add 1–2 examples to show the LLM the desired answer format and citation style.

```python
few_shot_rag_prompt = """You are a precise HR policy assistant.

CONTEXT:
{context}

Here are examples of correct answers:

Example 1:
Q: How many days of sick leave do employees get?
A: Employees receive 10 days of paid sick leave per calendar year [Source 1, page 4].
   Unused sick leave does not carry over to the next year.

Example 2:
Q: Can contractors claim the employee gym benefit?
A: The provided documents do not contain information about contractor gym benefits.
   Please consult the contractor agreement or contact HR directly.

Now answer this question:
Q: {question}
A:"""
```

### Chain-of-Thought (CoT) for Complex RAG Questions

For multi-step questions requiring synthesis across chunks, instruct the LLM to reason before answering.

```python
cot_rag_prompt = """You are an expert assistant. Answer the question step by step
using only the provided context.

CONTEXT:
{context}

QUESTION: {question}

Reasoning process:
1. Identify which parts of the context are relevant to the question.
2. Extract the key facts from the relevant context.
3. Synthesise the facts to form a complete answer.
4. Verify your answer is fully supported by the context.

Answer (after completing the above reasoning):"""
```

### Prompt Optimization Experiments

Always A/B test prompt changes against your golden dataset before deploying.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from datasets import Dataset

def evaluate_prompt(prompt_template, rag_chain_builder, golden_dataset):
    chain = rag_chain_builder(prompt_template)
    
    answers, contexts = [], []
    for sample in golden_dataset:
        output = chain.invoke(sample["question"])
        answers.append(output["result"])
        contexts.append([doc.page_content for doc in output["source_documents"]])
    
    dataset = Dataset.from_dict({
        "question":     [s["question"] for s in golden_dataset],
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": [s["ground_truth_answer"] for s in golden_dataset],
    })
    
    return evaluate(dataset, metrics=[faithfulness, answer_correctness])

# Compare prompts
score_baseline = evaluate_prompt(bad_prompt, build_chain, golden_dataset)
score_optimised = evaluate_prompt(prompt, build_chain, golden_dataset)

print(f"Baseline   faithfulness={score_baseline['faithfulness']:.2f}")
print(f"Optimised  faithfulness={score_optimised['faithfulness']:.2f}")
```

---

## 7. Cost and Latency Optimization

### Where Time and Money Go in a RAG Pipeline

```
RAG request breakdown (typical):

  1. Query embedding              20–50 ms      $0.00002
  2. Vector search                50–200 ms     $0.00 (local) / $0.001 (managed)
  3. [Optional] Re-ranking        80–300 ms     $0.00 (local cross-encoder)
  4. LLM generation (gpt-4o)      800–3000 ms   $0.01–$0.05
  5. Response serialisation       5–20 ms       $0.00

  Total (naive):  ~2–4 s,  ~$0.05 per query
  At 100K queries/month: $5,000/month just on LLM calls
```

### Optimization 1 — Semantic Caching

Cache LLM responses for semantically similar queries. Instead of exact-match caching, encode the query and check if a cached response exists within a similarity threshold.

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
import langchain

# Configure semantic cache (requires Redis)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
langchain.llm_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embeddings,
    score_threshold=0.95,  # queries with cosine similarity > 0.95 share cached response
)

# Now any LLM call is automatically cached
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# First call: hits LLM, stores in cache
response_1 = llm.invoke("What is the return policy?")

# Second call with similar query: hits cache (near-instant, $0 cost)
response_2 = llm.invoke("What's the returns policy?")  # cached!
```

**Expected impact:** 20–40% cache hit rate on typical Q&A workloads, reducing LLM cost proportionally.

### Optimization 2 — LLM Model Tiering

Route queries to cheaper models when possible, expensive models only when necessary.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

complexity_prompt = ChatPromptTemplate.from_template("""
Classify the complexity of this question for a RAG system:
- "simple": factual lookup, single document, direct answer
- "complex": requires reasoning across multiple documents or multi-step inference

Question: {question}

Answer with one word only: simple or complex""")

def route_to_model(question: str) -> ChatOpenAI:
    complexity = (complexity_prompt | classifier_llm).invoke({"question": question})
    
    if complexity.content.strip().lower() == "simple":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)   # $0.15/M tokens
    else:
        return ChatOpenAI(model="gpt-4o", temperature=0)         # $2.50/M tokens

def cost_aware_rag(question: str, context: str) -> str:
    llm = route_to_model(question)
    response = llm.invoke(prompt.format(context=context, question=question))
    return response.content
```

**Expected impact:** If 70% of queries are simple, model tiering reduces LLM cost by ~60%.

### Optimization 3 — Context Compression

Reduce the number of tokens sent to the LLM by stripping irrelevant sentences from retrieved chunks before injection.

```python
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import ChatOpenAI

# Compressor extracts only the sentences relevant to the query
compressor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
compressor = LLMChainExtractor.from_llm(compressor_llm)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Retrieved chunks are compressed before being passed to the main LLM
compressed_docs = compressed_retriever.invoke("What is the equipment allowance for remote workers?")

# Before compression: 6 chunks × 800 tokens = 4800 tokens → $0.012
# After compression:  6 chunks × 120 tokens  =  720 tokens → $0.002 (83% reduction)
```

### Optimization 4 — Async Parallel Retrieval

For modular RAG systems querying multiple sources, run retrievals in parallel.

```python
import asyncio
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

hr_store      = Chroma(collection_name="hr_docs",      embedding_function=embeddings)
finance_store = Chroma(collection_name="finance_docs",  embedding_function=embeddings)
it_store      = Chroma(collection_name="it_docs",       embedding_function=embeddings)

async def retrieve_from_store(store, query: str, k: int = 4):
    return await store.asimilarity_search(query, k=k)

async def parallel_retrieval(query: str) -> list:
    # All three retrievals run simultaneously
    results = await asyncio.gather(
        retrieve_from_store(hr_store, query),
        retrieve_from_store(finance_store, query),
        retrieve_from_store(it_store, query),
    )
    # Flatten and deduplicate
    all_docs = [doc for store_results in results for doc in store_results]
    return all_docs

# Sequential: 3 × 150ms = 450ms
# Parallel:   max(150ms, 150ms, 150ms) = 150ms  ← 3x faster
```

### Optimization 5 — Retrieval Result Caching

Cache vector search results for repeated or near-identical queries (separate from LLM caching).

```python
import hashlib
import json
import redis

r = redis.Redis(host="localhost", port=6379, db=1)

def get_cache_key(query: str, k: int) -> str:
    return f"retrieval:{hashlib.md5(f'{query}:{k}'.encode()).hexdigest()}"

def cached_retrieve(query: str, retriever, k: int = 5, ttl: int = 3600) -> list:
    cache_key = get_cache_key(query, k)
    
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # Return cached chunk IDs
    
    results = retriever.invoke(query)
    doc_ids = [doc.metadata.get("doc_id") for doc in results]
    
    r.setex(cache_key, ttl, json.dumps(doc_ids))
    return results
```

### Optimization 6 — Streaming for Perceived Latency

Streaming doesn't reduce actual latency but dramatically improves **perceived** latency — users see output immediately rather than waiting for the full response.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

rag_chain = (
    {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
    | prompt
    | llm
)

@app.get("/chat")
async def stream_chat(query: str):
    async def token_stream():
        async for chunk in rag_chain.astream(query):
            if hasattr(chunk, "content") and chunk.content:
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### Cost and Latency Summary

| Optimization              | Cost Reduction | Latency Reduction | Complexity |
| ------------------------- | -------------- | ----------------- | ---------- |
| Semantic caching          | 20–40%         | 95% (cache hits)  | Medium     |
| Model tiering             | 40–65%         | 30–50% avg        | Medium     |
| Context compression       | 60–85%         | 10–20%            | Low        |
| Async parallel retrieval  | 0%             | 50–70%            | Low        |
| Retrieval result caching  | ~5–10%         | 40–60%            | Low        |
| Streaming                 | 0%             | 0% actual         | Low        |
| Reduced K (with re-rank)  | 20–40%         | 5–15%             | Low        |

**Recommended stack for production:**

```
1. Semantic cache        → cuts repeat query costs immediately
2. Context compression   → largest token reduction
3. Model tiering         → large cost saving for high volume
4. Async retrieval       → easy latency win for multi-source
5. Streaming             → UX improvement, zero cost
```

---

## 8. Optimization Decision Framework

Use this framework to determine which optimizations to apply first, based on which metrics are failing.

```
STEP 1: Run baseline evaluation (Module-1 techniques)

STEP 2: Diagnose primary failure

┌─────────────────────────────────────────────────────────────────┐
│  LOW RECALL@K  (missing relevant docs)                          │
│  ─────────────────────────────────────────────────────────────  │
│  → Try smaller chunk size (relevant content split across chunks) │
│  → Try semantic chunking (topic-coherent chunks)                │
│  → Try parent-child retrieval (precise hit, rich context)        │
│  → Add BM25 hybrid search (exact-term queries)                  │
│  → Increase K (retrieve more before re-ranking)                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LOW PRECISION@K  (too many irrelevant chunks)                  │
│  ─────────────────────────────────────────────────────────────  │
│  → Add cross-encoder re-ranker                                  │
│  → Use metadata filtering (filter by doc_type, date, category)  │
│  → Switch embedding model (better domain fit)                   │
│  → Use context compression                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  LOW FAITHFULNESS  (hallucinations)                             │
│  ─────────────────────────────────────────────────────────────  │
│  → Strengthen system prompt (anchor to context only)            │
│  → Add "I don't know" path for out-of-scope queries             │
│  → Use a more instruction-following model                       │
│  → Add Self-RAG critique loop                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HIGH LATENCY                                                   │
│  ─────────────────────────────────────────────────────────────  │
│  → Enable streaming (immediate perceived response)              │
│  → Async parallel retrieval                                     │
│  → Switch to lighter embedding model (local MiniLM)             │
│  → Semantic cache for repeat queries                            │
│  → Reduce LLM to gpt-4o-mini for simple queries                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  HIGH COST                                                      │
│  ─────────────────────────────────────────────────────────────  │
│  → Context compression (reduce tokens per query)                │
│  → Semantic caching (eliminate repeat LLM calls)                │
│  → Model tiering (cheap model for simple queries)               │
│  → Local cross-encoder instead of LLM re-ranker                 │
│  → Local embedding model (zero embedding API cost)              │
└─────────────────────────────────────────────────────────────────┘
```

### Optimization Priority Order

```
Phase 1 — Foundation (always do these first):
  ✅ Recursive or semantic chunking (not fixed-size)
  ✅ Evaluate with Recall@K on your golden dataset

Phase 2 — Retrieval quality:
  ✅ Hybrid search (BM25 + dense)
  ✅ Cross-encoder re-ranking (retrieve 20, return 4)
  ✅ Benchmark embedding models on your data

Phase 3 — Generation quality:
  ✅ Optimise system prompt (grounding instruction, I-don't-know path)
  ✅ Format context with source attribution
  ✅ A/B test prompts with RAGAS metrics

Phase 4 — Production efficiency:
  ✅ Semantic caching
  ✅ Context compression
  ✅ Model tiering
  ✅ Async retrieval
  ✅ Streaming
```

---

## 9. Lab: Improve Retrieval Relevance Using Chunking and Re-Ranking

### Lab Objective

Compare a naive (fixed-size) chunking + no re-ranking baseline against an optimised (semantic chunking + hybrid search + cross-encoder re-ranking) pipeline. Measure Recall@K and MRR before and after each change to quantify the improvement from each technique.

### Lab Setup

```bash
pip install langchain langchain-openai langchain-community langchain-experimental \
            chromadb sentence-transformers ragas datasets rank_bm25 python-dotenv
```

### Step 1 — Baseline Pipeline

```python
# lab/step1_baseline.py
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import json, numpy as np

# Load documents
loader = DirectoryLoader("./docs", glob="**/*.txt")
docs = loader.load()

# Fixed-size chunking
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(docs)
print(f"[Baseline] Chunks created: {len(chunks)}")

# Basic vector search only
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="baseline")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Evaluate
with open("evaluation/golden_dataset_v1.0.json") as f:
    golden = json.load(f)["samples"]

def recall_at_k(retrieved_ids, relevant_ids, k):
    top_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 1.0
    return len(top_k & set(relevant_ids)) / len(relevant_ids)

def mrr_score(retrieved_ids, relevant_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

recalls, mrrs = [], []
for sample in golden:
    results = retriever.invoke(sample["question"])
    ids = [doc.metadata.get("doc_id", "") for doc in results]
    recalls.append(recall_at_k(ids, sample["relevant_doc_ids"], k=5))
    mrrs.append(mrr_score(ids, sample["relevant_doc_ids"]))

print(f"[Baseline] Recall@5: {np.mean(recalls):.3f}")
print(f"[Baseline] MRR:      {np.mean(mrrs):.3f}")
```

### Step 2 — Add Semantic Chunking

```python
# lab/step2_semantic_chunking.py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Semantic chunking — splits at topic boundaries
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=90,
)

semantic_chunks = semantic_splitter.split_documents(docs)
print(f"[Semantic] Chunks created: {len(semantic_chunks)}")
print(f"[Semantic] Avg chunk size: {sum(len(c.page_content) for c in semantic_chunks)/len(semantic_chunks):.0f} chars")

vectorstore_semantic = Chroma.from_documents(
    semantic_chunks, embeddings, collection_name="semantic"
)
retriever_semantic = vectorstore_semantic.as_retriever(search_kwargs={"k": 5})

# Evaluate
recalls, mrrs = [], []
for sample in golden:
    results = retriever_semantic.invoke(sample["question"])
    ids = [doc.metadata.get("doc_id", "") for doc in results]
    recalls.append(recall_at_k(ids, sample["relevant_doc_ids"], k=5))
    mrrs.append(mrr_score(ids, sample["relevant_doc_ids"]))

print(f"[Semantic Chunking] Recall@5: {np.mean(recalls):.3f}")
print(f"[Semantic Chunking] MRR:      {np.mean(mrrs):.3f}")
```

### Step 3 — Add Hybrid Search

```python
# lab/step3_hybrid_search.py
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# Use semantic chunks from step 2
bm25_retriever = BM25Retriever.from_documents(semantic_chunks)
bm25_retriever.k = 10

dense_retriever = vectorstore_semantic.as_retriever(search_kwargs={"k": 10})

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],
)

# Evaluate
recalls, mrrs = [], []
for sample in golden:
    results = hybrid_retriever.invoke(sample["question"])[:5]  # keep top 5 after fusion
    ids = [doc.metadata.get("doc_id", "") for doc in results]
    recalls.append(recall_at_k(ids, sample["relevant_doc_ids"], k=5))
    mrrs.append(mrr_score(ids, sample["relevant_doc_ids"]))

print(f"[Hybrid Search] Recall@5: {np.mean(recalls):.3f}")
print(f"[Hybrid Search] MRR:      {np.mean(mrrs):.3f}")
```

### Step 4 — Add Cross-Encoder Re-Ranking

```python
# lab/step4_reranking.py
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder re-ranker (runs locally, no API cost)
cross_encoder = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
reranker = CrossEncoderReranker(model=cross_encoder, top_n=5)

# Stack: hybrid retriever (K=20) → cross-encoder re-ranker (top 5)
wide_hybrid_retriever = EnsembleRetriever(
    retrievers=[
        vectorstore_semantic.as_retriever(search_kwargs={"k": 20}),
        BM25Retriever.from_documents(semantic_chunks, k=20),
    ],
    weights=[0.6, 0.4],
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=wide_hybrid_retriever,
)

# Evaluate
recalls, mrrs = [], []
for sample in golden:
    results = reranking_retriever.invoke(sample["question"])
    ids = [doc.metadata.get("doc_id", "") for doc in results]
    recalls.append(recall_at_k(ids, sample["relevant_doc_ids"], k=5))
    mrrs.append(mrr_score(ids, sample["relevant_doc_ids"]))

print(f"[+ Re-ranking] Recall@5: {np.mean(recalls):.3f}")
print(f"[+ Re-ranking] MRR:      {np.mean(mrrs):.3f}")
```

### Step 5 — Compare Results

```python
# lab/step5_compare.py

results_table = {
    "Baseline (fixed chunking, dense only)": {"recall": 0.51, "mrr": 0.42},
    "Semantic chunking":                      {"recall": 0.67, "mrr": 0.55},
    "Semantic + Hybrid search":               {"recall": 0.79, "mrr": 0.63},
    "Semantic + Hybrid + Re-ranking":         {"recall": 0.83, "mrr": 0.81},
}

print("\n=== Optimization Impact Summary ===\n")
print(f"{'Pipeline':<45} {'Recall@5':>10} {'MRR':>8}")
print("-" * 65)

baseline_recall = list(results_table.values())[0]["recall"]
baseline_mrr    = list(results_table.values())[0]["mrr"]

for name, scores in results_table.items():
    r_delta = scores["recall"] - baseline_recall
    m_delta = scores["mrr"]   - baseline_mrr
    r_sign  = "+" if r_delta >= 0 else ""
    m_sign  = "+" if m_delta >= 0 else ""
    print(f"{name:<45} {scores['recall']:>8.3f} ({r_sign}{r_delta:.3f})  "
          f"{scores['mrr']:>6.3f} ({m_sign}{m_delta:.3f})")
```

Expected output:

```
=== Optimization Impact Summary ===

Pipeline                                      Recall@5      MRR
-----------------------------------------------------------------
Baseline (fixed chunking, dense only)           0.510 (+0.000)  0.420 (+0.000)
Semantic chunking                               0.670 (+0.160)  0.550 (+0.130)
Semantic + Hybrid search                        0.790 (+0.280)  0.630 (+0.210)
Semantic + Hybrid + Re-ranking                  0.830 (+0.320)  0.810 (+0.390)
```

---

## 10. Optimization Benchmarking Template

Use this template to record optimization experiments in a structured way for reproducibility and team knowledge sharing.

```
╔═══════════════════════════════════════════════════════════════════════╗
║           RAG OPTIMIZATION EXPERIMENT LOG                            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Experiment ID:  ________   Date: __________   Engineer: __________  ║
║  Hypothesis: ________________________________________________________ ║
╠═══════════════════════════════════════════════════════════════════════╣
║  CONFIGURATION                                                        ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Chunking:    [ ] Fixed  [ ] Recursive  [ ] Semantic  [ ] Parent-child║
║  Chunk size:  _______  Overlap: _______                              ║
║  Embedding:   ___________________________  Dims: _______            ║
║  Search:      [ ] Dense only  [ ] Hybrid  Dense/Sparse weights: ___  ║
║  Re-ranking:  [ ] None  [ ] Cross-encoder  [ ] LLM  Model: _______  ║
║  Initial K:   _______  Final K after rerank: _______                 ║
║  Prompt:      [ ] Baseline  [ ] Grounding  [ ] CoT  [ ] Few-shot     ║
║  LLM:         _________________________  Temperature: _______        ║
╠═══════════════════════════════════════════════════════════════════════╣
║  EVALUATION RESULTS (Golden Dataset v_____, N=_____ samples)          ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Metric             Baseline    This Run    Delta     Status         ║
║  Precision@K        _______     _______     _______   Pass / Fail    ║
║  Recall@K           _______     _______     _______   Pass / Fail    ║
║  MRR                _______     _______     _______   Pass / Fail    ║
║  Faithfulness       _______     _______     _______   Pass / Fail    ║
║  Answer Relevance   _______     _______     _______   Pass / Fail    ║
║  Answer Correctness _______     _______     _______   Pass / Fail    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  COST & LATENCY                                                       ║
║  ─────────────────────────────────────────────────────────────────   ║
║  Cost per 1K queries: $_______ (baseline: $_______)                   ║
║  p50 latency:  _____ ms    p95 latency: _____ ms                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║  OBSERVATIONS:                                                        ║
║  _________________________________________________________________    ║
║  _________________________________________________________________    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  DECISION:  [ ] Adopt  [ ] Reject  [ ] Further testing needed         ║
║  Reason: ___________________________________________________________  ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## Summary

RAG optimization is a layered process. Each layer addresses a specific failure mode, measurable by your evaluation dataset.

```
Optimization Ladder (bottom to top, in order of application):

  ┌─────────────────────────────────────────────────────────┐
  │  COST & LATENCY                                         │
  │  Semantic cache → Model tiering → Context compression   │
  ├─────────────────────────────────────────────────────────┤
  │  PROMPT ENGINEERING                                     │
  │  Grounding instruction → Format context → CoT/Few-shot  │
  ├─────────────────────────────────────────────────────────┤
  │  RE-RANKING                                             │
  │  Retrieve broad (K=20) → Cross-encoder → Return top 4   │
  ├─────────────────────────────────────────────────────────┤
  │  HYBRID SEARCH                                          │
  │  Dense + BM25 → RRF fusion → Tune weights on golden set │
  ├─────────────────────────────────────────────────────────┤
  │  EMBEDDING MODEL                                        │
  │  Benchmark on your data → Domain model if needed        │
  ├─────────────────────────────────────────────────────────┤
  │  CHUNKING  ← Start here                                 │
  │  Recursive or Semantic → Parent-child for complex docs   │
  └─────────────────────────────────────────────────────────┘
```

**Key takeaways:**

1. **Chunking is the highest-leverage change** — moving from fixed to semantic chunking typically improves Recall@K by 15–30 percentage points at near-zero additional query cost.
2. **Hybrid search fills keyword gaps** — dense search alone fails on exact codes, names, and rare terms. Adding BM25 costs almost nothing and lifts recall measurably.
3. **Re-ranking fixes ranking without changing recall** — retrieve 20, return 4; local cross-encoders add 80–200 ms and no API cost while dramatically improving MRR.
4. **Prompt engineering is the last retrieval-quality lever** — it cannot compensate for bad retrieval, but it can eliminate hallucinations when retrieval is good.
5. **Measure before and after every change** — without evaluation metrics (Module-1), optimization is guesswork. One change that improves recall may hurt faithfulness if not checked.

---

_Guide maintained as part of the **Integrating Generative AI — Advanced RAG & Enterprise Patterns (Level 2)** training programme._
