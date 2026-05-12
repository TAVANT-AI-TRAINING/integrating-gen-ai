# Demo 02: Hybrid Search — BM25 + Dense Vector Retrieval

**Level**: Intermediate  
**Port**: 8002  
**Concept**: Pure vector search fails on exact keywords. BM25 fails on paraphrasing. Hybrid search fuses both using Reciprocal Rank Fusion (RRF).

---

## Why Hybrid Search?

| Query Type                               | Dense (vector) | BM25 (keyword) | Hybrid |
| ---------------------------------------- | -------------- | -------------- | ------ |
| "What are consequences of late payment?" | ✅             | ❌             | ✅     |
| "two-factor authentication minimum 12"   | ❌             | ✅             | ✅     |
| "GDPR Article 17"                        | ❌             | ✅             | ✅     |
| "What is the remote work flexibility?"   | ✅             | ❌             | ✅     |
| Mixed (acronym + concept)                | ❌             | ❌             | ✅     |

---

## How RRF Works

```
RRF_score(document d) = Σ  1 / (k + rank_retriever(d))
                        retrievers

k = 60  (smoothing constant)

Example — query "SSL TLS 1.2 handshake":
  Dense ranking:  doc_A(1), doc_B(2), doc_C(5)
  BM25 ranking:   doc_B(1), doc_A(3), doc_D(2)

  RRF scores:
    doc_A: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0156 = 0.0320  ← winner
    doc_B: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
```

---

## Quick Start

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env

uv sync
uv run uvicorn main:app --reload --port 8002
open http://localhost:8002/docs
```

---

## Key Demo: Compare All 3 Modes

```bash
# 1. Ingest documents first
curl -s -X POST http://localhost:8002/ingest/file \
  -F "file=@Documents/guidelines.txt" | python3 -m json.tool

curl -s -X POST http://localhost:8002/ingest/file \
  -F "file=@Documents/policy.txt" | python3 -m json.tool

# 2. Try a semantic query — dense should win
curl -s -X POST http://localhost:8002/retrieve/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the consequences of missing a project deadline?",
    "k": 3,
    "dense_weight": 0.6,
    "sparse_weight": 0.4
  }' | python3 -m json.tool

# 3. Try a keyword query — BM25 should win
curl -s -X POST http://localhost:8002/retrieve/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "two-factor authentication",
    "k": 3,
    "dense_weight": 0.6,
    "sparse_weight": 0.4
  }' | python3 -m json.tool
```

**Look at `overlap_analysis`** in the response — it shows which chunks each retriever found uniquely.

---

## Tune Hybrid Weights

```bash
# More weight on BM25 (good for keyword-heavy domains)
curl -s -X POST http://localhost:8002/retrieve/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "12 characters password security", "k": 4, "dense_weight": 0.3, "sparse_weight": 0.7}' \
  | python3 -m json.tool

# More weight on dense (good for semantic/conceptual domains)
curl -s -X POST http://localhost:8002/retrieve/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "work life balance flexibility", "k": 4, "dense_weight": 0.8, "sparse_weight": 0.2}' \
  | python3 -m json.tool
```

---

## Compare RAG Answers by Retrieval Mode

```bash
for MODE in dense sparse hybrid; do
  echo "=== $MODE ===";
  curl -s -X POST http://localhost:8002/generate/rag \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"What are the password requirements?\", \"retrieval_mode\": \"$MODE\", \"include_sources\": false}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'])"
  echo;
done
```

---

## Environment Variables

| Variable         | Default | Description                          |
| ---------------- | ------- | ------------------------------------ |
| `OPENAI_API_KEY` | —       | Required                             |
| `DENSE_WEIGHT`   | `0.6`   | Weight for dense retriever in hybrid |
| `SPARSE_WEIGHT`  | `0.4`   | Weight for BM25 in hybrid            |
| `CHUNK_SIZE`     | `1000`  | Chunk size for ingestion             |
| `CHUNK_OVERLAP`  | `200`   | Chunk overlap for ingestion          |

> **Note on BM25 persistence**: BM25 is in-memory only. On startup, it is automatically rebuilt from the persisted ChromaDB data, so it survives server restarts as long as ChromaDB data exists (`./chroma_db/`).

---

## Connection to the Course Guide

This demo implements **Section 4 — Hybrid Search: Keyword + Vector** from `guides/06-rag-optimization-techniques.md`, specifically the `EnsembleRetriever` pattern with RRF fusion.

**Previous**: [Demo-13](../demo-13-chunking-strategies) — Chunking Strategies  
**Next**: [Demo-15](../demo-15-rag-optimization-pipeline) — Full Optimization Pipeline with Re-ranking
