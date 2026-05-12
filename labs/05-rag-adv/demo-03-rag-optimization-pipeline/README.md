# Demo 03: Full RAG Optimization Pipeline

**Level**: Advanced  
**Port**: 8003  
**Concept**: Compose the complete optimization ladder and measure quality improvement at each rung using Recall@K and MRR.

---

## The Optimization Ladder

```
Stage 4 (reranked)        Recall@K ~0.85, MRR ~0.80
  SemanticChunker + Hybrid(fetch 20) + CrossEncoderReranker(top 4)
     ↑
Stage 3 (hybrid_search)   Recall@K ~0.75, MRR ~0.65
  SemanticChunker + EnsembleRetriever (dense 0.6 + BM25 0.4)
     ↑
Stage 2 (semantic)        Recall@K ~0.65, MRR ~0.55
  SemanticChunker(percentile 90) + dense-only
     ↑
Stage 1 (baseline)        Recall@K ~0.50, MRR ~0.40
  CharacterTextSplitter(1000) + dense-only
```

---

## Quick Start

> **First run note**: The cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB)
> downloads from HuggingFace automatically and is cached under `~/.cache/huggingface/`.
> Ensure network access for the initial startup.

```bash
cp .env.example .env
# Set OPENAI_API_KEY in .env

uv sync
uv run uvicorn main:app --reload --port 8003
open http://localhost:8003/docs
```

---

## Recommended Workflow

### Step 1 — Ingest documents into both collections

```bash
# Single /ingest/file call ingests into BOTH baseline and semantic collections
for FILE in Documents/guidelines.txt Documents/policy.txt; do
  curl -s -X POST http://localhost:8003/ingest/file \
    -F "file=@$FILE" | python3 -m json.tool
done
```

The response shows:

- `baseline_chunks`: fixed-size CharacterTextSplitter chunks (fast, uniform)
- `semantic_chunks_count`: SemanticChunker chunks (topic-coherent, slower to index)

### Step 2 — See one query across all 4 stages

```bash
curl -s -X POST http://localhost:8003/retrieve/pipeline \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the password requirements?", "k": 4}' \
  | python3 -m json.tool
```

### Step 3 — Run the golden dataset evaluation

```bash
curl -s -X POST http://localhost:8003/optimize/evaluate \
  -H "Content-Type: application/json" \
  -d '{"k": 4, "dense_weight": 0.6, "sparse_weight": 0.4, "initial_fetch_k": 20}' \
  | python3 -m json.tool
```

Look at `improvement_summary` for the Recall@K and MRR deltas from baseline to reranked.

---

## Cross-Encoder Re-ranking Explained

```
Standard retrieval (bi-encoder):
  query → embed → vector ──┐
                            ├── cosine_sim() → score  (fast, pre-computed)
  document → embed → vector ─┘

Cross-encoder re-ranking:
  [query + document concatenated] → CrossEncoder model → relevance score 0-1
  (slower, no pre-computation, but sees query-document interaction directly)
```

Cross-encoder is run locally (no API calls) — it adds ~100–500ms latency for
re-ranking 20 candidates but dramatically improves MRR.

---

## Evaluate Your Own Query

```bash
curl -s -X POST http://localhost:8003/optimize/custom-query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many sick days do employees get?",
    "k": 4,
    "relevant_keywords": ["sick leave", "10 days", "per year"]
  }' | python3 -m json.tool
```

Providing `relevant_keywords` (≥2 must match for a chunk to count as relevant)
computes Recall@K and MRR on the fly for each stage.

---

## Compare RAG Answers by Stage

```bash
for STAGE in baseline semantic_chunking hybrid_search reranked; do
  echo "=== $STAGE ===";
  curl -s -X POST http://localhost:8003/generate/rag \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"What are the code review requirements?\", \"pipeline_stage\": \"$STAGE\", \"include_sources\": false}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['answer'])"
  echo;
done
```

---

## Environment Variables

| Variable                               | Default                                | Description                               |
| -------------------------------------- | -------------------------------------- | ----------------------------------------- |
| `OPENAI_API_KEY`                       | —                                      | Required                                  |
| `COLLECTION_NAME_BASELINE`             | `pipeline_baseline`                    | ChromaDB collection for fixed-size chunks |
| `COLLECTION_NAME_SEMANTIC`             | `pipeline_semantic`                    | ChromaDB collection for semantic chunks   |
| `CROSS_ENCODER_MODEL`                  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace cross-encoder                 |
| `INITIAL_FETCH_K`                      | `20`                                   | Candidates fetched before re-ranking      |
| `RERANK_TOP_N`                         | `4`                                    | Final results after re-ranking            |
| `DENSE_WEIGHT`                         | `0.6`                                  | Hybrid search dense weight                |
| `SPARSE_WEIGHT`                        | `0.4`                                  | Hybrid search BM25 weight                 |
| `SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT` | `90`                                   | SemanticChunker sensitivity               |

---

## Connection to the Course Guide

This demo implements **Sections 2, 4, 5, and 9** from `guides/06-rag-optimization-techniques.md`:

- Section 2 (Chunking): Fixed vs Semantic strategies — two separate ChromaDB collections
- Section 4 (Hybrid Search): EnsembleRetriever with BM25 + dense
- Section 5 (Re-ranking): ContextualCompressionRetriever + CrossEncoderReranker
- Section 9 (Lab): The 4-stage evaluation loop with Recall@K and MRR metrics

**Previous**: [Demo-14](../demo-14-hybrid-search) — Hybrid Search  
**Series start**: [Demo-13](../demo-13-chunking-strategies) — Chunking Strategies
