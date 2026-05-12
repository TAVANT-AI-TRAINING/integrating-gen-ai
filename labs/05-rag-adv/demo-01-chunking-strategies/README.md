# Demo 01: Chunking Strategies Showcase

**Level**: Beginner → Intermediate  
**Port**: 8001  
**Concept**: Chunking is the single highest-leverage optimization in most RAG systems. This demo makes the choice visible.

---

## What You'll Learn

| Strategy         | Class                            | Speed   | Coherence | When to Use                                  |
| ---------------- | -------------------------------- | ------- | --------- | -------------------------------------------- |
| **Fixed-size**   | `CharacterTextSplitter`          | Fastest | Low       | Prototyping, uniform docs                    |
| **Recursive**    | `RecursiveCharacterTextSplitter` | Fast    | Medium    | General text (default)                       |
| **Semantic**     | `SemanticChunker`                | Slower  | High      | Long-form, topic-diverse docs                |
| **Parent-Child** | `ParentDocumentRetriever`        | Medium  | High      | Rich docs needing both precision and context |

---

## Quick Start

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Install dependencies
uv sync

# 3. Start the server
uv run uvicorn main:app --reload --port 8001

# 4. Open the interactive docs
open http://localhost:8001/docs
```

---

## Try It: Compare All 4 Strategies

```bash
curl -s -X POST http://localhost:8001/chunk/compare \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Section 1: Remote Work Policy\nEmployees may work remotely up to 3 days per week with manager approval.\n\nSection 2: Code Review\nAll code changes must undergo peer review before merging. Reviews focus on quality, tests, and security.\n\nSection 3: Security\nUse strong passwords (12+ characters). Enable two-factor authentication. Never share credentials.\n\nSection 4: Meetings\nKeep meetings to 30 minutes or less. Always share an agenda beforehand.\n\nSection 5: Communication\nRespond to emails within 24 hours. Use Slack for quick questions.",
    "chunk_size": 300,
    "chunk_overlap": 50,
    "include_content": true
  }' | python3 -m json.tool
```

**What to observe:**

- `processing_time_ms` — semantic is 5–20x slower (it embeds every sentence)
- `chunk_count` — semantic produces fewer, more topically coherent chunks
- `comparison_summary.observation` — auto-generated insight comparing the strategies

---

## Workflow: Ingest → Retrieve → Generate

```bash
# 1. Ingest with recursive (default)
curl -s -X POST http://localhost:8001/ingest/file \
  -F "file=@Documents/policy.txt" \
  -F "strategy=recursive" | python3 -m json.tool

# 2. Verify ingestion
curl -s http://localhost:8001/retrieve/verify | python3 -m json.tool

# 3. Search
curl -s -X POST http://localhost:8001/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "How many vacation days?", "k": 3, "include_scores": true}' \
  | python3 -m json.tool

# 4. Generate a RAG answer
curl -s -X POST http://localhost:8001/generate/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "What health benefits does the company provide?", "k": 4}' \
  | python3 -m json.tool
```

---

## Try Semantic Chunking

```bash
# Ingest with semantic strategy — notice the longer response time
curl -s -X POST http://localhost:8001/ingest/file \
  -F "file=@Documents/guidelines.txt" \
  -F "strategy=semantic" | python3 -m json.tool
```

---

## Analyze a Single Strategy

```bash
curl -s -X POST http://localhost:8001/chunk/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "strategy": "semantic",
    "include_full_content": true
  }' | python3 -m json.tool
```

---

## Environment Variables

| Variable                               | Default       | Description                          |
| -------------------------------------- | ------------- | ------------------------------------ |
| `OPENAI_API_KEY`                       | —             | Required                             |
| `OPENAI_MODEL`                         | `gpt-4o-mini` | LLM model                            |
| `CHUNK_SIZE`                           | `1000`        | Default chunk size (chars)           |
| `CHUNK_OVERLAP`                        | `200`         | Default overlap (chars)              |
| `SEMANTIC_BREAKPOINT_THRESHOLD_TYPE`   | `percentile`  | `percentile` or `standard_deviation` |
| `SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT` | `90`          | Higher = fewer splits                |
| `PARENT_CHUNK_SIZE`                    | `2000`        | Parent chunk size for parent-child   |
| `CHILD_CHUNK_SIZE`                     | `400`         | Child chunk size for parent-child    |

---

## Connection to the Course Guide

This demo implements **Section 2 — Chunking Strategies** from `guides/06-rag-optimization-techniques.md`:

- Strategy 1 (Fixed-size) → `POST /chunk/compare` field `fixed`
- Strategy 2 (Recursive) → `POST /chunk/compare` field `recursive`
- Strategy 3 (Semantic) → `POST /chunk/compare` field `semantic`
- Strategy 4 (Parent-Child) → `POST /chunk/compare` field `parent_child`

**Next step**: [Demo-14](../demo-14-hybrid-search) — add hybrid BM25 + dense retrieval.
