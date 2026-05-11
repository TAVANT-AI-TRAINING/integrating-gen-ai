# Demo 02: Retrieval Metrics

**Module-1: RAG Evaluation Techniques** | Beginner-Intermediate Level

## Objective

Learn the 4 standard retrieval metrics by calculating them manually on a simple HR knowledge base. Understanding these formulas is essential before using automated frameworks like RAGAS.

## Metrics Covered

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@K** | `\|relevant ∩ retrieved[:K]\| / K` | Of K retrieved chunks, what fraction are correct? |
| **Recall@K** | `\|relevant ∩ retrieved[:K]\| / \|relevant\|` | Of all correct chunks, how many did we find? |
| **MRR** | `1 / position_of_first_relevant` | How early does the first correct chunk appear? |
| **MAP** | `mean(AP) across all queries` | Overall retrieval quality across all queries |

## No API Keys Required

This demo uses keyword-overlap retrieval (pure Python) — no LLM, no embeddings, no API keys. The goal is to understand the metric math, not to build a perfect retriever.

## Expected Output

```
======================================================================
  PER-QUERY RESULTS
======================================================================

  [1] How many days per week can employees work remotely?
       Relevant chunks : ['chunk_001', 'chunk_002']
       Retrieved (top5): ['chunk_001', 'chunk_002', ...]
       P@1=1.00  P@3=0.67  P@5=0.40  R@3=1.00  R@5=1.00  RR=1.00  AP=1.00


======================================================================
  AGGREGATE SCORES
======================================================================

  MAP  (Mean Average Precision)  = 0.847
  MRR  (Mean Reciprocal Rank)    = 0.938
  Mean Precision@3               = 0.583
  Mean Recall@3                  = 0.844
```

## Setup

```bash
cd demo-02-retrieval-metrics

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies (minimal — just python-dotenv)
uv pip install -e .
```

## Running the Demo

```bash
# No .env file needed!
uv run python main.py
```

## Features

✅ **Zero API calls** — pure Python, runs instantly  
✅ **Formula in every docstring** — see the math inline  
✅ **Golden dataset** — 8 HR queries with manually labelled relevant chunks  
✅ **Score interpretation** — production benchmarks explained  

## Key Concepts

### Precision vs Recall Trade-off

- **High Precision, Low Recall**: Retrieved few chunks but they're correct. User gets accurate but incomplete information.
- **Low Precision, High Recall**: Retrieved many chunks, most wrong. More noise in LLM context → more hallucination risk.
- **Target**: High Precision@3 AND High Recall@5 (retrieve few but correct + catch most relevant)

### MRR vs MAP

- **MRR**: Good for QA where you need the *first* correct answer fast.
- **MAP**: Better for comprehensive retrieval where you need *all* relevant chunks.

## Curriculum Connection

This demo covers:

> *"Retrieval metrics: Precision@K, Recall@K, MRR, MAP"*

After this demo, proceed to:
- **demo-03**: Generation metrics (Faithfulness, Relevance, Groundedness)
- **demo-05**: RAGAS automates Context Precision and Context Recall (equivalent to Precision@K and Recall@K but using LLM relevance judgments)

## Related Demos

- **demo-01**: RAG Failure Modes (why poor retrieval causes failures)
- **demo-03**: Generation Metrics (evaluate the LLM's output)
- **demo-05**: RAGAS Framework (automates retrieval + generation evaluation)
