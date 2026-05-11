# Demo 01: RAG Failure Modes

**Module-1: RAG Evaluation Techniques** | Beginner Level

## Objective

Understand the 4 most common failure modes in RAG systems **before** applying evaluation metrics. You can't measure what you can't name — this demo gives you the vocabulary and intuition for why RAG evaluation is critical in production.

## Why This Matters

A RAG system can fail silently: it returns an answer without errors, but the answer is wrong, incomplete, or fabricated. These failures are invisible without systematic evaluation.

## The 4 Failure Modes

| # | Failure Mode | Root Cause | Detected By |
|---|---|---|---|
| 1 | **Hallucination** | LLM invents facts when context is empty | Faithfulness metric |
| 2 | **Poor Retrieval** | Wrong chunks retrieved for the query | Precision@K, Context Precision |
| 3 | **Irrelevant Chunks** | Right topic, wrong specific fact | Recall@K, Context Recall |
| 4 | **Incomplete Answer** | Only partial context retrieved | Answer Correctness |

## Expected Output

```
======================================================================
  FAILURE MODE: 1. HALLUCINATION
  Caught by metric: Faithfulness  (is every claim grounded in the context?)
======================================================================

  Query: What is the company stock option vesting schedule?
  Chunks used: (none — empty context)

  LLM Answer:
    The company stock options typically vest over a 4-year period with a
    1-year cliff... [fabricated answer]

  ⚠  FAILURE DETECTED: No relevant document was retrieved, yet the LLM
     generated a confident-sounding answer...
```

## Prerequisites

- Python 3.12+
- OpenAI API key

## Setup

```bash
# Navigate to this demo
cd demo-01-rag-failure-modes

# Create virtual environment
uv venv && source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Running the Demo

```bash
uv run python main.py
```

## Features

✅ **Zero external eval tools** — pure Python + OpenAI  
✅ **Controlled failures** — each failure is deliberately triggered so you can see it clearly  
✅ **Metric mapping** — every failure maps to a specific metric from the curriculum  
✅ **HR domain** — same documents as demo-12-rag-fastapi-service  

## Use Cases

- **Teaching**: Show students WHY evaluation matters before introducing metrics
- **Debugging**: Use the failure patterns to diagnose real RAG system issues
- **Design**: Use the metric mapping table to choose which metrics to implement

## Curriculum Connection

This demo covers the opening section of **Module-1: RAG Evaluation Techniques**:

> *"Why evaluation is critical for production RAG systems"*  
> *"Common failure modes: hallucination, poor retrieval, irrelevant chunks, incomplete answers"*

After this demo, proceed to:
- **demo-02**: Measure retrieval quality (Precision@K, Recall@K, MRR, MAP)
- **demo-03**: Measure generation quality (Faithfulness, Relevance, Groundedness)

## Related Demos

- **demo-02**: Retrieval Metrics
- **demo-03**: Generation Metrics (LLM-as-judge)
- **demo-05**: RAGAS evaluation framework (automates all these metrics)
- **Base**: `labs/03-rag/demo-12-rag-fastapi-service` — the RAG system being evaluated
