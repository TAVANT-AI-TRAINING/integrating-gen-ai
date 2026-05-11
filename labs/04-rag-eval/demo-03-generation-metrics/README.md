# Demo 03: Generation Metrics — LLM-as-Judge

**Module-1: RAG Evaluation Techniques** | Intermediate Level

## Objective

Learn how to evaluate the **quality of generated answers** using the LLM-as-judge pattern. While demo-02 measured retrieval quality, this demo measures whether the LLM produced a good answer from the retrieved context.

## The LLM-as-Judge Pattern

```
User Query ──→ RAG System ──→ Answer
                                │
                                ▼
                         Judge LLM (GPT)
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
               Faithfulness Relevance Groundedness
               (0.0 - 1.0) (0.0 - 1.0) (0.0 - 1.0)
```

A second LLM call evaluates the first LLM's output. This scales to thousands of test cases without human reviewers.

## Metrics Covered

| Metric | Question | Catches |
|--------|----------|---------|
| **Faithfulness** | Is every claim supported by the context? | Hallucination |
| **Relevance** | Does the answer address the question? | Off-topic answers |
| **Groundedness** | Does the answer stay within the context? | Over-generation |
| **Correctness** | Is the answer factually accurate vs ground truth? | Wrong facts |

## Score Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Faithfulness | ≥ 0.70 | Critical — hallucinations damage user trust |
| Relevance | ≥ 0.70 | Critical — irrelevant answers waste user time |
| Groundedness | ≥ 0.70 | Important — extra information is hard to verify |
| Correctness | ≥ 0.60 | Slightly lower — LLM phrasings can vary |

## Expected Output

```
  PER-CASE SCORECARD

  CASE 1: Correct Answer
  Query: How many days can employees work remotely per week?
    Faithfulness   0.95  ✓ PASS  — All claims are explicitly stated in context
    Relevance      0.95  ✓ PASS  — Answer directly addresses the question
    Groundedness   0.95  ✓ PASS  — Stays within the context provided
    Correctness    0.90  ✓ PASS  — Matches the ground truth facts

  CASE 2: Hallucinated Answer (Faithfulness Failure)
  ...
    Faithfulness   0.20  ✗ FAIL  — Claims about "unlimited PTO" not in context
```

## Setup

```bash
cd demo-03-generation-metrics

uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Add your OPENAI_API_KEY
```

## Running the Demo

```bash
uv run python main.py
```

**Cost**: ~20 OpenAI API calls (4 metrics × 5 test cases). Approximately $0.02-0.05 with gpt-4o-mini.

## Features

✅ **LLM-as-judge** — industry-standard evaluation pattern  
✅ **4 metrics** — covers the full generation quality spectrum  
✅ **Pydantic models** — type-safe score objects with reasoning  
✅ **5 deliberate failure cases** — see each metric fail in isolation  
✅ **JSON response format** — structured LLM output, no regex parsing  

## Curriculum Connection

This demo covers:

> *"Generation metrics: Faithfulness, Relevance, Groundedness, Answer Correctness"*  
> *"Human evaluation vs automated evaluation"*

The LLM-as-judge pattern is the automated evaluation alternative to human review. Demo-04 will compare both approaches.

## Related Demos

- **demo-02**: Retrieval metrics (evaluate what was retrieved)
- **demo-04**: Golden dataset (scale up test case creation)
- **demo-05**: RAGAS (runs these same metrics using a framework)
- **demo-06**: LangSmith (run and track these metrics in the cloud)
