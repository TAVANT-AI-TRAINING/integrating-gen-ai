# Demo 05: RAGAS Evaluation Framework

**Module-1: RAG Evaluation Techniques** | Intermediate-Advanced Level

## Objective

Use the **RAGAS** (Retrieval-Augmented Generation Assessment) framework to evaluate a complete RAG pipeline with standardized, reproducible metrics. RAGAS is widely used in production RAG systems and provides industry-standard benchmarks.

## RAGAS Metrics

| Metric | Measures | Maps To |
|--------|----------|---------|
| **Faithfulness** | Answer claims grounded in context | demo-03 faithfulness |
| **Response Relevancy** | Answer addresses the question | demo-03 relevance |
| **Context Precision** | Retrieved chunks are useful | demo-02 Precision@K |
| **Context Recall** | Context contains all needed info | demo-02 Recall@K |
| **Answer Correctness** | Answer matches ground truth | demo-03 correctness |

## Architecture

```
8 HR Documents → ChromaDB → Retrieval → LLM Generation
                                              │
                                              ▼
                                        RAGAS Framework
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         Faithfulness   Context         Answer
                         Response       Precision       Correctness
                         Relevancy      Context Recall
```

## Important: RAGAS API Version

This demo uses **RAGAS 0.2.x+** (breaking change from 0.1.x):

```python
# RAGAS 0.2.x+ (this demo)
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ...

sample = SingleTurnSample(
    user_input=query,
    response=answer,
    retrieved_contexts=contexts,   # list[str]
    reference=ground_truth,
)
dataset = EvaluationDataset(samples=[sample, ...])
result = evaluate(dataset=dataset, metrics=[Faithfulness(), ...])
```

## Expected Output

```
  RAGAS METRIC GLOSSARY
  ─────────────────────────────────────────────────────────────────
  Faithfulness       : Every claim in the answer is grounded in context.
  Response Relevancy : The answer actually addresses the question asked.
  Context Precision  : Retrieved chunks are mostly relevant to the query.
  Context Recall     : The context contains enough info to answer correctly.
  Answer Correctness : The answer matches the reference (ground truth) answer.
  ─────────────────────────────────────────────────────────────────

  AGGREGATE RAGAS SCORES
  faithfulness              avg=0.921  pass_rate=8/8  ✓ GOOD
  response_relevancy        avg=0.878  pass_rate=7/8  ✓ GOOD
  context_precision         avg=0.812  pass_rate=6/8  ✓ GOOD
  context_recall            avg=0.794  pass_rate=6/8  ✓ GOOD
  answer_correctness        avg=0.743  pass_rate=7/8  ✓ GOOD
```

## Setup

```bash
cd demo-05-ragas-evaluation

uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Add your OPENAI_API_KEY
```

## Running the Demo

```bash
uv run python main.py
```

**Cost**: ~40-50 API calls (retrieve + generate + 5 RAGAS metrics × 8 queries). Approximately $0.10-0.20 with gpt-4o-mini.  
**Time**: 3-5 minutes (RAGAS makes many LLM calls for evaluation).

## Output Files

| File | Contents |
|------|---------|
| `ragas_scores.json` | Mean and per-sample scores for all 5 metrics |
| `chroma_db_eval/` | ChromaDB vector store (auto-created) |

## Troubleshooting

**"No module named ragas"** → Run `uv pip install -e .`  
**"ChromaDB collection already exists"** → Normal — the demo skips re-ingestion  
**"API rate limit"** → Add `time.sleep(1)` between samples in `build_evaluation_dataset`  
**RAGAS import errors** → Ensure you have `ragas>=0.2.0` (check: `pip show ragas`)

## Curriculum Connection

This demo covers:

> *"Tools: RAGAS, LangSmith, DeepEval"*  
> *"Lab: Evaluate an existing RAG chatbot using sample queries and identify failures"*

RAGAS provides the standardized metrics. LangSmith (demo-06) provides the tracking infrastructure.

## Related Demos

- **demo-02**: Manual retrieval metrics (understand what RAGAS automates)
- **demo-03**: Manual LLM-as-judge (understand what RAGAS wraps)
- **demo-06**: LangSmith (track RAGAS scores over time)
- **demo-07**: Evaluation API (expose these scores via REST endpoint)
