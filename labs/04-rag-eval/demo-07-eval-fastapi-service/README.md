# Demo 07: RAG Evaluation FastAPI Service

**Module-1: RAG Evaluation Techniques** | Advanced Capstone

## Objective

The capstone of Module-1. Wrap all evaluation techniques from demos 01-06 into a **production-ready REST API** that can be called by any system — a CI/CD pipeline, monitoring dashboard, or QA team tool.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│               RAG Evaluation API (Port 8001)              │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  EVALUATION ENDPOINTS                                     │
│  ├── POST /eval/single  — evaluate one Q&A pair          │
│  │     ├── Auto-retrieve context from ChromaDB           │
│  │     ├── Faithfulness judge                            │
│  │     ├── Relevance judge                               │
│  │     ├── Groundedness judge                            │
│  │     └── Correctness judge (if ground_truth provided)  │
│  ├── POST /eval/batch   — batch evaluate golden dataset   │
│  ├── GET  /eval/report  — aggregate results & trends      │
│  └── DELETE /eval/results — clear results (testing)      │
│                                                           │
├──────────────────────────────────────────────────────────┤
│  ChromaDB (HR Knowledge Base) │  LangSmith (optional)     │
└──────────────────────────────────────────────────────────┘
```

## Module-1 Knowledge Used

| Demo | Concept Used In This Service |
|------|------------------------------|
| demo-01 | Failure modes motivate the evaluation thresholds |
| demo-02 | Retrieval drives the auto-context feature |
| demo-03 | LLM-as-judge pattern = the 4 metric functions |
| demo-04 | `golden_dataset.json` loadable via `golden_dataset_path` |
| demo-05 | Same metric patterns, now exposed via REST |
| demo-06 | LangSmith tracing wired to every evaluation call |

## Setup

```bash
cd demo-07-eval-fastapi-service

uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Required: OPENAI_API_KEY
# Optional: LANGSMITH_API_KEY + LANGCHAIN_TRACING_V2=true
```

## Starting the Server

```bash
# Option 1: With auto-reload (development)
uvicorn main:app --reload --port 8001

# Option 2: With uv
uv run uvicorn main:app --reload --port 8001

# Server runs on port 8001 (not 8000, to avoid conflict with demo-12)
```

## API Documentation

Open in browser after starting the server:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **Health**: http://localhost:8001/health

## Running the Test Client

```bash
# In a separate terminal (server must be running)
python test_eval_api.py
# or
uv run python test_eval_api.py
```

## Endpoints

### POST `/eval/single`

Evaluate one question-answer pair.

```bash
# With explicit context
curl -X POST http://localhost:8001/eval/single \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many days can employees work remotely?",
    "answer": "Employees can work remotely up to 3 days per week.",
    "context": "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval.",
    "ground_truth": "Up to 3 days per week with manager approval."
  }'

# Auto-retrieve context (no context field)
curl -X POST http://localhost:8001/eval/single \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the 401k match?",
    "answer": "The company matches 5% of 401k contributions."
  }'
```

Response:
```json
{
  "eval_id": "a1b2c3d4",
  "query": "How many days can employees work remotely?",
  "answer": "Employees can work remotely up to 3 days per week.",
  "faithfulness": {"score": 0.95, "reasoning": "...", "passed": true, "threshold": 0.7},
  "relevance":    {"score": 0.92, "reasoning": "...", "passed": true, "threshold": 0.7},
  "groundedness": {"score": 0.90, "reasoning": "...", "passed": true, "threshold": 0.7},
  "correctness":  {"score": 0.95, "reasoning": "...", "passed": true, "threshold": 0.6},
  "overall_pass": true,
  "eval_timestamp": "2026-05-11T10:00:00Z"
}
```

### POST `/eval/batch`

Batch evaluate a golden dataset.

```bash
# Use built-in 8-question dataset
curl -X POST http://localhost:8001/eval/batch \
  -H "Content-Type: application/json" \
  -d '{"use_builtin_dataset": true}'

# Use custom dataset from demo-04
curl -X POST http://localhost:8001/eval/batch \
  -H "Content-Type: application/json" \
  -d '{"golden_dataset_path": "../demo-04-golden-dataset/golden_dataset.json"}'
```

### GET `/eval/report`

```bash
curl http://localhost:8001/eval/report
```

Response:
```json
{
  "total_eval_runs": 5,
  "total_cases_evaluated": 13,
  "overall_pass_rate": 0.846,
  "aggregate_scores": {
    "faithfulness": 0.912,
    "relevance": 0.878,
    "groundedness": 0.891
  },
  "eval_run_ids": ["a1b2c3d4", "e5f6g7h8", ...]
}
```

## Features

✅ **Self-contained** — runs without demo-12 being active  
✅ **Auto-context** — retrieves context from ChromaDB if not provided  
✅ **LangSmith optional** — works offline; tracing enabled when `LANGSMITH_API_KEY` set  
✅ **Result persistence** — all evaluations saved to `eval_results.json`  
✅ **demo-04 compatible** — loads `golden_dataset.json` via `golden_dataset_path`  
✅ **Threshold configurable** — adjust via environment variables  

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | gpt-4o-mini | LLM model |
| `LANGSMITH_API_KEY` | No | — | Enable LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | false | Must be `true` to enable tracing |
| `LANGCHAIN_PROJECT` | No | rag-eval-demo-07 | LangSmith project name |
| `FAITHFULNESS_THRESHOLD` | No | 0.7 | Pass threshold |
| `RELEVANCE_THRESHOLD` | No | 0.7 | Pass threshold |
| `GROUNDEDNESS_THRESHOLD` | No | 0.7 | Pass threshold |
| `CORRECTNESS_THRESHOLD` | No | 0.6 | Pass threshold |
| `EVAL_RESULTS_FILE` | No | eval_results.json | Persistence file |

## Curriculum Connection

This demo is the **Module-1 Lab**:

> *"Lab / Demo: Evaluate an existing RAG chatbot using sample queries and identify failures."*

The service integrates every evaluation technique from Module-1 into a reusable tool.

## Related Demos

- **demo-03**: LLM-as-judge pattern (the evaluation core)
- **demo-04**: Golden dataset (loadable via `golden_dataset_path`)
- **demo-06**: LangSmith (integrated via `LANGSMITH_API_KEY`)
- **Base**: `labs/03-rag/demo-12-rag-fastapi-service` (the RAG system this evaluates)
