# Demo 06: LangSmith Observability Platform

**Module-1: RAG Evaluation Techniques** | Advanced Level

## Objective

Add **continuous observability** to your RAG system using LangSmith. While demos 02-05 measure quality at a point in time, LangSmith lets you track quality across every query, compare experiments, and debug failures visually.

## LangSmith Concepts

| Concept | Description |
|---------|-------------|
| **Trace** | One end-to-end execution of your RAG pipeline |
| **Project** | Group of traces (e.g., "rag-eval-demo-06") |
| **Dataset** | Saved Q&A pairs for benchmark evaluation |
| **Experiment** | One evaluation run against a dataset |
| **Evaluator** | Function that scores a trace against a reference |

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  @traceable  RAG Pipeline           │◄──── Auto-traced to LangSmith
│  ├── retrieve_documents()           │
│  └── generate_answer()              │
└─────────────────────────────────────┘
    │                   │
    ▼                   ▼
 Answer           LangSmith Dashboard
                  ├── Traces (all queries)
                  ├── Experiments (eval runs)
                  └── Scores per evaluator
```

## Getting Your LangSmith API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign in with GitHub or Google (free account)
3. Go to **Settings** → **API Keys** → **Create API Key**
4. Copy the key (starts with `ls__`)

## Setup

```bash
cd demo-06-langsmith-observability

uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and LANGSMITH_API_KEY
```

## Running the Demo

```bash
# With LangSmith (full tracing + evaluation)
uv run python main.py

# Without LangSmith (offline mode — still runs RAG, just no cloud tracing)
# Set LANGCHAIN_TRACING_V2=false in .env
uv run python main.py
```

## What the Demo Does

### Step 1: Single Trace Demo
Runs 3 sample HR queries through the traced RAG pipeline. After running, each query appears as a separate trace in your LangSmith project.

### Step 2: Create LangSmith Dataset
Creates a persistent dataset of 8 golden Q&A pairs in LangSmith. This dataset is reused across evaluation experiments.

### Step 3: Batch Evaluation Experiment
Runs all 8 dataset examples through the RAG pipeline, scoring each with 3 evaluators:
- **faithfulness** — Is the answer grounded in context?
- **relevance** — Does the answer address the question?
- **answer_similarity** — How close is the answer to the reference?

## Viewing Results in LangSmith

After running:
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Open **Projects** → `rag-eval-demo-06`
3. Click any trace to see inputs, outputs, latency, token usage
4. Click **Experiments** to see evaluation scores
5. Compare experiments after changing the RAG system

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `LANGSMITH_API_KEY` | For tracing | LangSmith API key (free) |
| `LANGCHAIN_TRACING_V2` | For tracing | Set `true` to enable |
| `LANGCHAIN_PROJECT` | No | Project name in LangSmith dashboard |

## Features

✅ **Works without LangSmith** — offline mode for API-key-free testing  
✅ **@traceable decorator** — automatic trace capture with no code changes  
✅ **Custom evaluators** — faithfulness, relevance, answer similarity  
✅ **Dataset management** — creates/reuses golden datasets automatically  
✅ **Experiment comparison** — run again after changes to compare scores  

## Curriculum Connection

This demo covers:

> *"Tools: RAGAS, LangSmith, DeepEval"*  
> *"Audit trails and observability"* (Module-3 preview)  
> *"Feedback loops and continuous improvement"* (Module-4 preview)

LangSmith bridges evaluation (Module-1) and LLMOps (Module-4).

## Related Demos

- **demo-03**: LLM-as-judge pattern (used inside evaluators here)
- **demo-05**: RAGAS (standardized metrics, complementary to LangSmith)
- **demo-07**: Evaluation FastAPI service (production-grade evaluation API)
