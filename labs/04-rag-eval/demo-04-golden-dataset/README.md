# Demo 04: Golden Dataset Creation

**Module-1: RAG Evaluation Techniques** | Intermediate Level

## Objective

Create a reusable **golden evaluation dataset** — the benchmark that all future evaluation runs are measured against. Learn to auto-generate Q&A pairs from documents and compare human vs automated evaluation.

## What is a Golden Dataset?

A golden dataset is a curated set of question-answer pairs where the answers are known to be correct (the "gold standard"). It enables:

- **Reproducible evaluation** — run the same questions every time you change the RAG system
- **Regression testing** — catch when system changes make things worse
- **Baseline comparison** — compare different chunking strategies, models, or prompts

## This Demo Produces

```
golden_dataset.json          # 18 Q&A pairs from 6 HR documents
                             # Used by demos 05, 06, and 07
```

## Steps Covered

1. **Auto-generate Q&A pairs** from 6 HR documents (3 pairs each = 18 total)
2. **Save to JSON** with metadata (version, created_at, model used)
3. **Reload and verify** round-trip integrity
4. **Compare human vs automated** scoring to understand trade-offs

## Q&A Difficulty Levels

| Level | Example | Purpose |
|-------|---------|---------|
| **Easy** | "How many vacation days?" | Tests basic fact retrieval |
| **Medium** | "What conditions qualify for remote work?" | Tests understanding |
| **Hard** | "How does parental leave differ between caregivers?" | Tests inference |

## Expected Output

```
======================================================================
  STEP 1: GENERATING GOLDEN DATASET
======================================================================
  Generating Q&A pairs from 6 documents (3 pairs each)...
    • Remote Work Policy...
    • Employee Benefits and Leave Policy...
    ...

  Generated 18 Q&A pairs from 6 documents

  Sample Q&A pairs generated:
  [1] [EASY] [factual]
       Q: How many days per week can employees work remotely?
       A: Employees can work remotely up to 3 days per week...

======================================================================
  STEP 3: HUMAN vs AUTOMATED EVALUATION COMPARISON
======================================================================
  Question                            Human    Auto
  ...
  Aggregate Results:
    Correlation (Pearson r) : 0.85
    Agreement Rate (±1.0)   : 80%
```

## Setup

```bash
cd demo-04-golden-dataset

uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
# Add your OPENAI_API_KEY
```

## Running the Demo

```bash
uv run python main.py
```

**Cost**: ~18 API calls for Q&A generation + ~5 for comparison. Approximately $0.03-0.08.

## Files Produced

| File | Purpose |
|------|---------|
| `golden_dataset.json` | The generated evaluation dataset — reuse in later demos |
| `sample_documents.py` | The canonical HR knowledge base (importable module) |

## Features

✅ **Auto Q&A generation** — scales to any document corpus  
✅ **Difficulty mixing** — easy/medium/hard ensures comprehensive testing  
✅ **Version control ready** — JSON format with metadata  
✅ **Human vs auto comparison** — understand when to use each approach  
✅ **Importable corpus** — `sample_documents.py` used by demos 05-07  

## Curriculum Connection

This demo covers:

> *"Golden datasets and benchmark creation"*  
> *"Human evaluation vs automated evaluation"*

Key insight: Automated scoring (correlation ≈ 0.80 with human reviewers) is sufficient for continuous evaluation. Human review is valuable for initial calibration and edge case analysis.

## Related Demos

- **demo-03**: LLM-as-judge (the automated eval technique)
- **demo-05**: RAGAS (uses this dataset format for standardized metrics)
- **demo-06**: LangSmith (upload this dataset for cloud-based tracking)
- **demo-07**: Eval API batch endpoint (loads this JSON file)
