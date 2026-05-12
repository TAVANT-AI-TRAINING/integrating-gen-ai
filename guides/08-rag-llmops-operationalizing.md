# Module-4: Operationalizing RAG Systems (LLMOps)

A comprehensive guide to taking RAG systems from prototype to production — covering the dev-to-production lifecycle, prompt and data versioning, CI/CD pipelines, production monitoring, hallucination management, feedback loops, incident handling, and the LLMOps toolchain.

---

## Table of Contents

1. [What Is LLMOps and Why It Differs from MLOps](#1-what-is-llmops-and-why-it-differs-from-mlops)
2. [Dev to Test to Production Lifecycle](#2-dev-to-test-to-production-lifecycle)
3. [Prompt Versioning](#3-prompt-versioning)
4. [Dataset and Embedding Version Control](#4-dataset-and-embedding-version-control)
5. [CI/CD Concepts for GenAI Applications](#5-cicd-concepts-for-genai-applications)
6. [Monitoring: Latency, Token Usage, Cost, Retrieval Quality](#6-monitoring-latency-token-usage-cost-retrieval-quality)
7. [Hallucination Management Strategies](#7-hallucination-management-strategies)
8. [Feedback Loops and Continuous Improvement](#8-feedback-loops-and-continuous-improvement)
9. [Incident Handling for GenAI Applications](#9-incident-handling-for-genai-applications)
10. [Tools: LangSmith, MLflow, Arize Phoenix, Weights & Biases](#10-tools-langsmith-mlflow-arize-phoenix-weights--biases)
11. [Lab: Production-Grade RAG Reference Architecture](#11-lab-production-grade-rag-reference-architecture)

---

## 1. What Is LLMOps and Why It Differs from MLOps

### The Operationalization Gap

Most RAG tutorials end at a working prototype. Closing the gap to production requires a discipline called **LLMOps** — the set of practices, tools, and processes for deploying, monitoring, and continuously improving LLM-based systems in production.

```
PROTOTYPE → PRODUCTION GAP:

  Prototype (day 1):            Production (month 6):
  ┌─────────────────┐           ┌──────────────────────────────────────┐
  │ Jupyter notebook│           │ Versioned prompts                    │
  │ Hardcoded prompt│           │ Evaluated embeddings                 │
  │ Local Chroma DB │           │ Pinecone / Weaviate (managed)        │
  │ One test query  │           │ 500K queries/month                   │
  │ No logging      │           │ Audit logs, LangSmith traces         │
  │ Manual testing  │           │ CI/CD pipeline with eval gate        │
  │ No cost control │           │ Cost dashboard, budget alerts        │
  │ No fallback     │           │ Fallback models, circuit breakers    │
  └─────────────────┘           └──────────────────────────────────────┘
```

### How LLMOps Differs from Classical MLOps

| Dimension             | Classical MLOps                       | LLMOps                                        |
| --------------------- | ------------------------------------- | --------------------------------------------- |
| **Model artefact**    | Trained weights, scikit-learn `.pkl`  | Prompt template + LLM API call                |
| **"Training"**        | Expensive GPU runs, versioned         | Prompt editing (free, but poorly tracked)     |
| **Evaluation**        | Accuracy, F1 on held-out set          | Faithfulness, relevance, correctness via LLM  |
| **Deployment unit**   | Docker container with model           | Prompt + retriever config + LLM endpoint      |
| **Drift detection**   | Feature/label distribution shift      | Retrieval quality drop, answer quality drift  |
| **Cost model**        | Infra cost (GPU/CPU)                  | Per-token API cost + vector search cost       |
| **Failure mode**      | Wrong prediction with a number        | Plausible-sounding hallucination              |
| **Rollback**          | Revert model version                  | Revert prompt + re-index if embedding changed |

### The LLMOps Flywheel

```
         ┌────────────────────────────────────────────┐
         │                                            │
    ┌────▼─────┐    ┌──────────┐    ┌─────────────┐  │
    │  Build   │───►│ Evaluate │───►│   Deploy    │  │
    │ & Version│    │ (CI/CD)  │    │ (Canary →   │  │
    │          │    │          │    │  Full)      │  │
    └──────────┘    └──────────┘    └──────┬──────┘  │
                                           │         │
    ┌──────────┐    ┌──────────┐    ┌──────▼──────┐  │
    │ Improve  │◄───│ Feedback │◄───│   Monitor   │  │
    │ (prompt/ │    │  Loop    │    │ (quality,   │  │
    │  data/   │    │          │    │  cost,      │  │
    │  model)  │    │          │    │  latency)   │  │
    └──────────┘    └──────────┘    └─────────────┘  │
         │                                            │
         └────────────────────────────────────────────┘
```

---

## 2. Dev to Test to Production Lifecycle

### The Three-Environment Model

```
DEV                    STAGING / TEST              PRODUCTION
────────               ──────────────              ──────────
Local machine          Shared infra                Customer traffic
Small doc subset        Full doc corpus             Full doc corpus
Hardcoded .env         Secrets manager             Secrets manager
Manual testing         Automated eval suite        Real-time monitoring
No SLA                 SLA validation              SLA enforced
Free-form prompt edits Prompt versioned + tested   Immutable deployed prompt
Local Chroma           Managed vector DB           Managed vector DB + replicas
```

### Environment Configuration Pattern

```python
# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Environment
    env: str = "development"         # development | staging | production
    
    # LLM
    llm_model:        str   = "gpt-4o-mini"
    llm_temperature:  float = 0.0
    llm_max_tokens:   int   = 1024
    
    # Embeddings
    embedding_model:  str = "text-embedding-3-small"
    embedding_dims:   int = 1536
    
    # Retrieval
    retrieval_k:      int   = 5
    retrieval_k_wide: int   = 20   # before re-ranking
    hybrid_alpha:     float = 0.6  # dense weight
    
    # Vector store
    vector_store_type:      str = "chroma"       # chroma | pinecone | weaviate
    vector_store_url:       str = "http://localhost:8000"
    vector_store_api_key:   str = ""
    vector_collection_name: str = "rag_dev"
    
    # Monitoring
    langsmith_project:      str = "rag-dev"
    langsmith_tracing:      bool = False
    log_level:              str = "DEBUG"
    
    # Cost controls
    monthly_token_budget:   int = 1_000_000   # tokens
    max_tokens_per_request: int = 8_000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    return Settings()

# Usage
settings = get_settings()
```

```bash
# .env.development
ENV=development
LLM_MODEL=gpt-4o-mini
LANGSMITH_TRACING=false
VECTOR_COLLECTION_NAME=rag_dev

# .env.staging
ENV=staging
LLM_MODEL=gpt-4o-mini
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-staging
VECTOR_COLLECTION_NAME=rag_staging

# .env.production
ENV=production
LLM_MODEL=gpt-4o-mini
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=rag-production
VECTOR_COLLECTION_NAME=rag_prod
MONTHLY_TOKEN_BUDGET=10000000
```

### Deployment Pipeline Overview

```
Developer push to feature branch
        │
        ▼
[GitHub Actions CI]
  ├── lint + type check
  ├── unit tests (prompt parsing, chunking logic)
  ├── RAG eval gate (Recall@K > 0.75, Faithfulness > 0.80)
  └── security scan (bandit, dependency audit)
        │
        ▼ (all gates pass)
[Merge to main → auto-deploy to STAGING]
  ├── Integration tests against staging vector DB
  ├── Load test (p95 latency < 3s at 50 rps)
  └── Smoke tests (golden query set)
        │
        ▼ (manual approval)
[Canary deploy to PRODUCTION — 10% traffic]
  ├── Monitor: error rate, latency, faithfulness
  └── After 30 min: promote to 100% OR rollback
```

---

## 3. Prompt Versioning

### Why Prompt Versioning Is Critical

A prompt is **code**. Changing a prompt changes system behaviour just as a code change does — but without version control it is invisible, irreversible, and untestable.

```
WITHOUT prompt versioning:                WITH prompt versioning:

  "Who changed the prompt?"                v1.2.0 — committed by alice
  "I don't know, someone edited           "Add grounding instruction — fixes
   the string in the Python file"          hallucination on policy queries"
                                           Recall@5: 0.71 → 0.79
  "The system was better last week"        Faithfulness: 0.68 → 0.87
  "I can't reproduce the old behaviour"    Rollback: git revert abc123
```

### Strategy 1 — Git-Based Prompt Versioning

Store prompts as versioned files in the repository. The simplest, most auditable approach.

```
prompts/
├── system/
│   ├── hr_assistant_v1.0.0.txt
│   ├── hr_assistant_v1.1.0.txt    ← current production
│   └── hr_assistant_v2.0.0.txt    ← in development
├── retrieval/
│   ├── query_rewrite_v1.0.0.txt
│   └── hyde_v1.0.0.txt
└── evaluation/
    ├── faithfulness_judge_v1.0.0.txt
    └── relevance_judge_v1.0.0.txt
```

```python
# prompts/loader.py
from pathlib import Path
import re

PROMPT_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str, version: str = "latest") -> str:
    """Load a versioned prompt template from disk."""
    if version == "latest":
        pattern   = f"{name}_v*.txt"
        matches   = sorted(PROMPT_DIR.glob(f"**/{pattern}"))
        if not matches:
            raise FileNotFoundError(f"No prompt found for '{name}'")
        path = matches[-1]   # lexicographic sort → highest version last
    else:
        path = PROMPT_DIR / f"{name}_v{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt '{name}' version '{version}' not found")
    
    return path.read_text(encoding="utf-8").strip()

def get_prompt_version(name: str, version: str = "latest") -> str:
    """Return the version string being used."""
    if version == "latest":
        pattern = f"{name}_v*.txt"
        matches = sorted(PROMPT_DIR.glob(f"**/{pattern}"))
        version_str = re.search(r"v(\d+\.\d+\.\d+)", matches[-1].name)
        return version_str.group(1) if version_str else "unknown"
    return version

# Usage
system_prompt_text = load_prompt("system/hr_assistant")
system_version     = get_prompt_version("system/hr_assistant")
print(f"Loaded system prompt version {system_version}")
```

### Strategy 2 — LangSmith Prompt Hub

LangSmith provides a hosted prompt registry with versioning, A/B testing, and deployment targeting.

```python
from langsmith import Client
from langchain import hub

client = Client()

# Push a new prompt version
from langchain_core.prompts import ChatPromptTemplate

new_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise HR assistant.
Answer ONLY using the CONTEXT below.
If the answer is not in the context, say "I don't have that information."

CONTEXT:
{context}"""),
    ("human", "{question}"),
])

# Push to LangSmith hub
client.push_prompt(
    "hr-rag-system-prompt",
    object=new_prompt,
    description="v1.2 — adds explicit I-don't-know path, improves faithfulness",
    tags=["production-candidate", "v1.2"],
)

# Pull specific version in application code
prompt_v1_1 = hub.pull("acmecorp/hr-rag-system-prompt:v1.1")   # pinned
prompt_latest = hub.pull("acmecorp/hr-rag-system-prompt")       # latest
```

### Strategy 3 — Prompt Metadata Registry

For teams not using LangSmith, maintain a local JSON registry mapping versions to file paths and evaluation scores.

```python
# prompts/registry.json
{
  "hr_assistant": {
    "current_production": "v1.1.0",
    "versions": {
      "v1.0.0": {
        "file": "system/hr_assistant_v1.0.0.txt",
        "deployed_at": "2025-01-10",
        "eval_faithfulness": 0.68,
        "eval_recall_k5": 0.71,
        "notes": "Initial version"
      },
      "v1.1.0": {
        "file": "system/hr_assistant_v1.1.0.txt",
        "deployed_at": "2025-02-15",
        "eval_faithfulness": 0.87,
        "eval_recall_k5": 0.79,
        "notes": "Add grounding instruction + I-don't-know path"
      }
    }
  }
}
```

```python
import json
from pathlib import Path

class PromptRegistry:
    def __init__(self, registry_path: str = "prompts/registry.json"):
        self.registry_path = Path(registry_path)
        with open(self.registry_path) as f:
            self.registry = json.load(f)
    
    def get_production_prompt(self, name: str) -> tuple[str, str]:
        """Returns (prompt_text, version)."""
        entry   = self.registry[name]
        version = entry["current_production"]
        file    = entry["versions"][version]["file"]
        text    = (Path("prompts") / file).read_text()
        return text, version
    
    def register_new_version(self, name: str, version: str, file: str,
                              eval_scores: dict, notes: str):
        self.registry[name]["versions"][version] = {
            "file": file,
            "deployed_at": None,          # set on promotion to production
            **eval_scores,
            "notes": notes,
        }
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def promote_to_production(self, name: str, version: str):
        from datetime import date
        self.registry[name]["current_production"] = version
        self.registry[name]["versions"][version]["deployed_at"] = str(date.today())
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
        print(f"Promoted {name} v{version} to production.")

registry = PromptRegistry()
prompt_text, version = registry.get_production_prompt("hr_assistant")
```

### Prompt Changelog Best Practices

```
PROMPT CHANGELOG FORMAT:

## v1.2.0 — 2025-03-01 (alice@acmecorp.com)
### Changed
- Added explicit grounding instruction: "Answer ONLY from the context below"
- Added "I don't know" path for out-of-scope questions
### Evaluation
- Faithfulness: 0.87 → 0.92  (+0.05)
- Answer Relevance: 0.82 → 0.85  (+0.03)
- Regression: No regressions on golden_dataset_v2.json (200 samples)
### Deployed
- Staging: 2025-03-01
- Production: 2025-03-03 (after 48h canary)

## v1.1.0 — 2025-02-15 (bob@acmecorp.com)
...
```

---

## 4. Dataset and Embedding Version Control

### The Dataset Versioning Problem

```
WITHOUT dataset versioning:                WITH dataset versioning:

  "Why did recall drop this month?"         Documents corpus: v2.3.1
  "Someone updated the HR docs folder"      Embeddings: index-20250301-v2.3.1
  "Which docs were added/removed?"          Eval dataset: golden_v2.json (200 samples)
  "Was the eval dataset changed too?"       All linked by experiment manifest
  "We can't reproduce last month's scores"  Full reproducibility guaranteed
```

### Document Corpus Versioning

Track every change to the source document set with a manifest file.

```python
# ingest/corpus_manifest.py
import hashlib
import json
from pathlib import Path
from datetime import datetime, UTC

def compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def build_corpus_manifest(docs_dir: str) -> dict:
    docs_path = Path(docs_dir)
    files = sorted(docs_path.rglob("*.*"))
    
    manifest = {
        "version":    f"v{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now(UTC).isoformat(),
        "docs_dir":   docs_dir,
        "total_files": len(files),
        "files": {},
    }
    
    for f in files:
        manifest["files"][str(f.relative_to(docs_path))] = {
            "sha256":        compute_file_hash(str(f)),
            "size_bytes":    f.stat().st_size,
            "modified_time": datetime.fromtimestamp(f.stat().st_mtime, UTC).isoformat(),
        }
    
    return manifest

def save_manifest(manifest: dict, output_path: str = "ingest/corpus_manifest.json"):
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Corpus manifest saved: {manifest['total_files']} files, version {manifest['version']}")

def detect_corpus_changes(old_manifest: dict, new_manifest: dict) -> dict:
    old_files = old_manifest["files"]
    new_files = new_manifest["files"]
    
    added    = set(new_files) - set(old_files)
    removed  = set(old_files) - set(new_files)
    modified = {
        f for f in set(old_files) & set(new_files)
        if old_files[f]["sha256"] != new_files[f]["sha256"]
    }
    
    return {
        "added":    sorted(added),
        "removed":  sorted(removed),
        "modified": sorted(modified),
        "changed":  bool(added or removed or modified),
    }

# Usage
manifest = build_corpus_manifest("./docs")
save_manifest(manifest)
```

### Embedding Index Versioning

Every time the corpus changes OR the embedding model changes, you need a new index. Name indices to encode both.

```python
# ingest/index_manager.py
import json
from datetime import datetime, UTC

class EmbeddingIndexManager:
    """Tracks which corpus version + embedding model produced each index."""
    
    INDEX_REGISTRY = "ingest/index_registry.json"
    
    def __init__(self):
        try:
            with open(self.INDEX_REGISTRY) as f:
                self.registry = json.load(f)
        except FileNotFoundError:
            self.registry = {"current": None, "indices": {}}
    
    def register_index(self, corpus_version: str, embedding_model: str,
                        collection_name: str, chunk_size: int, chunk_overlap: int,
                        eval_recall_k5: float = None) -> str:
        index_id = f"{corpus_version}__{embedding_model.replace('/', '-')}__{chunk_size}"
        
        self.registry["indices"][index_id] = {
            "corpus_version":   corpus_version,
            "embedding_model":  embedding_model,
            "collection_name":  collection_name,
            "chunk_size":       chunk_size,
            "chunk_overlap":    chunk_overlap,
            "created_at":       datetime.now(UTC).isoformat(),
            "eval_recall_k5":   eval_recall_k5,
            "status":           "candidate",
        }
        
        self._save()
        return index_id
    
    def promote_to_production(self, index_id: str):
        if index_id not in self.registry["indices"]:
            raise KeyError(f"Index '{index_id}' not found")
        
        # Mark old production as archived
        if self.registry["current"]:
            old = self.registry["current"]
            self.registry["indices"][old]["status"] = "archived"
        
        self.registry["current"] = index_id
        self.registry["indices"][index_id]["status"] = "production"
        self.registry["indices"][index_id]["promoted_at"] = datetime.now(UTC).isoformat()
        self._save()
        print(f"Index {index_id} promoted to production.")
    
    def get_production_collection(self) -> str:
        if not self.registry["current"]:
            raise RuntimeError("No production index registered")
        return self.registry["indices"][self.registry["current"]]["collection_name"]
    
    def _save(self):
        with open(self.INDEX_REGISTRY, "w") as f:
            json.dump(self.registry, f, indent=2)

# Usage
mgr = EmbeddingIndexManager()
index_id = mgr.register_index(
    corpus_version   = "v20250301-143000",
    embedding_model  = "text-embedding-3-small",
    collection_name  = "rag_prod_20250301",
    chunk_size       = 800,
    chunk_overlap    = 100,
    eval_recall_k5   = 0.83,
)
mgr.promote_to_production(index_id)
```

### Evaluation Dataset Versioning with DVC

For teams managing large evaluation datasets, DVC (Data Version Control) provides Git-like versioning for data files.

```bash
# Install DVC
pip install dvc dvc-s3   # or dvc-gcs, dvc-azure

# Initialise DVC alongside git
dvc init
git add .dvc .dvcignore
git commit -m "initialise DVC"

# Track evaluation dataset
dvc add evaluation/golden_dataset.json
git add evaluation/golden_dataset.json.dvc evaluation/.gitignore
git commit -m "add golden dataset v1.0 (200 samples)"

# Push to remote storage (S3, GCS, Azure Blob)
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc push

# Later: add new samples
# (edit golden_dataset.json)
dvc add evaluation/golden_dataset.json
git commit -m "golden dataset v2.0 — add 50 production failure cases"
dvc push

# Reproduce exact v1.0 dataset for comparison
git checkout <v1.0-commit>
dvc pull
```

### Experiment Manifest

Link all versioned artefacts together for full reproducibility.

```python
# experiments/manifest.py
import json
from datetime import datetime, UTC

def create_experiment_manifest(
    experiment_name: str,
    prompt_version:  str,
    corpus_version:  str,
    index_id:        str,
    eval_dataset_git_sha: str,
    config:          dict,
    results:         dict,
) -> dict:
    return {
        "experiment_name":         experiment_name,
        "created_at":              datetime.now(UTC).isoformat(),
        "artefacts": {
            "prompt_version":          prompt_version,
            "corpus_version":          corpus_version,
            "embedding_index_id":      index_id,
            "eval_dataset_git_sha":    eval_dataset_git_sha,
        },
        "config":  config,
        "results": results,
    }

# Example
manifest = create_experiment_manifest(
    experiment_name      = "chunking-800-hybrid-rerank",
    prompt_version       = "v1.1.0",
    corpus_version       = "v20250301-143000",
    index_id             = "v20250301-143000__text-embedding-3-small__800",
    eval_dataset_git_sha = "abc123def456",
    config = {
        "chunk_size": 800, "chunk_overlap": 100,
        "hybrid_alpha": 0.6, "reranker": "ms-marco-MiniLM-L-6-v2",
        "llm": "gpt-4o-mini",
    },
    results = {
        "precision_k5": 0.79, "recall_k5": 0.83,
        "faithfulness": 0.91, "answer_correctness": 0.78,
        "p95_latency_ms": 1840, "cost_per_1k_queries_usd": 0.92,
    },
)

with open(f"experiments/{manifest['experiment_name']}.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

---

## 5. CI/CD Concepts for GenAI Applications

### Why GenAI CI/CD Is Different

Classical CI/CD validates **deterministic** behaviour. GenAI CI/CD must validate **probabilistic** behaviour — outputs vary, quality is measured by LLM judges, and a "passing" threshold is a statistical boundary, not a binary.

```
Classical CI gate:        GenAI CI gate:
  test_login() → PASS     faithfulness_mean > 0.80  → 0.87 PASS
  test_login() → FAIL     faithfulness_mean > 0.80  → 0.73 FAIL
  (deterministic)         (statistical — run N=50 samples, check mean)
```

### GitHub Actions RAG Evaluation Pipeline

```yaml
# .github/workflows/rag-eval.yml
name: RAG Quality Gate

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  rag-quality-gate:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run prompt linting
        run: python scripts/lint_prompts.py   # validate prompt syntax/format

      - name: Run unit tests
        run: pytest tests/unit/ -v

      - name: Run RAG evaluation gate
        env:
          OPENAI_API_KEY:  ${{ secrets.OPENAI_API_KEY }}
          LANGCHAIN_API_KEY: ${{ secrets.LANGCHAIN_API_KEY }}
          LANGCHAIN_PROJECT: rag-ci-${{ github.run_id }}
          LANGCHAIN_TRACING_V2: "true"
        run: python scripts/eval_gate.py --threshold-file thresholds.json

      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-report-${{ github.run_id }}
          path: evaluation/results/ci_report_*.json

      - name: Comment PR with eval results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = JSON.parse(
              fs.readFileSync('evaluation/results/latest_report.json', 'utf8')
            );
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## RAG Evaluation Results\n\n` +
                    `| Metric | Score | Threshold | Status |\n` +
                    `|--------|-------|-----------|--------|\n` +
                    `| Faithfulness | ${report.faithfulness.toFixed(3)} | ≥ 0.80 | ${report.faithfulness >= 0.80 ? '✅' : '❌'} |\n` +
                    `| Recall@5 | ${report.recall_k5.toFixed(3)} | ≥ 0.75 | ${report.recall_k5 >= 0.75 ? '✅' : '❌'} |\n` +
                    `| Answer Correctness | ${report.answer_correctness.toFixed(3)} | ≥ 0.70 | ${report.answer_correctness >= 0.70 ? '✅' : '❌'} |\n`
            });
```

### The Evaluation Gate Script

```python
# scripts/eval_gate.py
import argparse, json, sys
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_correctness
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Load golden dataset
with open("evaluation/golden_dataset.json") as f:
    golden = json.load(f)["samples"]

# Load current RAG pipeline
from app.rag_pipeline import build_rag_chain
rag_chain = build_rag_chain()

answers, contexts = [], []
for sample in golden:
    output = rag_chain.invoke(sample["question"])
    answers.append(output["result"])
    contexts.append([d.page_content for d in output["source_documents"]])

dataset = Dataset.from_dict({
    "question":     [s["question"]           for s in golden],
    "answer":       answers,
    "contexts":     contexts,
    "ground_truth": [s["ground_truth_answer"] for s in golden],
})

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

results = evaluate(dataset, metrics=[faithfulness, context_recall, answer_correctness],
                   llm=ragas_llm, embeddings=ragas_emb)
scores = {
    "faithfulness":       float(results["faithfulness"]),
    "recall_k5":          float(results["context_recall"]),
    "answer_correctness": float(results["answer_correctness"]),
}

# Load thresholds
parser = argparse.ArgumentParser()
parser.add_argument("--threshold-file", default="thresholds.json")
args = parser.parse_args()

with open(args.threshold_file) as f:
    thresholds = json.load(f)

# Save report
import os, datetime
os.makedirs("evaluation/results", exist_ok=True)
ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
report = {"timestamp": ts, **scores, "thresholds": thresholds, "passed": True}

failures = []
for metric, threshold in thresholds.items():
    score = scores.get(metric, 0)
    if score < threshold:
        failures.append(f"{metric}: {score:.3f} < {threshold} (threshold)")
        report["passed"] = False

with open(f"evaluation/results/ci_report_{ts}.json", "w") as f:
    json.dump(report, f, indent=2)
with open("evaluation/results/latest_report.json", "w") as f:
    json.dump(report, f, indent=2)

if failures:
    print("❌ RAG QUALITY GATE FAILED:")
    for f in failures:
        print(f"   {f}")
    sys.exit(1)
else:
    print("✅ RAG quality gate passed.")
    for metric, score in scores.items():
        print(f"   {metric}: {score:.3f} ≥ {thresholds.get(metric, 'N/A')}")
    sys.exit(0)
```

```json
// thresholds.json
{
  "faithfulness":       0.80,
  "recall_k5":          0.75,
  "answer_correctness": 0.70
}
```

### Canary Deployment for RAG

```python
# deployment/canary.py — traffic splitting between prompt versions

import random
from app.rag_pipeline import build_rag_chain

# Load two versions of the pipeline
chain_v1 = build_rag_chain(prompt_version="v1.1.0")  # current production
chain_v2 = build_rag_chain(prompt_version="v1.2.0")  # canary

CANARY_TRAFFIC_PERCENT = 10   # 10% to new version

def route_request(query: str, user: dict) -> tuple[str, str]:
    """Returns (answer, version_used)."""
    use_canary = random.random() < (CANARY_TRAFFIC_PERCENT / 100)
    
    if use_canary:
        return chain_v2.invoke(query), "v1.2.0"
    else:
        return chain_v1.invoke(query), "v1.1.0"

# Monitor canary performance separately in LangSmith
# by tagging traces with the version label
```

---

## 6. Monitoring: Latency, Token Usage, Cost, Retrieval Quality

### The Production Monitoring Stack

```
┌──────────────────────────────────────────────────────────┐
│  REAL-TIME METRICS COLLECTION                            │
│                                                          │
│  Per-request:                                            │
│    latency_ms, tokens_in, tokens_out, model, retrieval_k │
│    faithfulness_score (sampled), cache_hit               │
│                                                          │
│  Aggregated (rolling 1h, 24h, 7d windows):              │
│    p50/p95/p99 latency, total_tokens, cost_usd           │
│    mean_faithfulness, error_rate, cache_hit_rate         │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│  ALERTING                                                │
│  faithfulness < 0.75 → PagerDuty (quality degradation)  │
│  error_rate   > 1%   → Slack (pipeline failure)         │
│  cost > budget       → Slack (budget alert)             │
│  p95_latency > 3s    → Slack (performance alert)        │
└──────────────────────────────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────────────────────────────┐
│  DASHBOARDS                                              │
│  LangSmith / Grafana / Arize Phoenix / W&B               │
└──────────────────────────────────────────────────────────┘
```

### Instrumented RAG Pipeline

```python
# monitoring/instrumented_chain.py
import time, json
from dataclasses import dataclass, field
from datetime import datetime, UTC
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

@dataclass
class RequestMetrics:
    request_id:       str   = ""
    timestamp:        str   = ""
    model:            str   = ""
    tokens_input:     int   = 0
    tokens_output:    int   = 0
    total_tokens:     int   = 0
    cost_usd:         float = 0.0
    retrieval_k:      int   = 0
    retrieval_ms:     float = 0.0
    generation_ms:    float = 0.0
    total_latency_ms: float = 0.0
    cache_hit:        bool  = False
    error:            str   = ""

# Token cost lookup (USD per 1000 tokens, input/output)
TOKEN_COSTS = {
    "gpt-4o":            (0.0025, 0.010),
    "gpt-4o-mini":       (0.00015, 0.0006),
    "text-embedding-3-small": (0.00002, 0.0),
}

def compute_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    if model not in TOKEN_COSTS:
        return 0.0
    cost_in, cost_out = TOKEN_COSTS[model]
    return (tokens_in / 1000 * cost_in) + (tokens_out / 1000 * cost_out)

class MetricsCallbackHandler(BaseCallbackHandler):
    def __init__(self, metrics: RequestMetrics):
        self.metrics = metrics
    
    def on_llm_end(self, response, **kwargs):
        usage = getattr(response, "llm_output", {}).get("token_usage", {})
        self.metrics.tokens_input  = usage.get("prompt_tokens", 0)
        self.metrics.tokens_output = usage.get("completion_tokens", 0)
        self.metrics.total_tokens  = usage.get("total_tokens", 0)
        self.metrics.cost_usd = compute_cost(
            self.metrics.model,
            self.metrics.tokens_input,
            self.metrics.tokens_output,
        )

class MetricsStore:
    """In-memory rolling metrics store. Replace with Prometheus/InfluxDB in production."""
    
    def __init__(self):
        self.records: list[RequestMetrics] = []
    
    def record(self, m: RequestMetrics):
        self.records.append(m)
        if len(self.records) > 10_000:
            self.records = self.records[-5_000:]
    
    def rolling_summary(self, last_n: int = 1000) -> dict:
        import statistics
        recent = self.records[-last_n:]
        if not recent:
            return {}
        
        latencies = [r.total_latency_ms for r in recent if not r.error]
        costs     = [r.cost_usd         for r in recent]
        tokens    = [r.total_tokens     for r in recent]
        errors    = [r for r in recent if r.error]
        
        return {
            "sample_size":      len(recent),
            "error_rate":       len(errors) / len(recent),
            "latency_p50_ms":   statistics.median(latencies) if latencies else 0,
            "latency_p95_ms":   sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "cost_total_usd":   sum(costs),
            "cost_avg_usd":     sum(costs) / len(costs) if costs else 0,
            "tokens_avg":       sum(tokens) / len(tokens) if tokens else 0,
            "cache_hit_rate":   sum(1 for r in recent if r.cache_hit) / len(recent),
        }

metrics_store = MetricsStore()

def instrumented_rag_invoke(query: str, rag_chain, model: str = "gpt-4o-mini") -> dict:
    import uuid
    m = RequestMetrics(
        request_id = str(uuid.uuid4()),
        timestamp  = datetime.now(UTC).isoformat(),
        model      = model,
    )
    
    t_total_start = time.time()
    
    try:
        t_retrieval_start = time.time()
        # retrieval timing is embedded in the chain; separate if using custom retriever
        result = rag_chain.invoke(
            query,
            config={"callbacks": [MetricsCallbackHandler(m)]},
        )
        m.total_latency_ms = (time.time() - t_total_start) * 1000
    except Exception as e:
        m.error = str(e)
        m.total_latency_ms = (time.time() - t_total_start) * 1000
        raise
    finally:
        metrics_store.record(m)
    
    return result
```

### Retrieval Quality Monitoring (Sampled)

Running full RAGAS evaluation on every request is too slow and expensive. Use **sampled evaluation** — run quality metrics on a random sample of production traffic.

```python
import random
from ragas import evaluate
from ragas.metrics import faithfulness

SAMPLING_RATE = 0.05   # evaluate 5% of production requests

quality_scores: list[float] = []

def production_rag_with_quality_sampling(query: str, rag_chain) -> str:
    output = rag_chain.invoke(query, return_source_documents=True)
    answer = output["result"]
    docs   = output["source_documents"]
    
    if random.random() < SAMPLING_RATE:
        context = [d.page_content for d in docs]
        try:
            from datasets import Dataset
            dataset = Dataset.from_dict({
                "question": [query],
                "answer":   [answer],
                "contexts": [context],
            })
            result = evaluate(dataset, metrics=[faithfulness], llm=ragas_llm)
            score  = float(result["faithfulness"])
            quality_scores.append(score)
            
            if score < 0.70:
                alert_quality_degradation(query, answer, score)
        except Exception:
            pass   # never let monitoring break the main path
    
    return answer

def get_rolling_faithfulness(window: int = 100) -> float:
    if not quality_scores:
        return -1.0
    return sum(quality_scores[-window:]) / len(quality_scores[-window:])
```

### Cost and Token Budget Enforcement

```python
from threading import Lock

class TokenBudgetManager:
    def __init__(self, monthly_budget_usd: float = 100.0):
        self.monthly_budget_usd = monthly_budget_usd
        self.spent_usd          = 0.0
        self.lock               = Lock()
    
    def record_cost(self, cost_usd: float):
        with self.lock:
            self.spent_usd += cost_usd
    
    def check_budget(self) -> dict:
        with self.lock:
            pct = self.spent_usd / self.monthly_budget_usd * 100
            return {
                "spent_usd":   round(self.spent_usd, 4),
                "budget_usd":  self.monthly_budget_usd,
                "pct_used":    round(pct, 1),
                "over_budget": self.spent_usd > self.monthly_budget_usd,
            }
    
    def enforce_budget(self):
        status = self.check_budget()
        if status["over_budget"]:
            raise RuntimeError(
                f"Monthly token budget exhausted "
                f"(${status['spent_usd']:.2f} / ${status['budget_usd']:.2f}). "
                f"Request rejected."
            )
        if status["pct_used"] > 80:
            send_budget_alert(status)   # warn at 80%

budget_manager = TokenBudgetManager(monthly_budget_usd=200.0)
```

---

## 7. Hallucination Management Strategies

### Why Hallucination Never Fully Disappears

Even with optimised retrieval and grounding prompts (Module-2 and Module-3), hallucination remains a residual risk. Production RAG systems need active, ongoing hallucination management — not just a one-time prompt fix.

```
HALLUCINATION SOURCES IN PRODUCTION RAG:

  Source 1 — Parametric knowledge bleed:
    LLM "knows" the answer from training and overrides retrieved context.

  Source 2 — Context window saturation:
    Too many chunks injected; LLM loses track of what the context says.

  Source 3 — Ambiguous retrieval:
    Retrieved chunks are partially relevant; LLM fills gaps with guesses.

  Source 4 — Out-of-distribution queries:
    Query type the system was never evaluated on; retrieval returns noise.

  Source 5 — Model updates:
    LLM provider silently updates the model; new version hallucinates differently.
```

### Strategy 1 — Factual Consistency Checking (Post-Generation)

Use an LLM judge to verify every claim in the answer before delivering it.

```python
from langchain_openai import ChatOpenAI
import json

checker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CONSISTENCY_PROMPT = """You are a factual consistency checker.

Retrieved Context:
{context}

Generated Answer:
{answer}

For each factual claim in the Generated Answer:
1. Identify the claim.
2. Check whether it is directly supported, partially supported, or contradicted by the Context.

Return JSON:
{{
  "claims": [
    {{
      "claim": "text of claim",
      "status": "supported" | "unsupported" | "contradicted",
      "evidence": "the context sentence that supports/contradicts it, or null"
    }}
  ],
  "overall_faithfulness": <0.0-1.0>,
  "safe_to_deliver": true/false
}}"""

def check_factual_consistency(context: str, answer: str) -> dict:
    result = checker_llm.invoke(
        CONSISTENCY_PROMPT.format(context=context, answer=answer)
    )
    return json.loads(result.content)

def hallucination_guarded_response(query: str, rag_chain,
                                    faithfulness_threshold: float = 0.80) -> str:
    output  = rag_chain.invoke(query, return_source_documents=True)
    answer  = output["result"]
    context = "\n\n".join(d.page_content for d in output["source_documents"])
    
    check = check_factual_consistency(context, answer)
    
    if not check["safe_to_deliver"] or check["overall_faithfulness"] < faithfulness_threshold:
        unsupported = [c["claim"] for c in check["claims"] if c["status"] != "supported"]
        
        # Option A: Return with caveat
        caveat = (
            "\n\n⚠️ Note: Parts of this answer could not be verified against the "
            "source documents. Please confirm with your HR team before acting on this."
        )
        return answer + caveat
    
    return answer
```

### Strategy 2 — Retrieval Sufficiency Check (Pre-Generation)

Before generating, verify the retrieved context is actually sufficient to answer the question. If not, say so rather than guessing.

```python
SUFFICIENCY_PROMPT = """You are assessing whether a set of retrieved documents
contains sufficient information to answer a question accurately.

Question: {question}

Retrieved Context:
{context}

Assessment:
- Does the context contain a direct answer to the question?
- Is the context relevant to the question?

Return JSON:
{{
  "sufficient": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "suggested_response": "if not sufficient, what to tell the user"
}}"""

def check_context_sufficiency(question: str, context: str) -> dict:
    result = checker_llm.invoke(
        SUFFICIENCY_PROMPT.format(question=question, context=context)
    )
    return json.loads(result.content)

def sufficiency_guarded_rag(query: str, retriever, rag_chain) -> str:
    docs    = retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    
    sufficiency = check_context_sufficiency(query, context)
    
    if not sufficiency["sufficient"] or sufficiency["confidence"] < 0.6:
        return (
            f"I don't have sufficient information in the available documents "
            f"to answer this accurately. "
            f"{sufficiency.get('suggested_response', 'Please contact your HR team directly.')}"
        )
    
    return rag_chain.invoke({"question": query, "context": context})
```

### Strategy 3 — Citation-Grounded Generation

Force the LLM to cite the specific source sentence for every claim, making verification automatic.

```python
CITATION_PROMPT = """You are a precise assistant. Answer the question using ONLY
the numbered sources below. After each factual claim, cite the source number in brackets.

{formatted_sources}

Question: {question}

Answer with citations (e.g. "The return window is 30 days [Source 1]."):"""

def citation_rag(query: str, docs: list) -> dict:
    formatted_sources = "\n\n".join(
        f"[Source {i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )
    
    answer = (ChatOpenAI(model="gpt-4o-mini", temperature=0)
              .invoke(CITATION_PROMPT.format(
                  formatted_sources=formatted_sources,
                  question=query))
              ).content
    
    # Verify all cited sources actually exist
    import re
    cited = set(re.findall(r"\[Source (\d+)\]", answer))
    available = set(str(i+1) for i in range(len(docs)))
    hallucinated_citations = cited - available
    
    if hallucinated_citations:
        # LLM cited a source that doesn't exist
        answer += "\n\n[Warning: Citation verification failed. Please check sources manually.]"
    
    return {
        "answer":   answer,
        "sources":  docs,
        "cited":    sorted(cited),
        "hallucinated_citations": sorted(hallucinated_citations),
    }
```

### Strategy 4 — Hallucination Rate Tracking and Threshold Alerting

```python
from collections import deque
from datetime import datetime, UTC

class HallucinationTracker:
    def __init__(self, window: int = 200, alert_threshold: float = 0.15):
        self.window    = window
        self.threshold = alert_threshold
        self.scores:   deque = deque(maxlen=window)
        self.timestamps: deque = deque(maxlen=window)
    
    def record(self, faithfulness_score: float):
        self.scores.append(faithfulness_score)
        self.timestamps.append(datetime.now(UTC).isoformat())
    
    @property
    def hallucination_rate(self) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for s in self.scores if s < 0.70) / len(self.scores)
    
    @property
    def mean_faithfulness(self) -> float:
        if not self.scores:
            return 1.0
        return sum(self.scores) / len(self.scores)
    
    def check_alert(self) -> bool:
        rate = self.hallucination_rate
        if rate > self.threshold:
            send_hallucination_alert(rate, self.mean_faithfulness)
            return True
        return False

hallucination_tracker = HallucinationTracker(window=200, alert_threshold=0.15)
```

---

## 8. Feedback Loops and Continuous Improvement

### The Improvement Cycle

```
Production traffic
      │
      ├── User explicit feedback  (thumbs up/down, star rating)
      ├── User implicit feedback  (query rephrasing, session abandonment)
      └── Automated quality scores (sampled faithfulness, relevance)
      │
      ▼
Feedback store
      │
      ├── Negative feedback → priority annotation queue
      ├── Confirmed failures → added to golden dataset
      └── Aggregate trends → inform next sprint
      │
      ▼
Improvement action
      ├── Add failing cases to golden dataset
      ├── Adjust chunking / embedding / reranking
      ├── Update prompts (version + evaluate)
      └── Retrain or swap embedding model
      │
      ▼
CI/CD eval gate → deploy → back to production traffic
```

### Collecting User Feedback

```python
# api/feedback.py
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, UTC
import json, os

app = FastAPI()

class FeedbackPayload(BaseModel):
    request_id:  str
    query:       str
    answer:      str
    rating:      str           # "thumbs_up" | "thumbs_down" | "1" - "5"
    comment:     str = ""
    user_id:     str = "anonymous"

FEEDBACK_FILE = "feedback/user_feedback.jsonl"

@app.post("/feedback")
async def submit_feedback(payload: FeedbackPayload):
    os.makedirs("feedback", exist_ok=True)
    record = {
        "timestamp":  datetime.now(UTC).isoformat(),
        **payload.model_dump(),
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": "recorded", "request_id": payload.request_id}
```

### Feedback Triage Pipeline

```python
# feedback/triage.py
import json
from pathlib import Path

def load_feedback(path: str = "feedback/user_feedback.jsonl") -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records

def triage_feedback(records: list[dict]) -> dict:
    negative = [r for r in records if r["rating"] in ("thumbs_down", "1", "2")]
    positive = [r for r in records if r["rating"] in ("thumbs_up", "4", "5")]
    
    # Cluster negative feedback by query similarity
    # (simplified — in production use embedding clustering)
    categories = {
        "wrong_answer":   [r for r in negative if "wrong" in r.get("comment", "").lower()
                                                or "incorrect" in r.get("comment", "").lower()],
        "incomplete":     [r for r in negative if "incomplete" in r.get("comment", "").lower()
                                                or "missing" in r.get("comment", "").lower()],
        "hallucination":  [r for r in negative if "made up" in r.get("comment", "").lower()
                                                or "hallucin" in r.get("comment", "").lower()],
        "out_of_scope":   [r for r in negative if "not relevant" in r.get("comment", "").lower()],
        "uncategorised":  [],
    }
    
    categorised = set()
    for cat, items in categories.items():
        if cat != "uncategorised":
            for item in items:
                categorised.add(item["request_id"])
    
    categories["uncategorised"] = [
        r for r in negative if r["request_id"] not in categorised
    ]
    
    return {
        "total":            len(records),
        "positive":         len(positive),
        "negative":         len(negative),
        "satisfaction_rate": len(positive) / len(records) if records else 0,
        "categories":       {k: len(v) for k, v in categories.items()},
        "priority_queue":   negative[:20],   # top 20 for human review
    }

def promote_to_golden_dataset(feedback_record: dict, correct_answer: str,
                               relevant_doc_ids: list[str]):
    """Promote an annotated failure case into the golden dataset."""
    golden_sample = {
        "question":           feedback_record["query"],
        "ground_truth_answer": correct_answer,
        "relevant_doc_ids":   relevant_doc_ids,
        "source":             "production_feedback",
        "original_bad_answer": feedback_record["answer"],
        "failure_category":   feedback_record.get("comment", ""),
    }
    
    golden_path = Path("evaluation/golden_dataset.json")
    with open(golden_path) as f:
        dataset = json.load(f)
    
    dataset["samples"].append(golden_sample)
    dataset["version"] = f"v_extended_{datetime.now(UTC).strftime('%Y%m%d')}"
    
    with open(golden_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Promoted failure case to golden dataset. "
          f"Total samples: {len(dataset['samples'])}")
```

### Implicit Feedback via Query Rephrasing Detection

```python
# feedback/implicit.py
from collections import defaultdict

class SessionRephraseDetector:
    """
    Detects when a user rephrases a query within the same session —
    a strong signal the previous answer was unsatisfactory.
    """
    
    def __init__(self):
        self.sessions: dict[str, list[dict]] = defaultdict(list)
    
    def record_query(self, session_id: str, query: str, request_id: str):
        self.sessions[session_id].append({
            "query":      query,
            "request_id": request_id,
        })
    
    def detect_rephrases(self, session_id: str,
                          similarity_threshold: float = 0.85) -> list[dict]:
        """Returns pairs of (original, rephrase) queries in the session."""
        from sentence_transformers import SentenceTransformer, util
        
        history = self.sessions.get(session_id, [])
        if len(history) < 2:
            return []
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        rephrases = []
        
        for i in range(1, len(history)):
            prev    = history[i - 1]
            current = history[i]
            
            emb_prev    = model.encode(prev["query"])
            emb_current = model.encode(current["query"])
            sim         = float(util.cos_sim(emb_prev, emb_current))
            
            if sim > similarity_threshold:
                rephrases.append({
                    "original_query":    prev["query"],
                    "original_req_id":   prev["request_id"],
                    "rephrase_query":    current["query"],
                    "rephrase_req_id":   current["request_id"],
                    "similarity":        sim,
                })
        
        return rephrases
```

---

## 9. Incident Handling for GenAI Applications

### GenAI Incident Types

```
SEVERITY LEVELS:

  SEV-1 (Critical — immediate response):
    └── Mass hallucination: faithfulness drops below 0.50 across all requests
    └── Data leakage: confirmed cross-user document exposure
    └── System outage: RAG API returning errors for > 5% of requests
    └── Security breach: prompt injection attack succeeded

  SEV-2 (High — response within 1 hour):
    └── Quality degradation: faithfulness drops below threshold on > 20% of requests
    └── Latency spike: p95 > 2× baseline sustained for > 15 min
    └── Cost overrun: actual spend > 150% of forecast

  SEV-3 (Medium — response within 1 business day):
    └── Single query type hallucinating (e.g., only date-related queries)
    └── Retrieval recall drop for specific document category
    └── Intermittent LLM provider errors (< 1% error rate)
```

### Incident Response Runbook

```python
# incidents/runbook.py

class RAGIncidentResponder:
    
    def handle_hallucination_spike(self, current_faithfulness: float):
        """SEV-1/SEV-2 runbook for sudden hallucination increase."""
        
        print("=== HALLUCINATION INCIDENT RUNBOOK ===")
        print(f"Current faithfulness: {current_faithfulness:.2f}")
        
        steps = [
            "1. CHECK recent prompt changes (git log prompts/ -5)",
            "2. CHECK recent embedding index changes (index_registry.json)",
            "3. CHECK LLM provider status page for model updates",
            "4. CHECK if affected queries share a pattern (query log analysis)",
            "5. RUN faithfulness check on golden dataset with current config",
            "6. IF prompt changed recently: ROLLBACK prompt version",
            "7. IF embedding changed: ROLLBACK to previous index",
            "8. IF provider issue: SWITCH to fallback model",
            "9. MONITOR for 30 min post-fix before closing incident",
        ]
        for step in steps:
            print(f"  {step}")
    
    def handle_latency_spike(self, current_p95_ms: float, baseline_p95_ms: float):
        """SEV-2 runbook for latency degradation."""
        
        ratio = current_p95_ms / baseline_p95_ms
        print(f"=== LATENCY INCIDENT RUNBOOK (ratio: {ratio:.1f}x) ===")
        
        steps = [
            "1. CHECK vector store response times (retrieval latency separately)",
            "2. CHECK LLM API response times (generation latency separately)",
            "3. CHECK if re-ranker model is causing bottleneck",
            "4. CHECK concurrent request count vs baseline",
            "5. IF vector store slow: check index fragmentation, scale read replicas",
            "6. IF LLM API slow: check provider status, consider switching region",
            "7. IF re-ranker slow: reduce initial K or switch to lighter model",
            "8. ENABLE semantic cache if not already active",
        ]
        for step in steps:
            print(f"  {step}")
    
    def handle_security_incident(self, incident_type: str, user_id: str):
        """SEV-1 runbook for security incidents."""
        
        print(f"=== SECURITY INCIDENT RUNBOOK: {incident_type} ===")
        
        if incident_type == "prompt_injection":
            steps = [
                f"1. BLOCK user_id={user_id} immediately",
                "2. REVIEW audit log for all requests from this user (last 24h)",
                "3. CHECK if injection succeeded (did system prompt leak?)",
                "4. REVIEW injection detection patterns — update if new pattern found",
                "5. NOTIFY security team with full audit log extract",
                "6. REVIEW other users' requests for similar patterns",
            ]
        elif incident_type == "data_leakage":
            steps = [
                f"1. IDENTIFY affected user_id={user_id} and the leaked document",
                "2. DETERMINE which user(s) saw data they shouldn't have",
                "3. NOTIFY DPO within 1 hour (GDPR 72-hour breach notification clock starts)",
                "4. AUDIT access control filter configuration",
                "5. TEMPORARILY restrict access to the affected data category",
                "6. DOCUMENT incident for GDPR breach register",
            ]
        else:
            steps = ["1. Escalate to CISO immediately"]
        
        for step in steps:
            print(f"  {step}")
```

### Circuit Breaker for LLM Failures

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED   = "closed"    # normal — requests pass through
    OPEN     = "open"      # tripped — requests rejected immediately
    HALF_OPEN = "half_open" # testing — one request allowed through

class LLMCircuitBreaker:
    def __init__(self, failure_threshold: int = 5,
                 recovery_timeout_s: float = 60.0,
                 success_threshold: int = 2):
        self.failure_threshold  = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.success_threshold  = success_threshold
        
        self.state             = CircuitState.CLOSED
        self.failure_count     = 0
        self.success_count     = 0
        self.last_failure_time: float = 0.0
    
    def call(self, fn, *args, fallback=None, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout_s:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                if fallback:
                    return fallback(*args, **kwargs)
                raise RuntimeError("Circuit breaker OPEN — LLM unavailable")
        
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if fallback:
                return fallback(*args, **kwargs)
            raise
    
    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count    += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Fallback: use a smaller/local model when primary LLM is down
def fallback_llm(query: str) -> str:
    return ("I'm temporarily unable to process your request. "
            "Please try again in a few minutes or contact support.")

breaker = LLMCircuitBreaker(failure_threshold=5, recovery_timeout_s=60)

def resilient_llm_call(query: str, llm) -> str:
    return breaker.call(
        llm.invoke, query,
        fallback=fallback_llm,
    )
```

### Rollback Procedure

```python
# ops/rollback.py

class RAGRollbackManager:
    """Orchestrates prompt, index, and config rollback."""
    
    def __init__(self, prompt_registry: PromptRegistry,
                       index_manager: EmbeddingIndexManager):
        self.prompts = prompt_registry
        self.indices = index_manager
    
    def rollback_prompt(self, name: str, to_version: str):
        """Rollback prompt to a previous version."""
        available = list(self.prompts.registry[name]["versions"].keys())
        if to_version not in available:
            raise ValueError(f"Version {to_version} not found. Available: {available}")
        
        current = self.prompts.registry[name]["current_production"]
        self.prompts.promote_to_production(name, to_version)
        print(f"Prompt '{name}' rolled back: {current} → {to_version}")
    
    def rollback_index(self, to_index_id: str):
        """Switch active index to a previous version."""
        current = self.indices.registry["current"]
        self.indices.promote_to_production(to_index_id)
        print(f"Embedding index rolled back: {current} → {to_index_id}")
    
    def full_rollback(self, prompt_name: str, prompt_version: str,
                       index_id: str):
        """Roll back both prompt and index simultaneously."""
        self.rollback_prompt(prompt_name, prompt_version)
        self.rollback_index(index_id)
        print("Full rollback complete. Monitor faithfulness scores for next 30 min.")
```

---

## 10. Tools: LangSmith, MLflow, Arize Phoenix, Weights & Biases

### LangSmith

LangSmith is LangChain's native observability and evaluation platform. Best for: tracing, prompt management, evaluation datasets, and CI/CD integration.

```python
# Setup
import os
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_API_KEY"]     = "ls__your_key"
os.environ["LANGCHAIN_PROJECT"]     = "rag-production"

from langsmith import Client
client = Client()

# 1. Log custom metrics alongside traces
from langchain_core.runnables import RunnableConfig
from langsmith import traceable

@traceable(name="rag-query", tags=["production"])
def rag_with_langsmith(query: str, user_id: str) -> str:
    result = rag_chain.invoke(query)
    # Add custom feedback to the trace
    run_id = ...  # captured from the callback
    client.create_feedback(
        run_id=run_id,
        key="faithfulness",
        score=compute_faithfulness_score(result),
    )
    return result["result"]

# 2. Run evaluations on a LangSmith dataset
from langsmith.evaluation import evaluate

results = evaluate(
    lambda inputs: rag_chain.invoke(inputs["question"]),
    data="HR Golden Dataset v2",
    evaluators=["qa", "cot_qa"],
    experiment_prefix="prompt-v1.2-candidate",
)

# 3. Compare two experiments
comparison = client.get_test_results(
    project_name="prompt-v1.2-candidate",
)
```

### MLflow

MLflow is the open-source experiment tracking standard. Best for: logging experiments, comparing runs, model registry, and deployment metadata.

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("rag-optimization")

with mlflow.start_run(run_name="semantic-chunking-hybrid-rerank") as run:
    # Log configuration
    mlflow.log_params({
        "chunk_size":        800,
        "chunk_overlap":     100,
        "embedding_model":   "text-embedding-3-small",
        "hybrid_alpha":      0.6,
        "reranker":          "ms-marco-MiniLM-L-6-v2",
        "llm_model":         "gpt-4o-mini",
        "prompt_version":    "v1.1.0",
    })
    
    # Log evaluation results
    mlflow.log_metrics({
        "precision_k5":       0.79,
        "recall_k5":          0.83,
        "faithfulness":       0.91,
        "answer_correctness": 0.78,
        "p95_latency_ms":     1840,
        "cost_per_1k_usd":    0.92,
    })
    
    # Log the golden dataset and eval report as artefacts
    mlflow.log_artifact("evaluation/golden_dataset.json")
    mlflow.log_artifact("evaluation/results/latest_report.json")
    
    # Tag for easy filtering
    mlflow.set_tags({
        "corpus_version":    "v20250301-143000",
        "deployed_env":      "staging",
        "evaluation_passed": "true",
    })
    
    print(f"Run ID: {run.info.run_id}")
```

```bash
# View experiments in MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Navigate to http://localhost:5000
```

### Arize Phoenix

Arize Phoenix is purpose-built for LLM and RAG observability, with built-in UMAP embedding visualisation and retrieval analysis.

```python
# pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Launch Phoenix (runs locally)
session = px.launch_app()

# Instrument LangChain — all RAG calls auto-traced
register(project_name="rag-production")
LangChainInstrumentor().instrument()

# Phoenix now captures:
#   - Query → retrieved chunks → LLM response
#   - Embedding distances for each retrieved chunk
#   - Token counts, latency, cost
#   - UMAP visualisation of query/chunk embedding clusters

# After running queries, explore at http://localhost:6006
print(f"Phoenix UI: {session.url}")

# Log custom evaluations to Phoenix
from phoenix.trace import SpanEvaluations
import pandas as pd

evals_df = pd.DataFrame({
    "span_id":     ["span-001", "span-002"],
    "label":       ["hallucinated", "faithful"],
    "score":       [0.3, 0.9],
    "explanation": ["Claim not in context", "Fully grounded"],
})

px.Client().log_evaluations(
    SpanEvaluations(eval_name="faithfulness", dataframe=evals_df)
)
```

### Weights & Biases (W&B)

W&B is strong for experiment tracking at scale and team collaboration. Best for: comparing many experiments, visualising metric trends, and sharing results across teams.

```python
import wandb

# Initialize W&B run
run = wandb.init(
    project   = "enterprise-rag",
    name      = "hybrid-rerank-v1.2",
    config    = {
        "chunk_size":      800,
        "embedding_model": "text-embedding-3-small",
        "hybrid_alpha":    0.6,
        "llm_model":       "gpt-4o-mini",
        "prompt_version":  "v1.2.0",
    },
    tags      = ["production-candidate", "hybrid-search", "reranking"],
)

# Log eval metrics
wandb.log({
    "eval/precision_k5":       0.79,
    "eval/recall_k5":          0.83,
    "eval/faithfulness":       0.91,
    "eval/answer_correctness": 0.78,
    "perf/p95_latency_ms":     1840,
    "cost/per_1k_queries_usd": 0.92,
})

# Log per-sample results as a table
columns = ["question", "faithfulness", "recall", "answer_correctness"]
data = [
    [s["question"], s["faithfulness"], s["recall"], s["correctness"]]
    for s in per_sample_results
]
table = wandb.Table(columns=columns, data=data)
wandb.log({"eval/per_sample": table})

# Log production faithfulness as time series
for i, score in enumerate(production_faithfulness_scores):
    wandb.log({"prod/faithfulness": score, "prod/request_index": i})

wandb.finish()
```

### Tool Selection Guide

| Use Case                             | Primary Tool    | Secondary       |
| ------------------------------------ | --------------- | --------------- |
| LangChain RAG — end-to-end tracing   | LangSmith       | Arize Phoenix   |
| Experiment comparison across teams   | W&B             | MLflow          |
| Embedding cluster visualisation      | Arize Phoenix   | —               |
| On-premises / self-hosted required   | MLflow          | Arize Phoenix   |
| CI/CD evaluation datasets            | LangSmith       | MLflow          |
| Production real-time monitoring      | LangSmith / Arize | W&B           |
| Cost tracking across experiments     | MLflow / W&B    | LangSmith       |

---

## 11. Lab: Production-Grade RAG Reference Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG REFERENCE ARCHITECTURE            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INGESTION PIPELINE (offline, triggered on corpus change)           │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐   ┌──────────┐  │
│  │ Document │──►│  Semantic  │──►│  Embedding   │──►│  Vector  │  │
│  │  Store   │   │  Chunker   │   │    Model     │   │    DB    │  │
│  │(SharePoint│  │(topic-aware│   │(text-emb-3-  │   │(Pinecone/│  │
│  │ /GDrive) │   │ splitting) │   │   small)     │   │ Weaviate)│  │
│  └──────────┘   └────────────┘   └──────────────┘   └──────────┘  │
│       │                                                             │
│  [Corpus manifest] [Index registry] [Content policy scan]          │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  QUERY PIPELINE (online, per request)                               │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│  [Auth + RBAC]──►[Injection Guard]──►[PII Redact]──►[Scope Check]  │
│                                                            │        │
│                                          ┌─────────────────▼─────┐ │
│                                          │  Hybrid Retrieval     │ │
│                                          │  Dense + BM25 (K=20)  │ │
│                                          └──────────┬────────────┘ │
│                                                     │              │
│                                          ┌──────────▼────────────┐ │
│                                          │  Cross-Encoder Rerank │ │
│                                          │  (top 5)              │ │
│                                          └──────────┬────────────┘ │
│                                                     │              │
│                                          ┌──────────▼────────────┐ │
│                                          │  Context Sufficiency  │ │
│                                          │  Check                │ │
│                                          └──────────┬────────────┘ │
│                                                     │              │
│                              ┌──────────────────────▼────────────┐ │
│                              │  LLM Generation                   │ │
│                              │  [Semantic Cache] → [gpt-4o-mini] │ │
│                              │  [Circuit Breaker] [Fallback LLM] │ │
│                              └──────────────────────┬────────────┘ │
│                                                     │              │
│                              ┌──────────────────────▼────────────┐ │
│                              │  Output Safety                    │ │
│                              │  [PII Redact] [Disclaimer] [Tox]  │ │
│                              └──────────────────────┬────────────┘ │
│                                                     │              │
│                    [Audit Log] [LangSmith Trace] [Metrics]         │
│                                                     │              │
│                                              Response to User      │
└─────────────────────────────────────────────────────────────────────┘
```

### Reference Implementation

```python
# app/production_rag.py

import os, time, uuid
from datetime import datetime, UTC
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.cache import RedisSemanticCache
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain

from config.settings import get_settings
from monitoring.instrumented_chain import MetricsCallbackHandler, RequestMetrics, metrics_store
from security.guards import check_injection, check_scope, redact_pii, apply_output_safety
from audit.logger import AuditLogger

settings    = get_settings()
audit_logger = AuditLogger()


@lru_cache()
def build_production_chain():
    """Build and cache the production RAG chain."""
    
    # Semantic cache (Redis)
    if os.environ.get("REDIS_URL"):
        embeddings_for_cache = OpenAIEmbeddings(model=settings.embedding_model)
        langchain.llm_cache = RedisSemanticCache(
            redis_url=os.environ["REDIS_URL"],
            embedding=embeddings_for_cache,
            score_threshold=0.95,
        )
    
    embeddings  = OpenAIEmbeddings(model=settings.embedding_model)
    vectorstore = Chroma(
        collection_name=settings.vector_collection_name,
        persist_directory=os.environ.get("VECTOR_STORE_PATH", "./chroma_db"),
        embedding_function=embeddings,
    )
    
    # Hybrid retriever
    all_docs    = vectorstore.get()
    from langchain.schema import Document
    docs_list   = [Document(page_content=t, metadata=m)
                   for t, m in zip(all_docs["documents"], all_docs["metadatas"])]
    bm25        = BM25Retriever.from_documents(docs_list, k=settings.retrieval_k_wide)
    dense       = vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k_wide})
    hybrid      = EnsembleRetriever(
        retrievers=[dense, bm25],
        weights=[settings.hybrid_alpha, 1 - settings.hybrid_alpha],
    )
    
    # Cross-encoder re-ranker
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker      = CrossEncoderReranker(model=cross_encoder, top_n=settings.retrieval_k)
    retriever     = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid,
    )
    
    # LLM with circuit breaker
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    
    # Prompt (loaded from versioned registry)
    from prompts.loader import load_prompt
    system_text = load_prompt("system/hr_assistant")
    prompt      = ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source {i+1}: {d.metadata.get('source','Doc').split('/')[-1]}]\n{d.page_content}"
            for i, d in enumerate(docs)
        )
    
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    
    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain, retriever


def handle_query(query: str, user: dict) -> dict:
    """Full production pipeline with security, monitoring, and audit."""
    
    request_id = str(uuid.uuid4())
    event = {
        "request_id":        request_id,
        "timestamp":         datetime.now(UTC).isoformat(),
        "user_id":           user.get("id", "anonymous"),
        "user_role":         user.get("role", ""),
        "injection_blocked": False,
        "scope_blocked":     False,
        "pii_in_query":      False,
        "answer":            "",
        "latency_ms":        0.0,
        "error":             "",
    }
    
    t_start = time.time()
    
    try:
        # ── Security pre-flight ──────────────────────────────────────
        if check_injection(query):
            event["injection_blocked"] = True
            event["answer"] = "I cannot process that request."
            return {"request_id": request_id, "answer": event["answer"]}
        
        if not check_scope(query):
            event["scope_blocked"] = True
            event["answer"] = ("This question is outside the scope of this assistant. "
                               "Please contact the appropriate team directly.")
            return {"request_id": request_id, "answer": event["answer"]}
        
        redacted_query      = redact_pii(query)
        event["pii_in_query"] = redacted_query != query
        
        # ── RAG pipeline ────────────────────────────────────────────
        chain, _ = build_production_chain()
        m        = RequestMetrics(request_id=request_id, model=settings.llm_model)
        
        raw_answer = chain.invoke(
            redacted_query,
            config={"callbacks": [MetricsCallbackHandler(m)]},
        )
        
        # ── Output safety ────────────────────────────────────────────
        safe_answer          = apply_output_safety(raw_answer, redacted_query)
        event["answer"]      = safe_answer
        
        metrics_store.record(m)
        
        return {"request_id": request_id, "answer": safe_answer}
    
    except Exception as e:
        event["error"] = str(e)
        raise
    
    finally:
        event["latency_ms"] = (time.time() - t_start) * 1000
        audit_logger.log(event)
```

### FastAPI Production Endpoint

```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query:      str
    user_id:    str
    user_role:  str = "user"
    session_id: str = ""
    stream:     bool = False

class QueryResponse(BaseModel):
    request_id: str
    answer:     str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    user = {"id": request.user_id, "role": request.user_role,
            "session_id": request.session_id, "clearance": 1}
    
    try:
        result = handle_query(request.query, user)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    return QueryResponse(**result)

@app.get("/health")
async def health():
    summary = metrics_store.rolling_summary(last_n=100)
    return {
        "status": "ok" if summary.get("error_rate", 0) < 0.05 else "degraded",
        "metrics": summary,
    }

@app.get("/metrics/summary")
async def metrics_summary():
    return metrics_store.rolling_summary(last_n=1000)
```

### Lab Deliverables and Checklist

```
production_rag/
├── app/
│   ├── main.py                    ← FastAPI endpoints (query, health, metrics)
│   └── production_rag.py          ← Full orchestration pipeline
├── config/
│   ├── settings.py                ← Pydantic settings with env overrides
│   └── thresholds.json            ← CI/CD quality gate thresholds
├── prompts/
│   ├── registry.json              ← Versioned prompt registry
│   └── system/
│       └── hr_assistant_v1.1.0.txt
├── ingest/
│   ├── corpus_manifest.json       ← Document corpus version
│   └── index_registry.json        ← Embedding index versions
├── evaluation/
│   ├── golden_dataset.json        ← Versioned golden dataset
│   └── results/
│       └── latest_report.json     ← Most recent CI/CD eval report
├── experiments/
│   └── *.json                     ← Experiment manifests (one per run)
├── audit/
│   └── rag_audit.jsonl            ← Structured production audit log
├── feedback/
│   └── user_feedback.jsonl        ← User feedback records
├── .github/workflows/
│   └── rag-eval.yml               ← GitHub Actions CI/CD pipeline
└── scripts/
    └── eval_gate.py               ← CI/CD evaluation gate script
```

**Production readiness checklist:**

```
Infrastructure
  [ ] Vector DB managed and replicated (not local Chroma)
  [ ] Secrets in secrets manager (not .env)
  [ ] Redis semantic cache deployed
  [ ] Health endpoint wired to load balancer

Versioning
  [ ] Prompt registry in place (git-versioned or LangSmith)
  [ ] Corpus manifest generated and committed
  [ ] Embedding index registry tracks corpus + model + config
  [ ] Experiment manifests link all artefacts for reproducibility

CI/CD
  [ ] GitHub Actions pipeline runs on every PR
  [ ] Eval gate blocks merge if thresholds not met
  [ ] Canary deployment configured (10% → 100%)
  [ ] Rollback procedure tested end-to-end

Monitoring
  [ ] Per-request metrics collected (latency, tokens, cost)
  [ ] Sampled faithfulness scoring active (5% of traffic)
  [ ] LangSmith / Arize Phoenix tracing enabled
  [ ] Alert thresholds configured and tested
  [ ] Monthly cost budget set and enforced

Reliability
  [ ] Circuit breaker on LLM API calls
  [ ] Fallback response for LLM outages
  [ ] Retrieval timeout with graceful degradation

Feedback Loop
  [ ] Thumbs up/down UI wired to /feedback endpoint
  [ ] Weekly triage of negative feedback
  [ ] Process for promoting failures to golden dataset
  [ ] Quarterly evaluation dataset refresh planned
```

---

## Summary

Operationalizing a RAG system means closing the gap between "it works in a demo" and "it works reliably, securely, and economically at production scale."

```
The LLMOps Maturity Ladder:

Level 0 — Prototype
  Jupyter notebook, hardcoded prompt, local vector DB, manual testing

Level 1 — Dev-Ready
  Config-driven settings, git-versioned prompts, basic evaluation

Level 2 — CI/CD
  GitHub Actions eval gate, prompt registry, index versioning,
  experiment manifests, staging environment

Level 3 — Production-Monitored
  Real-time metrics, sampled quality scoring, audit logs,
  LangSmith/Arize tracing, cost dashboards, alert thresholds

Level 4 — Continuously Improving
  User feedback loop, failure triage, golden dataset growth,
  quarterly model/embedding review, canary deployments

Level 5 — Enterprise-Governed
  Full security + governance stack (Module-3) + LLMOps stack,
  GDPR compliance, incident runbooks, quarterly review cadence
```

**Key takeaways:**

1. **Version everything** — prompts, corpora, indices, evaluation datasets. If you can't reproduce a score from 3 months ago, you can't diagnose regressions.
2. **Evaluation gates are the only reliable regression prevention** — CI/CD that only runs unit tests will not catch quality degradation in LLM systems.
3. **Monitor retrieval quality separately from generation quality** — a faithfulness drop caused by a bad index has a different fix than one caused by a bad prompt.
4. **The feedback loop is the product improvement engine** — every negative feedback record, triaged and promoted to the golden dataset, makes the next evaluation more realistic.
5. **Circuit breakers and fallbacks are non-negotiable** — LLM APIs have SLAs of 99.9%, meaning ~9 hours of potential downtime per year. Without fallbacks, your product inherits the provider's SLA.

---

_Guide maintained as part of the **Integrating Generative AI — Advanced RAG & Enterprise Patterns (Level 2)** training programme._
