# Module-1: RAG Evaluation Techniques

A comprehensive guide to measuring, diagnosing, and improving the quality of production RAG systems — covering retrieval metrics, generation metrics, evaluation tooling, and benchmark creation.

---

## Table of Contents

1. [Why Evaluation Is Critical for Production RAG](#1-why-evaluation-is-critical-for-production-rag)
2. [Common RAG Failure Modes](#2-common-rag-failure-modes)
3. [The RAG Evaluation Framework](#3-the-rag-evaluation-framework)
4. [Retrieval Metrics](#4-retrieval-metrics)
5. [Generation Metrics](#5-generation-metrics)
6. [Human Evaluation vs Automated Evaluation](#6-human-evaluation-vs-automated-evaluation)
7. [Golden Datasets and Benchmark Creation](#7-golden-datasets-and-benchmark-creation)
8. [Evaluation Tools: RAGAS, LangSmith, DeepEval](#8-evaluation-tools-ragas-langsmith-deepeval)
9. [End-to-End Evaluation Pipeline](#9-end-to-end-evaluation-pipeline)
10. [Lab: Evaluate an Existing RAG Chatbot](#10-lab-evaluate-an-existing-rag-chatbot)
11. [Evaluation Scorecard Template](#11-evaluation-scorecard-template)

---

## 1. Why Evaluation Is Critical for Production RAG

### The Silent Failure Problem

A RAG system that "looks like it works" during demos can silently fail in production. Unlike traditional software where crashes are visible, RAG failures are often:

- **Plausible but wrong** — the LLM generates confident-sounding incorrect answers
- **Partially correct** — retrieval returns 3 of 5 relevant chunks; the answer misses key facts
- **Inconsistent** — the same question answered differently on different runs
- **Degrading over time** — as the document corpus grows, retrieval quality drops undetected

```
Without evaluation:                  With evaluation:

   User asks question                   User asks question
         ↓                                     ↓
   System answers                       System answers
         ↓                                     ↓
   ??? (is it correct?)              Metrics measured → dashboards
         ↓                                     ↓
   Problem discovered months        Problem detected in hours
   later via customer complaints     → fix deployed in days
```

### What Evaluation Enables

| Capability              | Without Evaluation           | With Evaluation                         |
| ----------------------- | ---------------------------- | --------------------------------------- |
| Chunking strategy       | Guesswork                    | A/B test, pick the winner by recall@K   |
| Embedding model choice  | Default model                | Benchmarked against domain queries      |
| Re-ranker effectiveness | Unknown                      | MRR before vs after re-ranking          |
| Prompt changes          | Hope it doesn't break things | Regression suite catches regressions    |
| Production monitoring   | Alert when users complain    | Alert when faithfulness score drops     |
| LLM upgrade decisions   | Risk                         | Evaluate new model on golden dataset    |

### The Three Questions Every Production RAG Must Answer

```
1. Did we retrieve the right documents?   → Retrieval Evaluation
2. Did the LLM use those documents well?  → Generation Evaluation
3. Is the final answer what the user needed? → End-to-End Evaluation
```

---

## 2. Common RAG Failure Modes

Understanding where RAG breaks is the foundation of knowing what to measure.

### Failure Mode Taxonomy

```
RAG FAILURE MODES
│
├── RETRIEVAL FAILURES
│   ├── Wrong chunks retrieved (irrelevance)
│   ├── Relevant chunks missed (low recall)
│   ├── Duplicate / redundant chunks
│   └── Correct chunks, wrong ordering (low MRR)
│
├── GENERATION FAILURES
│   ├── Hallucination (facts not in context)
│   ├── Ignoring retrieved context (model overrides with prior knowledge)
│   ├── Incomplete answers (some context used, some ignored)
│   └── Answer format mismatch (verbose vs concise)
│
└── SYSTEM FAILURES
    ├── Latency spikes under load
    ├── Embedding model drift (embedding model updated, index stale)
    └── Context window exceeded (too many chunks injected)
```

### Failure Mode Details

#### Hallucination

The LLM generates a fact that is not present in — or directly contradicts — the retrieved context.

```
Retrieved context: "The return window is 30 days for physical products."

User question:    "Can I return a digital download?"

Hallucinated answer: "Yes, you can return digital downloads within 14 days."
                      ↑ Not in context — LLM fabricated this from general knowledge
```

**Detection metric:** Faithfulness score (see Section 5)

#### Poor Retrieval — Irrelevant Chunks

The embedding model retrieves semantically adjacent but contextually wrong chunks.

```
Query: "What is the password reset process?"

Retrieved chunk: "Passwords must be at least 12 characters and changed every 90 days."
                  ↑ About password policy, NOT the reset process — similar words, wrong content
```

**Detection metric:** Precision@K (see Section 4)

#### Poor Retrieval — Missed Relevant Chunks

The most relevant document exists in the corpus but was not retrieved in top-K.

```
Query: "How do I escalate a billing dispute?"

Corpus contains: A specific 3-page billing dispute escalation procedure document
Retrieved:        General customer service FAQ — the key document was ranked 12th

Result: Incomplete, generic answer
```

**Detection metric:** Recall@K (see Section 4)

#### Incomplete Answers

The LLM receives the right context but only uses part of it.

```
Retrieved context: [4 chunks covering: eligibility, process, timeline, exceptions]

Answer generated: "You are eligible if you joined before 2022." 
                   ↑ Only used chunk 1 of 4 — missed process, timeline, exceptions
```

**Detection metric:** Answer completeness / coverage (see Section 5)

---

## 3. The RAG Evaluation Framework

### Two-Axis Evaluation Model

```
                         RETRIEVAL QUALITY
                    Low              High
                ┌──────────────┬──────────────┐
           High │  Fabricating │  Working     │
GENERATION      │  confidently │  correctly   │
QUALITY         │  (dangerous) │  (goal)      │
                ├──────────────┼──────────────┤
           Low  │  Broken      │  Retrieval ok│
                │  (both fail) │  LLM fails   │
                │              │  (fix prompt)│
                └──────────────┴──────────────┘
```

### Evaluation Loop

```
                    ┌─────────────────────────────────┐
                    │         Golden Dataset           │
                    │  (questions + ground truth       │
                    │   answers + relevant doc IDs)    │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │         Run RAG System           │
                    │  For each question:              │
                    │  - Record retrieved chunks       │
                    │  - Record generated answer       │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       Compute Metrics            │
                    │  Retrieval: P@K, R@K, MRR, MAP   │
                    │  Generation: Faithfulness,        │
                    │  Relevance, Groundedness, AC      │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │       Diagnose & Improve         │
                    │  Low recall → fix chunking       │
                    │  Low faithfulness → fix prompt   │
                    │  Low precision → add re-ranker   │
                    └─────────────────────────────────┘
```

---

## 4. Retrieval Metrics

Retrieval metrics measure how well your retrieval pipeline surfaces relevant documents. They require a **golden dataset** with known relevant document IDs per query.

### Precision@K

**What it measures:** Of the K documents retrieved, what fraction are actually relevant?

```
Precision@K = (Number of relevant docs in top-K) / K
```

**Example:**

```
Query: "What is the expense reimbursement policy?"
K = 5, Retrieved docs: [doc_3, doc_7, doc_1, doc_9, doc_2]
Relevant docs (ground truth): {doc_3, doc_1, doc_5}

Relevant retrieved: doc_3 (rank 1), doc_1 (rank 3) → 2 out of 5

Precision@5 = 2/5 = 0.40
```

**Interpretation:**
- `P@5 = 1.0` → every retrieved doc is relevant (perfect precision)
- `P@5 = 0.2` → 4 out of 5 retrieved docs are noise — likely causing confused LLM answers

```python
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k
```

### Recall@K

**What it measures:** Of all relevant documents that exist, what fraction appear in the top-K results?

```
Recall@K = (Number of relevant docs in top-K) / (Total relevant docs in corpus)
```

**Example:**

```
Query: "What is the expense reimbursement policy?"
K = 5, Retrieved top-5: [doc_3, doc_7, doc_1, doc_9, doc_2]
All relevant docs in corpus: {doc_3, doc_1, doc_5}   ← 3 exist

Relevant retrieved: doc_3, doc_1 (doc_5 was rank 8, not retrieved)

Recall@5 = 2/3 = 0.67
```

**Interpretation:**
- `R@5 = 1.0` → all relevant docs were found (perfect recall)
- `R@5 = 0.33` → you're missing 2/3 of relevant material → answers will be incomplete

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)
```

### Precision vs Recall Trade-off

```
Increasing K improves Recall but hurts Precision:

K=3:  P@3 = 0.67, R@3 = 0.33  ← fewer chunks, more precise but miss some
K=5:  P@5 = 0.40, R@5 = 0.67  ← balanced
K=10: P@10= 0.30, R@10= 1.00  ← all relevant found but lots of noise

The LLM suffers from noise (low P) as much as from missing context (low R).
Production recommendation: optimise for R@K first, then use re-ranking to restore P.
```

### MRR — Mean Reciprocal Rank

**What it measures:** How high up does the *first* relevant document appear in the results? Rewards systems that rank the best document first.

```
MRR = (1/|Q|) × Σ (1 / rank of first relevant doc for query q)
```

**Example:**

```
Query 1: First relevant doc at rank 1 → 1/1 = 1.00
Query 2: First relevant doc at rank 3 → 1/3 = 0.33
Query 3: First relevant doc at rank 2 → 1/2 = 0.50

MRR = (1.00 + 0.33 + 0.50) / 3 = 0.61
```

**Why MRR matters for RAG:** LLMs pay more attention to context that appears earlier in the prompt. If the most relevant document is ranked 8th, the LLM may under-use it even if it's technically retrieved.

```python
def mrr(queries_results: list[dict]) -> float:
    """
    queries_results: list of {"retrieved": [...doc_ids...], "relevant": set(...)}
    """
    total = 0.0
    for result in queries_results:
        for rank, doc_id in enumerate(result["retrieved"], start=1):
            if doc_id in result["relevant"]:
                total += 1.0 / rank
                break
    return total / len(queries_results)
```

### MAP — Mean Average Precision

**What it measures:** Average Precision computed across all queries, rewarding both finding all relevant docs AND ranking them high.

```
AP@K for a single query = (1/R) × Σ [P@i × rel(i)]

Where:
  R = total number of relevant docs
  P@i = precision at rank i
  rel(i) = 1 if doc at rank i is relevant, 0 otherwise

MAP = mean of AP@K across all queries
```

**Example:**

```
Query: Relevant docs = {doc_3, doc_1, doc_5}
Retrieved order: [doc_3, doc_7, doc_1, doc_9, doc_5]

Rank 1: doc_3 relevant → P@1 = 1/1 = 1.00
Rank 2: doc_7 irrelevant
Rank 3: doc_1 relevant → P@3 = 2/3 = 0.67
Rank 4: doc_9 irrelevant
Rank 5: doc_5 relevant → P@5 = 3/5 = 0.60

AP = (1.00 + 0.67 + 0.60) / 3 = 0.76
```

```python
def average_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    if not relevant_ids:
        return 1.0
    return precision_sum / len(relevant_ids)

def map_score(queries_results: list[dict]) -> float:
    return sum(
        average_precision(r["retrieved"], r["relevant"])
        for r in queries_results
    ) / len(queries_results)
```

### Retrieval Metrics Summary

| Metric      | Asks                                           | Best for                         | Range  |
| ----------- | ---------------------------------------------- | -------------------------------- | ------ |
| Precision@K | "Are all retrieved chunks relevant?"           | Minimising LLM noise             | 0 → 1 |
| Recall@K    | "Did we find all relevant chunks?"             | Minimising incomplete answers    | 0 → 1 |
| MRR         | "Is the best chunk at the top?"               | Ranking quality for first hit    | 0 → 1 |
| MAP         | "Are all relevant chunks ranked high?"         | Overall ranking quality          | 0 → 1 |

---

## 5. Generation Metrics

Generation metrics measure the quality of the LLM's answer given the retrieved context. These can be computed automatically using an **LLM-as-judge** pattern.

### Faithfulness

**Definition:** Are all factual claims in the generated answer supported by the retrieved context? This is the primary hallucination metric.

```
Faithfulness = (claims supported by context) / (total claims in answer)

Score of 1.0 = fully grounded
Score of 0.0 = fully hallucinated
```

**How to compute:**

```python
from langchain_openai import ChatOpenAI
import json

faithfulness_prompt = """You are an expert evaluator assessing factual grounding.

Retrieved Context:
{context}

Generated Answer:
{answer}

Task:
1. Extract every distinct factual claim made in the Generated Answer.
2. For each claim, assess whether it is directly supported by the Retrieved Context.
3. Return a JSON object with:
   - "claims": list of claims found
   - "supported": list of claims that are supported by context
   - "unsupported": list of claims NOT in context (potential hallucinations)
   - "faithfulness_score": supported_count / total_claims (float 0-1)

Return only valid JSON."""

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def compute_faithfulness(context: str, answer: str) -> dict:
    result = llm.invoke(faithfulness_prompt.format(
        context=context,
        answer=answer
    ))
    return json.loads(result.content)

# Example usage
result = compute_faithfulness(
    context="The standard return window is 30 days. Refunds are processed in 5-7 business days.",
    answer="You can return items within 30 days and get a refund in 3-5 days."
)
# → faithfulness_score: 0.5 (30-day claim supported, 3-5 days is wrong — context says 5-7)
```

### Answer Relevance

**Definition:** Does the generated answer actually address the user's question? A faithful answer can still be off-topic.

```
High relevance: Answer addresses exactly what was asked
Low relevance:  Answer is factually correct but doesn't answer the question
```

```python
relevance_prompt = """You are evaluating whether an answer addresses the question.

Question: {question}
Answer: {answer}

Rate the relevance of the answer to the question on a scale of 1-5:
5 = Directly and completely answers the question
4 = Mostly answers the question with minor gaps
3 = Partially answers, significant parts of the question unaddressed
2 = Tangentially related but doesn't really answer
1 = Does not address the question at all

Return JSON: {{"score": <1-5>, "reasoning": "<brief explanation>"}}"""

def compute_relevance(question: str, answer: str) -> dict:
    result = llm.invoke(relevance_prompt.format(question=question, answer=answer))
    return json.loads(result.content)
```

### Groundedness

**Definition:** Is the answer anchored in the retrieved context, or does it rely on the LLM's parametric (training) knowledge? Groundedness is a stricter form of faithfulness — it also penalises answers that are true but not in the context.

```
Faithfulness:  "Is every claim in the answer supported by context?"
Groundedness:  "Is the entire answer derived from context, not LLM memory?"

Difference:
  Context: "Paris is the capital of France."
  Question: "What is the Eiffel Tower made of?"
  Answer: "The Eiffel Tower is made of iron."  ← True globally, but NOT from context

  Faithfulness: claim cannot be verified against context → unsupported
  Groundedness: 0 — answer comes from LLM training knowledge, not retrieval
```

```python
groundedness_prompt = """You are evaluating if an answer is grounded in retrieved context.

Retrieved Context:
{context}

Question: {question}
Answer: {answer}

Groundedness check:
- A grounded answer uses ONLY information from the retrieved context.
- An ungrounded answer introduces facts from general knowledge not in context.

Return JSON:
{{
  "groundedness_score": <0.0 to 1.0>,
  "grounded_statements": ["..."],
  "ungrounded_statements": ["..."],
  "verdict": "grounded" | "partially_grounded" | "ungrounded"
}}"""

def compute_groundedness(context: str, question: str, answer: str) -> dict:
    result = llm.invoke(groundedness_prompt.format(
        context=context, question=question, answer=answer
    ))
    return json.loads(result.content)
```

### Answer Correctness

**Definition:** Compared to the ground truth answer, how correct is the generated answer? Requires a golden dataset with reference answers.

```
Answer Correctness combines:
  - Semantic similarity to ground truth
  - Factual overlap (shared key facts)
  - Absence of contradictions
```

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(answer: str, ground_truth: str) -> float:
    emb_a = model.encode(answer, convert_to_tensor=True)
    emb_gt = model.encode(ground_truth, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_gt))

correctness_prompt = """Compare the Generated Answer to the Reference Answer.

Reference Answer (ground truth): {reference}
Generated Answer: {answer}

Evaluate:
1. Are all key facts from the reference present in the generated answer?
2. Does the generated answer introduce any facts that contradict the reference?
3. Is there important information in the reference missing from the answer?

Return JSON:
{{
  "factual_overlap_score": <0.0-1.0>,
  "contradiction_present": true/false,
  "missing_key_facts": ["..."],
  "overall_correctness": <0.0-1.0>
}}"""

def compute_correctness(answer: str, reference: str) -> dict:
    llm_result = json.loads(llm.invoke(
        correctness_prompt.format(reference=reference, answer=answer)
    ).content)
    semantic_score = semantic_similarity(answer, reference)
    llm_result["semantic_similarity"] = semantic_score
    return llm_result
```

### Generation Metrics Summary

| Metric             | Measures                                          | Requires Ground Truth | Range  |
| ------------------ | ------------------------------------------------- | --------------------- | ------ |
| Faithfulness       | Claims grounded in retrieved context              | No                    | 0 → 1 |
| Answer Relevance   | Does the answer address the question?             | No                    | 1 → 5 |
| Groundedness       | Is the answer derived from context (not LLM RAM)? | No                    | 0 → 1 |
| Answer Correctness | Factual match against reference answer            | Yes                   | 0 → 1 |

---

## 6. Human Evaluation vs Automated Evaluation

### Comparison

| Dimension             | Human Evaluation                       | Automated Evaluation (LLM-as-judge)      |
| --------------------- | -------------------------------------- | ---------------------------------------- |
| **Cost**              | High ($10–100 per annotated sample)    | Low (cents per sample)                   |
| **Scale**             | Hundreds of samples                    | Tens of thousands of samples             |
| **Speed**             | Days to weeks                          | Minutes                                  |
| **Consistency**       | Variable (inter-rater disagreement)    | Consistent (same prompt → same score)    |
| **Nuance**            | Catches subtle quality issues          | Can miss domain-specific nuance          |
| **Bias**              | Annotator bias                         | LLM judge bias (favours verbose answers) |
| **Ground truth**      | Gold standard                          | Approximate, needs calibration           |
| **When to use**       | Calibrating metrics, high-stakes audit | Continuous monitoring, A/B testing       |

### When to Use Each

```
USE HUMAN EVALUATION WHEN:
  ├── Setting up your evaluation framework for the first time
  │   (calibrate automated metrics against human judgement)
  ├── High-stakes domain: medical, legal, financial
  ├── Launching a new product version (release gate)
  └── Automated metrics show regression (investigate the cause)

USE AUTOMATED EVALUATION WHEN:
  ├── CI/CD pipeline (run on every code change)
  ├── A/B testing chunking strategies or embedding models
  ├── Monitoring production quality continuously
  ├── Evaluating large test sets (>500 queries)
  └── Rapid iteration during development
```

### Human Annotation Setup

```python
# Annotation schema for human evaluators
annotation_schema = {
    "query": str,
    "retrieved_chunks": list[str],
    "generated_answer": str,
    "ground_truth_answer": str,    # optional
    
    # Human annotations (Likert scale 1-5)
    "faithfulness": int,           # 1=hallucinated, 5=fully grounded
    "relevance": int,              # 1=off-topic, 5=directly answers
    "completeness": int,           # 1=major gaps, 5=comprehensive
    "clarity": int,                # 1=confusing, 5=clear and well-structured
    "overall_quality": int,        # 1=unacceptable, 5=excellent
    
    # Free-text
    "failure_type": str,           # "hallucination|retrieval_miss|irrelevant|ok"
    "annotator_notes": str,
}
```

### Calibrating Automated Metrics

Run human evaluation and automated evaluation on the same 100-200 samples, then measure agreement:

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr

human_faithfulness = [4, 5, 2, 3, 5, 1, 4, ...]  # human scores 1-5
auto_faithfulness  = [0.8, 0.95, 0.3, 0.6, 1.0, 0.1, 0.75, ...]  # 0-1

# Pearson correlation (linear)
r, p = pearsonr(human_faithfulness, auto_faithfulness)
print(f"Pearson r={r:.2f}, p={p:.4f}")

# Spearman rank correlation (ordinal)
rho, p = spearmanr(human_faithfulness, auto_faithfulness)
print(f"Spearman ρ={rho:.2f}, p={p:.4f}")

# Aim for: r > 0.7 or ρ > 0.7 for automated metrics to be trusted
```

---

## 7. Golden Datasets and Benchmark Creation

A **golden dataset** is the foundation of reliable RAG evaluation. Without it, you have no objective measurement baseline.

### What a Golden Dataset Contains

```
For each evaluation sample:
┌────────────────────────────────────────────────────────────┐
│ question:        "What is the maternity leave entitlement?" │
│ relevant_doc_ids: ["policy_hr_007", "policy_hr_008"]        │
│ ground_truth_answer: "Employees are entitled to 16 weeks... │
│                       of which 6 are fully paid..."         │
│ difficulty:       "medium"                                   │
│ category:         "HR policy"                               │
│ requires_multi_hop: false                                   │
└────────────────────────────────────────────────────────────┘
```

### Golden Dataset Creation Strategies

#### Strategy 1 — Manual Curation (Highest Quality)

Domain experts write questions and answers from scratch, referencing specific documents.

```
Best for:     High-stakes domains (medical, legal, financial)
Effort:       High — 2-4 hours per 100 samples
Coverage:     Covers exactly what matters to your domain
Pitfall:      Expert availability, annotation consistency
```

#### Strategy 2 — LLM-Generated + Human Verified (Balanced)

Use an LLM to generate Q&A pairs from your documents, then have humans verify and curate.

```python
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader

qa_generation_prompt = """You are creating evaluation questions for a RAG system.

Document chunk:
{chunk}

Generate 3 question-answer pairs that:
1. Can be answered using ONLY the information in this chunk
2. Cover different aspects of the content
3. Range in difficulty: one factual, one inferential, one application

Return JSON array:
[
  {{
    "question": "...",
    "answer": "...",
    "difficulty": "easy|medium|hard",
    "question_type": "factual|inferential|application"
  }},
  ...
]"""

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

def generate_qa_pairs(chunk_text: str, doc_id: str) -> list[dict]:
    result = llm.invoke(qa_generation_prompt.format(chunk=chunk_text))
    pairs = json.loads(result.content)
    for pair in pairs:
        pair["source_doc_id"] = doc_id
        pair["human_verified"] = False  # flag for human review queue
    return pairs

# Generate across your document corpus
all_qa_pairs = []
for doc_id, chunk in document_chunks.items():
    pairs = generate_qa_pairs(chunk, doc_id)
    all_qa_pairs.extend(pairs)

print(f"Generated {len(all_qa_pairs)} Q&A pairs for human review")
```

#### Strategy 3 — Production Query Mining (Most Realistic)

Capture real user queries from production, annotate the correct answers, and build a dataset from actual usage patterns.

```python
# Capture real queries with their retrieved chunks and answers
production_log_schema = {
    "timestamp": "2025-01-15T10:23:45Z",
    "query": str,
    "retrieved_chunk_ids": list[str],
    "generated_answer": str,
    "user_feedback": "thumbs_up|thumbs_down|null",
    "session_id": str
}

# Queries with thumbs_down are priority annotation targets
# Queries with thumbs_up can be bootstrapped as positive examples
def build_from_production_logs(logs: list[dict]) -> list[dict]:
    golden = []
    negative_feedback = [l for l in logs if l["user_feedback"] == "thumbs_down"]
    
    for log in negative_feedback:
        golden.append({
            "question": log["query"],
            "relevant_doc_ids": [],        # to be annotated
            "ground_truth_answer": "",     # to be annotated
            "source": "production_negative",
            "needs_annotation": True
        })
    return golden
```

### Dataset Composition Best Practices

```
Recommended golden dataset structure (200 samples minimum):

Category Distribution:
  ├── 40% Factual (direct lookup, single document)
  ├── 25% Inferential (requires reasoning over retrieved content)
  ├── 20% Multi-hop (answer spans multiple documents)
  ├── 10% Edge cases (ambiguous, out-of-scope, no-answer)
  └──  5% Adversarial (designed to expose failure modes)

Difficulty Distribution:
  ├── 30% Easy (answer is a direct sentence in one chunk)
  ├── 50% Medium (requires identifying the right document and synthesising)
  └── 20% Hard (multi-step, requires understanding relationships)
```

### Saving and Versioning the Dataset

```python
import json
from datetime import datetime

def save_golden_dataset(samples: list[dict], version: str = "v1.0"):
    dataset = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "total_samples": len(samples),
        "samples": samples
    }
    
    filepath = f"evaluation/golden_dataset_{version}.json"
    with open(filepath, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {filepath}")

# Version control your evaluation dataset alongside your code
# git add evaluation/golden_dataset_v1.0.json
```

---

## 8. Evaluation Tools: RAGAS, LangSmith, DeepEval

### RAGAS

RAGAS (Retrieval Augmented Generation Assessment) is the most widely used open-source RAG evaluation framework. It computes multiple metrics in one pass using an LLM judge.

#### Installation and Setup

```bash
pip install ragas langchain-openai
```

#### Core RAGAS Metrics

| RAGAS Metric          | Measures                                    | Needs Ground Truth |
| --------------------- | ------------------------------------------- | ------------------ |
| `faithfulness`        | Hallucination — claims vs context           | No                 |
| `answer_relevancy`    | Does the answer address the question?       | No                 |
| `context_recall`      | Did retrieval find all relevant docs?       | Yes                |
| `context_precision`   | Are retrieved docs relevant?                | Yes                |
| `answer_correctness`  | Factual match to reference answer           | Yes                |
| `answer_similarity`   | Semantic similarity to reference            | Yes                |

#### Running RAGAS Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Prepare evaluation dataset in RAGAS format
eval_data = {
    "question": [
        "What is the standard return window?",
        "Who is eligible for the employee discount?",
        "How do I escalate a billing dispute?",
    ],
    "answer": [
        "The standard return window is 30 days from purchase.",
        "All full-time employees are eligible for a 20% discount.",
        "Billing disputes should be escalated to billing@company.com within 14 days.",
    ],
    "contexts": [
        ["Returns must be made within 30 days of purchase with a valid receipt."],
        ["Full-time employees receive a 20% discount on all products."],
        ["To escalate a billing dispute, contact billing@company.com. Disputes must be raised within 14 days."],
    ],
    "ground_truth": [
        "Items can be returned within 30 days of purchase.",
        "Full-time employees qualify for the 20% discount.",
        "Contact billing@company.com within 14 days to escalate billing disputes.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Configure LLM and embeddings for RAGAS judge
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

print(results)
# Output: {'faithfulness': 0.92, 'answer_relevancy': 0.88, 'context_recall': 0.95, ...}

# Convert to DataFrame for analysis
df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy", "context_recall"]])
```

#### Identifying Low-Scoring Samples

```python
# Find questions where faithfulness is low (potential hallucinations)
low_faithfulness = df[df["faithfulness"] < 0.7].sort_values("faithfulness")
print("Questions with potential hallucinations:")
print(low_faithfulness[["question", "answer", "faithfulness"]].to_string())

# Find questions with low context recall (retrieval failures)
low_recall = df[df["context_recall"] < 0.6].sort_values("context_recall")
print("\nQuestions where retrieval likely missed relevant docs:")
print(low_recall[["question", "context_recall"]].to_string())
```

### LangSmith

LangSmith is LangChain's observability and evaluation platform — it traces every RAG call and enables running evaluators over traces.

#### Setup

```bash
pip install langsmith
export LANGCHAIN_API_KEY="ls__your_key"
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="rag-evaluation"
```

#### Tracing RAG Calls Automatically

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
# With LANGCHAIN_TRACING_V2=true, every invoke is automatically traced

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
)

# This call is automatically logged to LangSmith
result = qa_chain.invoke("What is the return policy for electronics?")
# → Trace visible at smith.langchain.com with: query, retrieved docs, LLM call, answer
```

#### Running Evaluators on a Dataset in LangSmith

```python
from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate, LangChainStringEvaluator

client = Client()

# Create a dataset in LangSmith
dataset_name = "RAG Evaluation - HR Policies"
dataset = client.create_dataset(dataset_name)

# Add examples
client.create_examples(
    inputs=[{"query": q} for q in questions],
    outputs=[{"answer": a} for a in ground_truth_answers],
    dataset_id=dataset.id,
)

# Define the RAG chain to evaluate
def rag_pipeline(inputs: dict) -> dict:
    result = qa_chain.invoke(inputs["query"])
    return {"answer": result["result"]}

# Run evaluation with built-in evaluators
results = ls_evaluate(
    rag_pipeline,
    data=dataset_name,
    evaluators=[
        LangChainStringEvaluator("qa"),          # correctness vs ground truth
        LangChainStringEvaluator("cot_qa"),      # chain-of-thought correctness
    ],
    experiment_prefix="baseline-gpt4o-mini",
)
```

### DeepEval

DeepEval is an open-source evaluation framework with a broader set of metrics and a pytest-like interface, making it easy to integrate into CI/CD.

#### Installation

```bash
pip install deepeval
```

#### Core DeepEval Metrics

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

# Define a test case
test_case = LLMTestCase(
    input="What is the expense reimbursement process?",
    actual_output="Submit your receipts via the finance portal within 30 days.",
    expected_output="Employees must submit expense receipts via the finance portal within 30 days of incurring the expense.",
    retrieval_context=[
        "All expense claims must be submitted through the Finance Portal. "
        "Receipts must be uploaded within 30 days of the expense date."
    ],
)

# Evaluate with multiple metrics
metrics = [
    FaithfulnessMetric(threshold=0.8, model="gpt-4o"),
    AnswerRelevancyMetric(threshold=0.7, model="gpt-4o"),
    ContextualRecallMetric(threshold=0.8, model="gpt-4o"),
    HallucinationMetric(threshold=0.1, model="gpt-4o"),  # lower = less hallucination
]

evaluate([test_case], metrics)
```

#### Integrating DeepEval into pytest (CI/CD)

```python
# test_rag_quality.py
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

@pytest.mark.parametrize("query,expected,context", [
    (
        "What is the return window?",
        "The return window is 30 days.",
        ["Products can be returned within 30 days of purchase."],
    ),
    (
        "How do I reset my password?",
        "Use the 'Forgot Password' link on the login page.",
        ["To reset your password, click 'Forgot Password' on the login page."],
    ),
])
def test_rag_answer_quality(query, expected, context):
    # Run your actual RAG chain
    actual = rag_chain.invoke(query)
    
    test_case = LLMTestCase(
        input=query,
        actual_output=actual,
        expected_output=expected,
        retrieval_context=context,
    )
    
    assert_test(test_case, [
        FaithfulnessMetric(threshold=0.8),
        AnswerRelevancyMetric(threshold=0.7),
    ])

# Run: pytest test_rag_quality.py --tb=short
```

### Tool Comparison

| Dimension              | RAGAS               | LangSmith                   | DeepEval                       |
| ---------------------- | ------------------- | --------------------------- | ------------------------------ |
| **Open-source**        | Yes                 | Partial (SDK open, UI paid) | Yes                            |
| **Hosted dashboard**   | No                  | Yes                         | Yes (cloud tier)               |
| **Metric breadth**     | RAG-focused         | General + custom            | RAG + safety + bias            |
| **CI/CD integration**  | Manual              | SDK-based                   | pytest plugin (native)         |
| **Production tracing** | No                  | Yes (automatic)             | No                             |
| **Best for**           | Quick offline evals | End-to-end observability    | Quality gates in CI/CD         |

**Recommended combination:**
- **RAGAS** for offline benchmark evaluation and chunking/embedding experiments
- **LangSmith** for production monitoring and tracing
- **DeepEval** for CI/CD quality gates and regression prevention

---

## 9. End-to-End Evaluation Pipeline

Combining retrieval and generation metrics into a single automated evaluation run.

```python
import json
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

@dataclass
class EvaluationResult:
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]
    
    # Retrieval metrics
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    
    # Generation metrics
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    groundedness: float = 0.0
    answer_correctness: float = 0.0

class RAGEvaluator:
    def __init__(self, rag_chain, judge_llm, k: int = 4):
        self.rag_chain = rag_chain
        self.judge_llm = judge_llm
        self.k = k

    def evaluate_single(self, sample: dict) -> EvaluationResult:
        # Run RAG
        rag_output = self.rag_chain.invoke(sample["question"])
        answer = rag_output["result"]
        retrieved_ids = [
            doc.metadata.get("doc_id", "unknown")
            for doc in rag_output["source_documents"]
        ]
        context = "\n\n".join([
            doc.page_content for doc in rag_output["source_documents"]
        ])

        result = EvaluationResult(
            question=sample["question"],
            generated_answer=answer,
            ground_truth=sample["ground_truth_answer"],
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=sample["relevant_doc_ids"],
        )

        # Retrieval metrics
        relevant_set = set(sample["relevant_doc_ids"])
        result.precision_at_k = precision_at_k(retrieved_ids, relevant_set, self.k)
        result.recall_at_k = recall_at_k(retrieved_ids, relevant_set, self.k)
        result.mrr = 1.0 / next(
            (i + 1 for i, d in enumerate(retrieved_ids) if d in relevant_set),
            float("inf"),
        )

        # Generation metrics
        result.faithfulness = compute_faithfulness(context, answer)["faithfulness_score"]
        result.answer_relevance = compute_relevance(sample["question"], answer)["score"] / 5.0
        result.groundedness = compute_groundedness(context, sample["question"], answer)["groundedness_score"]
        result.answer_correctness = semantic_similarity(answer, sample["ground_truth_answer"])

        return result

    def evaluate_dataset(self, golden_dataset: list[dict]) -> dict:
        results = [self.evaluate_single(s) for s in golden_dataset]

        summary = {
            "total_samples": len(results),
            "retrieval": {
                "precision_at_k": sum(r.precision_at_k for r in results) / len(results),
                "recall_at_k": sum(r.recall_at_k for r in results) / len(results),
                "mrr": sum(r.mrr for r in results if r.mrr != float("inf")) / len(results),
            },
            "generation": {
                "faithfulness": sum(r.faithfulness for r in results) / len(results),
                "answer_relevance": sum(r.answer_relevance for r in results) / len(results),
                "groundedness": sum(r.groundedness for r in results) / len(results),
                "answer_correctness": sum(r.answer_correctness for r in results) / len(results),
            },
            "individual_results": [r.__dict__ for r in results],
        }

        return summary

# Run evaluation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
)

with open("evaluation/golden_dataset_v1.0.json") as f:
    dataset = json.load(f)["samples"]

evaluator = RAGEvaluator(rag_chain=rag_chain, judge_llm=llm)
report = evaluator.evaluate_dataset(dataset)

print(f"Retrieval P@4:     {report['retrieval']['precision_at_k']:.2f}")
print(f"Retrieval R@4:     {report['retrieval']['recall_at_k']:.2f}")
print(f"Retrieval MRR:     {report['retrieval']['mrr']:.2f}")
print(f"Faithfulness:      {report['generation']['faithfulness']:.2f}")
print(f"Answer Relevance:  {report['generation']['answer_relevance']:.2f}")
print(f"Answer Correctness:{report['generation']['answer_correctness']:.2f}")
```

### Interpreting Results and Diagnosis

```
DIAGNOSIS MATRIX:

Low Retrieval Recall + Low Faithfulness
  → Retrieval is fundamentally broken
  → Fix: Review chunking strategy, try smaller chunks, add BM25 hybrid

Low Retrieval Precision + Low Faithfulness
  → Too much noise in context confusing the LLM
  → Fix: Add re-ranker, reduce K, use context compression

High Retrieval Recall + Low Faithfulness
  → LLM is ignoring retrieved context (using parametric knowledge)
  → Fix: Strengthen prompt ("Answer ONLY from the context below"),
         try a more instruction-following model

High Retrieval Recall + High Faithfulness + Low Answer Relevance
  → Context is correct but LLM answer doesn't address the question
  → Fix: Improve prompt with clearer task framing, add few-shot examples

High All Retrieval + High Faithfulness + Low Answer Correctness
  → Ground truth answers may be stale, or question requires
    information across multiple documents
  → Fix: Review ground truth, add multi-hop retrieval
```

---

## 10. Lab: Evaluate an Existing RAG Chatbot

### Lab Objective

Set up an evaluation pipeline for an existing RAG chatbot, identify at least 3 failure types across the golden dataset, and produce a diagnostic report with prioritised fixes.

### Lab Setup

```bash
# Install dependencies
pip install ragas deepeval langchain langchain-openai langchain-community \
            chromadb sentence-transformers datasets python-dotenv

# Create evaluation directory
mkdir -p evaluation/results
```

### Step 1 — Create a Mini Golden Dataset

```python
# evaluation/create_golden_dataset.py
import json

# Simulated golden dataset for a company HR chatbot
golden_dataset = [
    {
        "id": "q001",
        "question": "How many days of annual leave do employees get?",
        "relevant_doc_ids": ["hr_leave_policy_001"],
        "ground_truth_answer": "Full-time employees receive 25 days of annual leave per year.",
        "difficulty": "easy",
        "category": "HR policy"
    },
    {
        "id": "q002",
        "question": "What is the process for requesting parental leave?",
        "relevant_doc_ids": ["hr_parental_001", "hr_parental_002"],
        "ground_truth_answer": "Submit a parental leave request via the HR portal at least 8 weeks before the expected start date.",
        "difficulty": "medium",
        "category": "HR policy"
    },
    {
        "id": "q003",
        "question": "Can a contractor claim the employee wellness benefit?",
        "relevant_doc_ids": ["hr_benefits_001"],
        "ground_truth_answer": "No. The wellness benefit is available to permanent employees only. Contractors are not eligible.",
        "difficulty": "medium",
        "category": "Benefits"
    },
    {
        "id": "q004",
        "question": "What is the maximum claimable amount for home office equipment?",
        "relevant_doc_ids": ["hr_expenses_001"],
        "ground_truth_answer": "Employees can claim up to £500 per year for home office equipment.",
        "difficulty": "easy",
        "category": "Expenses"
    },
    {
        "id": "q005",
        "question": "Who approves an employee's training budget request, and what is the deadline?",
        "relevant_doc_ids": ["hr_training_001", "hr_training_002"],
        "ground_truth_answer": "Training budget requests are approved by the line manager and must be submitted by 31st January each year.",
        "difficulty": "hard",
        "category": "Training"
    },
]

with open("evaluation/golden_dataset_v1.0.json", "w") as f:
    json.dump({"version": "v1.0", "samples": golden_dataset}, f, indent=2)

print(f"Golden dataset saved: {len(golden_dataset)} samples")
```

### Step 2 — Run the RAG Chain

```python
# evaluation/run_rag.py
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate

# Load the golden dataset
with open("evaluation/golden_dataset_v1.0.json") as f:
    dataset = json.load(f)["samples"]

# Setup RAG chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
)

# Run the RAG chain on every sample
raw_results = []
for sample in dataset:
    output = qa_chain.invoke(sample["question"])
    raw_results.append({
        "id": sample["id"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth_answer"],
        "relevant_doc_ids": sample["relevant_doc_ids"],
        "generated_answer": output["result"],
        "retrieved_docs": [
            {
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "content": doc.page_content
            }
            for doc in output["source_documents"]
        ],
    })

with open("evaluation/results/raw_results.json", "w") as f:
    json.dump(raw_results, f, indent=2)

print(f"Generated answers for {len(raw_results)} questions")
```

### Step 3 — Evaluate with RAGAS

```python
# evaluation/run_ragas.py
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

with open("evaluation/results/raw_results.json") as f:
    raw_results = json.load(f)

# Format for RAGAS
ragas_data = {
    "question":     [r["question"] for r in raw_results],
    "answer":       [r["generated_answer"] for r in raw_results],
    "contexts":     [[doc["content"] for doc in r["retrieved_docs"]] for r in raw_results],
    "ground_truth": [r["ground_truth"] for r in raw_results],
}

dataset = Dataset.from_dict(ragas_data)

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

results = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision, answer_correctness],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

df = results.to_pandas()
df.to_csv("evaluation/results/ragas_scores.csv", index=False)

# Print summary
print("\n=== RAGAS Evaluation Report ===")
print(f"Faithfulness:       {df['faithfulness'].mean():.2f}")
print(f"Answer Relevancy:   {df['answer_relevancy'].mean():.2f}")
print(f"Context Recall:     {df['context_recall'].mean():.2f}")
print(f"Context Precision:  {df['context_precision'].mean():.2f}")
print(f"Answer Correctness: {df['answer_correctness'].mean():.2f}")

# Flag worst performing samples
print("\n=== Samples Needing Attention ===")
worst = df.nsmallest(3, "faithfulness")[["question", "faithfulness", "context_recall"]]
print(worst.to_string())
```

### Step 4 — Diagnose and Report

```python
# evaluation/diagnose.py
import json
import pandas as pd

df = pd.read_csv("evaluation/results/ragas_scores.csv")

def classify_failure(row):
    if row["faithfulness"] < 0.6:
        return "HALLUCINATION"
    elif row["context_recall"] < 0.6:
        return "RETRIEVAL_MISS"
    elif row["context_precision"] < 0.5:
        return "IRRELEVANT_CHUNKS"
    elif row["answer_correctness"] < 0.6:
        return "INCOMPLETE_ANSWER"
    else:
        return "OK"

df["failure_type"] = df.apply(classify_failure, axis=1)
failure_counts = df["failure_type"].value_counts()

print("\n=== Failure Type Distribution ===")
print(failure_counts)

print("\n=== Prioritised Fixes ===")
for failure, count in failure_counts.items():
    if failure == "OK":
        continue
    print(f"\n{failure} ({count} samples):")
    if failure == "HALLUCINATION":
        print("  Fix: Strengthen system prompt to anchor answers to context only")
        print("  Fix: Add groundedness assertion in post-processing")
    elif failure == "RETRIEVAL_MISS":
        print("  Fix: Review chunk size — may be too large, splitting relevant content")
        print("  Fix: Try hybrid retrieval (add BM25 alongside vector search)")
    elif failure == "IRRELEVANT_CHUNKS":
        print("  Fix: Add a cross-encoder re-ranker to filter noisy chunks")
        print("  Fix: Reduce K or use context compression")
    elif failure == "INCOMPLETE_ANSWER":
        print("  Fix: Increase K to retrieve more chunks")
        print("  Fix: Check if relevant info spans multiple docs — may need multi-hop")

df.to_csv("evaluation/results/diagnosis_report.csv", index=False)
print("\nFull report saved to evaluation/results/diagnosis_report.csv")
```

### Expected Lab Outcomes

After completing the lab you should have:

```
evaluation/
├── golden_dataset_v1.0.json         ← benchmark questions + ground truth
├── results/
│   ├── raw_results.json             ← RAG chain outputs per question
│   ├── ragas_scores.csv             ← per-question metric scores
│   └── diagnosis_report.csv         ← failure type classification + fix priorities
```

And a printed report showing which failure modes dominate your system, with specific, prioritised fixes for each.

---

## 11. Evaluation Scorecard Template

Use this scorecard to track RAG system quality over time and across system versions.

```
╔══════════════════════════════════════════════════════════════════════╗
║              RAG EVALUATION SCORECARD                               ║
║  System: ___________________   Date: ___________   Version: _______ ║
╠══════════════════════════════════════════════════════════════════════╣
║  RETRIEVAL METRICS          Score   Target   Status                  ║
║  ─────────────────────────────────────────────────────              ║
║  Precision@K                _____   > 0.70   [ ] Pass  [ ] Fail     ║
║  Recall@K                   _____   > 0.75   [ ] Pass  [ ] Fail     ║
║  MRR                        _____   > 0.65   [ ] Pass  [ ] Fail     ║
║  MAP                        _____   > 0.60   [ ] Pass  [ ] Fail     ║
╠══════════════════════════════════════════════════════════════════════╣
║  GENERATION METRICS         Score   Target   Status                  ║
║  ─────────────────────────────────────────────────────              ║
║  Faithfulness               _____   > 0.85   [ ] Pass  [ ] Fail     ║
║  Answer Relevance           _____   > 0.80   [ ] Pass  [ ] Fail     ║
║  Groundedness               _____   > 0.80   [ ] Pass  [ ] Fail     ║
║  Answer Correctness         _____   > 0.75   [ ] Pass  [ ] Fail     ║
╠══════════════════════════════════════════════════════════════════════╣
║  FAILURE DISTRIBUTION       Count   % of total                       ║
║  ─────────────────────────────────────────────────────              ║
║  Hallucinations             _____   _______%                         ║
║  Retrieval misses           _____   _______%                         ║
║  Irrelevant chunks          _____   _______%                         ║
║  Incomplete answers         _____   _______%                         ║
║  OK                         _____   _______%                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  TOP ISSUES (narrative):                                             ║
║  1. ____________________________________________________________      ║
║  2. ____________________________________________________________      ║
║  3. ____________________________________________________________      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PRIORITISED ACTIONS:                                                ║
║  [ ] ____________________________________________________________     ║
║  [ ] ____________________________________________________________     ║
║  [ ] ____________________________________________________________     ║
╠══════════════════════════════════════════════════════════════════════╣
║  COMPARISON TO PREVIOUS VERSION:                                     ║
║  Faithfulness: _____ → _____  (Δ _____)                              ║
║  Recall@K:     _____ → _____  (Δ _____)                              ║
║  Correctness:  _____ → _____  (Δ _____)                              ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Baseline Targets for Production RAG

| Metric             | Minimum (Ship) | Good (Healthy) | Excellent (Best-in-class) |
| ------------------ | -------------- | -------------- | ------------------------- |
| Faithfulness       | 0.75           | 0.85           | 0.95+                     |
| Answer Relevance   | 0.70           | 0.80           | 0.90+                     |
| Context Recall@K   | 0.65           | 0.80           | 0.90+                     |
| Context Precision  | 0.55           | 0.70           | 0.85+                     |
| Answer Correctness | 0.65           | 0.75           | 0.90+                     |
| Hallucination Rate | < 20%          | < 10%          | < 3%                      |

---

## Summary

RAG evaluation is not optional for production systems — it is the mechanism that turns a demo into a reliable product.

```
The Evaluation Flywheel:

  Build RAG
     ↓
  Create Golden Dataset (start with 50-100 samples)
     ↓
  Measure: Precision@K, Recall@K, Faithfulness, Correctness
     ↓
  Diagnose: retrieval failure? generation failure? both?
     ↓
  Fix: chunking / embedding / re-ranker / prompt
     ↓
  Re-measure → confirm improvement, check for regressions
     ↓
  Add failing cases to golden dataset → grows over time
     ↑_______________________________________________|
```

**Key takeaways:**

1. **Retrieval first** — if retrieval is broken, no amount of prompt engineering fixes it. Measure Recall@K before tuning generation.
2. **Faithfulness is your hallucination alarm** — keep it above 0.85 in production. Set automated alerts.
3. **Golden datasets compound in value** — every bug you find and add to the dataset makes future regressions detectable in minutes.
4. **Use RAGAS for offline evals, LangSmith for production monitoring, DeepEval for CI/CD gates** — they complement each other.
5. **Human eval calibrates automated eval** — run human evaluation on 100–200 samples before trusting automated metrics.

---

_Guide maintained as part of the **Integrating Generative AI — Advanced RAG & Enterprise Patterns (Level 2)** training programme._
