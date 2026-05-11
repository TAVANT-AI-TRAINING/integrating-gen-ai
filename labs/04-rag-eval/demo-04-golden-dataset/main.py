"""
Demo 04: Golden Dataset Creation

A "golden dataset" is a curated set of question-answer pairs that represents
the correct behavior of your RAG system. It is the foundation of all
systematic evaluation — you cannot measure progress without a benchmark.

This demo covers:
  1. Auto-generating Q&A pairs from your documents using LLM
  2. Saving and loading a golden dataset (JSON format)
  3. Comparing human evaluation vs automated evaluation scores

Usage:
    uv run python main.py
"""

import json
import os
from datetime import datetime, timezone
from statistics import correlation, mean
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from sample_documents import DOCUMENTS

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GOLDEN_DATASET_FILE = os.getenv("GOLDEN_DATASET_FILE", "golden_dataset.json")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================================
# DATA TYPES
# ============================================================================
class QAPair(BaseModel):
    """One question-answer pair in the golden dataset."""
    question: str = Field(description="The evaluation question")
    ground_truth_answer: str = Field(description="The correct, reference answer")
    source_document_id: str = Field(description="ID of the document this Q&A is based on")
    difficulty: str = Field(description="easy | medium | hard")
    category: str = Field(description="factual | procedural | comparative")


class GoldenDataset(BaseModel):
    """A versioned, saved collection of Q&A pairs for evaluation."""
    name: str
    created_at: str
    version: str
    document_count: int
    qa_pairs: list[QAPair]
    metadata: dict


# ============================================================================
# QA GENERATION
# ============================================================================
GENERATION_PROMPT = """You are creating a golden evaluation dataset for a RAG system.

Given the document below, generate exactly {n_pairs} question-answer pairs for evaluating a RAG system.

Requirements:
- Mix difficulty levels: easy (single fact), medium (requires understanding), hard (requires inference)
- Mix categories: factual (specific facts), procedural (how-to steps), comparative (comparisons)
- Questions should be realistic — what would an employee actually ask?
- Answers must be directly answerable from the document (no external knowledge)
- Answers should be concise but complete

DOCUMENT (ID: {doc_id}):
{content}

Return your response as JSON with this exact schema:
{{
  "qa_pairs": [
    {{
      "question": "...",
      "ground_truth_answer": "...",
      "difficulty": "easy|medium|hard",
      "category": "factual|procedural|comparative"
    }}
  ]
}}"""


def generate_qa_pairs(document: dict, n_pairs: int = 3) -> list[QAPair]:
    """Generate Q&A pairs from a single document using LLM."""
    prompt = GENERATION_PROMPT.format(
        n_pairs=n_pairs,
        doc_id=document["id"],
        content=document["content"]
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    raw = json.loads(response.choices[0].message.content)
    pairs = []
    for item in raw.get("qa_pairs", []):
        pairs.append(QAPair(
            question=item["question"],
            ground_truth_answer=item["ground_truth_answer"],
            source_document_id=document["id"],
            difficulty=item.get("difficulty", "medium"),
            category=item.get("category", "factual"),
        ))
    return pairs


def build_golden_dataset(documents: list[dict], n_pairs_per_doc: int = 3) -> GoldenDataset:
    """Build a complete golden dataset from all documents."""
    print(f"\n  Generating Q&A pairs from {len(documents)} documents ({n_pairs_per_doc} pairs each)...")
    all_pairs = []
    for doc in documents:
        print(f"    • {doc['title']}...", flush=True)
        pairs = generate_qa_pairs(doc, n_pairs=n_pairs_per_doc)
        all_pairs.extend(pairs)

    return GoldenDataset(
        name="HR Knowledge Base Evaluation Dataset",
        created_at=datetime.now(timezone.utc).isoformat(),
        version="1.0",
        document_count=len(documents),
        qa_pairs=all_pairs,
        metadata={
            "model_used": OPENAI_MODEL,
            "n_pairs_per_doc": n_pairs_per_doc,
            "total_qa_pairs": len(all_pairs),
        }
    )


# ============================================================================
# SAVE / LOAD
# ============================================================================
def save_golden_dataset(dataset: GoldenDataset, filepath: str) -> None:
    """Save the dataset to a JSON file for reuse in later demos."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(dataset.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved {len(dataset.qa_pairs)} Q&A pairs to: {filepath}")


def load_golden_dataset(filepath: str) -> GoldenDataset:
    """Load a previously saved golden dataset."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return GoldenDataset(**data)


# ============================================================================
# HUMAN VS AUTOMATED EVALUATION COMPARISON
#
# In practice, you would send Q&A pairs to human reviewers and compare
# their scores to automated LLM-judge scores. Here we simulate human scores
# to show how the comparison works.
# ============================================================================

# Simulated human reviewer scores (1-5 scale):
# These represent what a human HR expert gave when reviewing 5 Q&A pairs.
# quality = overall answer quality, clarity = how clear, completeness = how complete
SIMULATED_HUMAN_SCORES: list[dict] = [
    {
        "question": "How many days per week can employees work remotely?",
        "answer": "Employees can work remotely up to 3 days per week with manager approval.",
        "human_quality": 5, "human_clarity": 5, "human_completeness": 4,
    },
    {
        "question": "What is the 401k company match percentage?",
        "answer": "The company matches 5% of employee contributions to the 401k plan.",
        "human_quality": 5, "human_clarity": 5, "human_completeness": 5,
    },
    {
        "question": "How many weeks of parental leave for primary caregiver?",
        "answer": "Primary caregivers receive 12 weeks of paid parental leave.",
        "human_quality": 5, "human_clarity": 4, "human_completeness": 5,
    },
    {
        "question": "What is the minimum password length?",
        "answer": "The minimum password length is 12 characters, and it must include uppercase, lowercase, numbers, and special characters.",
        "human_quality": 5, "human_clarity": 5, "human_completeness": 5,
    },
    {
        "question": "How many reviewer approvals are needed for a pull request?",
        "answer": "You need approval from at least 2 reviewers, including 1 senior engineer.",
        "human_quality": 4, "human_clarity": 4, "human_completeness": 4,
    },
]


def automated_quality_score(question: str, answer: str) -> dict:
    """
    Automated scoring using simple heuristics + LLM.
    In production, this would call your judge functions from demo-03.
    Here we use heuristics to keep the demo self-contained (no extra API calls).
    """
    # Heuristic 1: Answer length ratio (too short = incomplete, too long = verbose)
    ideal_length = 100
    length_score = min(len(answer) / ideal_length, 1.0)

    # Heuristic 2: Question word coverage in answer
    question_words = set(question.lower().split()) - {"what", "how", "when", "where", "why", "is", "are", "the", "a", "an"}
    answer_words = set(answer.lower().split())
    coverage = len(question_words & answer_words) / max(len(question_words), 1)

    # Combined automated score (normalized to 1-5 scale)
    raw_score = (length_score * 0.4) + (coverage * 0.6)
    auto_quality = 1 + (raw_score * 4)  # scale to 1-5

    return {
        "question": question,
        "auto_quality": round(auto_quality, 1),
        "length_score": round(length_score, 2),
        "coverage_score": round(coverage, 2),
    }


def compare_human_vs_automated(human_scores: list[dict]) -> dict:
    """
    Compare human evaluation scores against automated heuristic scores.
    Returns correlation, agreement rate, and disagreement analysis.
    """
    auto_scores = [
        automated_quality_score(h["question"], h["answer"])
        for h in human_scores
    ]

    human_q = [h["human_quality"] for h in human_scores]
    auto_q = [a["auto_quality"] for a in auto_scores]

    # Pearson correlation (requires ≥2 data points with variance)
    try:
        corr = correlation(human_q, auto_q)
    except Exception:
        corr = float("nan")

    # Agreement: within 1 point on 1-5 scale
    agreements = sum(1 for h, a in zip(human_q, auto_q) if abs(h - a) <= 1.0)
    agreement_rate = agreements / len(human_scores)

    # Find cases where they disagree most
    disagreements = [
        {
            "question": human_scores[i]["question"][:60] + "...",
            "human": human_q[i],
            "auto": auto_q[i],
            "gap": abs(human_q[i] - auto_q[i]),
        }
        for i in range(len(human_scores))
        if abs(human_q[i] - auto_q[i]) > 0.5
    ]
    disagreements.sort(key=lambda x: -x["gap"])

    return {
        "correlation": corr,
        "agreement_rate": agreement_rate,
        "agreements": agreements,
        "total": len(human_scores),
        "mean_human": mean(human_q),
        "mean_auto": mean(auto_q),
        "disagreements": disagreements,
        "per_case": [
            {"question": human_scores[i]["question"][:50], "human": human_q[i], "auto": round(auto_q[i], 1)}
            for i in range(len(human_scores))
        ],
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 04: Golden Dataset Creation")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print("""
  Steps:
    1. Auto-generate Q&A pairs from HR documents using LLM
    2. Save to JSON for reuse in demos 05-07
    3. Compare human vs automated evaluation scores
""")

    # ── Step 1: Generate Golden Dataset ─────────────────────────────────────
    print("=" * 70)
    print("  STEP 1: GENERATING GOLDEN DATASET")
    print("=" * 70)

    dataset = build_golden_dataset(DOCUMENTS, n_pairs_per_doc=3)

    print(f"\n  Generated {len(dataset.qa_pairs)} Q&A pairs from {dataset.document_count} documents")
    print(f"\n  Sample Q&A pairs generated:")
    for i, qa in enumerate(dataset.qa_pairs[:4], start=1):
        print(f"\n  [{i}] [{qa.difficulty.upper()}] [{qa.category}]")
        print(f"       Q: {qa.question}")
        print(f"       A: {qa.ground_truth_answer[:120]}...")

    # ── Step 2: Save and Reload ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 2: SAVE & RELOAD (Round-Trip Verification)")
    print("=" * 70)

    save_golden_dataset(dataset, GOLDEN_DATASET_FILE)
    loaded = load_golden_dataset(GOLDEN_DATASET_FILE)
    assert len(loaded.qa_pairs) == len(dataset.qa_pairs), "Round-trip mismatch!"
    print(f"  Reloaded {len(loaded.qa_pairs)} Q&A pairs successfully — round-trip OK")
    print(f"\n  Dataset metadata:")
    print(f"    Name       : {loaded.name}")
    print(f"    Version    : {loaded.version}")
    print(f"    Created At : {loaded.created_at}")
    print(f"    Total Pairs: {loaded.metadata['total_qa_pairs']}")

    # Difficulty breakdown
    diffs = {}
    for qa in loaded.qa_pairs:
        diffs[qa.difficulty] = diffs.get(qa.difficulty, 0) + 1
    print(f"    Difficulty : {diffs}")

    # ── Step 3: Human vs Automated Comparison ───────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 3: HUMAN vs AUTOMATED EVALUATION COMPARISON")
    print("=" * 70)
    print("""
  We ask: does automated scoring agree with human reviewers?
  Human scores were collected from an HR expert reviewer (1-5 scale).
  Automated scores use heuristics (length + keyword coverage).
""")

    comparison = compare_human_vs_automated(SIMULATED_HUMAN_SCORES)

    print(f"  {'Question':<52} {'Human':^7} {'Auto':^7}")
    print(f"  {'-'*52} {'-'*7} {'-'*7}")
    for c in comparison["per_case"]:
        print(f"  {c['question']:<52} {c['human']:^7} {c['auto']:^7.1f}")

    print(f"""
  Aggregate Results:
    Correlation (Pearson r) : {comparison['correlation']:.2f}
    Agreement Rate (±1.0)   : {comparison['agreement_rate']*100:.0f}%  ({comparison['agreements']}/{comparison['total']} cases)
    Mean Human Score        : {comparison['mean_human']:.1f} / 5.0
    Mean Automated Score    : {comparison['mean_auto']:.1f} / 5.0
""")

    if comparison["disagreements"]:
        print("  Where they DISAGREED most:")
        for d in comparison["disagreements"]:
            direction = "human > auto" if d["human"] > d["auto"] else "auto > human"
            print(f"    • {d['question']}...")
            print(f"      Human={d['human']}, Auto={d['auto']:.1f}, Gap={d['gap']:.1f} ({direction})")

    print(f"""
  Key Insight:
    Automated scoring agrees with humans {comparison['agreement_rate']*100:.0f}% of the time.
    For large-scale evaluation (1000+ pairs), automated scoring is necessary.
    For quality-sensitive cases, human review remains valuable for calibration.

  The golden dataset has been saved to: {GOLDEN_DATASET_FILE}
  Use it in demos 05-07 with: load_golden_dataset("{GOLDEN_DATASET_FILE}")

  Next Steps:
    • demo-05: Feed this dataset into RAGAS for standardized metric scores
    • demo-06: Upload this dataset to LangSmith for cloud-based evaluation
    • demo-07: Use this dataset in the evaluation API's batch endpoint
""")


if __name__ == "__main__":
    main()
