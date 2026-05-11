"""
Demo 03: Generation Metrics — LLM-as-Judge

Demo-02 measured whether the RIGHT chunks were retrieved.
This demo measures whether the LLM generated a GOOD ANSWER from those chunks.

We use the "LLM-as-judge" pattern: a second LLM call evaluates the first one.
This works because:
  • LLMs are good at reading comprehension and fact-checking
  • It scales better than human evaluation
  • It correlates well with human judgment (70-80% agreement)

Metrics covered:
  • Faithfulness    — Is every claim grounded in the context? (anti-hallucination)
  • Relevance       — Does the answer address the question asked?
  • Groundedness    — Does the answer stay within the context (no extras)?
  • Correctness     — How factually accurate vs the ground truth?

Usage:
    uv run python main.py
"""

import json
import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================================
# THRESHOLDS
# Scores below these values indicate a problem worth investigating.
# ============================================================================
FAITHFULNESS_THRESHOLD = 0.7
RELEVANCE_THRESHOLD = 0.7
GROUNDEDNESS_THRESHOLD = 0.7
CORRECTNESS_THRESHOLD = 0.6

# ============================================================================
# DATA TYPES
# ============================================================================
class EvalInput(BaseModel):
    """One test case: a question, the context retrieved, the answer generated, and the ground truth."""
    label: str = Field(description="Human-readable label for this test case")
    query: str = Field(description="The user's question")
    context: str = Field(description="The retrieved chunks joined into one string")
    answer: str = Field(description="The RAG system's generated answer")
    ground_truth: str = Field(description="The correct, reference answer")


class MetricScore(BaseModel):
    """Result from one LLM judge call."""
    score: float = Field(description="Score from 0.0 (worst) to 1.0 (best)")
    reasoning: str = Field(description="LLM explanation of the score")
    passed: bool = Field(description="Whether score meets the threshold")
    threshold: float = Field(description="The minimum acceptable score")


# ============================================================================
# JUDGE PROMPTS
# IMPORTANT: Each prompt must mention "JSON" — OpenAI requires this when using
# response_format={"type":"json_object"}. The prompt specifies the exact schema.
# ============================================================================

FAITHFULNESS_PROMPT = """You are an evaluation judge assessing RAG system quality.

TASK: Score the FAITHFULNESS of an answer based on its context.
Faithfulness = every factual claim in the answer is directly supported by the context.

SCORING GUIDE:
  1.0 = Every claim is explicitly stated in the context
  0.7 = Most claims are supported, 1-2 minor extrapolations
  0.4 = Several claims not found in context
  0.0 = Answer contains fabricated information not in context at all

Return your evaluation as JSON with this exact schema:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}"""

RELEVANCE_PROMPT = """You are an evaluation judge assessing RAG system quality.

TASK: Score the RELEVANCE of an answer to the question asked.
Relevance = the answer directly addresses what the user asked.

SCORING GUIDE:
  1.0 = Answer completely and directly addresses the question
  0.7 = Answer mostly addresses the question, minor tangents
  0.4 = Answer is partially relevant or addresses a different aspect
  0.0 = Answer does not address the question at all

Return your evaluation as JSON with this exact schema:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}"""

GROUNDEDNESS_PROMPT = """You are an evaluation judge assessing RAG system quality.

TASK: Score the GROUNDEDNESS of an answer in its context.
Groundedness = the answer does NOT introduce information beyond what is in the context,
even if that information might be correct.

SCORING GUIDE:
  1.0 = Answer stays strictly within the context, no added information
  0.7 = Answer mostly uses context, 1-2 common-knowledge additions
  0.4 = Answer adds multiple facts not in the context
  0.0 = Answer is largely based on external knowledge, not the context

Return your evaluation as JSON with this exact schema:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}"""

CORRECTNESS_PROMPT = """You are an evaluation judge assessing RAG system quality.

TASK: Score the FACTUAL CORRECTNESS of an answer compared to the ground truth.
Correctness = the answer contains the same key facts as the ground truth answer.

SCORING GUIDE:
  1.0 = Answer matches all key facts in the ground truth
  0.7 = Answer has most key facts correct, minor omissions
  0.4 = Answer has some correct facts but significant errors or omissions
  0.0 = Answer contradicts or completely misses the ground truth facts

Return your evaluation as JSON with this exact schema:
{"score": <float 0.0-1.0>, "reasoning": "<one sentence explanation>"}"""


# ============================================================================
# JUDGE FUNCTIONS
# Each function calls the LLM with a specific prompt and parses the score.
# ============================================================================

def _call_judge(system_prompt: str, user_content: str, threshold: float) -> MetricScore:
    """Shared helper — calls LLM judge and parses JSON response."""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(response.choices[0].message.content)
    score = float(raw.get("score", 0.0))
    reasoning = raw.get("reasoning", "No reasoning provided.")
    return MetricScore(
        score=score,
        reasoning=reasoning,
        passed=score >= threshold,
        threshold=threshold,
    )


def evaluate_faithfulness(eval_input: EvalInput) -> MetricScore:
    """Is every claim in the answer directly supported by the context?"""
    content = (
        f"CONTEXT:\n{eval_input.context}\n\n"
        f"ANSWER:\n{eval_input.answer}"
    )
    return _call_judge(FAITHFULNESS_PROMPT, content, FAITHFULNESS_THRESHOLD)


def evaluate_relevance(eval_input: EvalInput) -> MetricScore:
    """Does the answer address the question asked?"""
    content = (
        f"QUESTION:\n{eval_input.query}\n\n"
        f"ANSWER:\n{eval_input.answer}"
    )
    return _call_judge(RELEVANCE_PROMPT, content, RELEVANCE_THRESHOLD)


def evaluate_groundedness(eval_input: EvalInput) -> MetricScore:
    """Does the answer stay within the context (no information beyond it)?"""
    content = (
        f"CONTEXT:\n{eval_input.context}\n\n"
        f"ANSWER:\n{eval_input.answer}"
    )
    return _call_judge(GROUNDEDNESS_PROMPT, content, GROUNDEDNESS_THRESHOLD)


def evaluate_correctness(eval_input: EvalInput) -> MetricScore:
    """How factually correct is the answer compared to the ground truth?"""
    content = (
        f"GROUND TRUTH:\n{eval_input.ground_truth}\n\n"
        f"ANSWER TO EVALUATE:\n{eval_input.answer}"
    )
    return _call_judge(CORRECTNESS_PROMPT, content, CORRECTNESS_THRESHOLD)


# ============================================================================
# SAMPLE TEST CASES
# 5 deliberately varied cases — from correct to severely failing.
# These represent the full spectrum of RAG answer quality.
# ============================================================================
TEST_CASES: list[EvalInput] = [
    EvalInput(
        label="CASE 1: Correct Answer",
        query="How many days can employees work remotely per week?",
        context="Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core hours (10 AM - 3 PM).",
        answer="Employees can work remotely up to 3 days per week, with manager approval required. They must remain available during core hours from 10 AM to 3 PM.",
        ground_truth="Employees can work remotely up to 3 days per week with manager approval. Core hours availability (10 AM - 3 PM) is required.",
    ),
    EvalInput(
        label="CASE 2: Hallucinated Answer (Faithfulness Failure)",
        query="How many vacation days do employees get?",
        context="Leave Policy: Vacation — 15 days per year (increases with tenure). Sick leave — 10 days per year. Personal days — 3 days per year.",
        answer="Employees receive 15 vacation days per year, plus an additional 5 days after 5 years of service, 10 days after 10 years, and unlimited PTO after 15 years of service.",
        ground_truth="Employees get 15 vacation days per year, which increases with tenure.",
    ),
    EvalInput(
        label="CASE 3: Irrelevant Answer (Relevance Failure)",
        query="What are the password security requirements?",
        context="Security Guidelines: Use strong passwords (minimum 12 characters). Enable two-factor authentication. Never share credentials. Report security incidents immediately.",
        answer="The company provides comprehensive benefits including health insurance, a 401(k) plan with 5% match, and flexible spending accounts. Employees should review their benefits during open enrollment.",
        ground_truth="Passwords must be at least 12 characters. Two-factor authentication is required. Never share credentials.",
    ),
    EvalInput(
        label="CASE 4: Incomplete Answer (Correctness Failure)",
        query="Describe all leave types and their durations.",
        context="Leave Policy: Vacation — 15 days per year. Sick leave — 10 days per year. Personal days — 3 days per year. Parental leave — 12 weeks paid. Bereavement leave — 5 days.",
        answer="Employees receive 15 vacation days per year. Sick leave is available but the exact number of days was not specified in the provided information.",
        ground_truth="Vacation: 15 days/year. Sick leave: 10 days/year. Personal days: 3 days/year. Parental leave: 12 weeks paid. Bereavement leave: 5 days.",
    ),
    EvalInput(
        label="CASE 5: Over-Grounded Answer (Groundedness Failure)",
        query="What does the code review policy require?",
        context="Code Review Policy: All code changes must undergo peer review before merging. Reviews should focus on code quality, test coverage, documentation, and security.",
        answer="All code changes require peer review before merging, focusing on code quality, test coverage, documentation, and security. This is consistent with industry best practices like those used at Google and Meta, where code reviews are mandatory and typically require two approvals.",
        ground_truth="All code changes must undergo peer review. Reviews focus on code quality, test coverage, documentation, and security.",
    ),
]


# ============================================================================
# BATCH EVALUATOR
# ============================================================================
def run_batch_evaluation(test_cases: list[EvalInput]) -> list[dict]:
    """Run all 4 metrics for each test case. Returns structured results."""
    results = []
    for case in test_cases:
        print(f"  Evaluating: {case.label}...", flush=True)
        faith = evaluate_faithfulness(case)
        relev = evaluate_relevance(case)
        groun = evaluate_groundedness(case)
        corre = evaluate_correctness(case)
        results.append({
            "label": case.label,
            "query": case.query,
            "faithfulness": faith,
            "relevance": relev,
            "groundedness": groun,
            "correctness": corre,
        })
    return results


def print_scorecard(results: list[dict]):
    """Print a formatted per-case scorecard and aggregate summary."""
    metric_names = ["faithfulness", "relevance", "groundedness", "correctness"]

    print("\n" + "=" * 70)
    print("  PER-CASE SCORECARD")
    print("=" * 70)

    for r in results:
        print(f"\n  {r['label']}")
        print(f"  Query: {r['query']}")
        for m in metric_names:
            score_obj: MetricScore = r[m]
            status = "✓ PASS" if score_obj.passed else "✗ FAIL"
            print(f"    {m.capitalize():<14} {score_obj.score:.2f}  {status}  — {score_obj.reasoning[:70]}")

    # Aggregates
    print("\n\n" + "=" * 70)
    print("  AGGREGATE SCORES (mean across 5 test cases)")
    print("=" * 70)
    for m in metric_names:
        scores = [r[m].score for r in results]
        avg = sum(scores) / len(scores)
        passed = sum(1 for r in results if r[m].passed)
        threshold = results[0][m].threshold
        print(f"  {m.capitalize():<16} avg={avg:.2f}  pass_rate={passed}/{len(results)}  threshold={threshold}")

    # Pass/Fail Summary Table
    print("\n\n" + "=" * 70)
    print("  PASS/FAIL SUMMARY TABLE")
    print("=" * 70)
    header = f"  {'Case':<40} {'Faith':^7} {'Relev':^7} {'Groun':^7} {'Corr':^7}"
    print(header)
    print("  " + "-" * 68)
    for r in results:
        row = f"  {r['label']:<40}"
        for m in metric_names:
            icon = "✓" if r[m].passed else "✗"
            row += f" {icon:^7}"
        print(row)

    print(f"""
  ✓ = score ≥ threshold (0.7 for faithfulness/relevance/groundedness, 0.6 for correctness)
  ✗ = score < threshold — investigate this failure

  Next Steps:
    • demo-04: Build a golden dataset to automate test case creation
    • demo-05: Use RAGAS to run these metrics on a real ChromaDB + LLM pipeline
    • demo-06: Track metric trends over time with LangSmith
""")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 03: Generation Metrics — LLM-as-Judge")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print("""
  Pattern: Use a second LLM call to evaluate the first LLM's output.
  Metrics: Faithfulness, Relevance, Groundedness, Answer Correctness

  Evaluating 5 test cases (2-3 API calls per case)...
""")

    results = run_batch_evaluation(TEST_CASES)
    print_scorecard(results)


if __name__ == "__main__":
    main()
