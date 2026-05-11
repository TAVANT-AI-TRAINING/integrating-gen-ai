"""
Demo 01: RAG Failure Modes

Before measuring quality, you need to understand what can go wrong.
This demo shows the 4 most common failure modes in RAG systems:

  1. Hallucination     - LLM invents facts not in the context
  2. Poor Retrieval    - Wrong chunks retrieved for the query
  3. Irrelevant Chunks - Related-but-not-helpful chunks retrieved
  4. Incomplete Answer - Only part of the required context is retrieved

Run this demo BEFORE applying any evaluation metrics — it teaches you
WHY evaluation is necessary and WHAT each metric is designed to catch.

Usage:
    uv run python main.py
"""

import os
from openai import OpenAI
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
# KNOWLEDGE BASE
# HR document chunks — the "correct" knowledge our RAG system should use
# ============================================================================
KNOWLEDGE_BASE = [
    {
        "id": "chunk_001",
        "topic": "remote_work",
        "content": "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core hours (10 AM - 3 PM)."
    },
    {
        "id": "chunk_002",
        "topic": "remote_work",
        "content": "Remote Work Equipment: The company provides a laptop and headset for remote work. Employees are responsible for a stable internet connection. VPN is required for accessing company systems remotely."
    },
    {
        "id": "chunk_003",
        "topic": "benefits",
        "content": "Employee Benefits: Health insurance (medical, dental, vision), 401(k) with 5% company match, flexible spending accounts, life insurance, and an employee assistance program."
    },
    {
        "id": "chunk_004",
        "topic": "leave",
        "content": "Leave Policy: Vacation — 15 days per year (increases with tenure). Sick leave — 10 days per year. Personal days — 3 days per year. Parental leave — 12 weeks paid. Bereavement leave — 5 days."
    },
    {
        "id": "chunk_005",
        "topic": "code_review",
        "content": "Code Review Policy: All code changes must undergo peer review before merging. Reviews should focus on code quality, test coverage, documentation completeness, and security considerations."
    },
    {
        "id": "chunk_006",
        "topic": "security",
        "content": "Security Guidelines: Use strong passwords (minimum 12 characters). Enable two-factor authentication. Never share credentials. Report security incidents immediately to the IT security team."
    },
]

# ============================================================================
# SIMULATED RAG FUNCTION
# In a real system this would call your vector store + LLM pipeline.
# Here we pass the chunks directly so we can control exactly what context
# the LLM sees — which lets us demonstrate failures precisely.
# ============================================================================
def simple_rag(query: str, retrieved_chunks: list[dict]) -> str:
    """Call the LLM with the provided chunks as context."""
    if retrieved_chunks:
        context = "\n\n".join(
            f"[{c['topic'].upper()}]\n{c['content']}"
            for c in retrieved_chunks
        )
    else:
        context = "(No context provided)"

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful HR assistant. Answer the user's question "
                    "based ONLY on the provided context. If the context doesn't "
                    "contain the answer, say so clearly."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def print_demo_header(title: str, failure_mode: str, curriculum_metric: str):
    print("\n" + "=" * 70)
    print(f"  FAILURE MODE: {title}")
    print(f"  Caught by metric: {curriculum_metric}")
    print("=" * 70)


def print_result(query: str, chunks_used: list[dict], answer: str, issue: str):
    print(f"\n  Query: {query}")
    if chunks_used:
        print(f"  Chunks used ({len(chunks_used)}):")
        for c in chunks_used:
            print(f"    • [{c['id']}] {c['topic']}: {c['content'][:80]}...")
    else:
        print("  Chunks used: (none — empty context)")
    print(f"\n  LLM Answer:\n    {answer}")
    print(f"\n  ⚠  FAILURE DETECTED: {issue}")


# ============================================================================
# FAILURE MODE 1: HALLUCINATION
# The LLM is given an empty context but still tries to answer.
# It "hallucinates" facts that are not grounded in any source document.
# Caught by: Faithfulness metric (answer claims not in context)
# ============================================================================
def demo_hallucination():
    print_demo_header(
        "1. HALLUCINATION",
        "Empty context — LLM invents facts",
        "Faithfulness  (is every claim grounded in the context?)"
    )

    query = "What is the company stock option vesting schedule?"

    # Intentionally pass NO chunks — simulating a retrieval failure where
    # the vector store returns zero results for an uncommon query.
    answer = simple_rag(query, retrieved_chunks=[])

    print_result(
        query=query,
        chunks_used=[],
        answer=answer,
        issue=(
            "No relevant document was retrieved, yet the LLM generated a "
            "confident-sounding answer. A Faithfulness evaluator would score "
            "this near 0.0 — the answer has no support in the provided context."
        )
    )


# ============================================================================
# FAILURE MODE 2: POOR RETRIEVAL
# The retriever returns the wrong chunk — a mismatch between the query
# and the retrieved content. The LLM is forced to answer from irrelevant data.
# Caught by: Precision@K / Context Precision
# ============================================================================
def demo_poor_retrieval():
    print_demo_header(
        "2. POOR RETRIEVAL",
        "Wrong chunks retrieved for the query",
        "Precision@K  (what fraction of retrieved chunks are relevant?)"
    )

    query = "How many vacation days do employees get per year?"

    # The correct chunk is chunk_004 (leave policy).
    # We simulate poor retrieval by returning chunk_005 (code review) instead.
    wrong_chunks = [KNOWLEDGE_BASE[4]]  # chunk_005: code review

    answer = simple_rag(query, retrieved_chunks=wrong_chunks)

    print_result(
        query=query,
        chunks_used=wrong_chunks,
        answer=answer,
        issue=(
            "The retriever returned a code review chunk instead of the leave "
            "policy. The LLM correctly admitted it can't find the answer — but "
            "the user got no useful information. Precision@K=0 here (0 of 1 "
            "retrieved chunks were relevant)."
        )
    )


# ============================================================================
# FAILURE MODE 3: IRRELEVANT CHUNKS
# The retriever returns chunks that are topically close but do not contain
# the specific fact needed. The LLM may give a partially wrong answer.
# Caught by: Context Recall / Recall@K
# ============================================================================
def demo_irrelevant_chunks():
    print_demo_header(
        "3. IRRELEVANT CHUNKS",
        "Topically adjacent chunks but missing the key fact",
        "Recall@K  (what fraction of relevant chunks were actually retrieved?)"
    )

    query = "How many weeks of paid parental leave do employees receive?"

    # The answer is in chunk_004 (leave policy: "Parental leave — 12 weeks paid").
    # We simulate retrieval returning the benefits chunk (nearby topic) instead.
    nearby_chunks = [KNOWLEDGE_BASE[2]]  # chunk_003: general benefits

    answer = simple_rag(query, retrieved_chunks=nearby_chunks)

    print_result(
        query=query,
        chunks_used=nearby_chunks,
        answer=answer,
        issue=(
            "The benefits chunk mentions health insurance, 401k, etc. — "
            "but NOT parental leave duration. The LLM cannot find the answer "
            "because Recall@K=0: the relevant chunk was never retrieved. "
            "The answer will be incomplete or incorrect."
        )
    )


# ============================================================================
# FAILURE MODE 4: INCOMPLETE ANSWER
# The retriever finds SOME relevant chunks but misses others. The answer
# is partially correct but incomplete — which can be just as harmful as wrong.
# Caught by: Answer Correctness / Groundedness
# ============================================================================
def demo_incomplete_answer():
    print_demo_header(
        "4. INCOMPLETE ANSWER",
        "Only part of the required context was retrieved",
        "Answer Correctness  (does the answer match the full ground truth?)"
    )

    query = "Describe the full leave policy including all leave types and durations."

    # The complete answer requires chunk_004 (leave policy).
    # We simulate partial retrieval by only returning the remote work chunk —
    # a related HR topic but missing the actual leave details.
    partial_chunks = [KNOWLEDGE_BASE[0]]  # chunk_001: remote work only

    answer = simple_rag(query, retrieved_chunks=partial_chunks)

    print_result(
        query=query,
        chunks_used=partial_chunks,
        answer=answer,
        issue=(
            "The retriever found HR content (remote work) but missed the leave "
            "policy chunk. The answer is incomplete — users won't learn about "
            "vacation days, sick leave, parental leave, or bereavement. "
            "Answer Correctness would score this below 0.4."
        )
    )


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 01: RAG Failure Modes")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print("""
  This demo shows 4 common RAG failures using a simple HR knowledge base.
  Each failure maps to a specific evaluation metric that catches it.
  Run demos 02-07 to learn how to measure and prevent these failures.
""")

    demo_hallucination()
    demo_poor_retrieval()
    demo_irrelevant_chunks()
    demo_incomplete_answer()

    # Summary table
    print("\n\n" + "=" * 70)
    print("  SUMMARY: RAG Failure Modes → Evaluation Metrics")
    print("=" * 70)
    rows = [
        ("Hallucination",      "Empty context, LLM invents facts",       "Faithfulness"),
        ("Poor Retrieval",     "Wrong chunks retrieved",                  "Precision@K, Context Precision"),
        ("Irrelevant Chunks",  "Right topic, wrong fact",                 "Recall@K, Context Recall"),
        ("Incomplete Answer",  "Partial context, partial answer",         "Answer Correctness, Groundedness"),
    ]
    print(f"\n  {'Failure Mode':<22} {'Root Cause':<36} {'Metric'}")
    print(f"  {'-'*22} {'-'*36} {'-'*30}")
    for mode, cause, metric in rows:
        print(f"  {mode:<22} {cause:<36} {metric}")
    print(f"""
  Next Steps:
    • demo-02: Measure retrieval quality with Precision@K, Recall@K, MRR, MAP
    • demo-03: Measure answer quality with LLM-as-judge evaluators
    • demo-05: Automate evaluation with the RAGAS framework
    • demo-06: Trace and evaluate with LangSmith
""")


if __name__ == "__main__":
    main()
