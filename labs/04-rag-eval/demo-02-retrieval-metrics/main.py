"""
Demo 02: Retrieval Metrics — Precision@K, Recall@K, MRR, MAP

Retrieval is the foundation of RAG. If the wrong chunks are retrieved,
even the best LLM cannot generate a correct answer (demo-01 showed this).

This demo teaches the 4 standard retrieval metrics using a simple
keyword-overlap retriever — no LLM or embeddings required. The focus
is on understanding the MATH behind each metric.

Metrics covered:
  • Precision@K  — Of the K chunks retrieved, what fraction are relevant?
  • Recall@K     — Of all relevant chunks, what fraction did we retrieve?
  • MRR          — How early does the first relevant chunk appear?
  • MAP          — Overall retrieval quality across all queries

Usage:
    uv run python main.py
"""

# No external libraries needed — this demo is pure Python math!

# ============================================================================
# KNOWLEDGE BASE
# 15 HR document chunks — each has a unique ID, topic, and content.
# The chunk IDs are used by the golden dataset to specify which chunks
# are "correct" for each query.
# ============================================================================
KNOWLEDGE_BASE: list[dict] = [
    {"id": "chunk_001", "topic": "remote_work",      "content": "remote work policy employees authorized remotely 3 days week manager approval availability core hours"},
    {"id": "chunk_002", "topic": "remote_work",      "content": "remote work equipment laptop headset company provides stable internet VPN required company systems"},
    {"id": "chunk_003", "topic": "benefits",         "content": "employee benefits health insurance medical dental vision 401k retirement plan company match"},
    {"id": "chunk_004", "topic": "benefits",         "content": "benefits 401k five percent company match flexible spending accounts life insurance disability"},
    {"id": "chunk_005", "topic": "leave",            "content": "vacation days 15 days per year tenure sick leave 10 days personal days 3 days per year"},
    {"id": "chunk_006", "topic": "leave",            "content": "parental leave 12 weeks paid bereavement leave 5 days maternity paternity"},
    {"id": "chunk_007", "topic": "code_review",      "content": "code review peer review required before merging code quality test coverage documentation security"},
    {"id": "chunk_008", "topic": "code_review",      "content": "pull request review process approval required two reviewers senior engineer sign off"},
    {"id": "chunk_009", "topic": "security",         "content": "password strong minimum 12 characters two-factor authentication 2FA never share credentials"},
    {"id": "chunk_010", "topic": "security",         "content": "security incident report immediately IT security team suspicious activity phishing email"},
    {"id": "chunk_011", "topic": "meetings",         "content": "meetings 30 minutes or less agenda beforehand start end on time actionable notes"},
    {"id": "chunk_012", "topic": "meetings",         "content": "meeting guidelines virtual meetings camera optional Slack for quick questions email formal"},
    {"id": "chunk_013", "topic": "training",         "content": "professional development training budget 2000 dollars per employee conference attendance certification reimbursement"},
    {"id": "chunk_014", "topic": "training",         "content": "mentorship program learning development skills growth annual training plan career development"},
    {"id": "chunk_015", "topic": "expenses",         "content": "expense reimbursement submit receipts within 30 days manager approval required travel policy"},
]

# ============================================================================
# GOLDEN DATASET
# Each entry is a query paired with the IDs of ALL relevant chunks.
# This is our "ground truth" — what a perfect retriever should return.
# Building this dataset manually is the first step of any RAG evaluation.
# ============================================================================
GOLDEN_DATASET: list[dict] = [
    {
        "query": "How many days per week can employees work remotely?",
        "relevant_chunk_ids": ["chunk_001", "chunk_002"],
    },
    {
        "query": "How many vacation days do employees get per year?",
        "relevant_chunk_ids": ["chunk_005"],
    },
    {
        "query": "What is required before merging code changes?",
        "relevant_chunk_ids": ["chunk_007", "chunk_008"],
    },
    {
        "query": "What are the password security requirements?",
        "relevant_chunk_ids": ["chunk_009"],
    },
    {
        "query": "How long should meetings be?",
        "relevant_chunk_ids": ["chunk_011", "chunk_012"],
    },
    {
        "query": "What health insurance does the company provide?",
        "relevant_chunk_ids": ["chunk_003", "chunk_004"],
    },
    {
        "query": "How many weeks of parental leave do employees receive?",
        "relevant_chunk_ids": ["chunk_006"],
    },
    {
        "query": "What is the annual training budget per employee?",
        "relevant_chunk_ids": ["chunk_013", "chunk_014"],
    },
]

# ============================================================================
# SIMULATED RETRIEVER
# Uses simple keyword overlap to score and rank chunks.
# No embeddings or LLM required — the point is to evaluate retrieval quality,
# not to demonstrate a perfect retriever.
# ============================================================================
def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Retrieve the top-K chunk IDs using keyword overlap scoring.

    This is intentionally imperfect — it will sometimes retrieve wrong chunks,
    which lets us demonstrate non-trivial metric scores.

    Returns:
        List of chunk IDs ranked by relevance score (most relevant first).
    """
    query_words = set(query.lower().split())

    scores = []
    for chunk in KNOWLEDGE_BASE:
        chunk_words = set(chunk["content"].lower().split())
        overlap = len(query_words & chunk_words)
        scores.append((chunk["id"], overlap))

    # Sort by overlap (descending), then by chunk_id for determinism
    scores.sort(key=lambda x: (-x[1], x[0]))

    return [chunk_id for chunk_id, _ in scores[:k]]


# ============================================================================
# METRIC FUNCTIONS
# Each function includes the formula in its docstring.
# Read the docstrings carefully — they explain the math, not just the code.
# ============================================================================

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K = |relevant ∩ retrieved[:K]| / K

    Of the K chunks we retrieved, what fraction were actually relevant?
    A score of 1.0 means every retrieved chunk was relevant.
    A score of 0.0 means no retrieved chunk was relevant.
    """
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@K = |relevant ∩ retrieved[:K]| / |relevant|

    Of all the relevant chunks that exist, what fraction did we find?
    A score of 1.0 means we found every relevant chunk in the top K.
    A score of 0.0 means we missed every relevant chunk.
    """
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """
    Reciprocal Rank (RR) = 1 / position_of_first_relevant_chunk

    How early does the FIRST relevant chunk appear in the ranked list?
    RR = 1.0 if the very first result is relevant.
    RR = 0.5 if the second result is the first relevant one.
    RR = 0.0 if no relevant chunk appears in the results.

    MRR (Mean Reciprocal Rank) averages RR across all queries.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: list[str]) -> float:
    """
    Average Precision (AP) = sum of Precision@k for each position k
                             where document k is relevant, divided by
                             the total number of relevant documents.

    AP = (1/|relevant|) * sum(Precision@k * rel(k))
    where rel(k) = 1 if position k is relevant, 0 otherwise.

    AP rewards retrievers that find relevant chunks EARLY in the ranking.
    MAP (Mean Average Precision) averages AP across all queries.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank  # Precision@k at this relevant position
    return precision_sum / len(relevant)


def mean_average_precision(all_results: list[dict]) -> float:
    """
    MAP = mean of Average Precision across all queries.

    AP scores for each query are averaged to get an overall retrieval quality score.
    Higher MAP = better retriever overall.
    """
    if not all_results:
        return 0.0
    ap_scores = [r["average_precision"] for r in all_results]
    return sum(ap_scores) / len(ap_scores)


def mean_reciprocal_rank(all_results: list[dict]) -> float:
    """
    MRR = mean of Reciprocal Rank across all queries.

    Focuses on how quickly the retriever finds THE FIRST correct result.
    Useful for question-answering systems where you mainly care about the top result.
    """
    if not all_results:
        return 0.0
    rr_scores = [r["reciprocal_rank"] for r in all_results]
    return sum(rr_scores) / len(rr_scores)


# ============================================================================
# EVALUATION RUNNER
# ============================================================================
def evaluate_retrieval(golden_dataset: list[dict], k_values: list[int] = [1, 3, 5]) -> list[dict]:
    """Run retrieval for all queries and compute all metrics."""
    results = []

    for test_case in golden_dataset:
        query = test_case["query"]
        relevant = test_case["relevant_chunk_ids"]

        # Retrieve top-5 chunk IDs using keyword overlap
        retrieved = retrieve(query, k=max(k_values))

        result = {
            "query": query,
            "relevant": relevant,
            "retrieved": retrieved,
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "average_precision": average_precision(retrieved, relevant),
        }

        for k in k_values:
            result[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            result[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)

        results.append(result)

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 02: Retrieval Metrics")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print("""
  Metrics:
    Precision@K  — what fraction of K retrieved chunks are relevant?
    Recall@K     — what fraction of all relevant chunks were found?
    MRR          — how early does the first relevant chunk appear?
    MAP          — overall retrieval quality across all queries
""")

    k_values = [1, 3, 5]
    results = evaluate_retrieval(GOLDEN_DATASET, k_values=k_values)

    # ── Per-Query Results ────────────────────────────────────────────────────
    print("=" * 70)
    print("  PER-QUERY RESULTS")
    print("=" * 70)

    for i, r in enumerate(results, start=1):
        query_short = r["query"][:55] + "..." if len(r["query"]) > 55 else r["query"]
        print(f"\n  [{i}] {query_short}")
        print(f"       Relevant chunks : {r['relevant']}")
        print(f"       Retrieved (top5): {r['retrieved'][:5]}")
        print(f"       P@1={r['precision@1']:.2f}  P@3={r['precision@3']:.2f}  P@5={r['precision@5']:.2f}  "
              f"R@3={r['recall@3']:.2f}  R@5={r['recall@5']:.2f}  RR={r['reciprocal_rank']:.2f}  AP={r['average_precision']:.2f}")

    # ── Aggregate Scores ─────────────────────────────────────────────────────
    map_score = mean_average_precision(results)
    mrr_score = mean_reciprocal_rank(results)

    avg_p3 = sum(r["precision@3"] for r in results) / len(results)
    avg_r3 = sum(r["recall@3"] for r in results) / len(results)
    avg_p5 = sum(r["precision@5"] for r in results) / len(results)
    avg_r5 = sum(r["recall@5"] for r in results) / len(results)

    print("\n\n" + "=" * 70)
    print("  AGGREGATE SCORES")
    print("=" * 70)
    print(f"""
  MAP  (Mean Average Precision)  = {map_score:.3f}
  MRR  (Mean Reciprocal Rank)    = {mrr_score:.3f}
  Mean Precision@3               = {avg_p3:.3f}
  Mean Precision@5               = {avg_p5:.3f}
  Mean Recall@3                  = {avg_r3:.3f}
  Mean Recall@5                  = {avg_r5:.3f}
""")

    # ── Interpretation ───────────────────────────────────────────────────────
    print("=" * 70)
    print("  INTERPRETING THE SCORES")
    print("=" * 70)
    print(f"""
  MAP  = {map_score:.2f}  →  The retriever finds relevant chunks correctly {map_score*100:.0f}% of the time on average.
  MRR  = {mrr_score:.2f}  →  On average, the first relevant chunk appears at rank {1/mrr_score:.1f} (ideal: rank 1).

  Score Benchmarks (for a production RAG system):
    • Precision@3 ≥ 0.70 : Good  (our score: {avg_p3:.2f})
    • Recall@5    ≥ 0.80 : Good  (our score: {avg_r5:.2f})
    • MAP         ≥ 0.70 : Good  (our score: {map_score:.2f})
    • MRR         ≥ 0.80 : Good  (our score: {mrr_score:.2f})

  NOTE: Our keyword-based retriever is intentionally simple.
  A production system using vector embeddings (demo-12) scores higher.
  The goal here is to understand the METRIC FORMULAS, not maximize scores.

  Next Steps:
    • demo-03: Evaluate generation quality (Faithfulness, Relevance)
    • demo-05: Use RAGAS to run these metrics on a real vector store
""")


if __name__ == "__main__":
    main()
