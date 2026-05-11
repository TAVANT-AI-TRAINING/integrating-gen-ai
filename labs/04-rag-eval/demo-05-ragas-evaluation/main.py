"""
Demo 05: RAGAS Evaluation Framework

Previous demos showed you HOW to build evaluation metrics manually.
RAGAS (Retrieval-Augmented Generation Assessment) is a framework that
provides standard, reproducible implementations of those same metrics.

Why use RAGAS?
  • Standard metric definitions shared across the industry
  • Handles the LLM judge calls automatically
  • Produces comparable scores you can track over time
  • Used widely in production RAG systems

RAGAS Metrics (all scored 0.0 to 1.0, higher = better):
  • Faithfulness       — Are answer claims supported by context? (= demo-03's faithfulness)
  • Response Relevancy — Does the answer address the question? (= demo-03's relevance)
  • Context Precision  — Are the retrieved chunks actually useful? (= demo-02's precision)
  • Context Recall     — Does context contain all needed info? (= demo-02's recall)
  • Answer Correctness — How close is the answer to ground truth? (= demo-03's correctness)

Usage:
    uv run python main.py

IMPORTANT: This uses RAGAS 0.2.x+ API.
    The 0.1.x API (using HuggingFace datasets) is different and incompatible.
"""

import json
import logging
import os
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHROMA_PERSIST_DIR = "./chroma_db_eval"
COLLECTION_NAME = "company_policies_eval"
SCORE_THRESHOLD = 0.7

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")
    sys.exit(1)

# ============================================================================
# METRIC GLOSSARY
# Print this at startup so students understand what each score means
# ============================================================================
METRIC_GLOSSARY = """
  RAGAS METRIC GLOSSARY
  ─────────────────────────────────────────────────────────────────────
  Faithfulness       : Every claim in the answer is grounded in context.
                       Low score = hallucination. Target: ≥ 0.70

  Response Relevancy : The answer actually addresses the question asked.
                       Low score = irrelevant or off-topic answer. Target: ≥ 0.70

  Context Precision  : Retrieved chunks are mostly relevant to the query.
                       Low score = too much noise in context. Target: ≥ 0.70

  Context Recall     : The context contains enough info to answer correctly.
                       Low score = key information was not retrieved. Target: ≥ 0.70

  Answer Correctness : The answer matches the reference (ground truth) answer.
                       Low score = factually wrong or incomplete. Target: ≥ 0.60
  ─────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# KNOWLEDGE BASE
# Same HR document content as demo-12-rag-fastapi-service.
# Hardcoded here so this demo is self-contained (no file dependencies).
# ============================================================================
HR_DOCUMENTS = [
    "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core hours (10 AM - 3 PM). The company provides a laptop and headset. VPN is required for accessing company systems remotely. Eligibility requires completion of the 90-day probationary period.",
    "Employee Benefits: Health insurance (medical, dental, vision), 401(k) with 5% company match (vesting after 1 year), flexible spending accounts, life insurance at 2x annual salary, short and long-term disability insurance, and employee assistance program (EAP).",
    "Leave Policy: Vacation 15 days per year (increases to 20 days after 5 years, 25 days after 10 years). Sick leave 10 days per year. Personal days 3 per year. Parental leave 12 weeks paid for primary caregiver, 4 weeks paid for secondary caregiver. Bereavement leave 5 days for immediate family, 3 days for extended family.",
    "Code Review Policy: All code changes must undergo peer review before merging. Minimum 2 reviewer approvals required, including 1 senior engineer. All automated CI checks must pass. Reviews focus on code quality, test coverage (minimum 80%), documentation, and security. Use squash commits before merging.",
    "Security Guidelines: Passwords must be minimum 12 characters with uppercase, lowercase, numbers, and special characters. Password rotation every 90 days. Multi-factor authentication (MFA) required for all company accounts. Never share credentials. Report security incidents immediately to security@company.com or IT Security at ext. 5555.",
    "Professional Development: Annual training budget $2,000 per employee. Conference attendance 1-2 per year with manager approval. Certification reimbursement up to $500 per certification. Mentorship program available to all employees.",
    "Performance Review: Reviews twice per year (June and December). Rating scale: Exceeds Expectations (EE), Meets Expectations (ME), Partially Meets Expectations (PME), Does Not Meet Expectations (DNE). Merit increases: EE 5-8%, ME 3-5%, PME 0-2%, DNE none. Employees must complete self-assessment 2 weeks before review.",
    "Expense Reimbursement: Submit within 30 days. Receipts required for expenses over $25. Submit via Concur system with manager approval. Meal limit $75 per person, hotel limit $250 per night US domestic. Economy class for flights under 6 hours.",
]

# ============================================================================
# TEST DATASET
# These are Q&A pairs with known correct answers (ground truth).
# ============================================================================
TEST_DATASET = [
    {
        "query": "How many days per week can employees work remotely?",
        "ground_truth": "Employees can work remotely up to 3 days per week with manager approval.",
    },
    {
        "query": "What is the company's 401k match percentage?",
        "ground_truth": "The company matches 5% of employee contributions to the 401k plan.",
    },
    {
        "query": "How many weeks of paid parental leave does a primary caregiver receive?",
        "ground_truth": "Primary caregivers receive 12 weeks of paid parental leave.",
    },
    {
        "query": "What is the minimum password length required by the security policy?",
        "ground_truth": "Passwords must be at least 12 characters long.",
    },
    {
        "query": "How many reviewer approvals are required for a pull request?",
        "ground_truth": "A minimum of 2 reviewer approvals is required, including 1 senior engineer.",
    },
    {
        "query": "What is the annual training budget per employee?",
        "ground_truth": "Each employee receives $2,000 per year for professional development and training.",
    },
    {
        "query": "How many vacation days do employees get after 5 years?",
        "ground_truth": "After 5 years of service, vacation increases to 20 days per year.",
    },
    {
        "query": "What is the meal expense limit per person?",
        "ground_truth": "The meal expense limit is $75 per person per meal for client meals.",
    },
]


# ============================================================================
# CHROMADB SETUP
# ============================================================================
def setup_chromadb():
    """Initialize ChromaDB and ingest HR documents."""
    from chromadb import Settings
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    logger.info("Setting up ChromaDB vector store...")
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-small",
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=Settings(anonymized_telemetry=False),
    )

    # Check if already populated to avoid duplicate ingestion
    try:
        existing = vectorstore.similarity_search("test", k=1)
        if existing:
            logger.info("ChromaDB already populated — skipping ingestion")
            return vectorstore
    except Exception:
        pass

    # Ingest HR documents
    logger.info(f"Ingesting {len(HR_DOCUMENTS)} HR documents...")
    docs = [
        Document(page_content=text, metadata={"source": f"hr_doc_{i+1}"})
        for i, text in enumerate(HR_DOCUMENTS)
    ]
    vectorstore.add_documents(docs)
    logger.info("Ingestion complete")
    return vectorstore


# ============================================================================
# RAG PIPELINE
# ============================================================================
def retrieve_context(vectorstore, query: str, k: int = 3) -> list[str]:
    """Retrieve top-K relevant chunks for the query."""
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate an answer using the LLM with retrieved context."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    context_text = "\n\n".join(contexts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant. Answer based ONLY on the provided context. Be concise and accurate."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL,
        temperature=0.0,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context_text, "question": query})


# ============================================================================
# BUILD RAGAS EVALUATION DATASET
# RAGAS 0.2.x+ API:
#   from ragas import EvaluationDataset, SingleTurnSample, evaluate
# ============================================================================
def build_evaluation_dataset(test_cases: list[dict], vectorstore):
    """Build a RAGAS EvaluationDataset from test cases."""
    from ragas import EvaluationDataset, SingleTurnSample

    logger.info(f"Building evaluation dataset from {len(test_cases)} test cases...")
    samples = []
    for i, case in enumerate(test_cases, start=1):
        query = case["query"]
        ground_truth = case["ground_truth"]

        logger.info(f"  [{i}/{len(test_cases)}] Processing: {query[:50]}...")
        contexts = retrieve_context(vectorstore, query, k=3)
        answer = generate_answer(query, contexts)

        sample = SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


# ============================================================================
# RUN RAGAS EVALUATION
# ============================================================================
def run_ragas_evaluation(dataset) -> dict:
    """Run all 5 RAGAS metrics and return results."""
    from ragas import evaluate
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,
        ContextPrecision,
        ContextRecall,
        AnswerCorrectness,
    )

    logger.info("Running RAGAS evaluation (this calls the LLM for each metric)...")
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
    ]
    result = evaluate(dataset=dataset, metrics=metrics)
    return result


# ============================================================================
# REPORT
# ============================================================================
def print_score_report(result):
    """Print per-sample and aggregate RAGAS scores."""
    try:
        df = result.to_pandas()
    except Exception:
        logger.warning("Could not convert to pandas — printing raw result")
        print(result)
        return

    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]

    print("\n" + "=" * 70)
    print("  PER-SAMPLE RAGAS SCORES")
    print("=" * 70)

    for i, row in df.iterrows():
        query_short = str(row.get("user_input", ""))[:55]
        print(f"\n  [{i+1}] {query_short}...")
        for col in metric_cols:
            val = row.get(col)
            if val is not None:
                score = float(val)
                status = "✓" if score >= SCORE_THRESHOLD else "✗"
                print(f"       {col:<25} {score:.3f}  {status}")

    print("\n\n" + "=" * 70)
    print("  AGGREGATE RAGAS SCORES")
    print("=" * 70)

    weakest = None
    weakest_score = 1.0
    for col in metric_cols:
        scores = df[col].dropna().tolist()
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        passed = sum(1 for s in scores if s >= SCORE_THRESHOLD)
        status = "✓ GOOD" if avg >= SCORE_THRESHOLD else "✗ NEEDS WORK"
        print(f"  {col:<25} avg={avg:.3f}  pass_rate={passed}/{len(scores)}  {status}")
        if avg < weakest_score:
            weakest_score = avg
            weakest = col

    if weakest:
        print(f"\n  Weakest metric: {weakest} (avg={weakest_score:.3f})")
        print(f"  → Focus optimization efforts here first.")

    print(f"""
  Score Reference:
    ≥ 0.70 : Good — acceptable for production
    0.50-0.69 : Fair — needs improvement
    < 0.50 : Poor — significant issue to address

  Next Steps:
    • demo-06: Track these scores over time with LangSmith
    • demo-07: Expose evaluation via a REST API
""")

    # Save scores
    scores_file = "ragas_scores.json"
    scores_data = {}
    for col in metric_cols:
        vals = df[col].dropna().tolist()
        if vals:
            scores_data[col] = {
                "mean": sum(vals) / len(vals),
                "scores": vals,
            }
    with open(scores_file, "w") as f:
        json.dump(scores_data, f, indent=2)
    logger.info(f"Scores saved to: {scores_file}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 05: RAGAS Evaluation Framework")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print(METRIC_GLOSSARY)

    # Step 1: Setup
    vectorstore = setup_chromadb()

    # Step 2: Build dataset (retrieve + generate for each test case)
    print("\n" + "=" * 70)
    print("  BUILDING EVALUATION DATASET")
    print("=" * 70)
    dataset = build_evaluation_dataset(TEST_DATASET, vectorstore)
    logger.info(f"Dataset ready: {len(dataset.samples)} samples")

    # Step 3: Run RAGAS
    print("\n" + "=" * 70)
    print("  RUNNING RAGAS EVALUATION")
    print("=" * 70)
    result = run_ragas_evaluation(dataset)

    # Step 4: Print report
    print_score_report(result)


if __name__ == "__main__":
    main()
