"""
Demo 06: LangSmith Observability Platform

Previous demos evaluated RAG quality as a one-time snapshot.
LangSmith adds continuous observability:
  • Every RAG call is traced (inputs, outputs, latency, token usage)
  • Evaluation runs are stored and comparable over time
  • You can visually inspect individual traces to debug failures
  • Automated evaluators run on datasets in the cloud

LangSmith concepts:
  • Trace     — One end-to-end execution of your RAG pipeline
  • Project   — Group of traces (e.g., "hr-rag-production")
  • Dataset   — Saved Q&A pairs for benchmark evaluation
  • Experiment — One evaluation run against a dataset (generates scores)
  • Evaluator — Function that scores a run against a reference answer

Usage:
    uv run python main.py

Requirements:
    1. Set LANGSMITH_API_KEY in .env (free account at smith.langchain.com)
    2. Set LANGCHAIN_TRACING_V2=true to enable automatic tracing
    3. Set LANGCHAIN_PROJECT to group your traces

NOTE: This demo works WITHOUT LangSmith (prints a warning and runs offline).
      Tracing and dataset evaluation require LANGSMITH_API_KEY.
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
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-eval-demo-06")
CHROMA_PERSIST_DIR = "./chroma_db_eval"
COLLECTION_NAME = "company_policies_eval"
DATASET_NAME = "hr-rag-golden-dataset"
EXPERIMENT_PREFIX = "hr-rag-eval"

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. Copy .env.example to .env.")
    sys.exit(1)

LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY) and LANGCHAIN_TRACING_V2
if not LANGSMITH_ENABLED:
    if not LANGSMITH_API_KEY:
        logger.warning("LANGSMITH_API_KEY not set — running in offline mode (no tracing).")
    elif not LANGCHAIN_TRACING_V2:
        logger.warning("LANGCHAIN_TRACING_V2=false — tracing disabled. Set to 'true' to enable.")

# ============================================================================
# KNOWLEDGE BASE
# ============================================================================
HR_DOCUMENTS = [
    "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Core hours 10 AM - 3 PM required. VPN mandatory. 90-day probationary period must be completed for eligibility.",
    "Employee Benefits: Health insurance (medical, dental, vision), 401(k) with 5% company match (vesting after 1 year), FSA, life insurance at 2x salary, disability insurance, EAP.",
    "Leave Policy: Vacation 15 days/year (20 days after 5 years, 25 after 10 years). Sick leave 10 days. Personal days 3. Parental leave: primary caregiver 12 weeks paid, secondary 4 weeks paid. Bereavement 5 days immediate family.",
    "Code Review: All code changes require peer review before merging. Minimum 2 approvals, including 1 senior engineer. CI checks must pass. 80% test coverage for new code. Squash commits before merging.",
    "Security: Passwords minimum 12 characters, rotation every 90 days, MFA required. Never share credentials. Report incidents to security@company.com or ext. 5555.",
    "Professional Development: $2,000 annual training budget. Conference attendance 1-2/year. Certification reimbursement up to $500. Mentorship program available.",
    "Performance Review: Twice yearly (June, December). Ratings: EE, ME, PME, DNE. Merit increases: EE 5-8%, ME 3-5%, PME 0-2%, DNE 0%.",
    "Expenses: Submit within 30 days, receipts for >$25, via Concur. Meal limit $75/person, hotel $250/night, economy for flights <6h.",
]

# Golden Q&A pairs for the LangSmith dataset
GOLDEN_QA_PAIRS = [
    {"query": "How many days can employees work remotely?", "answer": "Up to 3 days per week with manager approval."},
    {"query": "What is the 401k company match?", "answer": "The company matches 5% of employee contributions."},
    {"query": "How many weeks of paid parental leave for primary caregivers?", "answer": "12 weeks of paid parental leave."},
    {"query": "What is the minimum password length?", "answer": "Minimum 12 characters with complexity requirements."},
    {"query": "How many reviewer approvals are needed for a pull request?", "answer": "Minimum 2 approvals including 1 senior engineer."},
    {"query": "What is the annual training budget per employee?", "answer": "$2,000 per employee per year."},
    {"query": "What is the vacation policy after 5 years?", "answer": "Vacation increases to 20 days per year after 5 years."},
    {"query": "What is the meal expense limit?", "answer": "$75 per person per meal for client meals."},
]


# ============================================================================
# CHROMADB SETUP
# ============================================================================
def setup_chromadb():
    """Initialize ChromaDB vector store with HR documents."""
    from chromadb import Settings
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=Settings(anonymized_telemetry=False),
    )
    try:
        existing = vectorstore.similarity_search("test", k=1)
        if existing:
            logger.info("ChromaDB already populated — skipping ingestion")
            return vectorstore
    except Exception:
        pass

    docs = [
        Document(page_content=text, metadata={"source": f"hr_doc_{i+1}"})
        for i, text in enumerate(HR_DOCUMENTS)
    ]
    vectorstore.add_documents(docs)
    logger.info(f"Ingested {len(docs)} HR documents into ChromaDB")
    return vectorstore


# ============================================================================
# TRACED RAG PIPELINE
# When LANGSMITH_ENABLED, the @traceable decorator sends execution data to
# LangSmith so you can inspect inputs/outputs/latency in the dashboard.
# When disabled, @traceable is a no-op pass-through.
# ============================================================================
def _get_traceable():
    """Return langsmith traceable decorator, or a no-op if disabled."""
    if LANGSMITH_ENABLED:
        from langsmith import traceable
        return traceable
    else:
        def noop(name=None, **kwargs):
            def decorator(fn):
                return fn
            return decorator
        return noop

traceable = _get_traceable()


@traceable(name="retrieve_documents")
def retrieve(query: str, vectorstore, k: int = 3) -> list[str]:
    """Retrieve top-K relevant document chunks for the query."""
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


@traceable(name="generate_answer")
def generate(query: str, contexts: list[str]) -> str:
    """Generate an answer from the query and retrieved context."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    context_text = "\n\n".join(contexts)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant. Answer based ONLY on the provided context. Be concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context_text, "question": query})


@traceable(name="rag_pipeline")
def rag_pipeline(inputs: dict, vectorstore=None) -> dict:
    """
    Full RAG pipeline: retrieve context → generate answer.

    Accepts dict input to match LangSmith evaluator signature.
    When called via client.evaluate(), LangSmith passes dataset examples
    as inputs and compares outputs to reference answers.
    """
    query = inputs.get("query", inputs.get("question", ""))
    contexts = retrieve(query, vectorstore)
    answer = generate(query, contexts)
    return {"answer": answer, "contexts": contexts, "query": query}


# ============================================================================
# LANGSMITH DATASET MANAGEMENT
# A LangSmith dataset is a persistent collection of Q&A pairs.
# You can add examples, run evaluations against it, and track progress.
# ============================================================================
def create_or_load_dataset(client, name: str) -> str:
    """
    Check if a LangSmith dataset with this name already exists.
    If yes, return it. If no, create it with the golden Q&A pairs.
    Returns the dataset name (used as reference in evaluate calls).
    """
    try:
        existing = list(client.list_datasets(dataset_name=name))
        if existing:
            logger.info(f"Dataset '{name}' already exists — using existing dataset")
            return name
    except Exception as e:
        logger.warning(f"Could not check datasets: {e}")

    logger.info(f"Creating LangSmith dataset '{name}' with {len(GOLDEN_QA_PAIRS)} examples...")
    dataset = client.create_dataset(
        dataset_name=name,
        description="HR knowledge base golden Q&A pairs for RAG evaluation",
    )
    examples = [
        {"inputs": {"query": qa["query"]}, "outputs": {"answer": qa["answer"]}}
        for qa in GOLDEN_QA_PAIRS
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples)
    logger.info(f"Dataset created with {len(examples)} examples")
    return name


# ============================================================================
# CUSTOM EVALUATORS
# LangSmith evaluators have signature: (run, example) -> dict
# They return {"key": str, "score": float} — score is stored in LangSmith.
# ============================================================================
def faithfulness_evaluator(run, example) -> dict:
    """Score whether the answer is grounded in the retrieved context."""
    from openai import OpenAI
    oa_client = OpenAI(api_key=OPENAI_API_KEY)

    outputs = run.outputs or {}
    answer = outputs.get("answer", "")
    contexts = outputs.get("contexts", [])
    context_text = "\n".join(contexts) if contexts else "(no context)"

    prompt = (
        "Score FAITHFULNESS of this answer based on the context (0.0-1.0). "
        "Return JSON: {\"score\": float, \"reasoning\": str}\n\n"
        f"Context: {context_text}\n\nAnswer: {answer}"
    )
    response = oa_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(response.choices[0].message.content)
    return {"key": "faithfulness", "score": float(raw.get("score", 0.0)), "comment": raw.get("reasoning", "")}


def relevance_evaluator(run, example) -> dict:
    """Score whether the answer addresses the question asked."""
    from openai import OpenAI
    oa_client = OpenAI(api_key=OPENAI_API_KEY)

    outputs = run.outputs or {}
    inputs = run.inputs or {}
    query = inputs.get("query", "")
    answer = outputs.get("answer", "")

    prompt = (
        "Score RELEVANCE of this answer to the question (0.0-1.0). "
        "Return JSON: {\"score\": float, \"reasoning\": str}\n\n"
        f"Question: {query}\n\nAnswer: {answer}"
    )
    response = oa_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(response.choices[0].message.content)
    return {"key": "relevance", "score": float(raw.get("score", 0.0)), "comment": raw.get("reasoning", "")}


def answer_similarity_evaluator(run, example) -> dict:
    """
    Score how similar the generated answer is to the reference answer.
    Uses semantic comparison via LLM rather than exact string matching.
    """
    from openai import OpenAI
    oa_client = OpenAI(api_key=OPENAI_API_KEY)

    outputs = run.outputs or {}
    reference = (example.outputs or {}).get("answer", "") if example.outputs else ""
    answer = outputs.get("answer", "")

    if not reference:
        return {"key": "answer_similarity", "score": 0.5, "comment": "No reference answer available"}

    prompt = (
        "Score FACTUAL SIMILARITY between the generated answer and reference (0.0-1.0). "
        "Return JSON: {\"score\": float, \"reasoning\": str}\n\n"
        f"Reference: {reference}\n\nGenerated: {answer}"
    )
    response = oa_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = json.loads(response.choices[0].message.content)
    return {"key": "answer_similarity", "score": float(raw.get("score", 0.0)), "comment": raw.get("reasoning", "")}


# ============================================================================
# DEMO: SINGLE TRACE
# Show how individual queries are traced when LANGSMITH_ENABLED
# ============================================================================
def demo_single_trace(vectorstore):
    """Run 3 sample queries through the traced pipeline."""
    sample_queries = [
        "How many vacation days do new employees get?",
        "What is required before merging a pull request?",
        "How do I report a security incident?",
    ]
    print("\n  Running 3 sample queries through traced RAG pipeline...")
    for query in sample_queries:
        logger.info(f"  Query: {query[:55]}...")
        result = rag_pipeline({"query": query}, vectorstore=vectorstore)
        print(f"\n  Q: {query}")
        print(f"  A: {result['answer'][:150]}...")


# ============================================================================
# BATCH EVALUATION
# ============================================================================
def run_batch_evaluation(client, dataset_name: str, vectorstore):
    """Run a full evaluation experiment against the LangSmith dataset."""
    import functools

    logger.info(f"Running batch evaluation against dataset: {dataset_name}")

    # Wrap rag_pipeline to inject vectorstore (client.evaluate passes only inputs)
    def pipeline_with_vs(inputs: dict) -> dict:
        return rag_pipeline(inputs, vectorstore=vectorstore)

    results = client.evaluate(
        pipeline_with_vs,
        data=dataset_name,
        evaluators=[faithfulness_evaluator, relevance_evaluator, answer_similarity_evaluator],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=2,
    )
    return results


# ============================================================================
# OFFLINE MODE
# When LANGSMITH is not enabled, run a simple local evaluation loop
# ============================================================================
def run_offline_evaluation(vectorstore):
    """Run evaluation locally without LangSmith (offline mode)."""
    logger.info("Running offline evaluation (no LangSmith tracing)...")
    results = []
    for qa in GOLDEN_QA_PAIRS[:4]:  # run a subset for demo speed
        query = qa["query"]
        expected = qa["answer"]
        result = rag_pipeline({"query": query}, vectorstore=vectorstore)
        answer = result["answer"]
        results.append({"query": query, "expected": expected, "answer": answer})

    print("\n" + "=" * 70)
    print("  OFFLINE EVALUATION RESULTS (no LangSmith)")
    print("=" * 70)
    for r in results:
        print(f"\n  Q: {r['query']}")
        print(f"  Expected: {r['expected']}")
        print(f"  Got:      {r['answer'][:120]}...")
    print(f"""
  To enable full LangSmith tracing and evaluation:
    1. Get a free API key at https://smith.langchain.com
    2. Add to .env:
       LANGSMITH_API_KEY=ls__your_key_here
       LANGCHAIN_TRACING_V2=true
       LANGCHAIN_PROJECT=rag-eval-demo-06
""")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 06: LangSmith Observability Platform")
    print("  Module-1: RAG Evaluation Techniques")
    print("=" * 70)
    print(f"""
  LangSmith enabled : {LANGSMITH_ENABLED}
  Project           : {LANGCHAIN_PROJECT}
  OpenAI Model      : {OPENAI_MODEL}
""")

    # Setup ChromaDB
    vectorstore = setup_chromadb()

    if LANGSMITH_ENABLED:
        from langsmith import Client

        print("=" * 70)
        print("  STEP 1: SINGLE TRACE DEMO")
        print("=" * 70)
        print(f"  Traces will appear at: https://smith.langchain.com/o/default/projects/p/{LANGCHAIN_PROJECT}")
        demo_single_trace(vectorstore)

        print("\n\n" + "=" * 70)
        print("  STEP 2: CREATE LANGSMITH DATASET")
        print("=" * 70)
        client = Client()
        dataset_name = create_or_load_dataset(client, DATASET_NAME)

        print("\n\n" + "=" * 70)
        print("  STEP 3: BATCH EVALUATION EXPERIMENT")
        print("=" * 70)
        logger.info("Starting batch evaluation — this runs all evaluators against all dataset examples...")
        results = run_batch_evaluation(client, dataset_name, vectorstore)

        print(f"""
  Batch evaluation complete!

  View results at:
    https://smith.langchain.com → Projects → {LANGCHAIN_PROJECT}

  What you'll see in LangSmith:
    • Each query as a separate trace with inputs/outputs
    • Faithfulness, Relevance, Answer Similarity scores per example
    • Aggregate scores for the experiment
    • Latency and token usage per call
    • Compare this experiment to future runs as you improve the RAG system

  Tips:
    • Click a low-scoring trace to inspect what went wrong
    • Add more evaluators (e.g., toxicity, conciseness) in the evaluators list
    • Run this script again after changing chunking/prompts to compare experiments
""")
    else:
        # Offline mode — still demonstrates the pipeline, just without LangSmith
        print("=" * 70)
        print("  RUNNING IN OFFLINE MODE (LangSmith not configured)")
        print("=" * 70)
        demo_single_trace(vectorstore)
        run_offline_evaluation(vectorstore)

    print("=" * 70)
    print("  Next: demo-07 — Evaluation FastAPI Service (REST API for eval)")
    print("=" * 70)


if __name__ == "__main__":
    main()
