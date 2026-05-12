"""
Demo 07: RAG Evaluation FastAPI Service

This is the capstone of Module-1. It takes everything learned in demos 01-06
and wraps it into a production-ready REST API:

  • POST /eval/single  — evaluate one Q&A pair with all 4 LLM-as-judge metrics
  • POST /eval/batch   — run batch evaluation against a golden dataset
  • GET  /eval/report  — retrieve aggregate evaluation results
  • DELETE /eval/results — clear stored results (for testing)

The service also:
  • Optionally traces every evaluation to LangSmith (if configured)
  • Auto-retrieves context from ChromaDB when not explicitly provided
  • Persists evaluation results to JSON for trend analysis
  • Runs on port 8001 to avoid conflict with demo-12 (port 8000)

Based on: labs/03-rag/demo-12-rag-fastapi-service

Usage:
    uvicorn main:app --reload --port 8001
    # Then: python test_eval_api.py

Interactive docs:
    http://localhost:8001/docs
    http://localhost:8001/redoc
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "rag-eval-demo-07")
EVAL_RESULTS_FILE = os.getenv("EVAL_RESULTS_FILE", "eval_results.json")
CHROMA_PERSIST_DIR = "./chroma_db_eval"
COLLECTION_NAME = "company_policies_eval"

FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
GROUNDEDNESS_THRESHOLD = float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.7"))
CORRECTNESS_THRESHOLD = float(os.getenv("CORRECTNESS_THRESHOLD", "0.6"))

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found. Copy .env.example to .env.")
    sys.exit(1)

LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY) and LANGCHAIN_TRACING_V2

# ============================================================================
# HR KNOWLEDGE BASE (same content as demo-12 Documents/)
# ============================================================================
HR_DOCUMENTS = [
    "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Core hours 10 AM - 3 PM required. VPN mandatory. 90-day probationary period must be completed for eligibility.",
    "Employee Benefits: Health insurance (medical, dental, vision), 401(k) with 5% company match (vesting after 1 year), FSA, life insurance at 2x salary, disability insurance, EAP.",
    "Leave Policy: Vacation 15 days/year (20 days after 5 years, 25 after 10 years). Sick leave 10 days. Personal days 3. Parental leave: primary caregiver 12 weeks paid, secondary 4 weeks paid. Bereavement 5 days immediate family.",
    "Code Review: All code changes require peer review before merging. Minimum 2 approvals, including 1 senior engineer. CI checks must pass. 80% test coverage for new code. Squash commits before merging.",
    "Security: Passwords minimum 12 characters, rotation every 90 days, MFA required. Never share credentials. Report incidents to security@company.com or ext. 5555.",
    "Professional Development: $2,000 annual training budget. Conference attendance 1-2/year. Certification reimbursement up to $500. Mentorship program available.",
    "Performance Review: Twice yearly (June, December). Ratings: EE, ME, PME, DNE. Merit increases: EE 5-8%, ME 3-5%, PME 0-2%, DNE 0%.",
    "Meetings: Keep to 30 minutes or less. Always share agenda beforehand. Start and end on time. Use Slack for quick questions, email for formal communication.",
    "Expenses: Submit within 30 days, receipts for >$25, via Concur. Meal limit $75/person, hotel $250/night, economy for flights <6h.",
]

# Built-in golden dataset for batch evaluation
BUILTIN_GOLDEN_DATASET = [
    {"query": "How many days can employees work remotely?", "ground_truth": "Up to 3 days per week with manager approval."},
    {"query": "What is the 401k company match?", "ground_truth": "The company matches 5% of employee contributions."},
    {"query": "How many weeks of paid parental leave for primary caregivers?", "ground_truth": "12 weeks of paid parental leave."},
    {"query": "What is the minimum password length?", "ground_truth": "Minimum 12 characters with complexity requirements."},
    {"query": "How many reviewer approvals for a pull request?", "ground_truth": "Minimum 2 approvals including 1 senior engineer."},
    {"query": "What is the annual training budget per employee?", "ground_truth": "$2,000 per employee per year."},
    {"query": "How long should meetings be?", "ground_truth": "Meetings should be 30 minutes or less."},
    {"query": "What is the hotel expense limit?", "ground_truth": "$250 per night for US domestic travel."},
]

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
from openai import OpenAI

openai_client = OpenAI(api_key=OPENAI_API_KEY)
vectorstore = None  # initialized on startup


def init_vectorstore():
    """Initialize ChromaDB and ingest HR documents."""
    from chromadb import Settings
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_core.documents import Document

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        client_settings=Settings(anonymized_telemetry=False),
    )
    try:
        existing = vs.similarity_search("test", k=1)
        if existing:
            logger.info("ChromaDB already populated — skipping ingestion")
            return vs
    except Exception:
        pass

    docs = [
        Document(page_content=text, metadata={"source": f"hr_doc_{i+1}"})
        for i, text in enumerate(HR_DOCUMENTS)
    ]
    vs.add_documents(docs)
    logger.info(f"Ingested {len(docs)} HR documents into ChromaDB")
    return vs

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class SingleEvalRequest(BaseModel):
    query: str = Field(..., description="The user's question", min_length=1)
    answer: str = Field(..., description="The RAG system's generated answer", min_length=1)
    context: Optional[str] = Field(default=None, description="Retrieved context. If not provided, auto-retrieved from ChromaDB.")
    ground_truth: Optional[str] = Field(default=None, description="Reference answer (enables Correctness metric)")


class MetricResult(BaseModel):
    score: float = Field(description="Score from 0.0 (worst) to 1.0 (best)")
    reasoning: str = Field(description="LLM explanation of the score")
    passed: bool = Field(description="Whether score meets the threshold")
    threshold: float = Field(description="Minimum acceptable score")


class SingleEvalResponse(BaseModel):
    eval_id: str
    query: str
    answer: str
    faithfulness: MetricResult
    relevance: MetricResult
    groundedness: MetricResult
    correctness: Optional[MetricResult] = Field(default=None, description="Only present when ground_truth was provided")
    overall_pass: bool
    eval_timestamp: str


class BatchEvalRequest(BaseModel):
    golden_dataset_path: Optional[str] = Field(default=None, description="Path to a golden_dataset.json file (from demo-04)")
    use_builtin_dataset: bool = Field(default=True, description="Use the built-in 8-question HR golden dataset")


class BatchEvalResponse(BaseModel):
    eval_id: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    aggregate_scores: Dict[str, float]
    per_case_results: List[SingleEvalResponse]
    timestamp: str


class EvalReport(BaseModel):
    total_eval_runs: int
    latest_eval_id: Optional[str]
    total_cases_evaluated: int
    overall_pass_rate: float
    aggregate_scores: Dict[str, float]
    eval_run_ids: List[str]


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    langsmith_enabled: bool
    langsmith_project: str
    thresholds: Dict[str, float]
    stored_result_count: int
    vector_db: str


# ============================================================================
# EVALUATION STORAGE
# Results stored in memory + persisted to JSON file
# ============================================================================
eval_results_store: list[dict] = []


def save_results():
    """Persist evaluation results to JSON file."""
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(eval_results_store, f, indent=2, default=str)


def load_results():
    """Load persisted results on startup."""
    global eval_results_store
    if Path(EVAL_RESULTS_FILE).exists():
        with open(EVAL_RESULTS_FILE, "r") as f:
            eval_results_store = json.load(f)
        logger.info(f"Loaded {len(eval_results_store)} stored evaluation results")


# ============================================================================
# LLM JUDGE FUNCTIONS
# Reuse the same patterns established in demo-03.
# Optionally decorated with @traceable for LangSmith tracing.
# ============================================================================
def _get_traceable():
    if LANGSMITH_ENABLED:
        from langsmith import traceable
        return traceable
    def noop(name=None, **kwargs):
        def decorator(fn): return fn
        return decorator
    return noop

traceable = _get_traceable()


def _call_judge(system_prompt: str, user_content: str) -> dict:
    """Call OpenAI with JSON response format and return raw dict."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


@traceable(name="judge_faithfulness")
def judge_faithfulness(query: str, context: str, answer: str) -> MetricResult:
    """Score whether every answer claim is grounded in the context."""
    raw = _call_judge(
        system_prompt=(
            "Score FAITHFULNESS: every claim in the answer must be directly "
            "supported by the context. Score 0.0-1.0. "
            "Return JSON: {\"score\": float, \"reasoning\": string}"
        ),
        user_content=f"Context:\n{context}\n\nAnswer:\n{answer}",
    )
    score = float(raw.get("score", 0.0))
    return MetricResult(score=score, reasoning=raw.get("reasoning", ""), passed=score >= FAITHFULNESS_THRESHOLD, threshold=FAITHFULNESS_THRESHOLD)


@traceable(name="judge_relevance")
def judge_relevance(query: str, answer: str) -> MetricResult:
    """Score whether the answer addresses the question asked."""
    raw = _call_judge(
        system_prompt=(
            "Score RELEVANCE: does the answer address the user's question? "
            "Score 0.0-1.0. "
            "Return JSON: {\"score\": float, \"reasoning\": string}"
        ),
        user_content=f"Question:\n{query}\n\nAnswer:\n{answer}",
    )
    score = float(raw.get("score", 0.0))
    return MetricResult(score=score, reasoning=raw.get("reasoning", ""), passed=score >= RELEVANCE_THRESHOLD, threshold=RELEVANCE_THRESHOLD)


@traceable(name="judge_groundedness")
def judge_groundedness(query: str, context: str, answer: str) -> MetricResult:
    """Score whether the answer stays within the context (no hallucinated extras)."""
    raw = _call_judge(
        system_prompt=(
            "Score GROUNDEDNESS: does the answer stay within the provided context "
            "without adding external information? Score 0.0-1.0. "
            "Return JSON: {\"score\": float, \"reasoning\": string}"
        ),
        user_content=f"Context:\n{context}\n\nAnswer:\n{answer}",
    )
    score = float(raw.get("score", 0.0))
    return MetricResult(score=score, reasoning=raw.get("reasoning", ""), passed=score >= GROUNDEDNESS_THRESHOLD, threshold=GROUNDEDNESS_THRESHOLD)


@traceable(name="judge_correctness")
def judge_correctness(query: str, answer: str, ground_truth: str) -> MetricResult:
    """Score factual correctness against the ground truth answer."""
    raw = _call_judge(
        system_prompt=(
            "Score FACTUAL CORRECTNESS: compare the answer to the ground truth reference. "
            "Score 0.0-1.0. "
            "Return JSON: {\"score\": float, \"reasoning\": string}"
        ),
        user_content=f"Ground Truth:\n{ground_truth}\n\nAnswer to Evaluate:\n{answer}",
    )
    score = float(raw.get("score", 0.0))
    return MetricResult(score=score, reasoning=raw.get("reasoning", ""), passed=score >= CORRECTNESS_THRESHOLD, threshold=CORRECTNESS_THRESHOLD)


# ============================================================================
# RAG PIPELINE (for auto-context retrieval)
# ============================================================================
@traceable(name="auto_retrieve_context")
def auto_retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve context from ChromaDB when caller doesn't provide it."""
    if vectorstore is None:
        return "(vector store not initialized)"
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


# ============================================================================
# CORE EVALUATION LOGIC
# ============================================================================
def evaluate_single(request: SingleEvalRequest) -> SingleEvalResponse:
    """Run all applicable judge functions for one Q&A pair."""
    eval_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    # Get context — either from request or auto-retrieved
    context = request.context
    if not context:
        logger.info(f"[{eval_id}] No context provided — auto-retrieving from ChromaDB")
        context = auto_retrieve_context(request.query)

    # Run judges
    faith = judge_faithfulness(request.query, context, request.answer)
    relev = judge_relevance(request.query, request.answer)
    groun = judge_groundedness(request.query, context, request.answer)
    corre = None
    if request.ground_truth:
        corre = judge_correctness(request.query, request.answer, request.ground_truth)

    # Overall pass: all provided metrics must pass
    metric_passes = [faith.passed, relev.passed, groun.passed]
    if corre:
        metric_passes.append(corre.passed)
    overall_pass = all(metric_passes)

    return SingleEvalResponse(
        eval_id=eval_id,
        query=request.query,
        answer=request.answer,
        faithfulness=faith,
        relevance=relev,
        groundedness=groun,
        correctness=corre,
        overall_pass=overall_pass,
        eval_timestamp=timestamp,
    )


# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="RAG Evaluation API",
    description=(
        "Evaluate RAG system quality with LLM-as-judge metrics. "
        "Part of Module-1: RAG Evaluation Techniques. "
        "Extends demo-12-rag-fastapi-service with evaluation endpoints."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "RAG Evaluation API",
        "version": "1.0.0",
        "description": "Module-1 capstone — LLM-as-judge evaluation for RAG systems",
        "endpoints": {
            "documentation": {"swagger": "/docs", "redoc": "/redoc"},
            "health": "GET /health",
            "evaluation": {
                "single": "POST /eval/single",
                "batch": "POST /eval/batch",
                "report": "GET /eval/report",
                "clear": "DELETE /eval/results",
            },
        },
        "metrics": {
            "faithfulness": f"threshold={FAITHFULNESS_THRESHOLD}",
            "relevance": f"threshold={RELEVANCE_THRESHOLD}",
            "groundedness": f"threshold={GROUNDEDNESS_THRESHOLD}",
            "correctness": f"threshold={CORRECTNESS_THRESHOLD} (only when ground_truth provided)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check service health, configuration, and stored result count."""
    return HealthResponse(
        status="healthy",
        llm_model=OPENAI_MODEL,
        langsmith_enabled=LANGSMITH_ENABLED,
        langsmith_project=LANGCHAIN_PROJECT,
        thresholds={
            "faithfulness": FAITHFULNESS_THRESHOLD,
            "relevance": RELEVANCE_THRESHOLD,
            "groundedness": GROUNDEDNESS_THRESHOLD,
            "correctness": CORRECTNESS_THRESHOLD,
        },
        stored_result_count=len(eval_results_store),
        vector_db="ChromaDB",
    )


@app.post("/eval/single", response_model=SingleEvalResponse, tags=["Evaluation"])
async def eval_single(request: SingleEvalRequest):
    """
    Evaluate one Q&A pair with LLM-as-judge metrics.

    **Metrics run**:
    - Faithfulness: Is every claim grounded in context?
    - Relevance: Does the answer address the question?
    - Groundedness: Does the answer stay within context?
    - Correctness: (Only if ground_truth provided) How accurate vs reference?

    **Auto-context**: If `context` is not provided, the service retrieves
    relevant chunks from ChromaDB automatically using the query.

    Returns scores (0.0-1.0), LLM reasoning, and pass/fail per metric.
    """
    try:
        result = evaluate_single(request)
        eval_results_store.append(result.model_dump())
        save_results()
        return result
    except Exception as e:
        logger.error(f"Single evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/eval/batch", response_model=BatchEvalResponse, tags=["Evaluation"])
async def eval_batch(request: BatchEvalRequest):
    """
    Run batch evaluation against a golden dataset.

    **Options**:
    - `use_builtin_dataset=true`: Evaluates the built-in 8-question HR golden dataset
    - `golden_dataset_path`: Path to a JSON file from demo-04 (golden_dataset.json format)

    Returns aggregate scores, per-case results, and pass rate.
    """
    try:
        # Load the dataset
        if request.golden_dataset_path:
            path = Path(request.golden_dataset_path)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {request.golden_dataset_path}")
            with open(path, "r") as f:
                raw = json.load(f)
            # Handle demo-04 GoldenDataset format
            if "qa_pairs" in raw:
                test_cases = [
                    {"query": qa["question"], "ground_truth": qa["ground_truth_answer"]}
                    for qa in raw["qa_pairs"]
                ]
            else:
                test_cases = raw  # assume list of {query, ground_truth}
        else:
            test_cases = BUILTIN_GOLDEN_DATASET

        batch_id = str(uuid.uuid4())[:8]
        per_case_results = []

        logger.info(f"[{batch_id}] Running batch evaluation: {len(test_cases)} cases")

        for i, case in enumerate(test_cases, start=1):
            logger.info(f"  [{i}/{len(test_cases)}] {case['query'][:50]}...")
            single_request = SingleEvalRequest(
                query=case["query"],
                answer=_generate_rag_answer(case["query"]),
                ground_truth=case.get("ground_truth"),
            )
            result = evaluate_single(single_request)
            per_case_results.append(result)

        # Calculate aggregates
        passed = sum(1 for r in per_case_results if r.overall_pass)
        pass_rate = passed / len(per_case_results)

        def _avg_score(metric: str) -> float:
            scores = [getattr(r, metric).score for r in per_case_results if getattr(r, metric, None)]
            return sum(scores) / len(scores) if scores else 0.0

        aggregate_scores = {
            "faithfulness": _avg_score("faithfulness"),
            "relevance": _avg_score("relevance"),
            "groundedness": _avg_score("groundedness"),
        }
        corr_results = [r for r in per_case_results if r.correctness]
        if corr_results:
            aggregate_scores["correctness"] = sum(r.correctness.score for r in corr_results) / len(corr_results)

        batch_response = BatchEvalResponse(
            eval_id=batch_id,
            total_cases=len(test_cases),
            passed_cases=passed,
            pass_rate=pass_rate,
            aggregate_scores=aggregate_scores,
            per_case_results=per_case_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        eval_results_store.append({"type": "batch", **batch_response.model_dump()})
        save_results()
        return batch_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@app.get("/eval/report", response_model=EvalReport, tags=["Evaluation"])
async def get_eval_report():
    """
    Get an aggregate report of all stored evaluation results.

    Shows overall pass rate, mean scores per metric, and list of all eval IDs.
    Useful for tracking quality trends across multiple evaluation runs.
    """
    if not eval_results_store:
        return EvalReport(
            total_eval_runs=0,
            latest_eval_id=None,
            total_cases_evaluated=0,
            overall_pass_rate=0.0,
            aggregate_scores={},
            eval_run_ids=[],
        )

    metric_totals: Dict[str, list] = {"faithfulness": [], "relevance": [], "groundedness": [], "correctness": []}
    total_cases = 0
    total_passed = 0
    eval_ids = []

    for entry in eval_results_store:
        if entry.get("type") == "batch":
            total_cases += entry.get("total_cases", 0)
            total_passed += entry.get("passed_cases", 0)
            for m, val in entry.get("aggregate_scores", {}).items():
                if m in metric_totals:
                    metric_totals[m].append(val)
            eval_ids.append(entry.get("eval_id", "unknown"))
        else:
            total_cases += 1
            if entry.get("overall_pass"):
                total_passed += 1
            for m in ["faithfulness", "relevance", "groundedness"]:
                val = entry.get(m, {}).get("score")
                if val is not None:
                    metric_totals[m].append(float(val))
            if entry.get("correctness"):
                metric_totals["correctness"].append(float(entry["correctness"]["score"]))
            eval_ids.append(entry.get("eval_id", "unknown"))

    agg = {m: sum(v) / len(v) for m, v in metric_totals.items() if v}
    return EvalReport(
        total_eval_runs=len(eval_results_store),
        latest_eval_id=eval_ids[-1] if eval_ids else None,
        total_cases_evaluated=total_cases,
        overall_pass_rate=total_passed / total_cases if total_cases else 0.0,
        aggregate_scores=agg,
        eval_run_ids=eval_ids,
    )


@app.delete("/eval/results", tags=["Evaluation"])
async def clear_results():
    """Clear all stored evaluation results. Useful for testing."""
    count = len(eval_results_store)
    eval_results_store.clear()
    if Path(EVAL_RESULTS_FILE).exists():
        Path(EVAL_RESULTS_FILE).unlink()
    return {"status": "cleared", "deleted_count": count}


# ============================================================================
# HELPER: GENERATE RAG ANSWER (used in batch eval)
# ============================================================================
def _generate_rag_answer(query: str) -> str:
    """Generate a RAG answer by retrieving context + calling LLM."""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    context = auto_retrieve_context(query)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful HR assistant. Answer based ONLY on the provided context. Be concise."),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ])
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_MODEL, temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================
@app.on_event("startup")
async def startup_event():
    global vectorstore
    print("\n" + "=" * 70)
    print("  RAG EVALUATION API — STARTUP")
    print("=" * 70)
    print(f"  LLM Model       : {OPENAI_MODEL}")
    print(f"  LangSmith       : {'ENABLED — ' + LANGCHAIN_PROJECT if LANGSMITH_ENABLED else 'DISABLED (set LANGSMITH_API_KEY to enable)'}")
    print(f"  Results File    : {EVAL_RESULTS_FILE}")
    print("=" * 70)
    print("  Thresholds:")
    print(f"    Faithfulness  : ≥ {FAITHFULNESS_THRESHOLD}")
    print(f"    Relevance     : ≥ {RELEVANCE_THRESHOLD}")
    print(f"    Groundedness  : ≥ {GROUNDEDNESS_THRESHOLD}")
    print(f"    Correctness   : ≥ {CORRECTNESS_THRESHOLD}")
    print("=" * 70)

    vectorstore = init_vectorstore()
    load_results()

    print(f"\n✓ Evaluation API Ready!")
    print(f"✓ Interactive docs : http://localhost:8001/docs")
    print(f"✓ Health check     : http://localhost:8001/health")
    print(f"✓ Stored results   : {len(eval_results_store)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
