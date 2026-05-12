"""
Demo 07: RAG Evaluation API — Python Test Client

Tests all evaluation endpoints of the RAG Evaluation FastAPI service.
Run AFTER starting the server:

    uvicorn main:app --reload --port 8001

Then run this script:
    python test_eval_api.py
"""

import json
import requests
from typing import Optional

BASE_URL = "http://localhost:8001"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_eval_result(data: dict):
    """Print a formatted evaluation result."""
    print(f"  eval_id       : {data.get('eval_id')}")
    print(f"  query         : {data.get('query', '')[:60]}...")
    print(f"  answer        : {data.get('answer', '')[:80]}...")
    print(f"  overall_pass  : {'✓ PASS' if data.get('overall_pass') else '✗ FAIL'}")
    print()
    for metric in ["faithfulness", "relevance", "groundedness", "correctness"]:
        m = data.get(metric)
        if m:
            status = "✓" if m["passed"] else "✗"
            print(f"  {metric:<16} {m['score']:.2f}  {status}  {m['reasoning'][:60]}...")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_health_check():
    """Test 1: Health Check"""
    print_section("Test 1: Health Check")
    r = requests.get(f"{BASE_URL}/health")
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ Status          : {data.get('status')}")
        print(f"  ✓ LLM Model       : {data.get('llm_model')}")
        print(f"  ✓ LangSmith       : {'Enabled' if data.get('langsmith_enabled') else 'Disabled'}")
        print(f"  ✓ Vector DB       : {data.get('vector_db')}")
        print(f"  ✓ Stored Results  : {data.get('stored_result_count')}")
        print(f"  ✓ Thresholds      : {data.get('thresholds')}")
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


def test_single_eval_correct():
    """Test 2: Evaluate a correct answer (should PASS all metrics)"""
    print_section("Test 2: Single Eval — Correct Answer (Expected: all PASS)")
    payload = {
        "query": "How many days can employees work remotely per week?",
        "answer": "Employees can work remotely up to 3 days per week with manager approval. They must remain available during core hours from 10 AM to 3 PM.",
        "context": "Remote Work Policy: Employees are authorized to work remotely up to 3 days per week with manager approval. Remote work requires maintaining availability during core hours (10 AM - 3 PM).",
        "ground_truth": "Employees can work remotely up to 3 days per week with manager approval.",
    }
    r = requests.post(f"{BASE_URL}/eval/single", json=payload)
    if r.status_code == 200:
        print(f"  ✓ HTTP {r.status_code}")
        print_eval_result(r.json())
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


def test_single_eval_hallucinated():
    """Test 3: Evaluate a hallucinated answer (should FAIL faithfulness)"""
    print_section("Test 3: Single Eval — Hallucinated Answer (Expected: faithfulness FAIL)")
    payload = {
        "query": "How many vacation days do employees get?",
        "answer": "Employees receive 15 vacation days per year, plus an additional 5 bonus days after their first year, and unlimited PTO after 10 years of service.",
        "context": "Leave Policy: Vacation — 15 days per year (increases with tenure). Sick leave — 10 days per year.",
        "ground_truth": "Employees receive 15 vacation days per year.",
    }
    r = requests.post(f"{BASE_URL}/eval/single", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ HTTP {r.status_code}")
        print_eval_result(data)
        faith_score = data.get("faithfulness", {}).get("score", 1.0)
        if faith_score < 0.7:
            print(f"  ✓ Faithfulness correctly identified as low ({faith_score:.2f} < 0.70)")
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


def test_single_eval_irrelevant():
    """Test 4: Evaluate an irrelevant answer (should FAIL relevance)"""
    print_section("Test 4: Single Eval — Irrelevant Answer (Expected: relevance FAIL)")
    payload = {
        "query": "What are the password security requirements?",
        "answer": "The company offers excellent health benefits including medical, dental, and vision coverage. The 401k plan includes a 5% company match.",
        "context": "Security Guidelines: Use strong passwords (minimum 12 characters). Enable two-factor authentication.",
    }
    r = requests.post(f"{BASE_URL}/eval/single", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ HTTP {r.status_code}")
        print_eval_result(data)
        rel_score = data.get("relevance", {}).get("score", 1.0)
        if rel_score < 0.7:
            print(f"  ✓ Relevance correctly identified as low ({rel_score:.2f} < 0.70)")
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


def test_single_eval_auto_context():
    """Test 5: Evaluate with auto-retrieved context (no context provided)"""
    print_section("Test 5: Single Eval — Auto Context Retrieval (no context in request)")
    payload = {
        "query": "What is the annual training budget for employees?",
        "answer": "Each employee receives $2,000 per year for professional development, training, and certifications.",
    }
    r = requests.post(f"{BASE_URL}/eval/single", json=payload)
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ HTTP {r.status_code} — context auto-retrieved from ChromaDB")
        print_eval_result(data)
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


def test_batch_eval():
    """Test 6: Run batch evaluation with built-in golden dataset"""
    print_section("Test 6: Batch Evaluation — Built-in 8-Question HR Dataset")
    print("  Running 8 test cases (this may take 1-2 minutes)...")

    payload = {"use_builtin_dataset": True}
    r = requests.post(f"{BASE_URL}/eval/batch", json=payload, timeout=180)

    if r.status_code == 200:
        data = r.json()
        print(f"\n  ✓ HTTP {r.status_code}")
        print(f"  eval_id         : {data['eval_id']}")
        print(f"  total_cases     : {data['total_cases']}")
        print(f"  passed_cases    : {data['passed_cases']}")
        print(f"  pass_rate       : {data['pass_rate']*100:.1f}%")
        print(f"\n  Aggregate Scores:")
        for metric, score in data.get("aggregate_scores", {}).items():
            status = "✓" if score >= 0.7 else "✗"
            print(f"    {metric:<16} {score:.3f}  {status}")
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text[:200]}")
    return r.status_code == 200


def test_get_report():
    """Test 7: Retrieve evaluation report"""
    print_section("Test 7: Get Evaluation Report")
    r = requests.get(f"{BASE_URL}/eval/report")
    if r.status_code == 200:
        data = r.json()
        print(f"  ✓ HTTP {r.status_code}")
        print(f"  total_eval_runs       : {data.get('total_eval_runs')}")
        print(f"  total_cases_evaluated : {data.get('total_cases_evaluated')}")
        print(f"  overall_pass_rate     : {data.get('overall_pass_rate', 0)*100:.1f}%")
        print(f"  latest_eval_id        : {data.get('latest_eval_id')}")
        print(f"\n  Aggregate Scores:")
        for metric, score in data.get("aggregate_scores", {}).items():
            print(f"    {metric:<16} {score:.3f}")
    else:
        print(f"  ✗ Status: {r.status_code} — {r.text}")
    return r.status_code == 200


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  Demo 07: RAG Evaluation API — Test Client")
    print("=" * 70)
    print(f"\n  Base URL: {BASE_URL}")
    print(f"  Ensure the server is running:")
    print(f"    uvicorn main:app --reload --port 8001")

    results = {}

    try:
        results["health"] = test_health_check()
        results["single_correct"] = test_single_eval_correct()
        results["single_hallucinated"] = test_single_eval_hallucinated()
        results["single_irrelevant"] = test_single_eval_irrelevant()
        results["single_auto_context"] = test_single_eval_auto_context()

        # Batch eval is optional (slow) — comment out if you want quick testing
        run_batch = input("\n  Run batch evaluation? (~2 min) [y/N]: ").strip().lower()
        if run_batch == "y":
            results["batch_eval"] = test_batch_eval()
            results["report"] = test_get_report()
        else:
            results["report"] = test_get_report()

    except requests.exceptions.ConnectionError:
        print("\n  ✗ Cannot connect to server!")
        print("    Start it with: uvicorn main:app --reload --port 8001")
        return

    # Summary
    print("\n\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for test, ok in results.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {test}")
    print(f"\n  Result: {passed}/{total} tests passed")
    print(f"\n  Interactive API docs: {BASE_URL}/docs")
    print("=" * 70)


if __name__ == "__main__":
    main()
