"""
Demo 02: Access Control & PII Protection — Test Client

Run the server first:
  uv run uvicorn main:app --reload --port 8002

Then run:
  uv run python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8002"
BANNER   = "=" * 65


def section(title: str) -> None:
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)


def show_query_result(data: dict) -> None:
    print(f"  user_clearance:       {data.get('user_clearance')} ({data.get('clearance_label')})")
    print(f"  query_pii_found:      {data.get('query_pii_found')}  types: {data.get('query_pii_types')}")
    print(f"  query_used:           {data.get('query_used', '')[:120]}")
    print(f"  context_count:        {data.get('context_count')}")
    print(f"  sources:              {data.get('sources')}")
    print(f"  answer_pii_redacted:  {data.get('answer_pii_redacted')}")
    print(f"\n  Answer: {data.get('answer', '')[:500]}")


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_health():
    section("1. Health check")
    resp = requests.get(f"{BASE_URL}/health")
    print(json.dumps(resp.json(), indent=2))


def test_store_stats():
    section("2. Store stats — documents by clearance level")
    resp = requests.get(f"{BASE_URL}/admin/store-stats")
    print(json.dumps(resp.json(), indent=2))


def test_rbac_junior_cannot_access_confidential():
    section("3. RBAC — Junior employee (clearance=0) queries salary data")
    print("  Query: 'What is the salary for a senior manager?'")
    print("  Expected: No confidential data returned\n")
    resp = requests.post(f"{BASE_URL}/query", json={
        "query": "What is the salary for a senior manager?",
        "user":  {"user_id": "alice", "clearance": 0, "role": "employee"},
        "k": 5,
    })
    show_query_result(resp.json())


def test_rbac_hr_admin_can_access_confidential():
    section("4. RBAC — HR Admin (clearance=3) queries same salary data")
    print("  Query: 'What is the salary for a senior manager?'")
    print("  Expected: CONFIDENTIAL compensation data returned\n")
    resp = requests.post(f"{BASE_URL}/query", json={
        "query": "What is the salary for a senior manager?",
        "user":  {"user_id": "bob_hr", "clearance": 3, "role": "hr_admin"},
        "k": 5,
    })
    show_query_result(resp.json())


def test_rbac_manager_sees_expense_not_salary():
    section("5. RBAC — Manager (clearance=2) can see expense policy but NOT compensation")
    resp = requests.post(f"{BASE_URL}/query", json={
        "query": "What are the expense approval limits for managers?",
        "user":  {"user_id": "carol_mgr", "clearance": 2, "role": "manager"},
        "k": 5,
    })
    show_query_result(resp.json())


def test_pii_in_query_redacted():
    section("6. PII in query — should be redacted before retrieval")
    print("  Query contains: name + SSN + email")
    resp = requests.post(f"{BASE_URL}/query", json={
        "query": "What is the sick leave policy for John Smith (SSN 123-45-6789, john.smith@acmecorp.com)?",
        "user":  {"user_id": "test_user", "clearance": 1, "role": "employee"},
        "k": 4,
    })
    data = resp.json()
    print(f"\n  Original query:  {data.get('query')}")
    print(f"  Redacted query:  {data.get('query_used')}")
    print(f"  PII types found: {data.get('query_pii_types')}")
    print(f"\n  Answer: {data.get('answer', '')[:300]}")


def test_source_attribution_sanitised():
    section("7. Source sanitisation — classification labels hidden from low-clearance user")
    print("  A clearance=0 user should see 'Internal HR Document', not 'level3_compensation_bands.txt'")
    print("  (Note: clearance=0 won't retrieve level3 docs, but if they did the source is sanitised)")
    resp = requests.post(f"{BASE_URL}/query", json={
        "query": "What benefits does AcmeCorp offer?",
        "user":  {"user_id": "new_hire", "clearance": 0, "role": "employee"},
        "k": 4,
    })
    data = resp.json()
    print(f"\n  sources shown to user: {data.get('sources')}")
    print(f"  answer: {data.get('answer', '')[:300]}")


def test_leak_risk_demo():
    section("8. /demo/leak-risk — side-by-side comparison with and without RBAC")
    resp = requests.get(
        f"{BASE_URL}/demo/leak-risk",
        params={"query": "What is the salary for a senior manager?"},
    )
    data = resp.json()

    print("\n  WITHOUT RBAC:")
    for doc in data["without_rbac"]["docs"]:
        print(f"    [{doc['clearance_label']}] {doc['source']}: {doc['preview'][:80]}...")

    print(f"\n  Risk: {data['without_rbac']['risk']}")

    print("\n  WITH RBAC (clearance=0 PUBLIC only):")
    for doc in data["with_rbac_clearance_0"]["docs"]:
        print(f"    [{doc['clearance_label']}] {doc['source']}: {doc['preview'][:80]}...")

    if not data["with_rbac_clearance_0"]["docs"]:
        print("    (no results — CONFIDENTIAL data correctly hidden)")
    print(f"\n  Protection: {data['with_rbac_clearance_0']['protected']}")


def main():
    print(f"\n{BANNER}")
    print("  Demo 02 — Access Control & PII  |  Test Client")
    print(f"{BANNER}")
    print(f"  Server: {BASE_URL}")

    try:
        test_health()
        test_store_stats()

        print("\n─── RBAC DEMONSTRATIONS ───")
        test_rbac_junior_cannot_access_confidential()
        test_rbac_hr_admin_can_access_confidential()
        test_rbac_manager_sees_expense_not_salary()

        print("\n─── PII PROTECTION DEMONSTRATIONS ───")
        test_pii_in_query_redacted()
        test_source_attribution_sanitised()

        print("\n─── LEAK RISK COMPARISON ───")
        test_leak_risk_demo()

        print(f"\n{BANNER}")
        print("  All tests complete. Open http://localhost:8002/docs to explore.")
        print(BANNER)

    except requests.exceptions.ConnectionError:
        print(f"\n  ERROR: Cannot connect to {BASE_URL}")
        print("  Start the server:  uv run uvicorn main:app --reload --port 8002")


if __name__ == "__main__":
    main()
