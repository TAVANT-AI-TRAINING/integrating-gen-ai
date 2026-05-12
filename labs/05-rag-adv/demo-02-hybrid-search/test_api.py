"""
Test client for Demo-14: Hybrid Search API
Run: python test_api.py (server must be running on port 8002)
"""

import requests

BASE_URL = "http://localhost:8002"


def print_section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def test_health():
    print_section("1. Health Check")
    r = requests.get(f"{BASE_URL}/health")
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"Dense weight: {data['default_dense_weight']}, Sparse weight: {data['default_sparse_weight']}")
    print(f"BM25 docs: {data['bm25_chunk_count']}")
    print("✓ Health check passed")


def test_ingest_files():
    print_section("2. Ingest Both Documents")
    for fname in ["Documents/guidelines.txt", "Documents/policy.txt"]:
        with open(fname, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/ingest/file",
                files={"file": (fname.split("/")[-1], f, "text/plain")},
            )
        r.raise_for_status()
        data = r.json()
        print(f"  {data['filename']}: {data['chunks_created']} chunks, BM25 total: {data['bm25_total_docs']}")
    print("✓ Ingestion passed")


def test_verify():
    print_section("3. Verify Both Indexes")
    r = requests.get(f"{BASE_URL}/retrieve/verify")
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"ChromaDB chunks: {data['chroma_chunk_count']}")
    print(f"BM25 chunks: {data['bm25_chunk_count']}")
    print(f"In sync: {data['bm25_in_sync']}")
    print("✓ Verify passed")


def test_dense_retrieval():
    print_section("4. Dense-Only Retrieval (semantic query)")
    payload = {"query": "flexible work arrangements from home", "k": 3}
    r = requests.post(f"{BASE_URL}/retrieve/dense", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Mode: {data['mode']}")
    print(f"Results: {data['count']}")
    for doc in data["results"]:
        print(f"  Rank {doc['rank']}: {doc['content_preview'][:80]}...")
    print("✓ Dense retrieval passed")


def test_sparse_retrieval():
    print_section("5. Sparse-Only Retrieval (BM25 keyword query)")
    payload = {"query": "two-factor authentication", "k": 3}
    r = requests.post(f"{BASE_URL}/retrieve/sparse", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Mode: {data['mode']}")
    print(f"Results: {data['count']}")
    for doc in data["results"]:
        print(f"  Rank {doc['rank']}: {doc['content_preview'][:80]}...")
    print("✓ Sparse retrieval passed")


def test_hybrid_retrieval():
    print_section("6. Hybrid Retrieval (EnsembleRetriever RRF)")
    payload = {
        "query": "password security 12 characters authentication",
        "k": 4,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
    }
    r = requests.post(f"{BASE_URL}/retrieve/hybrid", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Mode: {data['mode']}")
    print(f"Results: {data['count']}")
    for doc in data["results"]:
        print(f"  Rank {doc['rank']}: {doc['content_preview'][:80]}...")
    print("✓ Hybrid retrieval passed")


def test_compare_semantic_query():
    print_section("7. Compare — Semantic Query (dense should win)")
    payload = {
        "query": "What are the benefits of flexible working arrangements?",
        "k": 3,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "include_content": False,
    }
    r = requests.post(f"{BASE_URL}/retrieve/compare", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Dense count:  {data['results']['dense']['count']}")
    print(f"Sparse count: {data['results']['sparse']['count']}")
    print(f"Hybrid count: {data['results']['hybrid']['count']}")
    oa = data["overlap_analysis"]
    print(f"\nOverlap analysis:")
    print(f"  All three agree on:   {oa['all_three_overlap_count']} chunks")
    print(f"  Unique to dense:      {oa['unique_to_dense_count']}")
    print(f"  Unique to sparse:     {oa['unique_to_sparse_count']}")
    print(f"  Unique to hybrid:     {oa['unique_to_hybrid_count']}")
    print(f"\nInsight: {oa['insight']}")
    print("✓ Compare (semantic) passed")


def test_compare_keyword_query():
    print_section("8. Compare — Keyword Query (BM25 should win)")
    payload = {
        "query": "two-factor authentication password 12",
        "k": 3,
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "include_content": False,
    }
    r = requests.post(f"{BASE_URL}/retrieve/compare", json=payload)
    r.raise_for_status()
    data = r.json()
    oa = data["overlap_analysis"]
    print(f"All-three overlap: {oa['all_three_overlap_count']}")
    print(f"Unique to sparse (BM25): {oa['unique_to_sparse_count']}")
    print(f"Insight: {oa['insight']}")
    print("✓ Compare (keyword) passed")


def test_generate_modes():
    print_section("9. RAG Generation — Compare Modes")
    question = "What are the password security requirements?"
    for mode in ["dense", "sparse", "hybrid"]:
        payload = {
            "query": question,
            "k": 4,
            "retrieval_mode": mode,
            "include_sources": False,
        }
        r = requests.post(f"{BASE_URL}/generate/rag", json=payload)
        r.raise_for_status()
        data = r.json()
        print(f"\n  Mode: {mode}")
        print(f"  Answer: {data['answer'][:200]}...")
    print("\n✓ RAG generation (all modes) passed")


if __name__ == "__main__":
    print("\n🧪 Demo-14 Hybrid Search — API Tests")
    print("   Server: http://localhost:8002\n")

    tests = [
        test_health,
        test_ingest_files,
        test_verify,
        test_dense_retrieval,
        test_sparse_retrieval,
        test_hybrid_retrieval,
        test_compare_semantic_query,
        test_compare_keyword_query,
        test_generate_modes,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    print("=" * 60)
