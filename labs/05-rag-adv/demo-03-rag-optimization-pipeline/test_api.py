"""
Test client for Demo-15: Full RAG Optimization Pipeline
Run: python test_api.py (server must be running on port 8003)

Note: The evaluate test (test_evaluate_pipeline) will be slow on first run
because the cross-encoder re-ranks 20 documents for each of 8 golden queries.
"""

import requests

BASE_URL = "http://localhost:8003"


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
    print(f"Cross-encoder loaded: {data['cross_encoder_loaded']}")
    print(f"Cross-encoder model: {data['cross_encoder_model']}")
    print(f"Baseline chunks: {data['baseline_collection']['chunk_count']}")
    print(f"Semantic chunks: {data['semantic_collection']['chunk_count']}")
    print(f"Golden dataset queries: {data['golden_dataset_queries']}")
    print("✓ Health check passed")


def test_golden_dataset():
    print_section("2. View Golden Dataset")
    r = requests.get(f"{BASE_URL}/optimize/golden-dataset")
    r.raise_for_status()
    data = r.json()
    print(f"Samples: {data['sample_count']}")
    print(f"Matching method: {data['matching_method']}")
    print(f"Rule: {data['matching_rule']}")
    for s in data["samples"][:3]:
        print(f"  [{s['id']}] {s['query'][:60]}...")
    print("✓ Golden dataset passed")


def test_ingest_both_files():
    print_section("3. Ingest Both Files (into both collections)")
    for fname in ["Documents/guidelines.txt", "Documents/policy.txt"]:
        with open(fname, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/ingest/file",
                files={"file": (fname.split("/")[-1], f, "text/plain")},
            )
        r.raise_for_status()
        data = r.json()
        print(f"  {data['filename']}: baseline={data['baseline_chunks']}, semantic={data['semantic_chunks_count']}")
    print("✓ Dual ingestion passed")


def test_verify():
    print_section("4. Verify Both Collections")
    r = requests.get(f"{BASE_URL}/retrieve/verify")
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"Baseline: {data['baseline_collection']['chunks']} chunks")
    print(f"Semantic: {data['semantic_collection']['chunks']} chunks")
    print(f"Ready for evaluation: {data['ready_for_evaluation']}")
    print("✓ Verify passed")


def test_pipeline_retrieval():
    print_section("5. Pipeline Retrieval (one query, 4 stages)")
    payload = {
        "query": "What are the password security requirements?",
        "k": 4,
        "include_content": False,
    }
    r = requests.post(f"{BASE_URL}/retrieve/pipeline", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Query: {data['query']}")
    for stage_name, stage in data["stages"].items():
        print(f"\n  [{stage_name}] {stage['count']} results")
        if stage["documents"]:
            print(f"    Top: {stage['documents'][0]['content_preview'][:80]}...")
    print("\n✓ Pipeline retrieval passed")


def test_rerank_retrieval():
    print_section("6. Re-ranking Retrieval (Stage 4 only)")
    payload = {
        "query": "How many days can employees work from home?",
        "k": 4,
        "initial_fetch_k": 10,
    }
    r = requests.post(f"{BASE_URL}/retrieve/rerank", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Mode: {data['mode']}")
    print(f"Fetched {data['fetched']} → returned top {data['returned']}")
    for doc in data["documents"]:
        print(f"  Rank {doc['rank']}: {doc['content_preview'][:80]}...")
    print("✓ Re-ranking retrieval passed")


def test_custom_query_with_metrics():
    print_section("7. Custom Query with Metrics")
    payload = {
        "query": "What sick leave do employees get?",
        "k": 4,
        "relevant_keywords": ["sick leave", "10 days"],
        "include_content": False,
    }
    r = requests.post(f"{BASE_URL}/optimize/custom-query", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Query: {data['query']}")
    print(f"Metrics computed: {data['metrics_computed']}")
    print(f"\n{'Stage':<25} {'Recall@K':>10} {'MRR':>8}")
    print("-" * 45)
    for name, s in data["stages"].items():
        print(f"{name:<25} {s.get('recall_at_k', 'n/a'):>10} {s.get('mrr', 'n/a'):>8}")
    print("\n✓ Custom query with metrics passed")


def test_evaluate_pipeline():
    print_section("8. Full Pipeline Evaluation (Golden Dataset) — may be slow")
    print("   Running 8 queries × 4 stages (including cross-encoder re-ranking)...")
    payload = {"k": 4, "dense_weight": 0.6, "sparse_weight": 0.4, "initial_fetch_k": 20}
    r = requests.post(f"{BASE_URL}/optimize/evaluate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    print(f"\n{'Pipeline Stage':<30} {'Recall@K':>10} {'MRR':>8}")
    print("-" * 50)
    for stage_name, stage in data["pipeline_stages"].items():
        print(f"{stage_name:<30} {stage['recall_at_k']:>10.3f} {stage['mrr']:>8.3f}")

    imp = data["improvement_summary"]
    print(f"\nImprovement (baseline → reranked):")
    print(f"  Recall@K: {imp['recall_baseline']} → {imp['recall_reranked']} ({imp['recall_delta']})")
    print(f"  MRR:      {imp['mrr_baseline']} → {imp['mrr_reranked']} ({imp['mrr_delta']})")
    print(f"  Best stage (Recall): {imp['best_recall_stage']}")
    print(f"  Best stage (MRR):    {imp['best_mrr_stage']}")
    print("\n✓ Pipeline evaluation passed")


def test_generate_rag_all_stages():
    print_section("9. RAG Generation — All 4 Stages")
    question = "What code review requirements must be followed?"
    for stage in ["baseline", "semantic_chunking", "hybrid_search", "reranked"]:
        payload = {
            "query": question,
            "pipeline_stage": stage,
            "include_sources": False,
        }
        r = requests.post(f"{BASE_URL}/generate/rag", json=payload)
        r.raise_for_status()
        data = r.json()
        print(f"\n  [{stage}]")
        print(f"  Chunks used: {data['context_count']}")
        print(f"  Answer: {data['answer'][:200]}...")
    print("\n✓ RAG generation (all stages) passed")


if __name__ == "__main__":
    print("\n🧪 Demo-15 Full Optimization Pipeline — API Tests")
    print("   Server: http://localhost:8003")
    print("   ⚠  Test 8 (evaluate) may take 30–120s due to cross-encoder re-ranking\n")

    tests = [
        test_health,
        test_golden_dataset,
        test_ingest_both_files,
        test_verify,
        test_pipeline_retrieval,
        test_rerank_retrieval,
        test_custom_query_with_metrics,
        test_evaluate_pipeline,
        test_generate_rag_all_stages,
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
