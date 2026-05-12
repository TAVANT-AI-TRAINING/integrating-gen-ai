"""
Test client for Demo-13: Chunking Strategies API
Run: python test_api.py (server must be running on port 8001)
"""

import requests
import json

BASE_URL = "http://localhost:8001"
POLICY_TEXT = open("Documents/policy.txt").read()
GUIDELINES_TEXT = open("Documents/guidelines.txt").read()


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
    print(f"Strategies: {data['strategies_available']}")
    print(f"Semantic threshold: {data['semantic_breakpoint_type']} @ {data['semantic_breakpoint_amount']}")
    print("✓ Health check passed")


def test_chunk_compare_basic():
    print_section("2. Compare All 4 Strategies (no content)")
    payload = {
        "text": POLICY_TEXT,
        "chunk_size": 300,
        "chunk_overlap": 50,
        "include_content": False,
    }
    r = requests.post(f"{BASE_URL}/chunk/compare", json=payload)
    r.raise_for_status()
    data = r.json()

    print(f"Input text length: {data['input_text_length']} chars")
    print(f"\nStrategy         | Chunks | Avg Size | Time (ms)")
    print("-" * 55)
    for strategy in ["fixed", "recursive", "semantic"]:
        s = data[strategy]
        print(f"{strategy:<17}| {s['chunk_count']:>6} | {s['avg_chunk_size']:>8.1f} | {s['processing_time_ms']:>9.1f}")
    pc = data["parent_child"]
    print(f"parent_child     | {pc['parent_chunk_count']:>3}p {pc['child_chunk_count']:>2}c | {pc['avg_child_size']:>8.1f} | {pc['processing_time_ms']:>9.1f}")
    print(f"\nObservation: {data['comparison_summary']['observation']}")
    print("✓ Compare (no content) passed")


def test_chunk_compare_with_content():
    print_section("3. Compare Strategies (with chunk content)")
    payload = {
        "text": GUIDELINES_TEXT,
        "chunk_size": 400,
        "chunk_overlap": 80,
        "include_content": True,
    }
    r = requests.post(f"{BASE_URL}/chunk/compare", json=payload)
    r.raise_for_status()
    data = r.json()

    print(f"Fixed chunks: {data['fixed']['chunk_count']}")
    print(f"Recursive chunks: {data['recursive']['chunk_count']}")
    print(f"Semantic chunks: {data['semantic']['chunk_count']}")
    if data["recursive"]["chunks"]:
        print(f"\nFirst recursive chunk preview:")
        print(f"  {data['recursive']['chunks'][0][:150]}...")
    print("✓ Compare (with content) passed")


def test_chunk_analyze():
    print_section("4. Analyze Single Strategy (recursive)")
    payload = {
        "text": POLICY_TEXT,
        "strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
        "include_full_content": False,
    }
    r = requests.post(f"{BASE_URL}/chunk/analyze", json=payload)
    r.raise_for_status()
    data = r.json()

    print(f"Strategy: {data['strategy']}")
    print(f"Chunks: {data['chunk_count']}, Avg size: {data['avg_size']:.0f}, Processing: {data['processing_time_ms']:.1f} ms")
    print(f"Chunk sizes: {data['size_distribution']}")
    print("✓ Analyze passed")


def test_ingest_recursive():
    print_section("5. Ingest policy.txt (recursive strategy)")
    payload = {
        "text": POLICY_TEXT,
        "metadata": {"source": "policy.txt", "type": "hr_policy"},
        "strategy": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
    }
    r = requests.post(f"{BASE_URL}/ingest/text", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"Strategy used: {data['strategy']}")
    print(f"Chunks created: {data['chunks_created']}")
    print("✓ Ingest (recursive) passed")


def test_ingest_file():
    print_section("6. Ingest guidelines.txt file (fixed strategy)")
    with open("Documents/guidelines.txt", "rb") as f:
        r = requests.post(
            f"{BASE_URL}/ingest/file?strategy=fixed&chunk_size=500&chunk_overlap=50",
            files={"file": ("guidelines.txt", f, "text/plain")},
        )
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"Filename: {data['filename']}")
    print(f"Strategy: {data['strategy']}")
    print(f"Chunks created: {data['chunks_created']}")
    print("✓ File ingest passed")


def test_verify_store():
    print_section("7. Verify Vector Store")
    r = requests.get(f"{BASE_URL}/retrieve/verify")
    r.raise_for_status()
    data = r.json()
    print(f"Status: {data['status']}")
    print(f"Has data: {data['has_data']}")
    print(f"Chunk count: {data['chunk_count']}")
    if data.get("sample_chunk"):
        print(f"Sample: {data['sample_chunk']['content_preview'][:100]}...")
    print("✓ Verify store passed")


def test_retrieve_similarity():
    print_section("8. Similarity Search")
    payload = {"query": "How many vacation days?", "k": 3, "include_scores": True}
    r = requests.post(f"{BASE_URL}/retrieve/similarity", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Query: {data['query']}")
    print(f"Results: {data['count']}")
    for res in data["results"]:
        print(f"  Rank {res['rank']}: score={res.get('score', 'n/a'):.3f} | {res['content_preview'][:80]}...")
    print("✓ Similarity search passed")


def test_generate_rag():
    print_section("9. RAG Generation")
    payload = {
        "query": "What benefits does the company offer?",
        "k": 4,
        "include_sources": True,
        "temperature": 0.0,
    }
    r = requests.post(f"{BASE_URL}/generate/rag", json=payload)
    r.raise_for_status()
    data = r.json()
    print(f"Query: {data['query']}")
    print(f"Answer: {data['answer'][:300]}...")
    print(f"Context chunks used: {data['context_count']}")
    print("✓ RAG generation passed")


if __name__ == "__main__":
    print("\n🧪 Demo-13 Chunking Strategies — API Tests")
    print("   Server: http://localhost:8001\n")

    tests = [
        test_health,
        test_chunk_compare_basic,
        test_chunk_compare_with_content,
        test_chunk_analyze,
        test_ingest_recursive,
        test_ingest_file,
        test_verify_store,
        test_retrieve_similarity,
        test_generate_rag,
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
