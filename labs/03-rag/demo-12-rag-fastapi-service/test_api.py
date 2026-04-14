"""
Demo 12: RAG FastAPI Service - Python Test Client

This script demonstrates how to interact with the RAG API using Python.
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_response(response: requests.Response, show_full: bool = False):
    """Print formatted response"""
    if response.status_code == 200:
        print(f"✓ Status: {response.status_code}")
        data = response.json()
        if show_full:
            print(json.dumps(data, indent=2))
        else:
            # Print key information only
            if isinstance(data, dict):
                for key, value in data.items():
                    if key != "sources" and key != "results":
                        print(f"  {key}: {value}")
                    elif key == "results" and isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
                    elif key == "sources" and isinstance(value, list):
                        print(f"  {key}: {len(value)} items")
    else:
        print(f"✗ Status: {response.status_code}")
        print(f"  Error: {response.text}")

def test_health_check():
    """Test 1: Health Check"""
    print_section("Test 1: Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print_response(response)
    return response.json() if response.status_code == 200 else None

def test_verify_store(label: str = "Initial"):
    """Test 2: Verify Vector Store"""
    print_section(f"Test 2: Verify Vector Store ({label})")
    response = requests.get(f"{BASE_URL}/retrieve/verify")
    print_response(response)
    return response.json() if response.status_code == 200 else None

def test_ingest_text():
    """Test 3: Ingest Text"""
    print_section("Test 3: Ingest Text")
    
    payload = {
        "text": "Remote work policy: Employees can work remotely up to 3 days per week with manager approval. Core hours are 10 AM - 3 PM.",
        "metadata": {
            "source": "test_policy",
            "type": "remote_work"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/ingest/text",
        json=payload
    )
    print_response(response)
    return response.json() if response.status_code == 200 else None

def test_ingest_file(filename: str):
    """Test 4-5: Ingest File"""
    print_section(f"Test: Ingest File ({filename})")
    
    filepath = Path(f"Documents/{filename}")
    if not filepath.exists():
        print(f"⚠ File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        files = {'file': (filename, f, 'text/plain')}
        response = requests.post(
            f"{BASE_URL}/ingest/file",
            files=files
        )
    
    print_response(response)
    return response.json() if response.status_code == 200 else None

def test_similarity_search(query: str, k: int = 3, include_scores: bool = False):
    """Test 7-8: Similarity Search"""
    label = "With Scores" if include_scores else "Basic"
    print_section(f"Test: Similarity Search ({label})")
    
    payload = {
        "query": query,
        "k": k,
        "include_scores": include_scores
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieve/similarity",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"  query: {data['query']}")
        print(f"  count: {data['count']}")
        
        if data['results']:
            print(f"\n  First result:")
            result = data['results'][0]
            if 'score' in result:
                print(f"    score: {result['score']}")
                print(f"    relevance: {result['relevance']}")
            print(f"    preview: {result['content_preview'][:100]}...")
    else:
        print_response(response)
    
    return response.json() if response.status_code == 200 else None

def test_mmr_search(query: str, k: int = 4, fetch_k: int = 20):
    """Test 9: MMR Search"""
    print_section("Test: MMR Search (Diverse Results)")
    
    payload = {
        "query": query,
        "k": k,
        "fetch_k": fetch_k
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieve/mmr",
        json=payload
    )
    print_response(response)
    return response.json() if response.status_code == 200 else None

def test_rag_generation(query: str, k: int = 3, include_sources: bool = False):
    """Test 10-11: RAG Generation"""
    label = "With Sources" if include_sources else "Answer Only"
    print_section(f"Test: RAG Generation ({label})")
    
    payload = {
        "query": query,
        "k": k,
        "include_sources": include_sources,
        "temperature": 0.0
    }
    
    response = requests.post(
        f"{BASE_URL}/generate/rag",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"  query: {data['query']}")
        print(f"  context_count: {data['context_count']}")
        print(f"\n  Answer:")
        print(f"    {data['answer']}")
        
        if include_sources and data.get('sources'):
            print(f"\n  Sources: {len(data['sources'])} documents")
    else:
        print_response(response)
    
    return response.json() if response.status_code == 200 else None

def test_metadata_filter():
    """Test 12: Metadata Filtering"""
    print_section("Test: Similarity Search with Metadata Filter")
    
    payload = {
        "query": "policies",
        "k": 3,
        "filter": {"source": "guidelines.txt"}
    }
    
    response = requests.post(
        f"{BASE_URL}/retrieve/similarity",
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {response.status_code}")
        print(f"  query: {data['query']}")
        print(f"  count: {data['count']}")
        print(f"  (filtered by source: guidelines.txt)")
        
        if data['results']:
            print(f"\n  Results:")
            for i, result in enumerate(data['results'], 1):
                source = result['metadata'].get('source', 'Unknown')
                print(f"    [{i}] Source: {source}")
    else:
        print_response(response)
    
    return response.json() if response.status_code == 200 else None

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  Demo 12: RAG FastAPI Service - Python Test Client")
    print("=" * 70)
    print(f"\n  Base URL: {BASE_URL}")
    print(f"  Make sure the server is running!")
    print("  Start with: uvicorn main:app --reload")
    
    try:
        # Test 1: Health Check
        test_health_check()
        
        # Test 2: Initial Verify
        test_verify_store("Initial")
        
        # Test 3: Ingest Text
        test_ingest_text()
        
        # Test 4-5: Ingest Files
        test_ingest_file("guidelines.txt")
        test_ingest_file("policy.txt")
        
        # Test 6: Verify After Ingestion
        test_verify_store("After Ingestion")
        
        # Test 7: Basic Similarity Search
        test_similarity_search("What is the remote work policy?", k=3, include_scores=False)
        
        # Test 8: Similarity Search with Scores
        test_similarity_search("remote work guidelines", k=3, include_scores=True)
        
        # Test 9: MMR Search
        test_mmr_search("company policies", k=4, fetch_k=20)
        
        # Test 10: RAG Generation (Answer Only)
        test_rag_generation("How many days can I work remotely?", k=3, include_sources=False)
        
        # Test 11: RAG Generation (With Sources)
        test_rag_generation("What are the employee benefits?", k=4, include_sources=True)
        
        # Test 12: Metadata Filtering
        test_metadata_filter()
        
        # Summary
        print("\n" + "=" * 70)
        print("  ✓ All Tests Completed!")
        print("=" * 70)
        print("\n  Next steps:")
        print("    • Open http://localhost:8000/docs for interactive testing")
        print("    • Check README.md for more examples")
        print("    • Try modifying the test queries above")
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to the API server")
        print("  Make sure the server is running:")
        print("    uvicorn main:app --reload --port 8000")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
