#!/bin/bash

# Demo 12: RAG FastAPI Service - Test Script
# This script tests all API endpoints

BASE_URL="http://localhost:8000"

echo "========================================"
echo "Demo 12: RAG FastAPI Service - Testing"
echo "========================================"
echo

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Health Check
echo -e "${BLUE}[Test 1] Health Check${NC}"
curl -s $BASE_URL/health | jq '.'
echo
echo

# Test 2: Verify Vector Store (before ingestion)
echo -e "${BLUE}[Test 2] Verify Vector Store (Initial)${NC}"
curl -s $BASE_URL/retrieve/verify | jq '.'
echo
echo

# Test 3: Ingest Text
echo -e "${BLUE}[Test 3] Ingest Text${NC}"
curl -s -X POST $BASE_URL/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Remote work policy: Employees can work remotely up to 3 days per week with manager approval. Core hours are 10 AM - 3 PM.",
    "metadata": {"source": "test_policy", "type": "remote_work"}
  }' | jq '.'
echo
echo

# Test 4: Ingest File
echo -e "${BLUE}[Test 4] Ingest File (guidelines.txt)${NC}"
if [ -f "Documents/guidelines.txt" ]; then
    curl -s -X POST $BASE_URL/ingest/file \
      -F "file=@Documents/guidelines.txt" | jq '.'
else
    echo -e "${YELLOW}Warning: Documents/guidelines.txt not found${NC}"
fi
echo
echo

# Test 5: Ingest Another File
echo -e "${BLUE}[Test 5] Ingest File (policy.txt)${NC}"
if [ -f "Documents/policy.txt" ]; then
    curl -s -X POST $BASE_URL/ingest/file \
      -F "file=@Documents/policy.txt" | jq '.'
else
    echo -e "${YELLOW}Warning: Documents/policy.txt not found${NC}"
fi
echo
echo

# Test 6: Verify Vector Store (after ingestion)
echo -e "${BLUE}[Test 6] Verify Vector Store (After Ingestion)${NC}"
curl -s $BASE_URL/retrieve/verify | jq '.'
echo
echo

# Test 7: Similarity Search (without scores)
echo -e "${BLUE}[Test 7] Similarity Search (Basic)${NC}"
curl -s -X POST $BASE_URL/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the remote work policy?",
    "k": 3,
    "include_scores": false
  }' | jq '.query, .count, .results[0].content_preview'
echo
echo

# Test 8: Similarity Search (with scores)
echo -e "${BLUE}[Test 8] Similarity Search (With Scores)${NC}"
curl -s -X POST $BASE_URL/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "remote work guidelines",
    "k": 3,
    "include_scores": true
  }' | jq '.results[] | {score: .score, relevance: .relevance, preview: .content_preview}'
echo
echo

# Test 9: MMR Search
echo -e "${BLUE}[Test 9] MMR Search (Diverse Results)${NC}"
curl -s -X POST $BASE_URL/retrieve/mmr \
  -H "Content-Type: application/json" \
  -d '{
    "query": "company policies",
    "k": 4,
    "fetch_k": 20
  }' | jq '.query, .count'
echo
echo

# Test 10: RAG Generation (without sources)
echo -e "${BLUE}[Test 10] RAG Generation (Answer Only)${NC}"
curl -s -X POST $BASE_URL/generate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many days can I work remotely?",
    "k": 3,
    "include_sources": false,
    "temperature": 0.0
  }' | jq '.query, .answer, .context_count'
echo
echo

# Test 11: RAG Generation (with sources)
echo -e "${BLUE}[Test 11] RAG Generation (With Sources)${NC}"
curl -s -X POST $BASE_URL/generate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the employee benefits?",
    "k": 4,
    "include_sources": true,
    "temperature": 0.0
  }' | jq '{query: .query, answer: .answer, context_count: .context_count, source_count: (.sources | length)}'
echo
echo

# Test 12: Metadata Filtering
echo -e "${BLUE}[Test 12] Similarity Search with Metadata Filter${NC}"
curl -s -X POST $BASE_URL/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "policies",
    "k": 3,
    "filter": {"source": "guidelines.txt"}
  }' | jq '.results[] | {source: .metadata.source, preview: .content_preview}'
echo
echo

echo -e "${GREEN}========================================"
echo "All Tests Completed!"
echo "========================================${NC}"
echo
echo "Next steps:"
echo "  • Open http://localhost:8000/docs for interactive API testing"
echo "  • Check README.md for more examples"
echo "  • Try the Python test script: python test_api.py"
