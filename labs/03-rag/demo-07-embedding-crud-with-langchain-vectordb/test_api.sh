#!/bin/bash

# Test Script for LangChain Multi-VectorDB CRUD API
# This script demonstrates CRUD operations
# Works with BOTH ChromaDB and Pinecone!

echo "============================================"
echo "LangChain Multi-VectorDB CRUD API - Tests"
echo "============================================"
echo ""

# Base URL
BASE_URL="http://localhost:8000"

echo "1. Health Check (Shows Active Database)"
echo "-----------------------------------"
curl -s $BASE_URL/ | jq '.'
echo ""
echo ""

echo "2. CREATE - Add Documents"
echo "-----------------------------------"

echo "Adding vacation policy..."
curl -s -X POST $BASE_URL/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-vacation",
    "text": "Employees receive 15 days of paid vacation annually.",
    "metadata": {"category": "benefits", "department": "HR"}
  }' | jq '.'
echo ""

echo "Adding remote work policy..."
curl -s -X POST $BASE_URL/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-remote",
    "text": "Remote work is allowed up to 3 days per week with manager approval.",
    "metadata": {"category": "workplace", "department": "HR"}
  }' | jq '.'
echo ""

echo "Adding security policy..."
curl -s -X POST $BASE_URL/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-security",
    "text": "All company data must be encrypted and passwords must be changed every 90 days.",
    "metadata": {"category": "security", "department": "IT"}
  }' | jq '.'
echo ""
echo ""

echo "3. READ - Get a Document"
echo "-----------------------------------"
curl -s $BASE_URL/documents/policy-vacation | jq '.'
echo ""
echo ""

echo "4. LIST - Get All Documents"
echo "-----------------------------------"
curl -s $BASE_URL/documents | jq '.'
echo ""
echo ""

echo "5. QUERY - Semantic Search"
echo "-----------------------------------"

echo "Query: Can I work from home?"
curl -s -X POST $BASE_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Can I work from home?",
    "n_results": 2
  }' | jq '.'
echo ""

echo "Query: How much time off do I get?"
curl -s -X POST $BASE_URL/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "How much time off do I get?",
    "n_results": 2
  }' | jq '.'
echo ""
echo ""

echo "6. UPDATE - Update a Document"
echo "-----------------------------------"
curl -s -X PUT $BASE_URL/documents/policy-vacation \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Employees receive 20 days of paid vacation annually after 2 years of service.",
    "metadata": {"category": "benefits", "department": "HR", "updated": "2024-02-01"}
  }' | jq '.'
echo ""
echo ""

echo "7. READ - Verify Update"
echo "-----------------------------------"
curl -s $BASE_URL/documents/policy-vacation | jq '.'
echo ""
echo ""

echo "8. DELETE - Remove a Document"
echo "-----------------------------------"
curl -s -X DELETE $BASE_URL/documents/policy-security | jq '.'
echo ""
echo ""

echo "9. LIST - Verify Deletion"
echo "-----------------------------------"
curl -s $BASE_URL/documents | jq '.'
echo ""
echo ""

echo "============================================"
echo "All tests completed!"
echo "This worked with your configured database!"
echo "Try switching VECTOR_DB in .env! 🔄"
echo "============================================"
