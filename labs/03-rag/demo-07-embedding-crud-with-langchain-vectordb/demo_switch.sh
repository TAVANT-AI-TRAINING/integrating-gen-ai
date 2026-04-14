#!/bin/bash

# Demo Script: Switch Between Vector Databases
# This demonstrates the power of LangChain abstraction!

echo "================================================"
echo "LangChain Abstraction Demo"
echo "Switch Vector Databases Without Code Changes!"
echo "================================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo "   Run: cp .env.example .env"
    exit 1
fi

# Function to wait for user
wait_for_user() {
    echo ""
    read -p "Press ENTER to continue..."
    echo ""
}

# Function to check which DB is configured
get_configured_db() {
    grep "^VECTOR_DB=" .env | cut -d'=' -f2
}

echo "📋 Current Configuration:"
echo "------------------------"
CURRENT_DB=$(get_configured_db)
echo "VECTOR_DB=$CURRENT_DB"
echo ""

if [ "$CURRENT_DB" == "chromadb" ]; then
    echo "✓ You're using ChromaDB (Local)"
    echo "  - Storage: Local files"
    echo "  - Speed: Fast (~10ms)"
    echo "  - Cost: Free"
elif [ "$CURRENT_DB" == "pinecone" ]; then
    echo "✓ You're using Pinecone (Cloud)"
    echo "  - Storage: Cloud"
    echo "  - Speed: Network latency (~100ms)"
    echo "  - Cost: Free tier + paid"
else
    echo "❌ Unknown database: $CURRENT_DB"
    exit 1
fi

wait_for_user

echo "🎯 Demo Steps:"
echo "-------------"
echo "1. Start server with current database"
echo "2. Add a test document"
echo "3. Search for it"
echo "4. Show how to switch databases"
echo ""

wait_for_user

echo "Step 1: Starting server..."
echo "-------------------------"
echo "Running: uv run python main.py &"
echo ""

# Start server in background (this is just a demo - in reality, user starts it)
echo "ℹ️  Please start the server in another terminal:"
echo "   uv run python main.py"
echo ""

wait_for_user

echo "Step 2: Adding test document..."
echo "-------------------------------"
DOC_ID="demo-test-$(date +%s)"

curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d "{
    \"doc_id\": \"$DOC_ID\",
    \"text\": \"This is a test document created with $CURRENT_DB\",
    \"metadata\": {\"database\": \"$CURRENT_DB\", \"timestamp\": \"$(date)\"}
  }" | jq '.'

echo ""
echo "✓ Document added: $DOC_ID"

wait_for_user

echo "Step 3: Searching for the document..."
echo "-------------------------------------"

curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "test document",
    "n_results": 3
  }' | jq '.'

echo ""
echo "✓ Search completed!"

wait_for_user

echo "Step 4: How to Switch Databases"
echo "================================"
echo ""

if [ "$CURRENT_DB" == "chromadb" ]; then
    echo "To switch from ChromaDB to Pinecone:"
    echo "-----------------------------------"
    echo ""
    echo "1. Edit .env file:"
    echo "   VECTOR_DB=pinecone"
    echo "   PINECONE_API_KEY=your-key-here"
    echo ""
    echo "2. Restart server:"
    echo "   Ctrl+C (stop current server)"
    echo "   uv run python main.py"
    echo ""
    echo "3. That's it! Same code now uses Pinecone! 🎉"
    echo ""
    echo "No code changes needed!"
    
else
    echo "To switch from Pinecone to ChromaDB:"
    echo "-----------------------------------"
    echo ""
    echo "1. Edit .env file:"
    echo "   VECTOR_DB=chromadb"
    echo ""
    echo "2. Restart server:"
    echo "   Ctrl+C (stop current server)"
    echo "   uv run python main.py"
    echo ""
    echo "3. That's it! Same code now uses ChromaDB! 🎉"
    echo ""
    echo "No code changes needed!"
fi

echo ""
echo "================================================"
echo "The Power of LangChain Abstraction!"
echo ""
echo "Same API endpoints:"
echo "  ✓ POST /documents      (works with both)"
echo "  ✓ GET /documents/{id}  (works with both)"
echo "  ✓ POST /query          (works with both)"
echo "  ✓ DELETE /documents    (works with both)"
echo ""
echo "Different backends:"
echo "  • ChromaDB: Local, fast, free"
echo "  • Pinecone: Cloud, scalable, managed"
echo ""
echo "One configuration line:"
echo "  VECTOR_DB=chromadb|pinecone"
echo ""
echo "🦜🔗 That's LangChain! 🦜🔗"
echo "================================================"
