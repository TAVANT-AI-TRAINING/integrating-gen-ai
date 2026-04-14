#!/bin/bash
# Test script for demo-09-rag-ingestion-pipeline

echo "================================"
echo "Demo 09: RAG Ingestion Pipeline"
echo "================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo "Please edit .env and add your OPENAI_API_KEY"
    exit 1
fi

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here" .env || grep -q "test_key_for_demo" .env; then
    echo "⚠️  Please set a valid OPENAI_API_KEY in .env file"
    echo ""
    echo "Get your API key from: https://platform.openai.com/api-keys"
    echo ""
    echo "Then edit .env and replace:"
    echo "  OPENAI_API_KEY=your_openai_api_key_here"
    echo "with:"
    echo "  OPENAI_API_KEY=sk-your-actual-key"
    exit 1
fi

echo "[Test 1] Checking dependencies..."
uv sync
echo ""

echo "[Test 2] Running RAG ingestion pipeline..."
uv run python main.py
EXIT_CODE=$?

echo ""
echo "================================"
echo "Test Results"
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Pipeline completed successfully!"
    echo ""
    echo "Expected output:"
    echo "  - Documents loaded from PDF, text, web"
    echo "  - Documents chunked"
    echo "  - Embeddings generated"
    echo "  - Chunks stored in vector database"
    echo "  - Similarity search demonstrated"
else
    echo "✗ Pipeline failed with exit code $EXIT_CODE"
    echo ""
    echo "Common issues:"
    echo "  - Check OPENAI_API_KEY in .env"
    echo "  - Ensure Documents/ folder has files"
    echo "  - Check internet connection for web loading"
fi
