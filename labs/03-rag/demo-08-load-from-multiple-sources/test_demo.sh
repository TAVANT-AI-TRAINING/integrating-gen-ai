#!/bin/bash
# Test script for demo-08-load-from-multiple-sources

echo "================================"
echo "Testing Demo 08"
echo "================================"
echo ""

# Test 1: Run the main application
echo "[Test 1] Running main application..."
uv run python main.py
EXIT_CODE=$?

echo ""
echo "================================"
echo "Test Results"
echo "================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Demo ran successfully!"
    echo ""
    echo "Expected:"
    echo "  - 5 PDF pages loaded"
    echo "  - 2 text files loaded"
    echo "  - 1 web page loaded"
    echo "  - Total: 8 documents"
else
    echo "✗ Demo failed with exit code $EXIT_CODE"
fi
