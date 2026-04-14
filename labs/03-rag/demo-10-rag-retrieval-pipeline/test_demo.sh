#!/bin/bash

# Test Script for demo-10-rag-retrieval-pipeline
# Verifies project structure and retrieval-focused setup

echo "======================================================================="
echo "TESTING: demo-10-rag-retrieval-pipeline"
echo "======================================================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo ""
echo "1. Checking Project Structure..."
echo "-------------------------------------------------------------------"

# Check main.py
if [ -f "main.py" ]; then
    lines=$(wc -l < main.py)
    success "main.py exists ($lines lines)"
else
    error "main.py is missing!"
    exit 1
fi

# Check for retrieval-focused code
if grep -q "def similarity_search_basic" main.py; then
    success "similarity_search_basic() found"
fi

if grep -q "def similarity_search_with_score" main.py; then
    success "similarity_search_with_score() found"
fi

if grep -q "def max_marginal_relevance_search" main.py; then
    success "max_marginal_relevance_search() found"
fi

if grep -q "def analyze_retrieval_quality" main.py; then
    success "analyze_retrieval_quality() found"
fi

# Check that generation functions are NOT present
if ! grep -q "def generate_answer" main.py; then
    success "No generate_answer() - retrieval-only confirmed ✓"
else
    warning "Found generate_answer() - should be retrieval-only"
fi

if ! grep -q "ChatOpenAI" main.py; then
    success "No ChatOpenAI import - retrieval-only confirmed ✓"
else
    warning "Found ChatOpenAI - should be retrieval-only"
fi

# Check pyproject.toml
if [ -f "pyproject.toml" ]; then
    success "pyproject.toml exists"
else
    error "pyproject.toml is missing!"
    exit 1
fi

# Check .env.example
if [ -f ".env.example" ]; then
    success ".env.example exists"
else
    error ".env.example is missing!"
    exit 1
fi

# Check .env
if [ -f ".env" ]; then
    success ".env exists"
else
    warning ".env not found (copy from .env.example)"
fi

# Check Documents folder
if [ -d "Documents" ]; then
    doc_count=$(find Documents -type f \( -name "*.pdf" -o -name "*.txt" \) | wc -l)
    success "Documents/ folder exists ($doc_count files)"
else
    warning "Documents/ folder not found"
fi

# Check README
if [ -f "README.md" ]; then
    success "README.md exists"
else
    warning "README.md not found"
fi

# Check QUICKSTART
if [ -f "QUICKSTART.md" ]; then
    success "QUICKSTART.md exists"
else
    warning "QUICKSTART.md not found"
fi

echo ""
echo "2. Checking Configuration..."
echo "-------------------------------------------------------------------"

if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY=sk-" .env; then
        success "OPENAI_API_KEY is configured"
    else
        warning "OPENAI_API_KEY not set in .env"
    fi
    
    if grep -q "VECTOR_DB=" .env; then
        vector_db=$(grep "VECTOR_DB=" .env | cut -d '=' -f2)
        success "VECTOR_DB is set to: $vector_db"
    else
        warning "VECTOR_DB not set (will default to chromadb)"
    fi
else
    warning "Cannot check configuration (no .env file)"
fi

echo ""
echo "3. Analyzing Code Focus..."
echo "-------------------------------------------------------------------"

if [ -f "main.py" ]; then
    # Count retrieval functions
    retrieval_funcs=$(grep -c "def.*search\|def.*retriev" main.py)
    success "Retrieval functions: $retrieval_funcs"
    
    # Check for demonstration scenarios
    if grep -q "demonstrate_retrieval_scenarios" main.py; then
        success "demonstrate_retrieval_scenarios() found"
    fi
    
    # Count scenarios
    scenarios=$(grep -c "\[Scenario" main.py)
    success "Scenario demonstrations: $scenarios"
    
    # Verify no LLM generation
    if ! grep -q "llm.invoke\|ChatOpenAI" main.py; then
        success "Confirmed: No LLM generation (retrieval-only) ✓"
    else
        warning "May contain LLM generation code"
    fi
fi

echo ""
echo "4. Checking Python Environment..."
echo "-------------------------------------------------------------------"

if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    success "Python available: $python_version"
else
    error "Python not found!"
    exit 1
fi

if [ -n "$VIRTUAL_ENV" ]; then
    success "Virtual environment active: $VIRTUAL_ENV"
else
    warning "No virtual environment active"
fi

echo ""
echo "5. Checking Dependencies..."
echo "-------------------------------------------------------------------"

python -c "import langchain" 2>/dev/null && success "langchain installed" || warning "langchain not installed"
python -c "import openai" 2>/dev/null && success "openai installed" || warning "openai not installed"
python -c "import chromadb" 2>/dev/null && success "chromadb installed" || warning "chromadb not installed"
python -c "from langchain_openai import OpenAIEmbeddings" 2>/dev/null && success "langchain-openai installed" || warning "langchain-openai not installed"

echo ""
echo "6. Code Metrics..."
echo "-------------------------------------------------------------------"

if [ -f "main.py" ]; then
    total_lines=$(wc -l < main.py)
    code_lines=$(grep -v "^\s*#" main.py | grep -v "^\s*$" | wc -l)
    comment_lines=$(grep "^\s*#" main.py | wc -l)
    
    echo "Total lines: $total_lines"
    success "Code lines: $code_lines"
    success "Comment lines: $comment_lines"
    
    # Expected range for retrieval-only demo
    if [ $total_lines -ge 400 ] && [ $total_lines -le 500 ]; then
        success "Line count appropriate for retrieval-focused demo (400-500)"
    fi
fi

echo ""
echo "7. Verifying Demo Focus..."
echo "-------------------------------------------------------------------"

if [ -f "main.py" ]; then
    # Check retrieval strategies
    if grep -q "similarity_search_basic" main.py; then
        success "✓ Basic similarity search"
    fi
    
    if grep -q "similarity_search_with_score" main.py; then
        success "✓ Similarity search with scores"
    fi
    
    if grep -q "max_marginal_relevance" main.py; then
        success "✓ MMR search"
    fi
    
    if grep -q "retriever_interface" main.py; then
        success "✓ Retriever interface"
    fi
    
    if grep -q "analyze_retrieval_quality" main.py; then
        success "✓ Quality analysis"
    fi
    
    if grep -q "display_document_details" main.py; then
        success "✓ Document inspection"
    fi
fi

echo ""
echo "======================================================================="
echo ""

# Summary
errors=0
warnings=0

if [ ! -f "main.py" ] || [ ! -f "pyproject.toml" ]; then
    ((errors++))
fi

if [ ! -f ".env" ] || [ ! -d "Documents" ]; then
    ((warnings++))
fi

if [ $errors -gt 0 ]; then
    error "Test FAILED with $errors error(s)"
    exit 1
elif [ $warnings -gt 0 ]; then
    warning "Test PASSED with $warnings warning(s)"
    echo ""
    echo "Next steps:"
    [ ! -f ".env" ] && echo "  1. Copy .env.example to .env and configure"
    [ ! -d "Documents" ] && echo "  2. Create Documents/ folder and add documents"
    echo "  3. Run: uv run python main.py"
else
    success "All tests PASSED! ✅"
    echo ""
    echo "Demo Focus: RAG RETRIEVAL ONLY"
    echo "  • NO LLM generation"
    echo "  • Demonstrates 6 retrieval strategies"
    echo "  • Quality analysis and comparison"
    echo ""
    echo "Ready to run!"
    echo "  → uv run python main.py"
fi

echo "======================================================================="
