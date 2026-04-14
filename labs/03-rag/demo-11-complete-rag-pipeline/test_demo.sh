#!/bin/bash

# Test Script for demo-10-complete-rag-pipeline
# Verifies project structure and dependencies

echo "======================================================================="
echo "TESTING: demo-10-complete-rag-pipeline"
echo "======================================================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to print warning
warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo ""
echo "1. Checking Project Structure..."
echo "-------------------------------------------------------------------"

# Check main.py exists
if [ -f "main.py" ]; then
    lines=$(wc -l < main.py)
    success "main.py exists ($lines lines)"
else
    error "main.py is missing!"
    exit 1
fi

# Check pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    success "pyproject.toml exists"
else
    error "pyproject.toml is missing!"
    exit 1
fi

# Check .env.example exists
if [ -f ".env.example" ]; then
    success ".env.example exists"
else
    error ".env.example is missing!"
    exit 1
fi

# Check .env exists
if [ -f ".env" ]; then
    success ".env exists"
else
    warning ".env not found (copy from .env.example)"
fi

# Check Documents folder exists
if [ -d "Documents" ]; then
    doc_count=$(find Documents -type f \( -name "*.pdf" -o -name "*.txt" \) | wc -l)
    success "Documents/ folder exists ($doc_count files)"
else
    warning "Documents/ folder not found (create and add documents)"
fi

# Check README exists
if [ -f "README.md" ]; then
    success "README.md exists"
else
    warning "README.md not found"
fi

# Check QUICKSTART exists
if [ -f "QUICKSTART.md" ]; then
    success "QUICKSTART.md exists"
else
    warning "QUICKSTART.md not found"
fi

echo ""
echo "2. Checking Environment Configuration..."
echo "-------------------------------------------------------------------"

if [ -f ".env" ]; then
    # Check OPENAI_API_KEY
    if grep -q "OPENAI_API_KEY=sk-" .env; then
        success "OPENAI_API_KEY is configured"
    else
        warning "OPENAI_API_KEY not set in .env"
    fi
    
    # Check VECTOR_DB
    if grep -q "VECTOR_DB=" .env; then
        vector_db=$(grep "VECTOR_DB=" .env | cut -d '=' -f2)
        success "VECTOR_DB is set to: $vector_db"
    else
        warning "VECTOR_DB not set in .env (will default to chromadb)"
    fi
else
    warning "Cannot check environment (no .env file)"
fi

echo ""
echo "3. Checking Python Environment..."
echo "-------------------------------------------------------------------"

# Check Python version
if command -v python &> /dev/null; then
    python_version=$(python --version 2>&1)
    success "Python available: $python_version"
else
    error "Python not found!"
    exit 1
fi

# Check if virtual environment is active
if [ -n "$VIRTUAL_ENV" ]; then
    success "Virtual environment active: $VIRTUAL_ENV"
else
    warning "No virtual environment active (run: source .venv/bin/activate)"
fi

echo ""
echo "4. Analyzing Code Structure..."
echo "-------------------------------------------------------------------"

if [ -f "main.py" ]; then
    # Count functions
    function_count=$(grep -c "^def " main.py)
    success "Functions defined: $function_count"
    
    # Check for key functions
    if grep -q "def load_documents" main.py; then
        success "load_documents() function found"
    fi
    
    if grep -q "def chunk_documents" main.py; then
        success "chunk_documents() function found"
    fi
    
    if grep -q "def store_chunks" main.py; then
        success "store_chunks() function found"
    fi
    
    if grep -q "def retrieve_documents" main.py; then
        success "retrieve_documents() function found"
    fi
    
    if grep -q "def generate_answer" main.py; then
        success "generate_answer() function found (Generation step!)"
    fi
    
    if grep -q "def run_rag_pipeline" main.py; then
        success "run_rag_pipeline() function found"
    fi
fi

echo ""
echo "5. Checking Dependencies..."
echo "-------------------------------------------------------------------"

# Try importing key libraries
python -c "import langchain" 2>/dev/null && success "langchain installed" || warning "langchain not installed"
python -c "import openai" 2>/dev/null && success "openai installed" || warning "openai not installed"
python -c "import chromadb" 2>/dev/null && success "chromadb installed" || warning "chromadb not installed"
python -c "from pinecone import Pinecone" 2>/dev/null && success "pinecone-client installed" || warning "pinecone-client not installed"
python -c "from langchain_openai import OpenAIEmbeddings" 2>/dev/null && success "langchain-openai installed" || warning "langchain-openai not installed"
python -c "from langchain_chroma import Chroma" 2>/dev/null && success "langchain-chroma installed" || warning "langchain-chroma not installed"

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
    
    # Calculate comment ratio
    comment_ratio=$(awk "BEGIN {printf \"%.1f\", ($comment_lines/$total_lines)*100}")
    success "Documentation: ${comment_ratio}%"
fi

echo ""
echo "======================================================================="
echo echo ""

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
    echo "Ready to run!"
    echo "  → uv run python main.py"
fi

echo "======================================================================="
