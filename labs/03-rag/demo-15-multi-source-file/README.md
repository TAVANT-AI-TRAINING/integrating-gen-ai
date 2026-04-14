# Multi-Source Document Loader Practice

A **Python application** that demonstrates how to load documents from **multiple sources** (PDF, text files, and web pages) using **LangChain Document Loaders** with a **service-based architecture**.  
This practice exercise combines all three loading techniques in a single, modular application with reusable service classes.

---

## Features

- **Multi-Source Loading** – Loads documents from PDF, text files, and web pages in one script
- **Service-Based Architecture** – Modular, reusable service classes for each document type
- **Unified Document Structure** – All sources produce standardized Document objects
- **Metadata Tracking** – Preserves source information for each document type
- **Batch Processing** – Handles multiple files and URLs efficiently
- **Modular Design** – Reusable service classes and utility functions
- **Comprehensive Inspection** – Displays detailed statistics and document breakdowns
- **Error Handling** – Gracefully handles failures from individual sources with safe loading methods
- **Type Safety** – Full type hints for better code quality and IDE support

---

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) or `pip` for dependency installation
- Active internet connection (for web loading example)
- No external API keys required

---

## Installation

### Linux / macOS

1. Clone or navigate to your project directory:

   ```bash
   cd demo-1-multi-source-file
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

### Windows

1. Clone or navigate to your project directory:

   ```cmd
   cd demo-1-multi-source-file
   ```

2. Install dependencies:
   ```cmd
   uv sync
   ```

---

## How It Works

1. The script initializes **service classes** for three different document sources:
   - `PDFLoaderService` for PDF files
   - `TextLoaderService` for text files
   - `WebLoaderService` for web pages
2. **PDF Loading**: Uses `PDFLoaderService` to load PDF files from `Documents/` directory (creates one Document per page).
3. **Text File Loading**: Uses `TextLoaderService` to batch process all `.txt` files in `Documents/` (creates one Document per file).
4. **Web Page Loading**: Uses `WebLoaderService` to fetch content from LangChain documentation (creates one Document per URL).
5. All Documents are collected into a single list with standardized structure.
6. The script displays detailed statistics, metadata, and content previews for each document.
7. Key takeaways about service behavior and metadata preservation are highlighted.

---

## Running the Application

### Linux / macOS

**Using UV:**

```bash
uv run python main/multi_source_loader.py
# or run as module
uv run python -m main.multi_source_loader
# or use the entry point
uv run main.py
```

The script will load documents from all three sources and display comprehensive statistics.

---

## Programmatic Usage

### Load Documents from All Sources

```python
from main.multi_source_loader import load_all_sources, inspect_documents

# Load documents from PDF, text files, and web
documents = load_all_sources()

# Inspect the loaded documents
inspect_documents(documents)
```

### Use Services Independently

```python
from pathlib import Path
from main.services import PDFLoaderService, TextLoaderService, WebLoaderService

# Initialize services
docs_dir = Path("Documents")
pdf_service = PDFLoaderService(docs_directory=docs_dir)
text_service = TextLoaderService(docs_directory=docs_dir)
web_service = WebLoaderService()

# Load PDF documents
pdf_docs = pdf_service.load_pdfs(Path("Documents/company_policy.pdf"))  # Single file
# Or load multiple specific files
multiple_pdfs = pdf_service.load_pdfs([Path("file1.pdf"), Path("file2.pdf")])

# Load text documents
text_files = list(Path("Documents").glob("*.txt"))
text_docs = text_service.load_text_files(text_files)  # Multiple files
# Or load a specific file
specific_text = text_service.load_text_files(Path("Documents/policy.txt"))
# Or load multiple files
multiple_texts = text_service.load_text_files([Path("file1.txt"), Path("file2.txt")])

# Load web documents
web_docs = web_service.load_web_pages("https://example.com")  # Single URL
# Or load multiple URLs
urls = ["https://example.com", "https://another-site.com"]
multiple_web_docs = web_service.load_web_pages(urls)
```

### Access Individual Documents

```python
from main.multi_source_loader import load_all_sources

# Load all documents
documents = load_all_sources()

# Access documents by type
pdf_docs = [d for d in documents if d.metadata['source'].endswith('.pdf')]
txt_docs = [d for d in documents if d.metadata['source'].endswith('.txt')]
web_docs = [d for d in documents if 'http' in d.metadata['source']]

print(f"PDF pages: {len(pdf_docs)}")
print(f"Text files: {len(txt_docs)}")
print(f"Web pages: {len(web_docs)}")
```

---

## Sample Output

When you run the script, you'll see output like:

```
LO1 PRACTICE: LOADING DOCUMENTS FROM MULTIPLE SOURCES

======================================================================
LOADING DOCUMENTS FROM MULTIPLE SOURCES
======================================================================

[1] Loading PDF document...
    Loaded 2 page(s) from PDF
    Source: Documents/company_policy.pdf

[2] Loading text files...
    Found 2 text file(s)
    Loaded: guidelines.txt
    Loaded: policy.txt

[3] Loading web page...
    URL: https://python.langchain.com/docs/introduction/
    Loaded 1 document(s) from web
    Content length: 15,234 characters

======================================================================
DOCUMENT INSPECTION
======================================================================

Total documents loaded: 5

Document breakdown by type:
  PDF documents: 2
  Text documents: 2
  Web documents: 1

======================================================================
DETAILED DOCUMENT INFORMATION
======================================================================

--- Document 1 ---
Source: Documents/company_policy.pdf
Page: 0
Content length: 456 characters
Word count: 78 words
Preview: Company Policy Document Page 1: Introduction This document outlines...

--- Document 2 ---
Source: Documents/company_policy.pdf
Page: 1
Content length: 512 characters
Word count: 85 words
Preview: Company Policy Document Page 2: Detailed Policies Remote Work Policy...

--- Document 3 ---
Source: Documents/guidelines.txt
Content length: 115 characters
Word count: 17 words
Preview: All code must pass automated tests before merging. Security reviews...

--- Document 4 ---
Source: Documents/policy.txt
Content length: 107 characters
Word count: 18 words
Preview: Remote work requires manager approval. Employees must be available...

--- Document 5 ---
Source: https://python.langchain.com/docs/introduction/
Content length: 15,234 characters
Word count: 2,456 words
Preview: Introduction | LangChain LangChain is a framework for developing...

======================================================================
METADATA STRUCTURE EXAMPLE
======================================================================

First document metadata:
  source: Documents/company_policy.pdf
  page: 0

======================================================================
KEY TAKEAWAYS
======================================================================
1. PDFLoaderService creates one Document per page
2. TextLoaderService creates one Document per file
3. WebLoaderService creates one Document per URL
4. All services preserve metadata (source, page numbers, etc.)
5. Document objects have a standardized structure:
   - page_content: The actual text content
   - metadata: Dictionary with source information
6. Services provide modular, reusable components for document loading
======================================================================
```

---

## Understanding Different Services

### Service Granularities

| Service           | Input       | Output      | Granularity |
| ----------------- | ----------- | ----------- | ----------- |
| PDFLoaderService  | 1 PDF file  | N Documents | 1 per page  |
| TextLoaderService | 1 text file | 1 Document  | 1 per file  |
| WebLoaderService  | 1 URL       | 1 Document  | 1 per URL   |

### Service Methods

Each service provides a specific method that handles single files or multiple files safely:

**PDFLoaderService:**

- `load_pdfs(source)` - Method for loading PDFs
  - Single file: `load_pdfs(Path("file.pdf"))`
  - Multiple files: `load_pdfs([Path("file1.pdf"), Path("file2.pdf")])`
  - Always safe: Returns empty list on error, never raises exceptions

**TextLoaderService:**

- `load_text_files(source, encoding)` - Method for loading text files
  - Single file: `load_text_files(Path("file.txt"))`
  - Multiple files: `load_text_files([Path("file1.txt"), Path("file2.txt")])`
  - Encoding support: Optional encoding parameter (default: utf-8)
  - Always safe: Returns empty list on error, never raises exceptions

**WebLoaderService:**

- `load_web_pages(source)` - Method for loading web pages
  - Single URL: `load_web_pages("https://example.com")`
  - Multiple URLs: `load_web_pages(["https://example.com", "https://another.com"])`
  - Always safe: Returns empty list on error, continues with other URLs

### Metadata Preservation

All services preserve source information:

**PDF Document:**

```python
{
    'source': 'Documents/company_policy.pdf',
    'page': 0  # Page number (0-indexed)
}
```

**Text Document:**

```python
{
    'source': 'Documents/policy.txt'
}
```

**Web Document:**

```python
{
    'source': 'https://python.langchain.com/docs/introduction/',
    'title': 'Introduction | LangChain',
    'language': 'en'
}
```

### Standardized Document Structure

All Document objects follow the same pattern:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content...",
    metadata={"source": "file.pdf", "page": 0}
)
```

This standardization enables:

- Uniform processing regardless of source
- Metadata-based filtering (e.g., "show only page 5")
- Source tracking for citations

---

## Project Structure

```
demo-1-multi-source-file/
│
├── main/
│   ├── __init__.py                      # Package initialization
│   ├── multi_source_loader.py           # Main script with multi-source loading functions
│   └── services/                        # Document loader services
│       ├── __init__.py                  # Services package initialization
│       ├── pdf_loader_service.py        # PDF document loading service
│       ├── text_loader_service.py       # Text file loading service
│       └── web_loader_service.py        # Web page loading service
├── pytest.ini                           # Pytest configuration
├── pyproject.toml                       # UV dependency configuration
├── uv.lock                              # Dependency lock file
├── main.py                              # Entry point script
├── README.md                            # Project documentation
└── .gitignore                           # Git ignore rules
```

---

## Troubleshooting

| Issue                    | Solution                                                            |
| ------------------------ | ------------------------------------------------------------------- |
| **PDF Not Found**        | Ensure PDF files exist in `Documents/` directory                    |
| **Web Loading Fails**    | Check internet connection; script continues with other sources      |
| **Missing Dependencies** | Run `uv sync` to install all required packages                      |
| **Import Errors**        | Ensure you're in the correct directory and UV environment is active |

---

## Extension Exercises

### 1. Add More Document Types

Try loading:

- Markdown files (`.md`)
- CSV files
- JSON files

### 2. Filter Documents

Modify the code to:

- Show only PDF documents from page 1
- Display only documents > 100 characters
- Group by source type

### 3. Custom Metadata

Add custom metadata:

```python
doc.metadata['category'] = 'policy'
doc.metadata['department'] = 'HR'
doc.metadata['last_updated'] = '2024-01-15'
```

### 4. Batch Processing

Load multiple PDFs using the service:

```python
from pathlib import Path
from main.services import PDFLoaderService

pdf_service = PDFLoaderService()
# Load all PDFs from directory
pdf_files = list(Path("Documents").glob("*.pdf"))
all_pdf_docs = pdf_service.load_pdfs(pdf_files)
# Or load specific files
specific_pdfs = pdf_service.load_pdfs([
    Path("file1.pdf"),
    Path("file2.pdf"),
    Path("file3.pdf")
])
```

Or load multiple web pages:

```python
from main.services import WebLoaderService

web_service = WebLoaderService()
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]
all_web_docs = web_service.load_web_pages(urls)
```

---

## Understanding the Code

### Service-Based Architecture

The project uses a **service-based architecture** where each document type has its own service class. This provides:

- **Modularity**: Each service is self-contained and reusable
- **Separation of Concerns**: Each service handles one document type
- **Error Handling**: Services provide safe loading methods that never raise exceptions
- **Type Safety**: Full type hints for better code quality

### Loading Pattern

Each service follows a consistent pattern with specific methods:

```python
from pathlib import Path
from main.services import PDFLoaderService, TextLoaderService, WebLoaderService

# 1. Initialize service
pdf_service = PDFLoaderService()
text_service = TextLoaderService()
web_service = WebLoaderService()

# 2. Load documents (always safe - returns empty list on error)
pdf_docs = pdf_service.load_pdfs(Path("company_policy.pdf"))  # Single file
text_docs = text_service.load_text_files(Path("file.txt"))  # Single file
web_docs = web_service.load_web_pages("https://example.com")  # Single URL

# 3. Access content and metadata
for doc in pdf_docs:
    print(doc.page_content)  # The text
    print(doc.metadata)      # Source info
```

### Error Handling

All loading methods are **always safe** - they never raise exceptions:

```python
# Always returns empty list if loading fails, never raises exceptions
pdf_docs = pdf_service.load_pdfs(Path("nonexistent.pdf"))  # Returns []
text_docs = text_service.load_text_files(Path("missing.txt"))  # Returns []
web_docs = web_service.load_web_pages("invalid-url")  # Returns []

# For multiple files/URLs, continues processing even if some fail
pdf_docs = pdf_service.load_pdfs([Path("file1.pdf"), Path("file2.pdf")])  # Loads what it can
web_docs = web_service.load_web_pages(["url1", "url2", "url3"])  # Continues on errors
```

This ensures the script continues even if one source fails, providing graceful error handling throughout.

---

## Questions to Consider

1. **Why does PDFLoaderService create multiple documents for one PDF?**  
   To preserve page-level granularity for better retrieval and more precise citations

2. **How is metadata useful in RAG systems?**  
   For filtering, citations, and source tracking. Metadata allows you to know exactly which page or file a piece of information came from.

3. **What's the advantage of standardized Document objects?**  
   Uniform processing regardless of source type. You can process PDFs, text files, and web pages the same way.

4. **Why use service classes instead of direct loader calls?**  
   Services provide:
   - Clear, specific method names (`load_pdfs()`, `load_text_files()`, `load_web_pages()`)
   - Always-safe error handling (never raises exceptions)
   - Reusable, testable components
   - Consistent interfaces across document types
   - Easier to extend and maintain

---
