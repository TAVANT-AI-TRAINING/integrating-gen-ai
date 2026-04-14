# Demo 08: Load Documents from Multiple Sources

A simple demonstration of loading documents from different sources using **LangChain document loaders**.

## What You'll Learn

- Load PDF files (creates one Document per page)
- Load text files (creates one Document per file)
- Load web pages (creates one Document per URL)
- Understand standardized Document structure
- Inspect document metadata from different sources

## Features

- **Single-file implementation** - Easy to understand and modify
- **Three document loaders** - PDF, text, and web
- **Standardized output** - All loaders produce Document objects
- **Metadata preservation** - Source information tracked automatically
- **Error handling** - Graceful failures with clear messages
- **No API keys required** - Works without external services

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- Internet connection (for web loading example)

## Installation

```bash
# Navigate to project directory
cd demo-08-load-from-multiple-sources

# Install dependencies
uv sync
```

## Quick Start

```bash
# Run the demo
uv run python main.py
```

The script will:

1. Load a PDF file (page by page)
2. Load all .txt files from Documents/
3. Load content from a web page
4. Display statistics and document details

## Expected Output

```
======================================================================
DEMO 08: LOAD DOCUMENTS FROM MULTIPLE SOURCES
======================================================================

[1] Loading PDF document...
    File: Documents/company_policy.pdf
    ✓ Loaded 5 page(s)

[2] Loading text files...
    Directory: Documents
    Found 2 text file(s)
    ✓ Loaded: guidelines.txt
    ✓ Loaded: policy.txt

[3] Loading web page...
    URL: https://www.python.org/
    ✓ Loaded 1 document(s)
    ✓ Content length: 2,246 characters

======================================================================
DOCUMENT INSPECTION
======================================================================

✓ Total documents loaded: 8

Breakdown by type:
  • PDF pages: 5
  • Text files: 2
  • Web pages: 1

======================================================================
DOCUMENT DETAILS (First 3)
======================================================================

--- Document 1 ---
Source: Documents/company_policy.pdf
Page: 0
Length: 1,051 characters, 144 words
Preview: Company Policy Manual  Document Version: 1.0 Prepared By: Human...

--- Document 2 ---
Source: Documents/company_policy.pdf
Page: 1
Length: 1,171 characters, 175 words
Preview: Policy  • Employees must act with integrity and professionalism...

--- Document 3 ---
Source: Documents/company_policy.pdf
Page: 2
Length: 983 characters, 142 words
Preview: • Public Holidays: As per the official holiday calendar...
```

## Understanding the Code

### Document Loading Pattern

Each loader follows the same pattern:

```python
from langchain_community.document_loaders import PyPDFLoader

# 1. Create loader instance
loader = PyPDFLoader("file.pdf")

# 2. Load documents
documents = loader.load()

# 3. Access content and metadata
for doc in documents:
    print(doc.page_content)  # The text content
    print(doc.metadata)      # Source information
```

### Document Structure

All Document objects have the same structure:

```python
Document(
    page_content="The actual text content...",
    metadata={"source": "Documents/file.pdf", "page": 0}
)
```

### Loader Granularities

| Loader        | Input       | Output      | Granularity |
| ------------- | ----------- | ----------- | ----------- |
| PyPDFLoader   | 1 PDF file  | N Documents | 1 per page  |
| TextLoader    | 1 text file | 1 Document  | 1 per file  |
| WebBaseLoader | 1 URL       | 1 Document  | 1 per URL   |

### Metadata by Source Type

**PDF Document:**

```python
{
    'source': 'Documents/company_policy.pdf',
    'page': 0,
    'total_pages': 5
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
    'source': 'https://www.python.org/',
    'title': 'Welcome to Python.org',
    'language': 'en'
}
```

## Modifying the Demo

### Load Different PDF

```python
# In main.py, change PDF_FILE
PDF_FILE = DOCS_DIR / "your_file.pdf"
```

### Load Different Web Page

```python
# In main.py, change WEB_URL
WEB_URL = "https://your-website.com/"
```

### Load Multiple Web Pages

```python
def load_web_documents() -> List[Document]:
    urls = [
        "https://www.python.org/",
        "https://docs.python.org/",
        "https://pypi.org/"
    ]

    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs
```

### Filter by Source Type

```python
# After loading all documents
all_documents = pdf_docs + text_docs + web_docs

# Get only PDF documents
pdf_only = [d for d in all_documents if d.metadata['source'].endswith('.pdf')]

# Get only first page of PDFs
first_pages = [d for d in all_documents if d.metadata.get('page') == 0]

# Get documents with more than 100 words
long_docs = [d for d in all_documents if len(d.page_content.split()) > 100]
```

## Project Structure

```
demo-08-load-from-multiple-sources/
├── Documents/               # Sample documents
│   ├── company_policy.pdf   # Sample PDF
│   ├── guidelines.txt       # Sample text file
│   └── policy.txt           # Sample text file
├── main.py                  # Single-file demo (170 lines)
├── pyproject.toml           # Dependency configuration
├── .python-version          # Python version specification
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── QUICKSTART.md            # Quick reference guide
```

## Why This Matters for RAG

Loading documents from multiple sources is essential for RAG systems because:

1. **Real-world data is diverse** - PDFs, text files, web pages, databases
2. **Granularity differs** - Need to understand page-level vs file-level chunks
3. **Metadata enables filtering** - "Show only page 5 from the policy document"
4. **Standardization simplifies processing** - Same Document structure for all sources
5. **Source tracking enables citations** - Know where each piece of information came from

## Common Issues

| Issue             | Solution                                                     |
| ----------------- | ------------------------------------------------------------ |
| PDF not found     | Ensure `Documents/company_policy.pdf` exists                 |
| Web loading fails | Check internet connection; the demo continues gracefully     |
| Import errors     | Run `uv sync` to install dependencies                        |
| Encoding errors   | TextLoader uses UTF-8 by default; specify encoding if needed |

## Next Steps

- **Demo 09**: Document splitting and chunking strategies
- **Demo 10**: Creating embeddings from loaded documents
- **Demo 11**: Storing document chunks in vector databases

## Questions to Consider

1. **Why does PyPDFLoader create multiple documents for one PDF?**  
   → Preserves page-level granularity for precise retrieval and citations

2. **When would you need page-level vs file-level documents?**  
   → Large documents benefit from page-level; small files are fine as single documents

3. **How is metadata useful in RAG systems?**  
   → Enables filtering, source tracking, and citation generation

4. **What happens if one source fails to load?**  
   → The demo continues with other sources (graceful error handling)

---

**Compare with**: See `demo-08-multi-source-file/` for a service-based architecture (production pattern).
