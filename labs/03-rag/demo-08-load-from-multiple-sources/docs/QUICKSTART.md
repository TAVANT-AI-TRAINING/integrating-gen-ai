# Quick Start Guide

## Installation & Run

```bash
# Navigate to project
cd demo-08-load-from-multiple-sources

# Install dependencies
uv sync

# Run the demo
uv run python main.py
```

## What It Does

Loads documents from three sources:

1. **PDF file** → Creates one Document per page
2. **Text files** → Creates one Document per file
3. **Web page** → Creates one Document per URL

All produce standardized `Document` objects.

## Core Concepts

### Document Structure

```python
Document(
    page_content="Text content here...",
    metadata={"source": "file.pdf", "page": 0}
)
```

### Three Loaders

```python
from langchain_community.document_loaders import (
    PyPDFLoader,      # For PDF files
    TextLoader,        # For text files
    WebBaseLoader      # For web pages
)

# PDF: One Document per page
pdf_loader = PyPDFLoader("file.pdf")
pdf_docs = pdf_loader.load()  # Returns [doc1, doc2, doc3, ...]

# Text: One Document per file
text_loader = TextLoader("file.txt")
text_docs = text_loader.load()  # Returns [doc]

# Web: One Document per URL
web_loader = WebBaseLoader("https://example.com")
web_docs = web_loader.load()  # Returns [doc]
```

### Loader Comparison

| Loader        | Granularity | Typical Use Case              |
| ------------- | ----------- | ----------------------------- |
| PyPDFLoader   | Page-level  | Large documents, manuals      |
| TextLoader    | File-level  | Config files, small documents |
| WebBaseLoader | URL-level   | Articles, documentation       |

## Quick Modifications

### Change PDF File

```python
# In main.py line 18
PDF_FILE = DOCS_DIR / "your_file.pdf"
```

### Change Web URL

```python
# In main.py line 19
WEB_URL = "https://your-website.com/"
```

### Load Multiple URLs

```python
# Replace load_web_documents() function
def load_web_documents() -> List[Document]:
    urls = ["https://example.com/page1", "https://example.com/page2"]
    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        all_docs.extend(loader.load())
    return all_docs
```

## Accessing Document Content

```python
# Load all documents
all_documents = load_pdf_documents() + load_text_documents() + load_web_documents()

# Access content
for doc in all_documents:
    print(doc.page_content)      # The text
    print(doc.metadata['source']) # Where it came from

    # PDF-specific metadata
    if 'page' in doc.metadata:
        print(f"Page: {doc.metadata['page']}")
```

## Filtering Documents

```python
# Only PDFs
pdf_docs = [d for d in all_documents if d.metadata['source'].endswith('.pdf')]

# Only first pages
first_pages = [d for d in all_documents if d.metadata.get('page') == 0]

# Only long documents (>1000 characters)
long_docs = [d for d in all_documents if len(d.page_content) > 1000]

# By source
from_web = [d for d in all_documents if 'http' in d.metadata['source']]
```

## Typical Output

```
======================================================================
DEMO 08: LOAD DOCUMENTS FROM MULTIPLE SOURCES
======================================================================

[1] Loading PDF document...
    ✓ Loaded 5 page(s)

[2] Loading text files...
    ✓ Loaded: guidelines.txt
    ✓ Loaded: policy.txt

[3] Loading web page...
    ✓ Loaded 1 document(s)

======================================================================
DOCUMENT INSPECTION
======================================================================

✓ Total documents loaded: 8
  • PDF pages: 5
  • Text files: 2
  • Web pages: 1
```

## Error Handling

All loaders handle errors gracefully:

```python
# If PDF doesn't exist
pdf_docs = load_pdf_documents()  # Returns [] (empty list)

# If web page unreachable
web_docs = load_web_documents()  # Returns [] (empty list)

# Script continues with available sources
```

## Key Takeaways

✓ Different sources → Different granularities  
✓ All loaders → Same Document structure  
✓ Metadata → Source tracking & filtering  
✓ Page-level → Better for large documents  
✓ File-level → Fine for small documents  
✓ Standardization → Uniform processing

## Next Steps

1. Run the demo: `uv run python main.py`
2. Try loading your own PDF
3. Change the web URL
4. Filter documents by type
5. Inspect metadata structure

See `README.md` for detailed documentation.
