# Module-6: Multi-modal RAG & Enterprise Knowledge Assistants

A comprehensive guide to building multi-modal RAG systems that process text, PDFs, images, and tables — covering OCR and document intelligence pipelines, multi-source enterprise search, five production enterprise assistant use cases, personalization, memory, and a complete enterprise assistant solution blueprint.

---

## Table of Contents

1. [Why Multi-modal RAG Matters](#1-why-multi-modal-rag-matters)
2. [Text + PDF + Image + Table Retrieval](#2-text--pdf--image--table-retrieval)
3. [OCR and Document Intelligence Pipelines](#3-ocr-and-document-intelligence-pipelines)
4. [Multi-Source Enterprise Search Assistants](#4-multi-source-enterprise-search-assistants)
5. [Enterprise Assistant Use Cases](#5-enterprise-assistant-use-cases)
6. [Personalization and Memory Concepts](#6-personalization-and-memory-concepts)
7. [Lab: Enterprise Assistant Solution Blueprint](#7-lab-enterprise-assistant-solution-blueprint)

---

## 1. Why Multi-modal RAG Matters

### The Enterprise Document Reality

Most enterprise knowledge is not plain text. A survey of typical enterprise content:

```
ENTERPRISE CONTENT MIX:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Content Type         % of Knowledge Base   Challenge            │
  ├─────────────────────────────────────────────────────────────────┤
  │  Plain text / Markdown     ~15%             Handled by Naive RAG │
  │  PDFs with formatting      ~35%             Layout breaks chunks  │
  │  Tables and spreadsheets   ~20%             Numbers lost in text  │
  │  Images, diagrams, charts  ~15%             Invisible to embedding│
  │  Scanned documents / forms ~10%             No machine text at all│
  │  Presentations (PPTX)       ~5%             Slides lose context   │
  └─────────────────────────────────────────────────────────────────┘

  A text-only RAG system is blind to 50% of enterprise knowledge.
```

### Multi-modal RAG Architecture

```
INDEXING (offline):

  PDF / PPTX / HTML / Scanned Doc
         │
         ▼
  ┌─────────────────────────────────────────────────┐
  │  Document Intelligence Layer                     │
  │  (unstructured.io / Azure DI / AWS Textract)    │
  │                                                  │
  │  ├── Text extraction → RecursiveCharSplitter    │
  │  ├── Table extraction → structured text repr.   │
  │  ├── Image extraction → base64 / file store     │
  │  └── OCR (scanned pages) → clean text           │
  └─────────────────────────────────────────────────┘
         │
         ├── Text chunks  → text embedding → vector store
         ├── Table repr.  → text embedding → vector store
         └── Images       → vision LLM captioning
                             OR CLIP image embedding
                             → vector store

RETRIEVAL (online):

  User Query (text OR image)
         │
         ▼
  Embed query → search all modality stores
         │
         ├── text chunks matched
         ├── table matches
         └── image matches (by caption or CLIP embedding)
         │
         ▼
  Assemble multi-modal context
         │
         ▼
  Multi-modal LLM (GPT-4o / Claude 3.x)
         │
         ▼
  Answer (reasoning over text + tables + images)
```

---

## 2. Text + PDF + Image + Table Retrieval

### PDF Parsing with Unstructured.io

Unstructured.io provides layout-aware PDF parsing that preserves document structure (headings, lists, tables, images) rather than treating the PDF as a flat text stream.

```bash
pip install unstructured[pdf,docx,pptx,xlsx] unstructured-inference pillow
```

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Title, NarrativeText, Table, Image, ListItem, Header, Footer
)
from pathlib import Path
import base64

def parse_pdf_multimodal(filepath: str, output_dir: str = "./extracted") -> dict:
    """
    Parse a PDF into typed elements: text, tables, images.
    Returns a dict with three element lists ready for downstream processing.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    elements = partition_pdf(
        filename=filepath,
        strategy="hi_res",                  # layout-aware; use "fast" for speed
        infer_table_structure=True,          # extract tables with HTML structure
        extract_images_in_pdf=True,          # save images to output_dir
        extract_image_block_output_dir=output_dir,
        extract_image_block_types=["Image", "Table"],
        chunking_strategy="by_title",        # chunk by document section
        max_characters=2000,
        new_after_n_chars=1500,
    )
    
    result = {"text": [], "tables": [], "images": [], "metadata": {"source": filepath}}
    
    for el in elements:
        if isinstance(el, (NarrativeText, ListItem)) and el.text.strip():
            result["text"].append({
                "content":  el.text,
                "category": "text",
                "metadata": {
                    "source":      filepath,
                    "page_number": el.metadata.page_number if el.metadata else None,
                    "element_id":  el.id,
                },
            })
        
        elif isinstance(el, Table):
            result["tables"].append({
                "content":   el.text,            # plain text representation
                "html":      el.metadata.text_as_html if el.metadata else "",
                "category":  "table",
                "metadata": {
                    "source":      filepath,
                    "page_number": el.metadata.page_number if el.metadata else None,
                },
            })
        
        elif isinstance(el, Image):
            img_path = el.metadata.image_path if el.metadata else None
            if img_path and Path(img_path).exists():
                result["images"].append({
                    "path":      img_path,
                    "category":  "image",
                    "metadata": {
                        "source":      filepath,
                        "page_number": el.metadata.page_number if el.metadata else None,
                    },
                })
    
    print(f"Parsed '{Path(filepath).name}': "
          f"{len(result['text'])} text, "
          f"{len(result['tables'])} tables, "
          f"{len(result['images'])} images")
    return result
```

### Table Extraction and Representation

Tables must be converted to a text format that preserves their structure for embedding and LLM reasoning.

```python
import pandas as pd
from bs4 import BeautifulSoup
from langchain.schema import Document

def html_table_to_text(html: str) -> str:
    """Convert HTML table to a readable text representation."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return html
    
    rows = table.find_all("tr")
    lines = []
    
    for i, row in enumerate(rows):
        cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
        if i == 0:
            lines.append(" | ".join(cells))
            lines.append("-" * (sum(len(c) + 3 for c in cells)))
        else:
            lines.append(" | ".join(cells))
    
    return "\n".join(lines)

def table_to_document(table_element: dict, doc_context: str = "") -> Document:
    """
    Convert a parsed table into a Document for embedding.
    Prefix with context (surrounding section heading) for better retrieval.
    """
    text_repr = html_table_to_text(table_element.get("html", ""))
    if not text_repr.strip():
        text_repr = table_element.get("content", "")
    
    full_text = (
        f"TABLE from '{table_element['metadata']['source']}':\n\n"
        + (f"Context: {doc_context}\n\n" if doc_context else "")
        + text_repr
    )
    
    return Document(
        page_content=full_text,
        metadata={**table_element["metadata"], "element_type": "table"},
    )

# Example output:
"""
TABLE from 'q3_report.pdf':

Context: Quarterly Revenue Breakdown by Region

Region    | Q3 2024  | Q3 2023  | YoY Change
---------------------------------------
EMEA      | £4.2M    | £3.8M    | +10.5%
APAC      | £2.1M    | £1.9M    | +10.5%
Americas  | £5.6M    | £4.7M    | +19.1%
Total     | £11.9M   | £10.4M   | +14.4%
"""
```

### Image Processing — Caption-Based Retrieval

Generate text captions for images using a vision LLM, then embed the captions for semantic retrieval.

```python
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.schema import Document

vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)

IMAGE_CAPTION_PROMPT = """Analyse this image from a business document and provide:
1. A detailed description of what the image shows (chart type, key data points,
   diagram components, or photograph content).
2. Key numerical values or relationships visible in the image.
3. The likely business context or purpose of this image.

Be specific and factual. If it is a chart, report the axis labels and key values.
If it is a diagram, describe the components and their relationships."""

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def generate_image_caption(image_path: str, source_doc: str = "") -> str:
    """Use GPT-4o Vision to generate a detailed caption for retrieval."""
    img_b64 = encode_image_base64(image_path)
    
    # Detect image format from extension
    ext     = image_path.rsplit(".", 1)[-1].lower()
    mime    = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
               "gif": "gif",  "webp": "webp"}.get(ext, "jpeg")
    
    message = HumanMessage(content=[
        {"type": "image_url",
         "image_url": {"url": f"data:image/{mime};base64,{img_b64}", "detail": "high"}},
        {"type": "text", "text": IMAGE_CAPTION_PROMPT},
    ])
    
    caption = vision_llm.invoke([message]).content
    return caption

def image_to_document(image_element: dict) -> Document:
    """Convert image to a searchable Document via LLM-generated caption."""
    img_path = image_element["path"]
    source   = image_element["metadata"].get("source", "")
    page     = image_element["metadata"].get("page_number", "")
    
    try:
        caption = generate_image_caption(img_path, source)
    except Exception as e:
        caption = f"[Image from {source}, page {page}. Caption generation failed: {e}]"
    
    return Document(
        page_content=f"IMAGE from '{source}' (page {page}):\n\n{caption}",
        metadata={
            **image_element["metadata"],
            "element_type":  "image",
            "image_path":    img_path,
            "has_caption":   True,
        },
    )
```

### CLIP-Based Image Embedding (No Vision LLM Required)

For high-volume image indexing where LLM captioning is too expensive, use CLIP to embed images directly.

```python
# pip install transformers torch pillow

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np

class CLIPImageRetriever:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model     = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.image_index: list[dict] = []   # [{embedding, path, metadata}]
    
    def index_image(self, image_path: str, metadata: dict):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # normalise
        self.image_index.append({
            "embedding": emb.squeeze().numpy(),
            "path":      image_path,
            "metadata":  metadata,
        })
    
    def search(self, text_query: str, k: int = 3) -> list[dict]:
        """Find images most similar to a text query."""
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
        with torch.no_grad():
            query_emb = self.model.get_text_features(**inputs)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
        
        query_np = query_emb.squeeze().numpy()
        scores   = [
            np.dot(query_np, item["embedding"])
            for item in self.image_index
        ]
        top_k    = sorted(
            zip(scores, self.image_index),
            reverse=True
        )[:k]
        return [{"score": s, **item} for s, item in top_k]

clip_retriever = CLIPImageRetriever()

# Index all extracted images
for img in extracted["images"]:
    clip_retriever.index_image(img["path"], img["metadata"])

# Search: "bar chart showing revenue by region"
results = clip_retriever.search("bar chart revenue by region", k=3)
```

### Multi-Modal Vector Store (Unified Index)

Store all modalities — text, table captions, image captions — in a single vector store with `element_type` metadata for filtering.

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="multimodal_enterprise",
    persist_directory="./chroma_multimodal",
    embedding_function=embeddings,
)

def index_multimodal_document(parsed: dict) -> dict:
    """Index all modalities from a parsed document into a unified vector store."""
    all_docs = []
    
    # Text chunks
    for el in parsed["text"]:
        all_docs.append(Document(
            page_content=el["content"],
            metadata={**el["metadata"], "element_type": "text"},
        ))
    
    # Tables (as text representations)
    for el in parsed["tables"]:
        all_docs.append(table_to_document(el))
    
    # Images (as LLM-generated captions)
    for el in parsed["images"]:
        all_docs.append(image_to_document(el))
    
    vectorstore.add_documents(all_docs)
    
    return {
        "text_indexed":   len(parsed["text"]),
        "tables_indexed": len(parsed["tables"]),
        "images_indexed": len(parsed["images"]),
        "total":          len(all_docs),
    }

# Retrieval — unified search across all modalities
def multimodal_retrieve(query: str, k: int = 6,
                         filter_type: str = None) -> list[Document]:
    filter_dict = {"element_type": filter_type} if filter_type else None
    return vectorstore.similarity_search(
        query, k=k,
        filter=filter_dict,
    )

# Examples
text_results  = multimodal_retrieve("refund policy", filter_type="text")
table_results = multimodal_retrieve("Q3 revenue by region", filter_type="table")
image_results = multimodal_retrieve("org chart showing department structure", filter_type="image")
all_results   = multimodal_retrieve("Q3 financial performance")  # all modalities
```

### Multi-Modal Answer Generation

Pass both text and image content to a vision-capable LLM.

```python
def multimodal_rag_answer(query: str, k: int = 6) -> str:
    docs = multimodal_retrieve(query, k=k)
    
    # Separate by type
    text_docs  = [d for d in docs if d.metadata.get("element_type") == "text"]
    table_docs = [d for d in docs if d.metadata.get("element_type") == "table"]
    image_docs = [d for d in docs if d.metadata.get("element_type") == "image"]
    
    # Build text context
    text_context = "\n\n---\n\n".join(
        f"[Text, Source: {d.metadata.get('source','Doc')}]\n{d.page_content}"
        for d in text_docs + table_docs
    )
    
    # Build message content list (text + inline images for vision LLM)
    content: list = [
        {"type": "text",
         "text": f"Answer the question using the provided documents and images.\n\n"
                 f"Text and Table Context:\n{text_context}\n\n"
                 f"Question: {query}\n\nAnswer:"},
    ]
    
    # Attach actual images that were retrieved
    for img_doc in image_docs:
        img_path = img_doc.metadata.get("image_path")
        if img_path and Path(img_path).exists():
            img_b64 = encode_image_base64(img_path)
            ext     = img_path.rsplit(".", 1)[-1].lower()
            mime    = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext, "jpeg")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{mime};base64,{img_b64}", "detail": "high"},
            })
    
    message = HumanMessage(content=content)
    return vision_llm.invoke([message]).content
```

---

## 3. OCR and Document Intelligence Pipelines

### When OCR Is Needed

```
OCR IS REQUIRED WHEN:
  ├── Scanned PDFs (no embedded text layer)
  ├── Photographs of documents or whiteboards
  ├── Legacy forms and printed records
  ├── Handwritten documents
  └── Image-heavy PDFs where text is embedded in images

INDICATORS a PDF needs OCR:
  - pdfplumber extracts 0 characters from a page
  - Text extracted is garbled / single characters
  - Document was "printed to PDF" from a scan
```

### OCR Pipeline with Tesseract

```python
# pip install pytesseract pillow pdf2image
# brew install tesseract  (macOS) / apt-get install tesseract-ocr (Linux)

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from langchain.schema import Document
import re

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """Enhance image quality before OCR to improve accuracy."""
    # Convert to grayscale
    img = image.convert("L")
    # Increase contrast
    img = ImageEnhance.Contrast(img).enhance(2.0)
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    # Binarise (threshold)
    img = img.point(lambda x: 0 if x < 128 else 255, "1")
    return img

def ocr_pdf(filepath: str, dpi: int = 300, lang: str = "eng") -> list[Document]:
    """Convert a scanned PDF to searchable text via OCR."""
    pages = convert_from_path(filepath, dpi=dpi)
    
    documents = []
    for page_num, page_img in enumerate(pages, start=1):
        enhanced     = preprocess_image_for_ocr(page_img)
        raw_text     = pytesseract.image_to_string(enhanced, lang=lang,
                           config="--psm 6 --oem 3")  # psm 6=block, oem 3=LSTM
        cleaned_text = clean_ocr_text(raw_text)
        
        if len(cleaned_text.strip()) > 50:  # skip mostly-empty pages
            documents.append(Document(
                page_content=cleaned_text,
                metadata={
                    "source":       filepath,
                    "page_number":  page_num,
                    "element_type": "text",
                    "ocr":          True,
                },
            ))
        
        print(f"  OCR page {page_num}/{len(pages)}: {len(cleaned_text)} chars extracted")
    
    return documents

def clean_ocr_text(text: str) -> str:
    """Post-process OCR output to fix common artefacts."""
    # Remove form feed characters
    text = text.replace("\f", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix common OCR substitutions
    text = text.replace("|", "I").replace("0O", "00")
    # Remove lone characters on a line (usually noise)
    text = re.sub(r"^\s*[^a-zA-Z0-9]{1,2}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
```

### Azure Document Intelligence (Production OCR)

For production use, Azure Document Intelligence (formerly Form Recognizer) provides layout-aware OCR with table extraction, key-value pairs, and form understanding.

```python
# pip install azure-ai-documentintelligence

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langchain.schema import Document
import os

di_client = DocumentIntelligenceClient(
    endpoint   = os.environ["AZURE_DI_ENDPOINT"],
    credential = AzureKeyCredential(os.environ["AZURE_DI_KEY"]),
)

def azure_document_intelligence_parse(filepath: str) -> list[Document]:
    """
    Use Azure DI for layout-aware OCR with table and structure extraction.
    Handles scanned PDFs, images, and complex layouts.
    """
    with open(filepath, "rb") as f:
        poller = di_client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=f,
            content_type="application/octet-stream",
        )
    result = poller.result()
    
    documents = []
    
    # Extract paragraphs (with page and bounding box info)
    for para in result.paragraphs or []:
        if para.content.strip():
            page_num = para.bounding_regions[0].page_number if para.bounding_regions else 0
            documents.append(Document(
                page_content=para.content,
                metadata={
                    "source":       filepath,
                    "page_number":  page_num,
                    "element_type": "text",
                    "role":         para.role or "paragraph",  # title, sectionHeading, etc.
                },
            ))
    
    # Extract tables as structured text
    for i, table in enumerate(result.tables or []):
        table_text = render_azure_table(table)
        page_num   = table.bounding_regions[0].page_number if table.bounding_regions else 0
        documents.append(Document(
            page_content=f"TABLE {i+1} from '{filepath}' (page {page_num}):\n{table_text}",
            metadata={
                "source":       filepath,
                "page_number":  page_num,
                "element_type": "table",
            },
        ))
    
    return documents

def render_azure_table(table) -> str:
    """Render an Azure DI table object as pipe-delimited text."""
    # Build 2D grid
    grid = [[""]*table.column_count for _ in range(table.row_count)]
    for cell in table.cells:
        grid[cell.row_index][cell.column_index] = cell.content or ""
    
    lines = []
    for r, row in enumerate(grid):
        lines.append(" | ".join(row))
        if r == 0:
            lines.append("-" * (sum(len(c) + 3 for c in row)))
    return "\n".join(lines)
```

### AWS Textract Integration

```python
# pip install boto3

import boto3
import json

textract = boto3.client("textract", region_name="eu-west-1")

def textract_parse_document(s3_bucket: str, s3_key: str) -> list[Document]:
    """Parse a document stored in S3 using AWS Textract."""
    response = textract.analyze_document(
        Document={"S3Object": {"Bucket": s3_bucket, "Name": s3_key}},
        FeatureTypes=["TABLES", "FORMS", "LAYOUT"],
    )
    
    documents = []
    blocks    = {b["Id"]: b for b in response["Blocks"]}
    
    # Extract LINE blocks for text
    text_lines = [b["Text"] for b in response["Blocks"]
                  if b["BlockType"] == "LINE" and "Text" in b]
    if text_lines:
        documents.append(Document(
            page_content="\n".join(text_lines),
            metadata={"source": f"s3://{s3_bucket}/{s3_key}", "element_type": "text"},
        ))
    
    # Extract TABLE blocks
    for block in response["Blocks"]:
        if block["BlockType"] == "TABLE":
            table_text = extract_textract_table(block, blocks)
            if table_text:
                documents.append(Document(
                    page_content=f"TABLE:\n{table_text}",
                    metadata={"source": f"s3://{s3_bucket}/{s3_key}", "element_type": "table"},
                ))
    
    return documents

def extract_textract_table(table_block: dict, blocks: dict) -> str:
    """Reconstruct a table from Textract block relationships."""
    cells = []
    for rel in table_block.get("Relationships", []):
        if rel["Type"] == "CHILD":
            for child_id in rel["Ids"]:
                cell = blocks.get(child_id)
                if cell and cell["BlockType"] == "CELL":
                    text = " ".join(
                        blocks[w]["Text"]
                        for r in cell.get("Relationships", []) if r["Type"] == "CHILD"
                        for w in r["Ids"] if blocks.get(w, {}).get("BlockType") == "WORD"
                    )
                    cells.append((cell["RowIndex"], cell["ColumnIndex"], text))
    
    if not cells:
        return ""
    
    max_row = max(c[0] for c in cells)
    max_col = max(c[1] for c in cells)
    grid    = [[""] * (max_col + 1) for _ in range(max_row + 1)]
    for row, col, text in cells:
        grid[row][col] = text
    
    return "\n".join(" | ".join(row) for row in grid)
```

### Intelligent Document Routing

Detect the document type and route to the appropriate parsing strategy.

```python
import magic   # pip install python-magic
from pathlib import Path

def route_document(filepath: str) -> list[Document]:
    """Detect document type and apply the appropriate parsing strategy."""
    path     = Path(filepath)
    suffix   = path.suffix.lower()
    mime     = magic.from_file(filepath, mime=True)
    
    # Check if PDF is scanned (no embedded text)
    if suffix == ".pdf":
        is_scanned = detect_scanned_pdf(filepath)
        if is_scanned:
            print(f"  '{path.name}' → scanned PDF — applying OCR")
            return ocr_pdf(filepath)
        else:
            print(f"  '{path.name}' → native PDF — applying unstructured.io")
            parsed = parse_pdf_multimodal(filepath)
            return build_documents_from_parsed(parsed)
    
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        print(f"  '{path.name}' → image — applying OCR")
        img  = Image.open(filepath)
        text = pytesseract.image_to_string(preprocess_image_for_ocr(img))
        return [Document(page_content=clean_ocr_text(text),
                         metadata={"source": filepath, "element_type": "text", "ocr": True})]
    
    elif suffix in (".docx", ".pptx", ".xlsx"):
        from unstructured.partition.auto import partition
        elements = partition(filename=filepath)
        return [Document(page_content=el.text, metadata={"source": filepath})
                for el in elements if el.text.strip()]
    
    elif suffix in (".md", ".txt", ".rst"):
        text = path.read_text(encoding="utf-8", errors="replace")
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.create_documents([text], metadatas=[{"source": filepath}])
    
    else:
        print(f"  Unsupported file type: {suffix}")
        return []

def detect_scanned_pdf(filepath: str, sample_pages: int = 3) -> bool:
    """Detect if a PDF is image-only (scanned) by checking for embedded text."""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages[:sample_pages]:
                text = page.extract_text() or ""
                if len(text.strip()) > 50:
                    return False  # Has real text — not scanned
        return True  # No text found on sampled pages
    except Exception:
        return False  # Assume native PDF on error
```

---

## 4. Multi-Source Enterprise Search Assistants

### The Multi-Source Challenge

```
ENTERPRISE KNOWLEDGE LIVES IN MANY PLACES:

  ┌──────────────────────────────────────────────────────────────┐
  │  Source              Type        Connector                   │
  ├──────────────────────────────────────────────────────────────┤
  │  SharePoint / OneDrive  Docs      Microsoft Graph API        │
  │  Confluence             Wiki      Confluence REST API        │
  │  Jira                   Tickets   Jira REST API              │
  │  Salesforce             CRM       Salesforce SOQL API        │
  │  Slack                  Messages  Slack Events API           │
  │  GitHub / GitLab        Code+Docs Git API                    │
  │  PostgreSQL / MySQL     Structured SQL                       │
  │  S3 / Azure Blob        Files     Cloud storage SDK          │
  │  Email (Exchange)       Messages  Microsoft Graph            │
  └──────────────────────────────────────────────────────────────┘

  A true enterprise search assistant ingests and searches ALL of these.
```

### Source Connector Architecture

```python
from abc import ABC, abstractmethod
from langchain.schema import Document
from datetime import datetime, UTC

class SourceConnector(ABC):
    """Base class for all enterprise source connectors."""
    
    @abstractmethod
    def fetch_documents(self, since: datetime = None) -> list[Document]:
        """Fetch documents from the source, optionally only changed since `since`."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        pass
    
    def tag_documents(self, docs: list[Document], extra_meta: dict = None) -> list[Document]:
        """Add source metadata to all documents."""
        for doc in docs:
            doc.metadata.update({
                "source_connector": self.get_source_name(),
                "indexed_at":       datetime.now(UTC).isoformat(),
                **(extra_meta or {}),
            })
        return docs
```

### Confluence Connector

```python
# pip install atlassian-python-api

from atlassian import Confluence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import os

class ConfluenceConnector(SourceConnector):
    def __init__(self, url: str, username: str, api_token: str,
                 spaces: list[str] = None):
        self.client = Confluence(url=url, username=username, password=api_token)
        self.spaces = spaces or ["HR", "IT", "ENG", "LEGAL"]
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    def get_source_name(self) -> str:
        return "confluence"
    
    def fetch_documents(self, since: datetime = None) -> list[Document]:
        all_docs = []
        
        for space_key in self.spaces:
            pages = self.client.get_all_pages_from_space(
                space=space_key, status="current", expand="body.storage,version"
            )
            for page in pages:
                modified = page.get("version", {}).get("when", "")
                
                # Incremental sync: skip if not modified since last run
                if since and modified and modified < since.isoformat():
                    continue
                
                html      = page.get("body", {}).get("storage", {}).get("value", "")
                plain_text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
                plain_text = plain_text.strip()
                
                if len(plain_text) < 50:
                    continue
                
                chunks = self.splitter.create_documents(
                    [plain_text],
                    metadatas=[{
                        "source":       page.get("_links", {}).get("webui", ""),
                        "title":        page.get("title", ""),
                        "space":        space_key,
                        "page_id":      page.get("id", ""),
                        "modified_at":  modified,
                        "clearance_level": self._space_clearance(space_key),
                    }],
                )
                all_docs.extend(chunks)
        
        return self.tag_documents(all_docs)
    
    def _space_clearance(self, space_key: str) -> int:
        return {"HR": 2, "LEGAL": 3, "ENG": 1, "IT": 1}.get(space_key, 1)
```

### SharePoint Connector

```python
# pip install Office365-REST-Python-Client

from office365.runtime.auth.user_credential import UserCredential
from office365.sharepoint.client_context import ClientContext

class SharePointConnector(SourceConnector):
    def __init__(self, site_url: str, username: str, password: str,
                 library_names: list[str] = None):
        self.ctx       = ClientContext(site_url).with_credentials(
                             UserCredential(username, password))
        self.libraries = library_names or ["Documents", "HR Policies", "IT Documentation"]
        self.splitter  = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    def get_source_name(self) -> str:
        return "sharepoint"
    
    def fetch_documents(self, since: datetime = None) -> list[Document]:
        all_docs = []
        
        for lib_name in self.libraries:
            library = self.ctx.web.lists.get_by_title(lib_name)
            items   = library.items.select(["FileLeafRef", "FileRef",
                                            "Modified", "File_x0020_Size"])
            self.ctx.load(items).execute_query()
            
            for item in items:
                file_ref  = item.properties.get("FileRef", "")
                file_name = item.properties.get("FileLeafRef", "")
                modified  = item.properties.get("Modified", "")
                
                if not any(file_name.lower().endswith(ext)
                           for ext in [".pdf", ".docx", ".pptx", ".txt"]):
                    continue
                
                # Download and parse
                try:
                    file_content = self.ctx.web.get_file_by_server_relative_url(
                        file_ref).download().execute_query().content
                    
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(
                        suffix=os.path.splitext(file_name)[1], delete=False
                    ) as tmp:
                        tmp.write(file_content)
                        tmp_path = tmp.name
                    
                    docs = route_document(tmp_path)
                    for doc in docs:
                        doc.metadata.update({
                            "source":      file_ref,
                            "filename":    file_name,
                            "library":     lib_name,
                            "modified_at": modified,
                        })
                    all_docs.extend(docs)
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"  Failed to process {file_name}: {e}")
        
        return self.tag_documents(all_docs)
```

### Multi-Source Ingestion Orchestrator

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class EnterpriseIngestionOrchestrator:
    def __init__(self, connectors: list[SourceConnector], vectorstore):
        self.connectors  = connectors
        self.vectorstore = vectorstore
    
    def run_full_ingest(self, parallel: bool = True) -> dict:
        """Ingest from all sources, with optional parallelism."""
        stats = {}
        
        if parallel:
            with ThreadPoolExecutor(max_workers=len(self.connectors)) as pool:
                futures = {
                    pool.submit(self._ingest_source, c): c.get_source_name()
                    for c in self.connectors
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        stats[name] = future.result()
                    except Exception as e:
                        stats[name] = {"error": str(e)}
        else:
            for connector in self.connectors:
                stats[connector.get_source_name()] = self._ingest_source(connector)
        
        total = sum(s.get("indexed", 0) for s in stats.values() if isinstance(s, dict))
        print(f"\nIngestion complete. Total documents indexed: {total}")
        return stats
    
    def _ingest_source(self, connector: SourceConnector) -> dict:
        name = connector.get_source_name()
        print(f"Ingesting from {name}...")
        try:
            docs = connector.fetch_documents()
            if docs:
                self.vectorstore.add_documents(docs)
            return {"indexed": len(docs), "source": name}
        except Exception as e:
            print(f"Error ingesting {name}: {e}")
            return {"error": str(e), "source": name}
    
    def run_incremental_ingest(self, since: datetime) -> dict:
        """Only re-ingest documents modified since the last run."""
        stats = {}
        for connector in self.connectors:
            name = connector.get_source_name()
            docs = connector.fetch_documents(since=since)
            if docs:
                # Delete old versions of the same docs, re-add updated
                self._upsert_documents(docs, connector)
            stats[name] = {"updated": len(docs)}
        return stats
    
    def _upsert_documents(self, docs: list[Document], connector: SourceConnector):
        """Delete existing docs from this source, then add updated versions."""
        try:
            existing = self.vectorstore.get(
                where={"source_connector": connector.get_source_name()}
            )
            if existing["ids"]:
                self.vectorstore.delete(ids=existing["ids"])
        except Exception:
            pass
        self.vectorstore.add_documents(docs)
```

---

## 5. Enterprise Assistant Use Cases

### Use Case 1 — HR Knowledge Bot

```python
# assistants/hr_bot.py

HR_SYSTEM_PROMPT = """You are AcmeCorp's HR Knowledge Assistant.
You help employees understand HR policies, benefits, leave entitlements,
expense procedures, and workplace guidelines.

Rules:
1. Answer ONLY using the provided CONTEXT. Never use general knowledge.
2. If the answer is not in the context: "I don't have that information.
   Please contact HR at hr@acmecorp.com or call ext. 1234."
3. For sensitive topics (disciplinary, grievance, dismissal, medical):
   always end with: "Please speak directly with your HR Business Partner
   for formal guidance on this topic."
4. Never reveal specific employee personal data to other employees.
5. Confirm the user's department/location if policy differs by region.

CONTEXT:
{context}"""

HR_TOOLS = [
    "search_hr_policies",          # leave, benefits, expenses, procedures
    "search_employee_handbook",    # culture, conduct, values
    "lookup_hr_contacts",          # HR BPs by department/region
    "check_leave_balance",         # structured query to HR system (read-only)
]

HR_SCOPE_KEYWORDS = {
    "in_scope":  ["leave", "holiday", "sick", "parental", "benefit", "pension",
                  "expense", "bonus", "salary", "training", "performance",
                  "probation", "promotion", "onboarding", "offboarding",
                  "working hours", "flexible", "remote", "dress code"],
    "escalate":  ["disciplinary", "grievance", "harassment", "dismissal",
                  "redundancy", "discrimination", "tribunal", "whistleblowing"],
    "sensitive": ["medical", "disability", "pregnancy", "religion", "personal data"],
}

def hr_bot_query(question: str, user: dict) -> str:
    # Check escalation triggers
    q_lower = question.lower()
    if any(kw in q_lower for kw in HR_SCOPE_KEYWORDS["escalate"]):
        return (
            "This topic requires direct HR support. Please contact your "
            "HR Business Partner or email hr-relations@acmecorp.com. "
            "For urgent matters, call the confidential HR helpline: 0800-HR-HELP."
        )
    
    # Role-filtered retrieval
    docs    = get_retriever_for_user(vectorstore, user["clearance"], user["roles"]).invoke(question)
    context = format_docs_with_sources(docs)
    
    answer  = (ChatPromptTemplate.from_messages([
        ("system", HR_SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke({}).content
    
    # Append disclaimer for sensitive topics
    if any(kw in q_lower for kw in HR_SCOPE_KEYWORDS["sensitive"]):
        answer += ("\n\n⚠️ For matters involving health, disability, or personal "
                   "circumstances, please speak with your HR BP in confidence.")
    
    return answer
```

### Use Case 2 — IT Support Assistant

```python
# assistants/it_bot.py

IT_SYSTEM_PROMPT = """You are AcmeCorp's IT Support Assistant.
You help employees resolve IT issues, follow setup procedures, and
understand IT policies. You have access to the internal knowledge base,
runbooks, and can check system status.

Rules:
1. Always ask for the employee's device OS and version for troubleshooting.
2. For security incidents (phishing, malware, data breach): IMMEDIATELY
   direct to security@acmecorp.com and advise disconnecting from network.
3. Never instruct users to disable security tools or bypass MFA.
4. If troubleshooting steps fail after 3 attempts, escalate to a ticket.
5. Include ticket creation instructions for unresolved issues.

CONTEXT:
{context}"""

def it_support_query(question: str, session_context: dict = None) -> dict:
    """IT support with hybrid search (exact error codes + semantic)."""
    
    # Security incident detection — immediate escalation
    SECURITY_TERMS = ["phishing", "malware", "ransomware", "data breach",
                      "suspicious email", "virus", "hacked", "password stolen"]
    if any(t in question.lower() for t in SECURITY_TERMS):
        return {
            "answer": (
                "⚠️ SECURITY INCIDENT: Please take these immediate steps:\n"
                "1. Disconnect from WiFi/ethernet NOW\n"
                "2. Call IT Security: +44 20 XXXX XXXX (24/7)\n"
                "3. Email security@acmecorp.com from a PERSONAL device\n"
                "4. Do NOT power off your device — preserve evidence\n\n"
                "Do not attempt to resolve a security incident yourself."
            ),
            "escalated": True,
        }
    
    # Hybrid retrieval — crucial for IT (exact error codes + semantic)
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    
    it_bm25    = BM25Retriever.from_documents(it_chunks, k=10)
    it_dense   = vectorstore.as_retriever(
        search_kwargs={"k": 10, "filter": {"department": "IT"}}
    )
    it_hybrid  = EnsembleRetriever(retrievers=[it_dense, it_bm25], weights=[0.5, 0.5])
    
    docs    = it_hybrid.invoke(question)
    context = format_docs_with_sources(docs[:5])
    
    answer = (ChatPromptTemplate.from_messages([
        ("system", IT_SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke({}).content
    
    # Add ticket creation prompt if answer mentions unresolved issue
    if any(phrase in answer.lower() for phrase in
           ["contact support", "raise a ticket", "submit a request"]):
        answer += (
            "\n\n📋 **Create a support ticket:** "
            "https://jira.acmecorp.com/servicedesk/customer/portal/2 "
            "or call the IT helpdesk: ext. 5678"
        )
    
    return {"answer": answer, "escalated": False}
```

### Use Case 3 — Legal Assistant

```python
# assistants/legal_bot.py

LEGAL_SYSTEM_PROMPT = """You are AcmeCorp's internal Legal Knowledge Assistant.
You help employees understand standard company contracts, legal policies,
and regulatory requirements.

IMPORTANT LIMITATIONS — always enforce these:
1. You provide GENERAL INFORMATION ONLY from AcmeCorp's legal documents.
2. You do NOT provide legal advice. For any specific legal matter, employees
   MUST consult the Legal team at legal@acmecorp.com.
3. For any contract with a value > £50,000 or involving personal liability,
   always require Legal team sign-off.
4. Never interpret external law or regulations beyond what is stated in
   the provided context.
5. Mark every response: "ℹ️ This is general information, not legal advice."

CONTEXT:
{context}"""

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def legal_assistant_query(question: str, user: dict) -> str:
    """Legal assistant using multi-hop for complex contract questions."""
    
    # High-clearance retrieval (legal docs are CONFIDENTIAL)
    if user.get("clearance", 0) < 3 and user.get("role") not in ("legal", "executive"):
        return (
            "Access to detailed legal documents requires Legal team clearance. "
            "Please contact legal@acmecorp.com with your query."
        )
    
    # Check if multi-hop needed (clause cross-reference questions)
    MULTI_HOP_TRIGGERS = ["referenced in", "defined in", "pursuant to",
                           "as per section", "in accordance with", "under clause"]
    needs_multihop = any(t in question.lower() for t in MULTI_HOP_TRIGGERS)
    
    if needs_multihop:
        result  = multi_hop_rag(question, max_hops=4, k=3)
        context = "\n\n".join(h["retrieved"] for h in result["hops"])
    else:
        # Re-ranked retrieval for precision
        cross_enc  = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker   = CrossEncoderReranker(model=cross_enc, top_n=4)
        retriever  = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=vectorstore.as_retriever(search_kwargs={"k": 15})
        )
        docs    = retriever.invoke(question)
        context = format_docs_with_sources(docs)
    
    answer = (ChatPromptTemplate.from_messages([
        ("system", LEGAL_SYSTEM_PROMPT.format(context=context)),
        ("human", question),
    ]) | ChatOpenAI(model="gpt-4o", temperature=0)).invoke({}).content
    
    return answer + "\n\nℹ️ This is general information, not legal advice."
```

### Use Case 4 — Sales Enablement Bot

```python
# assistants/sales_bot.py

SALES_SYSTEM_PROMPT = """You are AcmeCorp's Sales Enablement Assistant.
You help the sales team find product information, competitive intelligence,
pricing guidelines, case studies, and talk tracks.

You have access to:
- Internal product documentation and positioning
- Approved competitive battle cards
- Customer case studies and testimonials
- Pricing and discount approval guidelines
- Sales scripts and objection-handling guides

Rules:
1. Never share unapproved pricing or discount levels externally.
2. For deals > £100K, always flag that SE/Solutions Architect involvement is required.
3. Competitive claims must cite internal battle cards — do not speculate.
4. If asked about a competitor gap we have: acknowledge it and redirect to our strengths.

CONTEXT:
{context}"""

from langchain_community.tools.tavily_search import TavilySearchResults

def sales_enablement_query(question: str, opportunity_context: dict = None) -> str:
    """
    Sales bot using CRAG: internal docs first, web search for competitor/market intel.
    """
    # For competitive / market questions → use CRAG with web fallback
    COMPETITIVE_TERMS = ["competitor", "vs ", "versus", "compare", "alternative",
                          "market share", "pricing", "how does", "differentiator"]
    
    if any(t in question.lower() for t in COMPETITIVE_TERMS):
        result  = run_crag(question)  # grade → web fallback if needed
        context = result["context"]
    else:
        # Internal knowledge only for product / process questions
        docs    = vectorstore.similarity_search(question, k=5,
                      filter={"department": "sales"})
        context = format_docs_with_sources(docs)
    
    # Enrich with opportunity context if provided
    opp_context = ""
    if opportunity_context:
        opp_context = (
            f"\n\nOpportunity Context: "
            f"Industry={opportunity_context.get('industry')}, "
            f"Size={opportunity_context.get('company_size')}, "
            f"Stage={opportunity_context.get('deal_stage')}\n"
        )
    
    answer = (ChatPromptTemplate.from_messages([
        ("system", SALES_SYSTEM_PROMPT.format(context=context + opp_context)),
        ("human", question),
    ]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke({}).content
    
    # Flag large deal requirement
    if opportunity_context and opportunity_context.get("deal_value", 0) > 100_000:
        answer += (
            "\n\n⚠️ **Deal size > £100K**: Solutions Architect involvement is required. "
            "Please request SE support via Salesforce."
        )
    
    return answer
```

### Use Case 5 — Internal Enterprise Search

```python
# assistants/enterprise_search.py

class EnterpriseSearchAssistant:
    """
    Universal enterprise search combining all knowledge sources
    with adaptive routing based on query intent.
    """
    
    SOURCES = {
        "hr_policies":   {"icon": "👤", "description": "HR policies and procedures"},
        "it_runbooks":   {"icon": "💻", "description": "IT documentation and runbooks"},
        "legal_docs":    {"icon": "⚖️",  "description": "Legal contracts and policies"},
        "sales_assets":  {"icon": "📈", "description": "Sales playbooks and case studies"},
        "engineering":   {"icon": "🔧", "description": "Technical architecture and ADRs"},
        "finance":       {"icon": "💰", "description": "Finance policies and reports"},
    }
    
    def search(self, query: str, user: dict,
               sources: list[str] = None,
               result_limit: int = 10) -> dict:
        """
        Unified search across all enterprise knowledge sources.
        """
        # Determine which sources to search
        active_sources = sources or self._auto_select_sources(query)
        
        # Build source-specific retriever with RBAC
        all_results = []
        
        for source in active_sources:
            source_filter = {
                "department":    source,
                "clearance_level": {"$lte": user.get("clearance", 1)},
            }
            source_docs = vectorstore.similarity_search(
                query, k=5, filter=source_filter
            )
            for doc in source_docs:
                doc.metadata["matched_source"] = source
            all_results.extend(source_docs)
        
        # Re-rank unified results
        if len(all_results) > result_limit:
            from sentence_transformers import CrossEncoder
            cross_enc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs     = [(query, d.page_content) for d in all_results]
            scores    = cross_enc.predict(pairs)
            all_results = [doc for _, doc in
                           sorted(zip(scores, all_results), reverse=True)][:result_limit]
        
        # Generate synthesised answer
        context = format_docs_with_sources(all_results[:6])
        answer  = (ChatPromptTemplate.from_messages([
            ("system", f"""You are AcmeCorp's enterprise search assistant.
Answer using only the provided context from company knowledge bases.
Cite the source for each piece of information.

CONTEXT:
{context}"""),
            ("human", query),
        ]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke({}).content
        
        # Format results for UI
        return {
            "answer":   answer,
            "sources":  [
                {
                    "title":   d.metadata.get("title", d.metadata.get("source", "")),
                    "source":  d.metadata.get("matched_source", ""),
                    "url":     d.metadata.get("source", ""),
                    "excerpt": d.page_content[:200] + "...",
                }
                for d in all_results[:6]
            ],
            "searched_sources": active_sources,
        }
    
    def _auto_select_sources(self, query: str) -> list[str]:
        """Route query to relevant source categories."""
        q = query.lower()
        selected = []
        
        if any(k in q for k in ["leave", "salary", "benefit", "hr", "holiday"]):
            selected.append("hr_policies")
        if any(k in q for k in ["error", "ssl", "vpn", "wifi", "laptop", "password", "access"]):
            selected.append("it_runbooks")
        if any(k in q for k in ["contract", "legal", "compliance", "gdpr", "regulation"]):
            selected.append("legal_docs")
        if any(k in q for k in ["pricing", "competitor", "case study", "customer", "sales"]):
            selected.append("sales_assets")
        if any(k in q for k in ["architecture", "api", "service", "deploy", "code"]):
            selected.append("engineering")
        if any(k in q for k in ["budget", "invoice", "expense", "finance", "cost"]):
            selected.append("finance")
        
        return selected or list(self.SOURCES.keys())  # fallback: search all
```

---

## 6. Personalization and Memory Concepts

### Types of Memory in Enterprise Assistants

```
MEMORY TAXONOMY:

  Short-term (conversation memory):
  └── What was said earlier in THIS session
      → ConversationBufferMemory, ConversationSummaryMemory
      → Cleared when session ends

  Long-term episodic memory:
  └── What THIS USER has asked, done, and learned in PAST sessions
      → Stored in a user profile vector store
      → Persists across sessions

  Semantic / knowledge memory:
  └── What the system knows about the domain (the knowledge base)
      → The RAG vector store
      → Shared across all users

  Preference memory:
  └── HOW this user likes to receive information
      → Response length preference, technical depth, language
      → Stored in a user profile database
```

### Conversation Memory (Short-Term)

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Summarise when conversation exceeds 2000 tokens
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)

conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory,
    return_source_documents=True,
    verbose=False,
)

# Session 1
result = conversational_chain.invoke({"question": "What is the parental leave policy?"})
print(result["answer"])

# Follow-up (uses conversation memory — no need to repeat context)
result = conversational_chain.invoke({"question": "How do I apply for it?"})
print(result["answer"])

# Even further follow-up
result = conversational_chain.invoke({"question": "And what about for adoption?"})
print(result["answer"])
```

### Long-Term User Memory Store

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from datetime import datetime, UTC
import json

class UserMemoryStore:
    """
    Stores each user's interaction history as searchable embeddings.
    Enables personalisation across sessions.
    """
    
    def __init__(self, persist_dir: str = "./user_memory"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.store      = Chroma(
            collection_name="user_memories",
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )
    
    def save_interaction(self, user_id: str, question: str,
                          answer: str, topics: list[str] = None):
        """Save a Q&A interaction to the user's memory."""
        memory_text = (
            f"User asked: {question}\n"
            f"Topics: {', '.join(topics or [])}\n"
            f"Summary: {answer[:200]}"
        )
        self.store.add_documents([Document(
            page_content=memory_text,
            metadata={
                "user_id":    user_id,
                "timestamp":  datetime.now(UTC).isoformat(),
                "question":   question,
                "topics":     json.dumps(topics or []),
            },
        )])
    
    def get_relevant_memories(self, user_id: str, current_question: str,
                               k: int = 3) -> list[str]:
        """Retrieve past interactions relevant to the current question."""
        results = self.store.similarity_search(
            current_question,
            k=k,
            filter={"user_id": user_id},
        )
        return [doc.page_content for doc in results]
    
    def get_user_profile_summary(self, user_id: str) -> str:
        """Build a summary of what this user typically asks about."""
        all_memories = self.store.get(where={"user_id": user_id})
        if not all_memories["documents"]:
            return ""
        
        # Ask LLM to summarise the user's interest pattern
        mem_text = "\n".join(all_memories["documents"][-20:])  # last 20 interactions
        summary  = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
            f"Summarise this user's typical questions and interests in 2-3 sentences:\n\n{mem_text}"
        ).content
        return summary

user_memory = UserMemoryStore()
```

### User Preference Memory

```python
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class UserPreferences:
    user_id:            str
    response_style:     str   = "balanced"   # "brief" | "balanced" | "detailed"
    technical_level:    str   = "intermediate"  # "beginner" | "intermediate" | "expert"
    preferred_language: str   = "en"
    include_sources:    bool  = True
    format_preference:  str   = "prose"      # "prose" | "bullet_points" | "numbered"
    domain_expertise:   list  = None         # e.g., ["HR", "Legal"]
    
    def __post_init__(self):
        if self.domain_expertise is None:
            self.domain_expertise = []

PREFS_DB_PATH = Path("./user_profiles")

def save_preferences(prefs: UserPreferences):
    PREFS_DB_PATH.mkdir(exist_ok=True)
    (PREFS_DB_PATH / f"{prefs.user_id}.json").write_text(
        json.dumps(asdict(prefs), indent=2)
    )

def load_preferences(user_id: str) -> UserPreferences:
    path = PREFS_DB_PATH / f"{user_id}.json"
    if path.exists():
        return UserPreferences(**json.loads(path.read_text()))
    return UserPreferences(user_id=user_id)  # default preferences

def personalise_prompt(base_prompt: str, prefs: UserPreferences) -> str:
    """Inject user preference instructions into the system prompt."""
    style_instructions = {
        "brief":    "Be concise. Answer in 2-3 sentences maximum.",
        "balanced": "Provide a complete but focused answer.",
        "detailed": "Provide a thorough, comprehensive answer with all relevant details.",
    }
    format_instructions = {
        "prose":         "Write in natural prose paragraphs.",
        "bullet_points": "Use bullet points for lists and key information.",
        "numbered":      "Use numbered steps for procedural information.",
    }
    level_instructions = {
        "beginner":     "Use simple language, avoid jargon, explain acronyms.",
        "intermediate": "Use standard business language.",
        "expert":       "Use technical terminology freely, skip basic explanations.",
    }
    
    personalisation = (
        f"\n\nUser Preferences:"
        f"\n- Style: {style_instructions[prefs.response_style]}"
        f"\n- Format: {format_instructions[prefs.format_preference]}"
        f"\n- Level: {level_instructions[prefs.technical_level]}"
        + (f"\n- Include document sources: yes" if prefs.include_sources else "")
    )
    
    return base_prompt + personalisation
```

### Personalised RAG Query

```python
def personalised_rag_query(question: str, user_id: str,
                             user_prefs: UserPreferences = None,
                             session_memory=None) -> dict:
    """
    Full personalised RAG query combining:
    - User preference-adapted prompt
    - Long-term user memory context
    - Short-term conversation memory
    """
    prefs   = user_prefs or load_preferences(user_id)
    
    # Retrieve relevant past interactions
    past_interactions = user_memory.get_relevant_memories(user_id, question, k=2)
    past_context      = ""
    if past_interactions:
        past_context = (
            "\n\nUser's past relevant queries (for context, not as answers):\n"
            + "\n".join(f"- {m}" for m in past_interactions)
        )
    
    # Build personalised prompt
    base_system = """You are an enterprise knowledge assistant.
Answer ONLY using the provided CONTEXT.

CONTEXT:
{context}"""
    
    personalised_system = personalise_prompt(base_system, prefs) + past_context
    
    # Retrieve documents
    docs    = vectorstore.similarity_search(question, k=5)
    context = format_docs_with_sources(docs)
    
    # Generate with optional conversation memory
    if session_memory:
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=session_memory,
        )
        result = chain.invoke({"question": question})
        answer = result["answer"]
    else:
        answer = (ChatPromptTemplate.from_messages([
            ("system", personalised_system.format(context=context)),
            ("human", question),
        ]) | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke({}).content
    
    # Save to long-term memory (async in production)
    user_memory.save_interaction(
        user_id=user_id,
        question=question,
        answer=answer,
        topics=extract_topics(question),
    )
    
    return {"answer": answer, "sources": docs, "personalised": True}

def extract_topics(text: str) -> list[str]:
    """Simple keyword extraction for memory tagging."""
    TOPIC_KEYWORDS = {
        "leave": ["leave", "holiday", "vacation", "sick", "parental"],
        "expenses": ["expense", "reimbursement", "claim", "receipt"],
        "IT": ["laptop", "password", "vpn", "access", "error"],
        "legal": ["contract", "gdpr", "compliance", "regulation"],
        "benefits": ["pension", "health", "insurance", "gym", "benefit"],
    }
    topics = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            topics.append(topic)
    return topics
```

---

## 7. Lab: Enterprise Assistant Solution Blueprint

### Blueprint Overview

Design and implement a complete multi-modal enterprise assistant for **AcmeCorp** — a mid-size technology company. The assistant must handle all five use cases from Section 5 under a unified API with multi-source ingestion, multi-modal document support, role-based access, personalisation, and memory.

### System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ACMECORP ENTERPRISE ASSISTANT                         │
├─────────────────────────────┬────────────────────────────────────────────┤
│  INGESTION LAYER            │  QUERY LAYER                               │
│                             │                                            │
│  ┌─────────────────────┐   │  ┌──────────────────────────────────────┐  │
│  │  Source Connectors  │   │  │  Auth + RBAC + Session Management    │  │
│  │  ├── SharePoint     │   │  └──────────────────┬───────────────────┘  │
│  │  ├── Confluence     │   │                      │                      │
│  │  ├── Jira           │   │  ┌──────────────────▼───────────────────┐  │
│  │  └── Local FS       │   │  │  Intent Router (Assistant Selector)  │  │
│  └────────┬────────────┘   │  │  HR / IT / Legal / Sales / Search    │  │
│           │                │  └──────────────────┬───────────────────┘  │
│  ┌────────▼────────────┐   │                      │                      │
│  │  Document Intel.    │   │  ┌──────────────────▼───────────────────┐  │
│  │  Text/Table/Image   │   │  │  Personalised RAG Pipeline           │  │
│  │  OCR (Azure DI)     │   │  │  Prefs + User Memory + Session Mem   │  │
│  └────────┬────────────┘   │  └──────────────────┬───────────────────┘  │
│           │                │                      │                      │
│  ┌────────▼────────────┐   │  ┌──────────────────▼───────────────────┐  │
│  │  Unified Vector DB  │   │  │  Output Safety + Audit Log           │  │
│  │  (Pinecone/Chroma)  │◄──┤  └──────────────────┬───────────────────┘  │
│  │  + RBAC metadata    │   │                      │                      │
│  └─────────────────────┘   │               Response to User             │
└─────────────────────────────┴────────────────────────────────────────────┘
```

### Intent Router

```python
# app/intent_router.py

from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

class AssistantType(str, Enum):
    HR      = "hr"
    IT      = "it"
    LEGAL   = "legal"
    SALES   = "sales"
    SEARCH  = "enterprise_search"

INTENT_PROMPT = ChatPromptTemplate.from_template("""Classify this enterprise query
into the most appropriate internal assistant:

- hr:              Leave, benefits, expenses, HR policies, performance, onboarding
- it:              Technical issues, passwords, VPN, software, hardware, IT policies
- legal:           Contracts, compliance, GDPR, regulations, legal policies
- sales:           Product info, pricing, competitors, case studies, sales process
- enterprise_search: Cross-domain or unclear — search all knowledge bases

Query: "{query}"

Return JSON: {{"assistant": "<type>", "confidence": 0.0-1.0}}""")

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def route_to_assistant(query: str) -> tuple[AssistantType, float]:
    result   = router_llm.invoke(INTENT_PROMPT.format(query=query))
    parsed   = json.loads(result.content)
    return AssistantType(parsed["assistant"]), parsed["confidence"]
```

### Unified FastAPI Application

```python
# app/main.py

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

app = FastAPI(title="AcmeCorp Enterprise Assistant", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class AssistantRequest(BaseModel):
    query:          str
    user_id:        str
    user_role:      str       = "employee"
    user_clearance: int       = 1
    session_id:     str       = ""
    assistant_hint: Optional[str] = None   # override auto-routing if known
    stream:         bool      = False

class AssistantResponse(BaseModel):
    answer:       str
    assistant:    str
    sources:      list
    request_id:   str
    personalised: bool = False

@app.post("/ask", response_model=AssistantResponse)
async def ask(req: AssistantRequest):
    import uuid
    request_id = str(uuid.uuid4())
    
    user = {
        "id":        req.user_id,
        "role":      req.user_role,
        "clearance": req.user_clearance,
        "session_id": req.session_id,
    }
    
    # Security pre-flight (from Module-3)
    from security.guards import check_injection, check_scope
    if check_injection(req.query):
        raise HTTPException(status_code=400, detail="Request rejected.")
    
    # Route to appropriate assistant
    if req.assistant_hint:
        assistant_type = AssistantType(req.assistant_hint)
    else:
        assistant_type, confidence = route_to_assistant(req.query)
    
    # Load user preferences
    prefs = load_preferences(req.user_id)
    
    # Dispatch to specialised assistant
    if assistant_type == AssistantType.HR:
        answer = hr_bot_query(req.query, user)
        sources = []
    elif assistant_type == AssistantType.IT:
        result  = it_support_query(req.query)
        answer, sources = result["answer"], []
    elif assistant_type == AssistantType.LEGAL:
        answer  = legal_assistant_query(req.query, user)
        sources = []
    elif assistant_type == AssistantType.SALES:
        answer  = sales_enablement_query(req.query)
        sources = []
    else:  # enterprise_search
        search_result = EnterpriseSearchAssistant().search(req.query, user)
        answer  = search_result["answer"]
        sources = search_result["sources"]
    
    # Personalise if preferences exist
    if prefs.response_style != "balanced" or prefs.format_preference != "prose":
        answer = apply_format_preferences(answer, prefs)
    
    # Audit log (from Module-3)
    from audit.logger import audit_logger
    audit_logger.log({
        "request_id": request_id,
        "user_id":    req.user_id,
        "assistant":  assistant_type.value,
        "query":      req.query[:500],
        "answer_len": len(answer),
    })
    
    return AssistantResponse(
        answer=answer,
        assistant=assistant_type.value,
        sources=sources,
        request_id=request_id,
        personalised=True,
    )

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/assistants")
async def list_assistants():
    return {
        "assistants": [
            {"id": "hr",    "name": "HR Knowledge Bot",     "description": "Leave, benefits, HR policies"},
            {"id": "it",    "name": "IT Support Assistant", "description": "Technical help and IT policies"},
            {"id": "legal", "name": "Legal Assistant",      "description": "Contracts and compliance (Legal clearance required)"},
            {"id": "sales", "name": "Sales Enablement Bot", "description": "Product info and competitive intel"},
            {"id": "enterprise_search", "name": "Enterprise Search", "description": "Search all knowledge bases"},
        ]
    }
```

### Ingestion Pipeline Script

```python
# scripts/ingest_all_sources.py

import os
from datetime import datetime, UTC

def main():
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="enterprise_knowledge",
        persist_directory="./chroma_enterprise",
        embedding_function=embeddings,
    )
    
    connectors = [
        ConfluenceConnector(
            url=os.environ["CONFLUENCE_URL"],
            username=os.environ["CONFLUENCE_USER"],
            api_token=os.environ["CONFLUENCE_TOKEN"],
            spaces=["HR", "IT", "ENG", "LEGAL"],
        ),
        SharePointConnector(
            site_url=os.environ["SHAREPOINT_SITE"],
            username=os.environ["SHAREPOINT_USER"],
            password=os.environ["SHAREPOINT_PASS"],
            library_names=["HR Policies", "Legal Documents", "IT Runbooks"],
        ),
    ]
    
    # Also ingest local multi-modal documents
    import glob
    local_docs_dir = "./docs"
    
    all_local_docs = []
    for filepath in glob.glob(f"{local_docs_dir}/**/*.*", recursive=True):
        docs = route_document(filepath)
        all_local_docs.extend(docs)
    
    if all_local_docs:
        vectorstore.add_documents(all_local_docs)
        print(f"Local docs indexed: {len(all_local_docs)}")
    
    orchestrator = EnterpriseIngestionOrchestrator(connectors, vectorstore)
    stats = orchestrator.run_full_ingest(parallel=True)
    
    print("\n=== Ingestion Report ===")
    for source, stat in stats.items():
        print(f"  {source}: {stat}")

if __name__ == "__main__":
    main()
```

### Blueprint Deliverables and Project Structure

```
acmecorp-enterprise-assistant/
├── app/
│   ├── main.py                         ← FastAPI application
│   ├── intent_router.py                ← Assistant type classifier
│   └── production_rag.py              ← Core RAG pipeline
├── assistants/
│   ├── hr_bot.py                       ← HR Knowledge Bot
│   ├── it_bot.py                       ← IT Support Assistant
│   ├── legal_bot.py                    ← Legal Assistant
│   ├── sales_bot.py                    ← Sales Enablement Bot
│   └── enterprise_search.py           ← Universal enterprise search
├── connectors/
│   ├── base.py                         ← SourceConnector ABC
│   ├── confluence.py                   ← Confluence connector
│   ├── sharepoint.py                   ← SharePoint connector
│   └── orchestrator.py                ← Ingestion orchestrator
├── multimodal/
│   ├── pdf_parser.py                   ← unstructured.io PDF parsing
│   ├── table_extraction.py            ← Table → text representation
│   ├── image_captioning.py            ← GPT-4o vision captioning
│   ├── clip_retriever.py              ← CLIP image embedding
│   └── ocr_pipeline.py                ← Tesseract + Azure DI OCR
├── memory/
│   ├── conversation_memory.py         ← Short-term session memory
│   ├── user_memory_store.py           ← Long-term user memory
│   └── user_preferences.py           ← User preference management
├── security/
│   ├── guards.py                       ← Injection, scope, PII (Module-3)
│   └── rbac.py                         ← Role-based access control
├── audit/
│   └── logger.py                       ← Structured audit logging
├── prompts/
│   ├── registry.json                   ← Versioned prompt registry
│   └── system/
│       ├── hr_assistant_v1.1.0.txt
│       ├── it_assistant_v1.0.0.txt
│       ├── legal_assistant_v1.0.0.txt
│       └── sales_assistant_v1.0.0.txt
├── evaluation/
│   ├── golden_dataset.json            ← Multi-assistant golden dataset
│   └── results/
├── scripts/
│   ├── ingest_all_sources.py          ← Full ingest pipeline
│   └── eval_gate.py                    ← CI/CD quality gate
├── config/
│   └── settings.py                     ← Pydantic env-aware settings
├── .github/workflows/
│   └── rag-eval.yml                    ← CI/CD pipeline
├── docker-compose.yml                  ← Chroma + Redis + App
└── requirements.txt
```

### Quick-Start Docker Compose

```yaml
# docker-compose.yml
version: "3.9"
services:
  chroma:
    image: chromadb/chroma:latest
    ports: ["8000:8000"]
    volumes: ["./chroma_data:/chroma/chroma"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  enterprise-assistant:
    build: .
    ports: ["8080:8080"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VECTOR_STORE_URL=http://chroma:8000
      - REDIS_URL=redis://redis:6379
      - LANGCHAIN_TRACING_V2=true
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=enterprise-assistant-prod
    depends_on: [chroma, redis]
    volumes: ["./docs:/app/docs", "./user_profiles:/app/user_profiles"]
```

### Solution Blueprint Evaluation Criteria

```
FUNCTIONAL COMPLETENESS:
  [ ] HR bot answers leave/benefit/expense queries correctly (Faithfulness > 0.85)
  [ ] IT bot handles both exact error codes and semantic queries (Recall@5 > 0.80)
  [ ] Legal bot correctly restricts access by clearance level
  [ ] Sales bot uses CRAG for competitive/time-sensitive queries
  [ ] Enterprise search retrieves from all active sources

MULTI-MODAL SUPPORT:
  [ ] PDFs with tables return structured data in answers
  [ ] Images/diagrams retrievable via text query (CLIP or caption search)
  [ ] Scanned PDFs successfully OCR'd and indexed
  [ ] Response correctly distinguishes text vs table vs image sources

PERSONALIZATION:
  [ ] Response style adapts to user preference (brief/detailed)
  [ ] Long-term memory correctly recalls past user interactions
  [ ] Conversation memory maintains context across follow-up questions

SECURITY & GOVERNANCE (Module-3):
  [ ] Injection attempts blocked
  [ ] Cross-clearance retrieval prevented by metadata filters
  [ ] PII redacted from queries, responses, and audit logs
  [ ] Full audit trail in JSONL format

PRODUCTION READINESS (Module-4):
  [ ] CI/CD eval gate configured and passing
  [ ] Prompts versioned in registry
  [ ] Semantic cache active (Redis)
  [ ] Health endpoint returns meaningful status
  [ ] LangSmith tracing enabled
```

---

## Summary

Multi-modal RAG and enterprise knowledge assistants represent the convergence of all previous modules into a production-grade, organisation-wide system.

```
THE ENTERPRISE ASSISTANT STACK:

  Module-1: Evaluation    → Quality gates on each assistant's golden dataset
  Module-2: Optimization  → Chunking + Hybrid + Re-ranking per source type
  Module-3: Security      → RBAC, PII, injection guards, audit trails
  Module-4: LLMOps        → Versioning, CI/CD, monitoring, feedback loops
  Module-5: Advanced Arch → CRAG for sales, Multi-hop for legal, Adaptive routing
  Module-6: Multi-modal   → OCR, tables, images, multi-source connectors
             + Enterprise  → Five specialist assistants under one intent router
             + Memory      → Conversation + user + preference memory
```

**Key takeaways:**

1. **50% of enterprise knowledge is non-text** — tables, images, and scanned documents are invisible to text-only RAG. Multi-modal indexing is not optional for enterprise deployment.
2. **Parse at the element level, not the page level** — treating a PDF page as a single chunk destroys table structure and loses image context. Use unstructured.io or Azure Document Intelligence for layout-aware parsing.
3. **Specialist assistants outperform generalist ones** — an HR bot with HR-tuned prompts, HR-filtered retrieval, and HR-specific escalation rules will consistently outperform a generic assistant on HR queries.
4. **Intent routing is the enterprise scalability lever** — a single query classifier distributing to specialist assistants scales the system to new domains by adding a new assistant module, without changing the core pipeline.
5. **Memory makes assistants feel intelligent** — conversation memory eliminates repetitive context-setting; user memory enables true personalisation; preference memory respects how each person wants to work.

---

_Guide maintained as part of the **Integrating Generative AI — Advanced RAG & Enterprise Patterns (Level 2)** training programme._
