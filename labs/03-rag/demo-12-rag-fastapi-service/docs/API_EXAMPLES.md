# API Examples - Demo 12

Practical examples for using the RAG FastAPI service.

## Getting Started

Start the server:
```bash
uv run uvicorn main:app --reload --port 8000
```

Base URL: `http://localhost:8000`

## Example 1: Basic RAG Workflow (cURL)

```bash
# Step 1: Ingest some documents
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Our company offers 15 vacation days per year, increasing to 20 days after 5 years of service. Vacation requests should be submitted at least 2 weeks in advance.",
    "metadata": {"source": "hr_policy", "section": "vacation"}
  }'

curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Remote work is permitted up to 3 days per week with manager approval. Employees must be available during core hours 10 AM - 3 PM Eastern time.",
    "metadata": {"source": "hr_policy", "section": "remote_work"}
  }'

# Step 2: Ask questions
curl -X POST http://localhost:8000/generate/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many vacation days do employees get?",
    "k": 3,
    "include_sources": true
  }' | jq '.'
```

## Example 2: File Upload (Python)

```python
import requests

BASE_URL = "http://localhost:8000"

# Upload a file
with open("company_policy.pdf", "rb") as f:
    files = {"file": ("company_policy.pdf", f, "application/pdf")}
    response = requests.post(f"{BASE_URL}/ingest/file", files=files)
    print(response.json())

# Result:
# {
#   "status": "success",
#   "filename": "company_policy.pdf",
#   "documents_loaded": 5,
#   "chunks_created": 47,
#   "message": "Successfully ingested company_policy.pdf into CHROMADB"
# }
```

## Example 3: Semantic Search with Filters

```bash
# Search only in specific documents
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "query": "vacation policy",
    "k": 5,
    "include_scores": true,
    "filter": {"section": "vacation"}
  }' | jq '.results[] | {score: .score, relevance: .relevance, source: .metadata.source}'
```

## Example 4: Python Client Class

```python
import requests
from typing import List, Dict, Any, Optional

class RAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health(self) -> Dict[str, Any]:
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ingest_text(self, text: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Ingest text content"""
        payload = {"text": text, "metadata": metadata or {}}
        response = requests.post(f"{self.base_url}/ingest/text", json=payload)
        response.raise_for_status()
        return response.json()
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a file"""
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(f"{self.base_url}/ingest/file", files=files)
            response.raise_for_status()
            return response.json()
    
    def search(self, query: str, k: int = 4, include_scores: bool = False) -> Dict[str, Any]:
        """Search for relevant chunks"""
        payload = {"query": query, "k": k, "include_scores": include_scores}
        response = requests.post(f"{self.base_url}/retrieve/similarity", json=payload)
        response.raise_for_status()
        return response.json()
    
    def ask(self, query: str, k: int = 4, include_sources: bool = True) -> Dict[str, Any]:
        """Ask a question using RAG"""
        payload = {
            "query": query,
            "k": k,
            "include_sources": include_sources,
            "temperature": 0.0
        }
        response = requests.post(f"{self.base_url}/generate/rag", json=payload)
        response.raise_for_status()
        return response.json()

# Usage
client = RAGClient()

# Check health
print(client.health())

# Ingest data
client.ingest_text(
    "Employees get 15 vacation days per year.",
    metadata={"source": "hr_policy"}
)

# Ask questions
result = client.ask("How many vacation days do employees get?")
print(f"Answer: {result['answer']}")
print(f"Used {result['context_count']} context chunks")
```

## Example 5: Batch Ingestion

```python
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

# Ingest all documents from a folder
docs_folder = Path("Documents")

for file_path in docs_folder.glob("*.{txt,pdf}"):
    print(f"Ingesting {file_path.name}...")
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{BASE_URL}/ingest/file", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Created {data['chunks_created']} chunks")
        else:
            print(f"  ✗ Error: {response.text}")
```

## Example 6: Interactive Q&A Session

```python
import requests

BASE_URL = "http://localhost:8000"

def ask_question(query: str):
    """Ask a question and display the answer"""
    response = requests.post(
        f"{BASE_URL}/generate/rag",
        json={
            "query": query,
            "k": 4,
            "include_sources": True,
            "temperature": 0.0
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nQuestion: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"\nSources used:")
        for i, source in enumerate(result.get('sources', []), 1):
            print(f"  [{i}] {source['metadata'].get('source', 'Unknown')}")
            print(f"      {source['content_preview'][:100]}...")
    else:
        print(f"Error: {response.text}")

# Interactive session
questions = [
    "How many vacation days do employees get?",
    "What is the remote work policy?",
    "What are the core working hours?",
]

for q in questions:
    ask_question(q)
    print("\n" + "="*70)
```

## Example 7: Async Python Client

```python
import aiohttp
import asyncio
from typing import List, Dict

class AsyncRAGClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    async def ask(self, query: str, k: int = 4) -> Dict:
        """Ask a question asynchronously"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate/rag",
                json={"query": query, "k": k, "include_sources": False}
            ) as response:
                return await response.json()
    
    async def ask_many(self, questions: List[str]) -> List[Dict]:
        """Ask multiple questions concurrently"""
        tasks = [self.ask(q) for q in questions]
        return await asyncio.gather(*tasks)

# Usage
async def main():
    client = AsyncRAGClient()
    
    questions = [
        "How many vacation days?",
        "What is the remote work policy?",
        "What are the benefits?",
    ]
    
    results = await client.ask_many(questions)
    
    for q, r in zip(questions, results):
        print(f"Q: {q}")
        print(f"A: {r['answer']}\n")

# Run
asyncio.run(main())
```

## Example 8: Testing Different Retrieval Strategies

```bash
# Strategy 1: Basic similarity search
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "vacation policy", "k": 3}' | jq '.count'

# Strategy 2: With relevance scores
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "vacation policy", "k": 3, "include_scores": true}' | \
  jq '.results[] | {score, relevance}'

# Strategy 3: MMR for diverse results
curl -X POST http://localhost:8000/retrieve/mmr \
  -H "Content-Type: application/json" \
  -d '{"query": "company policies", "k": 5, "fetch_k": 20}' | jq '.count'

# Strategy 4: With metadata filter
curl -X POST http://localhost:8000/retrieve/similarity \
  -H "Content-Type: application/json" \
  -d '{"query": "policies", "k": 3, "filter": {"source": "hr_policy"}}' | \
  jq '.results[] | .metadata.source'
```

## Example 9: Temperature Comparison

```python
import requests

BASE_URL = "http://localhost:8000"
query = "What are the benefits of remote work?"

# Test different temperatures
for temp in [0.0, 0.5, 1.0]:
    response = requests.post(
        f"{BASE_URL}/generate/rag",
        json={
            "query": query,
            "k": 3,
            "temperature": temp,
            "include_sources": False
        }
    )
    
    result = response.json()
    print(f"\nTemperature: {temp}")
    print(f"Answer: {result['answer'][:200]}...")
    print("-" * 70)
```

## Example 10: Error Handling

```python
import requests

BASE_URL = "http://localhost:8000"

def safe_ask(query: str) -> str:
    """Ask a question with proper error handling"""
    try:
        response = requests.post(
            f"{BASE_URL}/generate/rag",
            json={"query": query, "k": 3},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result['answer']
        
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to API server"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return "Error: No documents found. Please ingest documents first."
        else:
            return f"Error: {e.response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Usage
answer = safe_ask("How many vacation days?")
print(answer)
```

## Tips

### Performance
- Use `include_sources=False` when you only need the answer
- Adjust `k` based on your needs (lower is faster)
- Use metadata filters to narrow search scope

### Accuracy
- Start with `temperature=0.0` for consistent answers
- Increase `k` if answers lack context
- Use MMR when you need diverse perspectives

### Production
- Add authentication headers
- Implement retry logic
- Monitor response times
- Cache frequent queries

## Next Steps

1. Explore [Swagger UI](http://localhost:8000/docs) for interactive testing
2. Check [README.md](../README.md) for complete API reference
3. Review [QUICKSTART.md](QUICKSTART.md) for setup guide
