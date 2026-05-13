# Demo 02: Access Control & PII Protection

**Level:** Intermediate  
**Topic:** Role-Based Retrieval and PII Redaction

A RAG system without access control leaks confidential documents to any authenticated
user who asks the right question. This demo shows how to prevent it.

---

## The Problem

```
Without RBAC:
  User (junior analyst, clearance=0) asks: "What is the salary for a senior manager?"
  RAG retrieves chunks from level3_compensation_bands.txt (CONFIDENTIAL)
  LLM reveals: "Senior Managers earn £110,000 – £160,000 plus 20-35% bonus"
  → GDPR violation, HR data breach

With RBAC:
  Same query, same user (clearance=0)
  Chroma filter: { clearance_level: { $lte: 0 } }
  RAG retrieves NOTHING from level3 document
  LLM responds: "I don't have sufficient information..."
  → Protected
```

---

## Document Clearance Levels

| Level | Label | Who Can Access |
|-------|-------|---------------|
| 0 | PUBLIC | Everyone |
| 1 | INTERNAL | All employees |
| 2 | RESTRICTED | Managers and above |
| 3 | CONFIDENTIAL | HR admins and executives only |

---

## PII Redaction — Three Layers

| When | What | How |
|------|------|-----|
| Query time | PII in user's search query | Redact before embedding |
| Response time | PII in the LLM answer | Redact before returning to user |
| Source citation | Classification labels in filenames | Replace with "Internal HR Document" |

> **Production upgrade:** Replace regex patterns with
> [Microsoft Presidio](https://github.com/microsoft/presidio) for language-aware
> PII detection: `pip install presidio-analyzer presidio-anonymizer spacy`

---

## Quick Start

```bash
cp .env.example .env        # add your OPENAI_API_KEY
uv sync
uv run uvicorn main:app --reload --port 8002
uv run python test_api.py
```

Open **http://localhost:8002/docs**

---

## Key Experiments

### 1. RBAC in action — same query, different clearance
```bash
# Junior employee (clearance=0) — should get NO salary data
curl -s -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the salary for a senior manager?","user":{"user_id":"alice","clearance":0,"role":"employee"}}' \
  | python3 -m json.tool

# HR Admin (clearance=3) — should get full compensation bands
curl -s -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the salary for a senior manager?","user":{"user_id":"bob","clearance":3,"role":"hr_admin"}}' \
  | python3 -m json.tool
```

### 2. Leak risk comparison — see exactly what RBAC prevents
```bash
curl -s "http://localhost:8002/demo/leak-risk?query=salary+senior+manager" \
  | python3 -m json.tool
```

### 3. PII redaction in query
```bash
curl -s -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the sick leave policy for John Smith (SSN 123-45-6789)?",
    "user": {"user_id": "test", "clearance": 1, "role": "employee"}
  }' | python3 -m json.tool
```
Check `query_used` in the response — the SSN will be `[REDACTED:SSN_US]`.

---

## How RBAC Works in Chroma

```python
# At ingest: tag every document with its clearance level
chunk.metadata["clearance_level"] = 3   # CONFIDENTIAL

# At query time: filter to user's clearance
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"clearance_level": {"$lte": user.clearance}}
    }
)
# A user with clearance=1 can only retrieve documents with clearance_level <= 1
```

The key insight: **access control must live in the vector database, not the
application layer**. Application-layer checks can be bypassed; metadata filters
in Chroma/Pinecone are enforced at the storage level.

---

## Next Steps

- **Demo 03** — Full enterprise stack: injection + RBAC + PII + audit logging + responsible AI (advanced)
