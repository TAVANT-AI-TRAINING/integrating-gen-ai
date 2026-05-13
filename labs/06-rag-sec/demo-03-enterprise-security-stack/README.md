# Demo 03: Full Enterprise RAG Security Stack

**Level:** Advanced  
**Topic:** Production-grade security, governance, and responsible AI

Assembles all 11 security layers into a single pipeline modelled on the requirements
for enterprise RAG deployment — suitable as the basis for a security review sign-off.

---

## Security Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────┐
│ INPUT GUARDRAILS (from Demo 01)             │
│   Layer 1 — Regex injection detection       │
│   Layer 2 — LLM-as-guard detector           │
│   Layer 3 — Scope enforcement               │
└─────────────────────────────────────────────┘
     │ (passes if not blocked)
     ▼
┌─────────────────────────────────────────────┐
│ DATA PROTECTION (from Demo 02)              │
│   Layer 4 — RBAC (Chroma metadata filter)   │
│   Layer 5 — PII redaction at query time     │
│   Layer 6 — PII redaction at response time  │
│   Layer 7 — Source attribution sanitisation │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│ RESPONSIBLE AI (new in Demo 03)             │
│   Layer 8 — Mandatory disclaimers           │
│   Layer 9 — Human escalation               │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│ GOVERNANCE (new in Demo 03)                 │
│  Layer 10 — JSONL audit trail (all events)  │
│  Layer 11 — Content policy scan at ingest   │
└─────────────────────────────────────────────┘
     │
     ▼
 Response to User
```

---

## Quick Start

```bash
cp .env.example .env        # add your OPENAI_API_KEY
uv sync
uv run uvicorn main:app --reload --port 8003
uv run python test_api.py
```

Open **http://localhost:8003/docs**

---

## Key Experiments

### 1. Run the full test suite (10 test cases, covers all layers)
```bash
curl -s -X POST http://localhost:8003/security/run-tests | python3 -m json.tool
```

### 2. Query as different users — RBAC in action
```bash
# Junior employee — gets no salary data
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the salary ranges?","user":{"user_id":"alice","clearance":0,"role":"employee"}}' \
  | python3 -m json.tool

# HR Admin — gets full compensation bands
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the salary ranges?","user":{"user_id":"hr_bob","clearance":3,"role":"hr_admin"}}' \
  | python3 -m json.tool
```

### 3. Responsible AI — see escalation trigger
```bash
curl -s -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"query":"I am facing a dismissal hearing. What should I do?","user":{"user_id":"emp1","clearance":1,"role":"employee"}}' \
  | python3 -m json.tool
```
The response will be an HR specialist referral instead of a direct AI answer.

### 4. View the audit log report
```bash
curl -s "http://localhost:8003/audit/report?hours=1" | python3 -m json.tool
```
The JSONL log at `./audit/rag_audit.jsonl` records every request including:
- User identity and clearance
- PII redacted query (not the raw query)
- Retrieved source document names
- Security events (injection attempts, scope rejections)
- Escalations and disclaimers applied

---

## Audit Event Schema

```json
{
  "event_id":           "uuid4",
  "timestamp":          "2025-01-01T12:00:00+00:00",
  "user_id":            "alice",
  "user_clearance":     1,
  "query_redacted":     "What is the leave policy for <PERSON>?",
  "query_pii_found":    true,
  "query_pii_types":    ["email"],
  "retrieved_sources":  ["leave_policy.txt"],
  "answer_pii_found":   false,
  "disclaimers_added":  [],
  "escalated":          false,
  "injection_detected": false,
  "scope_rejected":     false,
  "latency_ms":         342.1
}
```

---

## Responsible AI Controls

| Trigger Keywords | Disclaimer Added |
|-----------------|-----------------|
| disciplinary, grievance, dismissal, redundancy | HR specialist referral |
| legal, liability, contract | Legal advice disclaimer |
| medical, diagnosis, treatment | Medical advice disclaimer |
| investment, pension, tax | Financial advice disclaimer |

**Human escalation** fires when high-risk HR keywords are detected in the query
(grievance, dismissal, discrimination, harassment, etc.). Instead of returning
a direct answer, the system issues a ticket reference and routes to an HR specialist.

---

## Enterprise Governance Checklist

Before deploying to production, verify:

```
Security Review
  [x] Layer 1: Regex injection patterns tested against 12+ attack cases
  [x] Layer 2: LLM guard catches sophisticated attacks
  [x] Layer 3: Scope boundary clearly defined for this use case
  [x] Layer 4: RBAC filter verified — low clearance cannot retrieve high clearance docs
  [x] Layer 5-6: PII redacted in both query and response
  [x] Layer 7: No CONFIDENTIAL labels in source citations

Responsible AI
  [x] Layer 8: Disclaimers cover medical, legal, financial, and HR-process topics
  [x] Layer 9: Escalation path tested end-to-end (ticket created, user informed)

Governance
  [x] Layer 10: Audit log covers all required fields (who, what, when, retrieved, generated)
  [x] Layer 11: Document content scan rejects poisoned documents at ingest

Compliance
  [ ] Log retention policy defined (1 year minimum for most regulations)
  [ ] Right-to-erasure process implemented (purge user events by user_id)
  [ ] DPA signed with OpenAI, Chroma/Pinecone provider
  [ ] Penetration test performed against /query endpoint
```

---

## Learning Path

| Demo | Level | Covers |
|------|-------|--------|
| Demo 01 | Beginner | Prompt injection defences (Layers 1–4 input guards) |
| Demo 02 | Intermediate | RBAC + PII protection (Layers 4–7) |
| Demo 03 | Advanced | Full stack (Layers 1–11) + audit + responsible AI |
