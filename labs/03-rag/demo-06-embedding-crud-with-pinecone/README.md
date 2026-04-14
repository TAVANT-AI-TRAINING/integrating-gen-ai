# Demo 06: Embedding CRUD with Pinecone

A simple demonstration of CRUD (Create, Read, Update, Delete) operations with **Pinecone**, a managed cloud vector database, using OpenAI embeddings and FastAPI.

## 🎯 What You'll Learn

- How to use Pinecone (cloud vector database)
- Differences between Pinecone and ChromaDB
- Generate embeddings using OpenAI
- Perform CRUD operations with Pinecone
- Semantic similarity search in the cloud
- Build a RESTful API with FastAPI

## 🌟 Why Pinecone?

**Pinecone vs ChromaDB (Demo 04):**

| Feature         | ChromaDB (Demo 04)    | Pinecone (Demo 06)       |
| --------------- | --------------------- | ------------------------ |
| **Hosting**     | Local files           | Managed cloud            |
| **Scaling**     | Manual                | Automatic                |
| **Maintenance** | You handle it         | Pinecone handles it      |
| **Performance** | Good for dev          | Optimized for production |
| **Cost**        | Free                  | Free tier + paid plans   |
| **Best for**    | Development, learning | Production, scale        |

**Use Pinecone when:**

- ✅ Building production applications
- ✅ Need automatic scaling
- ✅ Want fully managed infrastructure
- ✅ Need high availability
- ✅ Global deployment required

**Use ChromaDB when:**

- ✅ Learning and development
- ✅ Local prototyping
- ✅ No cloud dependency needed
- ✅ Simple use cases
- ✅ Cost is primary concern

## 🚀 Setup

### 1. Get Pinecone API Key

1. Go to [https://www.pinecone.io/](https://www.pinecone.io/)
2. Sign up for free account
3. Create a new project
4. Copy your API key from the dashboard

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
OPENAI_API_EMBEDDING_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=company-policies
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

**Note:** The index will be created automatically on first run!

### 4. Run the Server

```bash
uv run python main.py
```

The server will start at `http://localhost:8000`

## 📚 API Endpoints

### Health Check

```bash
curl http://localhost:8000/
```

### CREATE - Add a Document

```bash
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-001",
    "text": "All employees must complete annual security training.",
    "metadata": {"category": "security", "date": "2024-01-15"}
  }'
```

### READ - Get a Document

```bash
curl http://localhost:8000/documents/policy-001
```

### UPDATE - Update a Document

```bash
curl -X PUT http://localhost:8000/documents/policy-001 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "All employees must complete annual security training and pass the assessment.",
    "metadata": {"category": "security", "updated": "2024-02-01"}
  }'
```

### DELETE - Remove a Document

```bash
curl -X DELETE http://localhost:8000/documents/policy-001
```

### QUERY - Semantic Search

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What are the security requirements?",
    "n_results": 3
  }'
```

### LIST - Get Document Count

```bash
curl http://localhost:8000/documents
```

**Note:** Pinecone doesn't support listing all vectors directly. This is a design choice for performance at scale.

## 🧪 Try It Out

### Example: Add Company Policies

```bash
# Add vacation policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-vacation",
    "text": "Employees receive 15 days of paid vacation annually.",
    "metadata": {"category": "benefits", "department": "HR"}
  }'

# Add remote work policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-remote",
    "text": "Remote work is allowed up to 3 days per week with manager approval.",
    "metadata": {"category": "workplace", "department": "HR"}
  }'

# Add security policy
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "policy-security",
    "text": "All company data must be encrypted and passwords must be changed every 90 days.",
    "metadata": {"category": "security", "department": "IT"}
  }'

# Search for related policies
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "Can I work from home?",
    "n_results": 2
  }'
```

## 🔍 Interactive API Documentation

FastAPI provides automatic interactive documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Project Structure

```
demo-06-embedding-crud-with-pinecone/
├── main.py              # Pinecone implementation
├── pyproject.toml       # Dependencies (includes pinecone-client)
├── .env.example         # Environment template
├── .env                 # Your configuration (git-ignored)
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── QUICKSTART.md        # Quick setup guide
└── PINECONE_VS_CHROMA.md # Comparison guide
```

## 🧠 How Pinecone Works

### 1. Cloud-Based Storage

Unlike ChromaDB (local files), Pinecone stores everything in the cloud:

- Vectors stored in managed infrastructure
- Automatic backups and replication
- Global availability

### 2. Serverless Architecture

```python
# Index created with serverless spec
pc.create_index(
    name="company-policies",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)
```

### 3. Upsert Operation

Pinecone uses "upsert" (insert or update):

```python
index.upsert(
    vectors=[{
        "id": "doc-001",
        "values": [0.1, 0.2, ...],  # Embedding
        "metadata": {"text": "...", "category": "..."}
    }]
)
```

### 4. Metadata Storage

Pinecone stores the original text in metadata:

```python
metadata = {
    "text": "Original document text",
    "category": "security",
    "date": "2024-01-15"
}
```

### 5. Fast Similarity Search

```python
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)
```

## 💡 Key Concepts

### What is Pinecone?

A fully managed vector database optimized for production use cases.

### Serverless vs Pod-Based

- **Serverless**: Auto-scaling, pay-per-use (what we use)
- **Pod-Based**: Dedicated resources, predictable costs

### Index

A collection of vectors with the same dimension and metric.

### Upsert

Insert or update operation - idempotent and efficient.

### Metadata Filtering

Filter search results based on metadata (not shown in basic demo).

## 🔧 Troubleshooting

### Error: "PINECONE_API_KEY not found"

Make sure you've created `.env` file with your Pinecone API key.

### Error: "Index already exists"

The code handles this automatically - it will use the existing index.

### Error: Port 8000 already in use

Stop other servers or change the port in `main.py`.

### Slow First Request

First query might be slow as Pinecone initializes the index.

## 🎓 Learning Notes

### Simple Design

This demo uses a single file to make it easier to understand. In production, you'd typically split into modules.

### Free Tier Limits

Pinecone free tier includes:

- 1 serverless index
- 100K vectors
- Sufficient for learning and small projects

### No "List All" Operation

Pinecone doesn't have a native "list all vectors" operation. This is by design for performance at scale. In production:

- Maintain a separate database for document listings
- Use query with filters
- Store document IDs externally

### Metadata Storage

Pinecone has a metadata size limit (~40KB per vector). For large documents, store only references and keep full text elsewhere.

## 📊 Performance Characteristics

| Operation  | Latency   | Notes                   |
| ---------- | --------- | ----------------------- |
| **Upsert** | ~50-200ms | Network overhead        |
| **Fetch**  | ~20-100ms | Single vector retrieval |
| **Query**  | ~50-200ms | Similarity search       |
| **Delete** | ~50-100ms | Immediate               |

## 🚀 Production Considerations

### Scaling

Pinecone automatically scales based on usage:

- No manual configuration needed
- Handles millions of vectors
- Global distribution available

### Cost Management

- Free tier: Good for development
- Paid tiers: Based on usage and performance
- Monitor usage in Pinecone dashboard

### Best Practices

1. **Batch operations**: Upsert multiple vectors at once
2. **Use namespaces**: Organize vectors by namespace
3. **Metadata filtering**: Reduce search space
4. **Monitor costs**: Check dashboard regularly
5. **Use sparse-dense hybrid**: For better search quality

## 🔄 Migration from ChromaDB

If you built with ChromaDB (Demo 04) and want to move to Pinecone:

1. **Export from ChromaDB**: Get all vectors and metadata
2. **Create Pinecone index**: Same dimension as ChromaDB
3. **Batch upsert**: Upload all vectors to Pinecone
4. **Update code**: Replace ChromaDB calls with Pinecone
5. **Test thoroughly**: Verify all operations work

## 📚 Next Steps

After mastering this demo:

1. Add metadata filtering to queries
2. Implement batch operations
3. Use namespaces for multi-tenancy
4. Add sparse-dense hybrid search
5. Monitor and optimize costs
6. Try different cloud providers (AWS, GCP, Azure)

## 🤝 Index Name

This demo uses `company-policies` as the index name, perfect for storing organizational policies and documentation.

## 🔗 Useful Links

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone Python Client](https://github.com/pinecone-io/pinecone-python-client)
- [Pinecone Dashboard](https://app.pinecone.io/)
- [Pricing](https://www.pinecone.io/pricing/)
