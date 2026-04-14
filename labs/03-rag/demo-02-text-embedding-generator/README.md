# Demo 2: Text Embedding Generator - OpenAI Embeddings

A Python application demonstrating how to generate **text embeddings** and calculate **semantic similarities** using **OpenAI** and LangChain. This example converts text into numerical vectors and compares them using cosine similarity to measure semantic relationships.

## Objective

To demonstrate the text embedding generation workflow:

1. **Initialize OpenAI Embeddings**: Configure LangChain's OpenAIEmbeddings client
2. **Generate Embeddings**: Convert text strings into high-dimensional vector representations
3. **Calculate Similarity**: Use cosine similarity to measure semantic relationships between texts
4. **Validate Inputs**: Handle edge cases with comprehensive error handling
5. **Display Results**: Show structured output with logging instead of print statements

## Project Structure

```
demo-2-text-embedding-generator/
├── .env                    # Environment variables (OpenAI API key)
├── .env.example            # Environment template
├── main.py                 # Single consolidated module with all functionality
├── pyproject.toml          # UV dependency configuration
├── uv.lock                 # Dependency lock file
├── README.md               # This file
└── .python-version         # Python version specification
```

## Setup Instructions

1. **Install Dependencies**

   ```bash
   uv sync
   ```

2. **Environment Configuration**

   Create a `.env` file in the project root:

   ```env
   OPENAI_API_EMBEDDING_KEY=your_openai_api_key_here
   OPENAI_MODEL=text-embedding-3-small
   ```

   **Get your API key:** [OpenAI Platform](https://platform.openai.com/api-keys)

3. **Run the Script**

   ```bash
   # Run the main script with example sentences
   uv run python main.py

   # Or run as module
   uv run python -m main
   ```

   The script will process sample sentences and display embedding statistics and similarity scores.

## Features

- ✅ **Text to Vector Conversion**: Generates high-dimensional embeddings using OpenAI
- ✅ **Semantic Similarity Calculation**: Computes cosine similarity between text pairs
- ✅ **Modular Function Design**: Reusable `get_embedding()` and `cosine_similarity()` functions
- ✅ **Structured Logging**: Clean, formatted output with timestamps instead of print statements
- ✅ **Input Validation**: Comprehensive error handling for empty text and invalid vectors
- ✅ **Environment-Based Configuration**: Secure API key management via `.env` file
- ✅ **OpenAI Integration**: Uses LangChain's OpenAIEmbeddings for embedding generation
- ✅ **Type Safety**: Proper type hints and validation throughout
- ✅ **Single File Architecture**: All functionality in one consolidated, easy-to-understand module

## Usage

### Basic Usage

Run the script to generate embeddings and calculate similarities:

```bash
uv run python main.py
```

### Programmatic Usage

#### Generate Embeddings

```python
from main import get_embedding, cosine_similarity

# Generate embedding for a single text
embedding = get_embedding("Artificial intelligence is transforming industries")
print(f"Embedding dimensions: {len(embedding)}")
```

#### Calculate Similarity

```python
from main import get_embedding, cosine_similarity

# Calculate similarity between two texts
text1 = "Artificial intelligence is transforming industries"
text2 = "Machine learning revolutionizes business processes"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
similarity = cosine_similarity(emb1, emb2)

print(f"Similarity: {similarity:.4f}")  # Higher values = more similar
```

## Use Cases

This text embedding functionality is useful for:

- **Semantic Search**: Finding documents or content similar to a query
- **Recommendation Systems**: Suggesting similar items based on content
- **Clustering**: Grouping similar texts together
- **Classification**: Categorizing documents by semantic content
- **RAG Systems**: Building retrieval-augmented generation pipelines
- **Content Analysis**: Understanding relationships between different texts

## Expected Behavior

When you run the script, you'll observe:

- ✅ **Embedding Generation**: Text converted to 1536-dimensional vectors
- ✅ **Vector Validation**: Proper dimension checking and magnitude calculation
- ✅ **Similarity Scores**: Cosine similarity values between -1 and 1
- ✅ **Structured Output**: Clean logging with timestamps and formatted results
- ✅ **Error Handling**: Graceful handling of empty inputs and API failures

## Expected Output

When you run the script, you'll see output like:

```
2024-01-15 10:30:15,123 - INFO - ======================================================================
2024-01-15 10:30:15,124 - INFO - GENERATING EMBEDDINGS
2024-01-15 10:30:15,125 - INFO - ======================================================================

2024-01-15 10:30:15,126 - INFO - Processing: Artificial intelligence is transforming industries
2024-01-15 10:30:16,234 - INFO -   → Embedding dimensions: 1536
2024-01-15 10:30:16,235 - INFO -   → First 10 values: [-0.1234, 0.5678, 0.9012, ...]
2024-01-15 10:30:16,236 - INFO -   → Vector magnitude: 1.0000

2024-01-15 10:30:16,237 - INFO - ======================================================================
2024-01-15 10:30:16,238 - INFO - SEMANTIC SIMILARITY ANALYSIS
2024-01-15 10:30:16,239 - INFO - ======================================================================

2024-01-15 10:30:16,240 - INFO - Two tech-related sentences:
2024-01-15 10:30:16,241 - INFO -   'Artificial intelligence is transforming industries'
2024-01-15 10:30:16,242 - INFO -   vs
2024-01-15 10:30:16,243 - INFO -   'Machine learning revolutionizes business processes'
2024-01-15 10:30:16,244 - INFO -   → Similarity: 0.8547
```

## Understanding Similarity Scores

Try different inputs to see how semantic similarity works:

**Examples:**

| Text Pair                                                  | Expected Similarity | Interpretation                    |
| ---------------------------------------------------------- | ------------------- | --------------------------------- |
| "AI transforms industries" vs "ML revolutionizes business" | 0.85+               | High similarity (same domain)     |
| "AI transforms industries" vs "Pizza is delicious"         | 0.10-0.30           | Low similarity (unrelated)        |
| "Dogs are loyal" vs "Cats are independent"                 | 0.50-0.70           | Medium similarity (same category) |
