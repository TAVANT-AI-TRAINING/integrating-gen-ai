# Demo 1: Sentence Transformers - Local Embedding Generation

A Python application demonstrating how to generate **text embeddings** using a local **SentenceTransformer** model, without requiring external API calls. This example shows how to work with embeddings entirely on your local machine.

## Objective

To demonstrate the local embedding generation workflow:

1. **Load Local Model**: Initialize and download a SentenceTransformer model (`all-MiniLM-L6-v2`)
2. **Generate Embeddings**: Convert text strings into high-dimensional vector representations
3. **Tokenization Analysis**: Show token IDs and token lengths for input texts
4. **Display Results**: Show embedding dimensions and sample values

## Project Structure

```
demo-1-sentence-transformers/
├── main.py                 # Main execution script
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

   Or using pip:

   ```bash
   pip install sentence-transformers>=5.2.0
   ```

2. **Run the Script**

   ```bash
   # Using uv
   uv run python main.py

   # Or directly with Python
   python main.py
   ```

   The script will automatically download the model on first run and process sample texts.

## Features

- ✅ **Local Model Usage**: Works entirely offline after initial model download
- ✅ **Text to Vector Conversion**: Generates 384-dimensional embeddings using SentenceTransformer
- ✅ **Tokenization Analysis**: Shows token IDs and token lengths for each text
- ✅ **No API Keys Required**: All processing happens locally on your machine
- ✅ **Fast and Efficient**: Optimized model for speed and efficiency

## Usage

### Basic Usage

Run the script to generate embeddings for sample texts:

```bash
uv run python main.py
```

### Programmatic Usage

```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = ["Hello world", "Embeddings can be used for semantic search"]
embeddings = model.encode(texts)

# Access embeddings
for i, emb in enumerate(embeddings):
    print(f"Text: {texts[i]}")
    print(f"Embedding Dimension: {len(emb)}")
    print(f"First 10 values: {emb[:10]}")
```

## Use Cases

This local embedding functionality is useful for:

- **Offline Development**: Work with embeddings without internet connectivity
- **Privacy-Sensitive Applications**: Keep all data processing on your local machine
- **Cost-Effective Solutions**: No API costs for embedding generation
- **Rapid Prototyping**: Quick iteration without external dependencies
- **Semantic Search**: Building local semantic search systems
- **Content Analysis**: Understanding relationships between different texts

## Expected Behavior

When you run the script, you'll observe:

- ✅ **Model Download**: Automatic download of `all-MiniLM-L6-v2` model on first run
- ✅ **Embedding Generation**: Text converted to 384-dimensional vectors
- ✅ **Tokenization**: Token IDs and lengths displayed for each text
- ✅ **Structured Output**: Clean display of embedding dimensions and sample values

## Expected Output

When you run the script, you'll see output like:

```
Text: Hello world
Embedding Dimension: 384
[0.1234, -0.5678, 0.9012, ...] ...
Text: Hello world
Tokens: [101, 7592, 2088, 102], Token Length: 4

Text: Embeddings can be used for semantic search
Embedding Dimension: 384
[0.2345, -0.6789, 0.0123, ...] ...
Text: Embeddings can be used for semantic search
Tokens: [101, 4567, 8901, ...], Token Length: 12
```

## Model Information

The `all-MiniLM-L6-v2` model:

- **Dimensions**: Produces 384-dimensional embeddings
- **Optimization**: Optimized for speed and efficiency
- **Offline Capability**: Works entirely offline after initial download
- **Use Cases**: Suitable for semantic search and similarity tasks
- **License**: Apache 2.0
