from sentence_transformers import SentenceTransformer


def main():
    # This model can be downloaded and used locally
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [
        "Hello world",
        "Embeddings can be used for semantic search",
        "Local models keep your data on your machine.",
    ]

    # Generate embeddings - wraps tokenization and embedding
    embeddings = model.encode(texts)

    for i, emb in enumerate(embeddings):
        print(f"Text: {texts[i]}")
        print(f"Embedding Dimension: {len(emb)}")
        print(emb[:10], "...")

    # Here we are only tokenizing to show token lengths
    tokens = [model.tokenizer.encode(text) for text in texts]

    for i, token in enumerate(tokens):
        print(f"Text: {texts[i]}")
        print(f"Tokens: {token}, Token Length: {len(token)}")


if __name__ == "__main__":
    main()
