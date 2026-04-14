"""
Text Embedding Generator - Consolidated Main Module

This module demonstrates text embedding generation and semantic similarity analysis
using OpenAI and LangChain in a single, unified file.

COMPLETE WORKFLOW:
Step 1: Configuration & Initialization (runs automatically on import)
Step 2: Text to Vector Conversion (get_embedding function)
Step 3: Similarity Calculation (cosine_similarity function)
Step 4: Sample execution with test sentences (when run as main)
"""
import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: CONFIGURATION & INITIALIZATION
# ============================================================================
# Load environment variables from .env file
try:
    load_dotenv()
    logger.info("✓ Environment variables loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load .env file: {e}")
    raise

# Extract configuration values from environment variables
OPENAI_API_EMBEDDING_KEY = os.getenv("OPENAI_API_EMBEDDING_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")  # Default model

# Validate all required environment variables are present
required_vars = {
    "OPENAI_API_EMBEDDING_KEY": OPENAI_API_EMBEDDING_KEY
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}. Please check your .env file."
    logger.error(f"✗ {error_msg}")
    raise ValueError(error_msg)

logger.info("✓ All required environment variables validated")

# Initialize OpenAI Embeddings model
try:
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_EMBEDDING_KEY,
        model=OPENAI_MODEL
    )
    logger.info(f"✓ OpenAI Embeddings model initialized successfully (model: {OPENAI_MODEL})")
except Exception as e:
    logger.error(f"✗ Failed to initialize OpenAIEmbeddings: {e}")
    raise


# ============================================================================
# STEP 2: TEXT TO VECTOR CONVERSION FUNCTION
# ============================================================================
def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text string.

    This function converts text into a high-dimensional vector representation
    that captures semantic meaning. The embedding can be used for similarity
    comparison with other text embeddings.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector (typically 1536 dimensions)

    Raises:
        ValueError: If text is empty or None, or if API returns empty response
        Exception: If API call fails
    """
    try:
        # Validate input text - prevents unnecessary API calls and catches errors early
        if not text or (isinstance(text, str) and not text.strip()):
            raise ValueError("Text cannot be empty or None")

        # Generate embedding using OpenAI API via LangChain
        embedding = embeddings_model.embed_query(text)

        # Validate the response - ensure we received valid data from the API
        if not embedding or len(embedding) == 0:
            raise ValueError("Empty embedding received from API")

        # Return the embedding vector
        return embedding

    except Exception as e:
        # Error handling: Log error safely without exposing full text
        text_preview = str(text)[:50] if text else "None or empty"
        logger.error(f"✗ Failed to get embedding for text '{text_preview}...': {str(e)}")
        raise


# ============================================================================
# STEP 3: SIMILARITY CALCULATION FUNCTION
# ============================================================================
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    This function implements the cosine similarity formula:
        similarity = (A · B) / (||A|| × ||B||)
    
    Where:
    - A · B is the dot product (sum of element-wise products)
    - ||A|| and ||B|| are the magnitudes (L2 norms = Euclidean distance from origin)
    
    The result measures the cosine of the angle between two vectors:
    - 1.0 = identical direction (most similar)
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite direction (least similar)
    
    For text embeddings, values typically range from 0.0 to 1.0 (rarely negative).

    Args:
        vec1: First vector as list of floats
        vec2: Second vector as list of floats

    Returns:
        Float between -1 and 1 (higher = more similar)

    Raises:
        ValueError: If vectors are empty, None, or have different dimensions
        ZeroDivisionError: If either vector has zero magnitude
    """
    try:
        # Validate both vectors - check they're not None, not empty, and have same dimensions
        if not vec1 or not vec2:
            raise ValueError("Vectors cannot be None or empty")

        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must have same dimensions. Got {len(vec1)} and {len(vec2)}")

        if len(vec1) == 0:
            raise ValueError("Vectors cannot be empty")

        # Convert lists to NumPy arrays - provides efficient mathematical operations for large vectors
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        # Calculate dot product (A · B) - sum of element-wise products: Σ(A[i] × B[i])
        dot_product = np.dot(vec1_array, vec2_array)
        
        # Calculate magnitudes (||A|| and ||B||) - L2 norm = Euclidean distance from origin = √(Σ(x²))
        magnitude1 = np.linalg.norm(vec1_array)
        magnitude2 = np.linalg.norm(vec2_array)

        # Check for zero magnitude (edge case) - prevents division by zero error
        if magnitude1 == 0 or magnitude2 == 0:
            raise ZeroDivisionError("Cannot calculate cosine similarity with zero-magnitude vector")

        # Apply cosine similarity formula - final calculation: (A · B) / (||A|| × ||B||)
        similarity = float(dot_product / (magnitude1 * magnitude2))
        return similarity

    except Exception as e:
        # Log unexpected errors
        logger.error(f"✗ Unexpected error calculating cosine similarity: {str(e)}")
        raise


# ============================================================================
# STEP 4: MAIN EXECUTION - DEMONSTRATION WITH SAMPLE SENTENCES
# ============================================================================
if __name__ == "__main__":
    try:
        # Define test sentences with varying semantic similarity
        sentences = {
            "tech1": "Artificial intelligence is transforming industries",
            "tech2": "Machine learning revolutionizes business processes",
            "food1": "Pizza is a delicious Italian dish",
            "animal1": "Dogs are loyal and friendly companions"
        }
        logger.info("Step 2: Defined sample sentences.")

        logger.info("=" * 70)
        logger.info("GENERATING EMBEDDINGS")
        logger.info("=" * 70)

        # Generate embeddings for all sentences
        embeddings = {}
        logger.info("Step 3: Starting embedding generation...")

        for key, sentence in sentences.items():
            try:
                logger.info(f"\nProcessing: '{sentence}'")
                embedding = get_embedding(sentence)
                embeddings[key] = embedding
                logger.info(f"  ✓ Generated embedding for '{key}'")

                # Display embedding properties
                logger.info(f"  → Embedding dimensions: {len(embedding)}")
                logger.info(f"  → First 10 values: {embedding[:10]}...")
                logger.info(f"  → Vector magnitude: {np.linalg.norm(embedding):.4f}")

            except Exception as e:
                logger.error(f"✗ Failed to process sentence '{key}': {str(e)}")
                continue

        if not embeddings:
            logger.error("No embeddings were successfully generated. Cannot proceed with similarity analysis.")
            sys.exit(1)
        logger.info("Step 3 & 4 Complete: All available embeddings generated and inspected.")

        # Calculate semantic similarities between sentence pairs
        logger.info("\n" + "=" * 70)
        logger.info("SEMANTIC SIMILARITY ANALYSIS")
        logger.info("=" * 70)

        comparisons = [
            ("tech1", "tech2", "Two tech-related sentences (expected high similarity)"),
            ("tech1", "food1", "Tech vs Food (expected low similarity)"),
            ("tech1", "animal1", "Tech vs Animals (expected low similarity)"),
            ("food1", "animal1", "Food vs Animals (expected medium similarity)")
        ]
        logger.info("Step 5: Starting similarity calculations...")

        for key1, key2, description in comparisons:
            try:
                if key1 not in embeddings or key2 not in embeddings:
                    logger.warning(f"Skipping comparison {key1} vs {key2} - missing embeddings")
                    continue

                similarity = cosine_similarity(embeddings[key1], embeddings[key2])
                logger.info(f"  ✓ Calculated similarity for '{key1}' vs '{key2}'")

                # Display similarity results
                logger.info(f"\n{description}:")
                logger.info(f"  '{sentences[key1]}'")
                logger.info("  vs")
                logger.info(f"  '{sentences[key2]}'")
                logger.info(f"  → Similarity: {similarity:.4f}")

            except Exception as e:
                logger.error(f"✗ Failed to calculate similarity between {key1} and {key2}: {str(e)}")
                continue

        logger.info("Step 5 & 6 Complete: All similarity analyses performed and results displayed.")
        logger.info("\n" + "=" * 70)
        logger.info("EMBEDDING ANALYSIS COMPLETE")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
