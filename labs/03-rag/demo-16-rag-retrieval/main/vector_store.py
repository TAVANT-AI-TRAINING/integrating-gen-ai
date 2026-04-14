import os
import sys
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_env() -> Dict[str, str]:
    """
    Load and validate required environment variables.
    
    Returns:
        Dictionary containing environment variables
    
    Raises:
        SystemExit: If required environment variables are missing
    """
    load_dotenv()
    required_envs = [
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_OPENAI_API_VERSION",
    ]
    missing = [name for name in required_envs if not os.getenv(name)]
    if missing:
        logger.error("Error: Missing required environment variables:")
        for name in missing:
            logger.error(f" - {name}")
        logger.error(
            "Please create a .env file based on .env.example and set the missing variables."
        )
        sys.exit(1)

    return {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT"),
        "DB_NAME": os.getenv("DB_NAME"),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "company_policies"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }


def get_connection_string(env: Dict[str, str]) -> str:
    """
    Build PostgreSQL connection string from environment variables.
    
    Args:
        env: Dictionary containing environment variables
    
    Returns:
        PostgreSQL connection string
    """
    return (
        f"postgresql+psycopg://{env['DB_USER']}:{env['DB_PASSWORD']}@"
        f"{env['DB_HOST']}:{env['DB_PORT']}/{env['DB_NAME']}"
    )


essential_embedding_keys = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


def get_embeddings(env: Dict[str, str]) -> Any:
    """
    Initialize Azure OpenAI embeddings from environment variables.
    
    Args:
        env: Dictionary containing environment variables
    
    Returns:
        AzureOpenAIEmbeddings instance
    
    Raises:
        SystemExit: If required embedding configuration is missing
    """
    for k in essential_embedding_keys:
        if not env.get(k):
            logger.error("Error: Missing Azure OpenAI configuration.")
            logger.error(f"Missing: {k}")
            sys.exit(1)
    return AzureOpenAIEmbeddings(
        azure_endpoint=env["AZURE_OPENAI_ENDPOINT"],
        api_key=env["AZURE_OPENAI_API_KEY"],
        azure_deployment=env["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        api_version=env["AZURE_OPENAI_API_VERSION"],
    )


def get_vector_store(connection_string: str, collection_name: str, embeddings: Any) -> Any:
    """
    Initialize PGVector store with connection and embeddings.
    
    Args:
        connection_string: PostgreSQL connection string
        collection_name: Name of the vector collection
        embeddings: Embeddings instance
    
    Returns:
        PGVector instance
    
    Raises:
        SystemExit: If vector store initialization fails
    """
    try:
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection_string,
        )
    except Exception as e:
        logger.error(
            "Error: Unable to initialize vector store. Verify database connection settings and credentials."
        )
        logger.error(f"Details: {e}")
        sys.exit(1)


__all__ = [
    "load_env",
    "get_connection_string",
    "get_embeddings",
    "get_vector_store",
]
