"""
FastAPI application instance for the embedding service.
This module defines the FastAPI app and includes all routes.

COMPLETE WORKFLOW (when server starts):
Step 1: Configuration & Initialization (runs automatically when embedding_service.py is imported)
  → Sets up ChromaDB and OpenAI connections

Step 2: Application Setup (this file)
  → Create FastAPI app instance
  → Configure lifespan events (startup/shutdown)
  → Register routes from routes.py

Step 3: Server Startup (main.py starts uvicorn)
  → Server listens for HTTP requests
  → Routes handle incoming API requests
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from main.routes import router

# ============================================================================
# STEP 2: APPLICATION SETUP
# ============================================================================
# This step sets up the FastAPI application:
# - Configure logging
# - Define lifespan events (startup/shutdown)
# - Create FastAPI app instance
# - Register routes from routes.py
# 
# NOTE: Step 1 (Configuration & Initialization) already completed
# This happened automatically when routes.py imported embedding_service.py
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define lifespan event handler (runs on server startup and shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup - runs when server starts
    try:
        logger.info("✓ API server starting: Embeddings and ChromaDB connection verified.")
        yield
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise HTTPException(status_code=500, detail="Server startup error")
    # Shutdown - runs when server stops
    logger.info("✓ API server shutting down.")

# Create FastAPI application instance
# This initializes the app with metadata and lifespan handler
app = FastAPI(
    title="Embedding API", 
    description="API for generating embeddings and storing/deleting documents in ChromaDB", 
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# CENTRALIZED EXCEPTION HANDLING
# ============================================================================
# Single exception handler for all endpoints
# Handles ValueError (404) and general exceptions (500)
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions (typically document not found)."""
    logger.error(f"✗ ValueError: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    logger.error(f"✗ Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Register routes from routes.py
# This connects all API endpoints defined in routes.py to the FastAPI app
app.include_router(router)

