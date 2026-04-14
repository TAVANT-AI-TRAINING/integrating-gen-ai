"""
Main entry point for running the FastAPI embedding service.
This module provides a simple way to run the server.

COMPLETE WORKFLOW (when server starts):
Step 1: Configuration & Initialization (runs automatically when app.py imports routes.py)
  → routes.py imports embedding_service.py
  → embedding_service.py sets up database and Azure OpenAI connections

Step 2: Application Setup (app.py)
  → Create FastAPI app instance
  → Register routes

Step 3: Server Startup (this file)
  → Start uvicorn server
  → Server listens on port 8000
  → Ready to handle API requests
"""

import uvicorn

if __name__ == "__main__":
    # Start the FastAPI server using uvicorn
    # This step: loads app.py → app.py loads routes.py → routes.py loads embedding_service.py
    # All initialization happens automatically through imports
    # Server starts and listens for HTTP requests on port 8000
    uvicorn.run("main.app:app", host="0.0.0.0", port=8000, reload=True)
