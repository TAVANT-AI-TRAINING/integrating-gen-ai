import os
import uvicorn


def get_app():
    # Import inside function to avoid side effects on module import
    from main.routes import app
    return app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run("main.routes.ingestion_routes:app", host=host, port=port, reload=reload)

