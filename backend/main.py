from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from api.github import router as github_router

app = FastAPI(
    title="AI Digital Shadow - Backend API",
    description="Backend API services for AI Digital Shadow for Software Developers",
    version="0.1.0",
)

# Register routers
app.include_router(github_router)


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {
        "project": "AI Digital Shadow for Software Developers",
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "repositories": "/github/repositories"
        }
    }
