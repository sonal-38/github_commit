from fastapi import FastAPI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.getcwd(), "backend", ".env"))

from api.github import router as github_router
from api.graph import router as graph_router
from api.vector import router as vector_router
from api.ai import router as ai_router
from api.knowledge import router as knowledge_router

app = FastAPI(
    title="AI Digital Shadow - Backend API",
    description="Backend API services for AI Digital Shadow for Software Developers",
    version="0.1.0",
)

# Register routers
app.include_router(github_router)
app.include_router(graph_router)
app.include_router(vector_router)
app.include_router(ai_router)
app.include_router(knowledge_router)


@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {
        "project": "AI Digital Shadow for Software Developers",
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "repositories": "/github/repositories",
            "commits": "/github/repositories/{owner}/{repo}/commits",
            "pull_requests": "/github/repositories/{owner}/{repo}/pull-requests",
            "pr_reviews": "/github/repositories/{owner}/{repo}/pull-requests/{pull_number}/reviews",
            "pr_comments": "/github/repositories/{owner}/{repo}/pull-requests/{pull_number}/comments",
            "issues": "/github/repositories/{owner}/{repo}/issues",
            "issue_comments": "/github/repositories/{owner}/{repo}/issues/{issue_number}/comments",
            "pr_files": "/github/repositories/{owner}/{repo}/pull-requests/{pull_number}/files",
            "ingest": "/github/repositories/{owner}/{repo}/ingest",
            "graph_build": "/graph/repositories/{owner}/{repo}/build",
            "graph_summary": "/graph/repositories/{owner}/{repo}/summary",
            "vector_index": "/vector/repositories/{owner}/{repo}/index",
            "vector_search": "/vector/search",
            "ai_ask": "/ai/ask",
            "knowledge_cluster": "/knowledge/repositories/{owner}/{repo}/cluster",
            "knowledge_documents": "/knowledge/repositories/{owner}/{repo}/documents",
            "knowledge_interpret": "/knowledge/repositories/{owner}/{repo}/interpret",
            "knowledge_graph": "/knowledge/repositories/{owner}/{repo}/graph"
        }
    }
