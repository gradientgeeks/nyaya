"""
API package for Legal Document Analysis System
Contains all API route modules
"""

from . import (
    health,
    queries,
    documents,
    classification,
    predictions,
    conversations,
    system,
    auth
)

# Optional import for document_query
try:
    from . import document_query
    __all__ = [
        "health",
        "queries", 
        "documents",
        "classification",
        "predictions",
        "conversations",
        "system",
        "auth",
        "document_query"
    ]
except ImportError:
    __all__ = [
        "health",
        "queries", 
        "documents",
        "classification",
        "predictions",
        "conversations",
        "system",
        "auth"
    ]