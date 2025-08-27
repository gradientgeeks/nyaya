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