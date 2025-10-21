"""
Nyaya Backend Application

A production FastAPI backend for role-aware legal RAG system.
"""

__version__ = "1.0.0"
__author__ = "Nyaya Team"
__description__ = "Role-Aware Legal RAG System with LangGraph & Pinecone"

from .main import app

__all__ = ["app"]
