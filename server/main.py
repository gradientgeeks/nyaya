"""
FastAPI Backend for Legal Document Analysis System

This module provides REST API endpoints for all system functionality including
document upload, role classification, RAG queries, conversation management,
and judgment prediction.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.core.agent_orchestrator import AgentOrchestrator
from src.core.legal_rag import LegalRAGSystem
from src.core.conversation_manager import ConversationManager
from src.core.document_processor import LegalDocumentProcessor
from src.core.prediction_module import JudgmentPredictor
from src.models.role_classifier import RoleClassifier

# Import API routers
from src.api import (
    health,
    queries,
    documents,
    classification,
    predictions,
    conversations,
    system
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Nyaya - Legal Document Analysis API",
    description="Intelligent Agent for Legal Document Analysis using RAG, Role Classifier, and Multi-turn Conversation Support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
orchestrator: Optional[AgentOrchestrator] = None
rag_system: Optional[LegalRAGSystem] = None
conversation_manager: Optional[ConversationManager] = None
document_processor: Optional[LegalDocumentProcessor] = None
role_classifier: Optional[RoleClassifier] = None
prediction_module: Optional[JudgmentPredictor] = None

@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global orchestrator, rag_system, conversation_manager, document_processor, role_classifier, prediction_module
    
    try:
        logger.info("Initializing Legal Document Analysis System...")
        
        # Initialize components
        logger.info("Loading Agent Orchestrator...")
        orchestrator = AgentOrchestrator()
        
        logger.info("Loading RAG System...")
        rag_system = orchestrator.rag_system
        
        logger.info("Loading Conversation Manager...")
        conversation_manager = orchestrator.conversation_manager
        
        logger.info("Loading Document Processor...")
        document_processor = orchestrator.document_processor
        
        logger.info("Loading Role Classifier...")
        role_classifier = orchestrator.rag_system.role_classifier
        
        logger.info("Loading Prediction Module...")
        prediction_module = JudgmentPredictor(rag_system)
        
        # Set global instances for route modules
        queries.set_orchestrator(orchestrator)
        documents.set_orchestrator(orchestrator)
        classification.set_role_classifier(role_classifier)
        predictions.set_prediction_module(prediction_module)
        conversations.set_conversation_manager(conversation_manager)
        
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Legal Document Analysis System...")

# Include routers
app.include_router(health.router)
app.include_router(queries.router)
app.include_router(documents.router)
app.include_router(classification.router)
app.include_router(predictions.router)
app.include_router(conversations.router)
app.include_router(system.router)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Main entry point
if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
