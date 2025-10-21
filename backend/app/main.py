"""
Nyaya: Role-Aware Legal RAG System
Main FastAPI Application

Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import Settings
from .api import router
from .api.routes import init_orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Initialize orchestrator and services
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("🚀 Starting Nyaya Legal RAG API...")
    
    try:
        # Load settings
        settings = Settings()
        logger.info("✅ Settings loaded")
        
        # Initialize orchestrator (this loads all models and connects to Pinecone)
        logger.info("🔧 Initializing orchestrator...")
        init_orchestrator(settings)
        logger.info("✅ Orchestrator initialized")
        
        logger.info("✅ Nyaya API is ready!")
        logger.info(f"   📚 Pinecone Index: {settings.pinecone_index_name}")
        logger.info(f"   🤖 LLM Model: {settings.llm_model_name}")
        logger.info(f"   📊 Embedding Model: {settings.embedding_model}")
        logger.info(f"   🏛️  Classifier: {settings.classifier_model_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("🛑 Shutting down Nyaya Legal RAG API...")
    # Add cleanup code here if needed


# Create FastAPI app
app = FastAPI(
    title="Nyaya Legal RAG API",
    description="""
    **Nyaya** (न्याय - Sanskrit for "justice") is a role-aware legal document analysis system.
    
    ## Features
    
    - 📄 **Document Upload & Classification**: Automatically classify legal documents into 7 rhetorical roles
    - 🔍 **Role-Aware RAG**: Query specific sections (Facts, Reasoning, Decision, etc.)
    - 🔗 **Similarity Search**: Find similar cases using role-weighted similarity
    - 🔮 **Outcome Prediction**: Predict case outcomes based on precedents
    - 💬 **Multi-Turn Conversations**: Context-aware follow-up questions
    
    ## Rhetorical Roles
    
    1. **Facts** - Background and case events
    2. **Issue** - Legal questions to resolve
    3. **Arguments of Petitioner** - Petitioner's claims
    4. **Arguments of Respondent** - Respondent's counter-arguments
    5. **Reasoning** - Court's legal analysis
    6. **Decision** - Final judgment
    7. **None** - Other content
    
    ## Quick Start
    
    1. **Create Session**: `POST /api/v1/sessions`
    2. **Upload Document**: `POST /api/v1/upload` (with PDF/TXT file)
    3. **Ask Questions**: `POST /api/v1/query` (role-aware Q&A)
    4. **Find Similar Cases**: `POST /api/v1/search`
    5. **Predict Outcome**: `POST /api/v1/predict`
    
    ## Technology Stack
    
    - **LangGraph**: Multi-agent orchestration
    - **Pinecone**: Vector database (384-dim embeddings)
    - **EmbeddingGemma**: Asymmetric text embeddings
    - **InLegalBERT**: Rhetorical role classification
    - **Vertex AI Gemini**: Answer generation
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://localhost:8080",
        "*"  # WARNING: In production, specify exact origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Nyaya Legal RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "description": "Role-aware legal document analysis system"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
