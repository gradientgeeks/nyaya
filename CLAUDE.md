# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nyaya** is an intelligent legal document analysis system that combines:
- **Role Classifier**: Segments legal documents into rhetorical roles (Facts, Issues, Arguments of Petitioner/Respondent, Reasoning, Decision)
- **RAG System**: Role-aware Retrieval-Augmented Generation for precise legal Q&A
- **Conversation Manager**: Multi-turn dialogue with context memory
- **Agent Orchestrator**: Routes queries to appropriate retrievers and tools
- **Prediction Module**: Provides probable judgments for pending cases

The system uses InLegalBERT for role classification and integrates with Vertex AI (Gemini) for embeddings and LLM generation.

## Architecture

### Directory Structure

```
nyaya/
├── client/           # React + TypeScript + Vite frontend
│   └── src/
│       ├── components/   # UI components
│       ├── contexts/     # React contexts (theme, etc.)
│       ├── types/        # TypeScript type definitions
│       └── data/         # Mock data
├── server/           # FastAPI backend
│   ├── main.py          # FastAPI app entry point
│   ├── requirements.txt # Python dependencies (uv-generated)
│   └── src/
│       ├── core/        # Core system components
│       │   ├── agent_orchestrator.py    # Query routing and orchestration
│       │   ├── legal_rag.py             # Role-aware RAG system
│       │   ├── conversation_manager.py  # Multi-turn conversation handling
│       │   ├── document_processor.py    # Document parsing and preprocessing
│       │   └── prediction_module.py     # Judgment prediction
│       ├── models/      # ML models
│       │   ├── role_classifier.py       # InLegalBERT, BiLSTM-CRF classifiers
│       │   └── training/                # Training scripts and data loaders
│       ├── api/         # API route modules
│       │   ├── queries.py               # General query endpoints
│       │   ├── documents.py             # Document management
│       │   ├── document_query.py        # Upload + simultaneous query
│       │   ├── classification.py        # Role classification
│       │   ├── predictions.py           # Judgment prediction
│       │   ├── conversations.py         # Conversation management
│       │   └── health.py                # Health checks
│       └── config/      # Configuration files
├── docs/             # Documentation
└── dataset/          # Training datasets
```

### Key Components

**Agent Orchestrator** ([server/src/core/agent_orchestrator.py](server/src/core/agent_orchestrator.py))
- `QueryRouter`: Classifies user queries by type and intent
- `QueryClassification`: Determines relevant rhetorical roles, tools, and confidence
- Routes to appropriate retrievers based on query content (facts, reasoning, decision, etc.)

**Legal RAG System** ([server/src/core/legal_rag.py](server/src/core/legal_rag.py))
- `LegalRAGSystem`: Multi-vector retriever with role-aware embeddings
- Uses ChromaDB for vector storage with InMemoryStore for document storage
- Integrates with Vertex AI embeddings (text-embedding-005) and Gemini 2.5 Flash
- Stores role-tagged documents with metadata

**Role Classifier** ([server/src/models/role_classifier.py](server/src/models/role_classifier.py))
- `InLegalBERTClassifier`: Fine-tuned InLegalBERT for role classification
- `BiLSTMCRFClassifier`: Hierarchical BiLSTM-CRF model
- Seven rhetorical roles: Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision, None

**Conversation Manager** ([server/src/core/conversation_manager.py](server/src/core/conversation_manager.py))
- Maintains short-term (last N turns) and long-term (vector DB) conversation memory
- Tracks document context across multi-turn queries
- Handles session-based conversations

## Development Commands

### Backend (Server)

The server uses **uv** for Python dependency management.

```bash
cd server

# Install dependencies (uv handles virtual environment automatically)
uv sync

# Run development server
python main.py
# Or with custom host/port:
HOST=0.0.0.0 PORT=8000 RELOAD=true python main.py

# The server runs on http://localhost:8000 by default
# API docs available at http://localhost:8000/docs
# Alternative docs at http://localhost:8000/redoc
```

**Environment Variables:**
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `RELOAD`: Auto-reload on code changes (default: true)

**Training Role Classifier:**
```bash
cd server/src/models/training
python train.py --model inlegalbert --epochs 10
```

### Frontend (Client)

```bash
cd client

# Install dependencies
npm install

# Run development server
npm run dev
# Runs on http://localhost:5173

# Build for production
npm run build

# Preview production build
npm run preview

# Lint code
npm run lint
```

## API Architecture

### Main Endpoints

**Document Upload with Query** (`/api/document-query/upload-and-ask`):
- Uploads PDF/TXT and asks a question simultaneously
- Returns document metadata, role classification, and answer
- Creates or continues a conversation session

**Follow-up Questions** (`/api/document-query/ask-followup`):
- Continues conversation about previously uploaded documents
- Maintains context from session_id

**General Query** (`/api/query`):
- Queries across all indexed documents
- Supports role-specific filtering

**Role Classification** (`/api/classify`):
- Classifies text into rhetorical roles
- Returns role labels with confidence scores

**Judgment Prediction** (`/api/predict`):
- Predicts probable outcomes for pending cases
- Returns probabilities and similar precedents

See [docs/DOCUMENT_QUERY_API.md](docs/DOCUMENT_QUERY_API.md) for detailed API documentation.

## Workflow Understanding

### Document Processing Flow
1. User uploads legal document (PDF/TXT)
2. Document Processor extracts and cleans text
3. Role Classifier labels each sentence with rhetorical role
4. Embeddings generated using Vertex AI
5. Role-tagged embeddings stored in ChromaDB with metadata
6. Document ready for querying

### Query Processing Flow
1. User query received
2. Query Router classifies intent and detects relevant roles
3. Agent Orchestrator selects appropriate tools
4. Role-aware retrieval from vector DB
5. LLM generates response using retrieved context
6. Response includes sources, confidence, and tools used

### Multi-turn Conversation
1. Initial query creates session with session_id
2. Conversation Manager tracks document context
3. Follow-up queries use session_id for context
4. System maintains both short-term and long-term memory

See [docs/EXAMPLE_FLOW_1.md](docs/EXAMPLE_FLOW_1.md) for a detailed walkthrough.

## Model Information

**Role Classification:**
- Primary model: InLegalBERT (law-ai/InLegalBERT)
- Alternative: BiLSTM-CRF with hierarchical sentence encoding
- Training notebook: [ROLE_CLASSIFIER_TRAINING.ipynb](ROLE_CLASSIFIER_TRAINING.ipynb)

**Embeddings & LLM:**
- Embeddings: Vertex AI text-embedding-005
- Generation: Gemini 2.5 Flash
- Requires Google Cloud credentials for Vertex AI

**Rhetorical Roles:**
- Facts: Background and events of the case
- Issue: Legal questions to be decided
- Arguments of Petitioner (AoP): Petitioner's claims
- Arguments of Respondent (AoR): Respondent's defense
- Reasoning: Court's legal analysis
- Decision: Final judgment/ruling

## Important Notes

### Backend Dependencies
- The server uses **uv** (not pip) for dependency management
- `requirements.txt` is auto-generated by uv - do not edit manually
- Use `uv add <package>` to add dependencies
- `pyproject.toml` and `uv.lock` are the source of truth

### Frontend Stack
- React 19 with TypeScript
- Vite for build tooling
- Tailwind CSS v4 for styling
- ESLint with TypeScript rules

### Vertex AI Configuration
- The system requires Google Cloud Vertex AI credentials
- Embeddings use `text-embedding-005` model
- LLM uses `gemini-2.5-flash` model
- Ensure GOOGLE_APPLICATION_CREDENTIALS is set or gcloud auth is configured

### Role-Aware RAG
- Always filter by role when querying for specific information
- Use parallel retrieval for multi-role queries (e.g., Facts + Reasoning)
- Role detection is automatic but can be explicitly specified via `role_filter` parameter

### Session Management
- Each conversation has a unique `session_id`
- Session tracks document context and conversation history
- Use same `session_id` for follow-up questions
- Sessions enable document-specific querying

## Citation

Based on research from:
- NYAYAANUMANA and INLegalLlama (COLING 2025)
- Pre-trained Language Models for Indian Law (ICAIL 2023)

See [docs/README.md](docs/README.md) for full citations and project background.
