# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Nyaya** (न्याय - Sanskrit for "justice") is a role-aware legal RAG system for analyzing Indian legal judgments. The key innovation is **role-aware retrieval**: instead of returning mixed content like standard RAG, Nyaya chunks documents by semantic role and retrieves precisely what users ask for.

**Example:**
- Standard RAG: "What were the facts?" → Returns Facts + Reasoning + Decision (mixed)
- Nyaya RAG: "What were the facts?" → Returns **only Facts-labeled sentences**

### Tech Stack
- **Backend:** FastAPI + LangGraph (multi-agent orchestration) + Python 3.12
- **Frontend:** React 19 + Vite + TypeScript + TailwindCSS v4
- **Vector DB:** Pinecone (serverless, 384-dim embeddings)
- **Embeddings:** EmbeddingGemma (`google/embeddinggemma-300M`)
- **LLM:** Google Vertex AI Gemini 1.5 Pro
- **Classification:** InLegalBERT (`law-ai/InLegalBERT`) | LegalBERT 

## Repository Structure

```
nyaya/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py            # FastAPI entry point
│   │   ├── agents/            # 4 LangGraph agents + orchestrator
│   │   ├── api/routes.py      # 5 REST endpoints
│   │   ├── services/          # Business logic (6 services)
│   │   ├── models/schemas.py  # 26 Pydantic models
│   │   └── core/config.py     # Settings management
│   ├── test_pinecone_embedding.py  # Integration test
│   ├── official_rag_pattern.py     # Reference implementation
│   └── .env                   # API keys (NOT committed)
├── client/                    # React frontend
│   ├── src/
│   │   ├── components/        # 8 UI components
│   │   ├── types/index.ts     # TypeScript interfaces
│   │   └── data/mockData.ts   # Mock data (7 roles)
│   └── package.json
├── docs/                      # Documentation
└── pyproject.toml             # Python dependencies (uv)
```

## The 7 Rhetorical Roles

Every sentence in legal judgments is classified into one of:
1. **Facts** - Background and case events
2. **Issue** - Legal questions to resolve
3. **Arguments of Petitioner** - Petitioner's claims
4. **Arguments of Respondent** - Respondent's counter-arguments
5. **Reasoning** - Court's legal analysis
6. **Decision** - Final judgment
7. **None** - Other content

## Common Commands

### Backend Development

```bash
# Run the FastAPI server
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main

# Test Pinecone integration
python test_pinecone_embedding.py

# Check Pinecone status
python check_pinecone_status.py

# Test API endpoints (uses shell script)
./test_api.sh
```

### Frontend Development

```bash
cd client
npm install          # Install dependencies
npm run dev          # Start dev server (http://localhost:5173)
npm run build        # Build for production
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

### Testing Individual Services

```bash
cd backend

# Test embedding service
python -c "from app.services.embedding_service import EmbeddingService; from app.core.config import Settings; s = Settings(); e = EmbeddingService(s); print(e.encode_query('test'))"

# Test Pinecone service
python -c "from app.services.pinecone_service import PineconeService; from app.core.config import Settings; s = Settings(); p = PineconeService(s); print(p.get_index_stats())"
```

## Key Architecture Patterns

### 1. Multi-Agent System (LangGraph)

The backend uses **LangGraph** for orchestrating 4 specialized agents:

```
User Query → Intent Detection (rule-based) → Orchestrator
                                                  ↓
                        ┌─────────────────────────┼─────────────────────┐
                        ↓                         ↓                     ↓
                Classification Agent      Similarity Agent      Prediction Agent      RAG Agent
                (Upload → Classify)       (Find similar)        (Predict outcome)     (Q&A)
```

**Intent detection is rule-based (NOT LLM)** - uses keyword matching for routing:
- Upload: `has_file=True`
- Similarity: "similar", "like this", "related"
- Prediction: "predict", "outcome", "chances"
- Role-specific QA: "facts", "reasoning", "decision"

**Reference:** `backend/app/agents/orchestrator.py` and `backend/app/services/intent_detection.py`

### 2. Role-Aware Retrieval (THE KEY INNOVATION)

Pinecone vectors store role metadata for filtered retrieval:

```python
# Upload with role metadata
vectors = [{
    "id": f"{case_id}_sent_{i}",
    "values": embedding.tolist(),  # 384-dim
    "metadata": {
        "text": "The petition is allowed.",
        "role": "Decision",           # One of 7 roles
        "confidence": 0.95,
        "case_id": "case_12345",
        "sentence_index": i
    }
}]

# Query with role filtering (Nyaya's differentiator)
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"role": {"$eq": "Reasoning"}},  # Only Reasoning sentences!
    include_metadata=True
)
```

**This is Nyaya's core innovation** - maintain role filtering in all features.

### 3. Asymmetric Encoding (EmbeddingGemma)

**CRITICAL:** Use different prompts for documents vs queries (+16.7% accuracy):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)

# Documents: "Retrieval-document" prompt
doc_embeddings = model.encode(
    [f"title: {title} | text: {content}" for title, content in docs],
    prompt_name="Retrieval-document",  # Important!
    normalize_embeddings=True
)

# Queries: "Retrieval-query" prompt (DIFFERENT!)
query_embedding = model.encode(
    query,
    prompt_name="Retrieval-query",  # Different from documents!
    normalize_embeddings=True
)
```

**Reference:** `backend/official_rag_pattern.py` (from Gemma Cookbook)

### 4. Environment Configuration

**Backend `.env` file (REQUIRED):**

```bash
# Pinecone
PINECONE_API_KEY=pcsk_xxxxx...
PINECONE_INDEX_NAME=nyaya-legal-rag
PINECONE_ENVIRONMENT=us-east-1

# Hugging Face (for EmbeddingGemma)
HF_TOKEN=hf_xxxxx...

# Google Cloud (for Vertex AI Gemini)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Model paths
CLASSIFIER_MODEL_PATH=./models/inlegalbert_classifier.pt
EMBEDDING_MODEL=google/embeddinggemma-300M
LLM_MODEL_NAME=gemini-1.5-pro
```

**First-time setup:**
```bash
cd backend
cp .env.example .env  # Then add your API keys

# Accept EmbeddingGemma license
# Visit: https://huggingface.co/google/embeddinggemma-300M
# Then login:
python -c "from huggingface_hub import login; login()"
```

### 5. FastAPI Endpoints

**5 Main Endpoints:**
- `POST /api/v1/sessions` - Create new conversation session
- `POST /api/v1/upload` - Upload PDF/TXT → Classify → Store
- `POST /api/v1/query` - Role-aware Q&A
- `POST /api/v1/search` - Find similar cases
- `POST /api/v1/predict` - Predict case outcome

**Monitoring:**
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - System statistics

**Reference:** `backend/app/api/routes.py`

## File Organization Conventions

### Backend
- `backend/*.py` - Standalone scripts (tests, demos, utilities)
- `backend/app/` - Main application package
- `backend/.env` - **Secrets (NEVER commit)**
- `backend/docs/` - Backend-specific documentation

### Frontend
- `client/src/components/` - Reusable UI components (PascalCase files)
- `client/src/types/index.ts` - **Centralized TypeScript interfaces**
- `client/src/data/mockData.ts` - Mock data for development
- `client/src/contexts/` - React contexts (theme, etc.)

### Naming Conventions
- **Vector IDs:** `{case_id}_sent_{index}` (e.g., `case_12345_sent_0`)
- **Pinecone namespaces:** `user_documents`, `training_data`, `demo`
- **React components:** PascalCase (`ChatMessage.tsx`)
- **Python:** snake_case (`test_pinecone_embedding.py`)

## Implementation Status

### ✅ Production Ready
**Backend:**
- Complete FastAPI server with 5 endpoints
- LangGraph multi-agent orchestration (4 agents)
- Pinecone integration with role metadata
- EmbeddingGemma integration (official pattern)
- Pydantic models (26 schemas)
- Rule-based intent detection

**Frontend:**
- Full React 19 + TypeScript UI
- 8 components with mock data
- Theme toggle (dark/light)
- Responsive design
- Type-safe interfaces

**Infrastructure:**
- Pinecone serverless index (us-east-1, 384-dim)
- Environment configuration
- Comprehensive documentation

### ⏳ In Progress / Needed
- InLegalBERT classifier integration (training exists, needs deployment)
- Frontend ↔ Backend API connection (currently mock data)
- File upload → Classification pipeline
- LangChain RAG with Vertex AI Gemini
- Session management for multi-turn conversations
- Authentication/authorization

## Critical Development Rules

### 1. Respect the Role-Aware Paradigm

This isn't standard RAG - it's role-aware RAG:

❌ **Wrong (standard RAG):**
```python
results = index.query(vector=emb, top_k=5)
```

✅ **Right (role-aware RAG):**
```python
results = index.query(
    vector=emb,
    top_k=5,
    filter={"role": {"$eq": "Reasoning"}}
)
```

### 2. Always Normalize Embeddings

❌ **Wrong:**
```python
embeddings = model.encode(texts)
```

✅ **Right:**
```python
embeddings = model.encode(texts, normalize_embeddings=True)
```

### 3. Use Asymmetric Prompts

❌ **Wrong (same prompt for both):**
```python
doc_emb = model.encode(doc, prompt_name="Retrieval-query")
query_emb = model.encode(query, prompt_name="Retrieval-query")
```

✅ **Right (different prompts):**
```python
doc_emb = model.encode(doc, prompt_name="Retrieval-document")
query_emb = model.encode(query, prompt_name="Retrieval-query")
```

### 4. Never Hardcode Secrets

❌ **Wrong:**
```python
api_key = "pcsk_xxxxx..."
```

✅ **Right:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
```

### 5. Two-Stage Pipeline

Respect the two-stage architecture:
1. **Stage 1: Classification** (InLegalBERT assigns roles to sentences)
2. **Stage 2: Retrieval** (Vector search filtered by role metadata)

Never skip Stage 1 for role-aware queries.

## Troubleshooting

### Backend Issues

**"Invalid authentication token" (Hugging Face)**
```bash
# 1. Accept license: https://huggingface.co/google/embeddinggemma-300M
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Login:
python -c "from huggingface_hub import login; login()"
# Or set in .env:
export HF_TOKEN=hf_xxxxx...
```

**"PineconeException: Index not found"**
```bash
# Check index exists
python backend/check_pinecone_status.py

# Or run test (creates index automatically)
python backend/test_pinecone_embedding.py
```

**"Dimension mismatch" in Pinecone**
```python
# Verify embedding dimension
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
print(model.get_sentence_embedding_dimension())  # Should be 384
```

**Slow first run**
- Expected: First run downloads ~1.2GB EmbeddingGemma model
- Subsequent runs use cached model (~/.cache/huggingface/)

### Frontend Issues

**"Cannot find module '@/components'"**
- This is normal - frontend uses mock data currently
- Run: `cd client && npm install && npm run dev`

## Key Documentation Files

- **`backend/README.md`** - Complete backend API documentation (367 lines)
- **`backend/SETUP.md`** - Quick start guide (400+ lines)
- **`backend/IMPLEMENTATION_SUMMARY.md`** - Implementation details (425 lines)
- **`.github/copilot-instructions.md`** - Comprehensive development guide (676 lines)
- **`backend/official_rag_pattern.py`** - Reference RAG implementation
- **`client/src/types/index.ts`** - All TypeScript interfaces
- **`client/src/data/mockData.ts`** - Example case structure with 7 roles

## Performance Expectations

- **Startup:** 15-30 seconds (model loading)
- **Classification:** 5-10 sent/sec (GPU), 1-2 sent/sec (CPU)
- **Embedding:** ~100 sent/sec
- **RAG query:** 2-5 seconds (Gemini API latency)

## Testing Checklist

Before committing changes:
- [ ] Backend scripts run without errors: `python test_pinecone_embedding.py`
- [ ] Frontend builds: `cd client && npm run build`
- [ ] Environment variables in `.env`, not hardcoded
- [ ] New dependencies added to `pyproject.toml` (backend) or `package.json` (frontend)
- [ ] TypeScript types updated if data structures changed
- [ ] Mock data updated if API contract changed

## Additional Resources

- **Full system workflow:** `docs/SYSTEM_WORKFLOW_EXPLANATION.md`
- **Quick start:** `docs/QUICK_START_GUIDE.md`
- **Architecture details:** `backend/docs/ARCHITECTURE.md`
- **Pinecone setup:** `backend/docs/PINECONE_SETUP.md`
- **Embedding patterns:** `backend/docs/README_EMBEDDINGS.md`

## Research Citations

This system implements research from:
- **NYAYAANUMANA and INLEGALLLAMA** (Nigam et al., COLING 2025)
- **InLegalBERT** (Paul et al., ICAIL 2023)
- **EmbeddingGemma** (Google Gemma Cookbook)
