"""
Nyaya Backend - Role-Aware Legal RAG System

Production FastAPI backend with:
- LangGraph multi-agent orchestration
- LangChain + Vertex AI Gemini for RAG
- Pinecone vector storage with role metadata
- InLegalBERT rhetorical role classification
- EmbeddingGemma for 384-dim embeddings

## Quick Start

1. **Setup Environment:**
   ```bash
   cd backend
   cp .env.example .env  # Add your API keys
   ```

2. **Install Dependencies:**
   ```bash
   # Already installed via uv
   # See pyproject.toml for full list
   ```

3. **Run Server:**
   ```bash
   # From backend/ directory
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   
   # Or using Python directly
   python -m app.main
   ```

4. **Access API:**
   - **API Docs**: http://localhost:8000/docs
   - **ReDoc**: http://localhost:8000/redoc
   - **Health Check**: http://localhost:8000/api/v1/health

## API Endpoints

### Session Management
- `POST /api/v1/sessions` - Create new session

### Document Operations
- `POST /api/v1/upload` - Upload & classify document (PDF/TXT)
- `POST /api/v1/query` - Role-aware Q&A
- `POST /api/v1/search` - Find similar cases
- `POST /api/v1/predict` - Predict case outcome

### Monitoring
- `GET /api/v1/health` - Health check
- `GET /api/v1/stats` - System statistics

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py  # LangGraph multi-agent coordinator
│   │   ├── classification_agent.py  # Upload → Classify → Store
│   │   ├── similarity_agent.py      # Find similar cases
│   │   ├── prediction_agent.py      # Predict outcomes
│   │   └── rag_agent.py            # Role-aware Q&A
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Pydantic settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models for API
│   └── services/
│       ├── __init__.py
│       ├── intent_detection.py      # Rule-based intent routing
│       ├── preprocessing.py         # Document preprocessing
│       ├── classification_service.py # InLegalBERT wrapper
│       ├── embedding_service.py     # EmbeddingGemma wrapper
│       ├── pinecone_service.py      # Pinecone operations
│       └── context_manager.py       # Session management
├── .env.example             # Environment variables template
└── README.md               # This file
```

## Environment Variables

Required in `.env`:

```bash
# Pinecone
PINECONE_API_KEY=pcsk_xxxxx...
PINECONE_INDEX_NAME=nyaya-legal-rag
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws

# Hugging Face (for EmbeddingGemma)
HF_TOKEN=hf_xxxxx...

# Google Cloud (for Vertex AI Gemini)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Model paths
CLASSIFIER_MODEL_PATH=./models/inlegalbert_classifier.pt
EMBEDDING_MODEL=google/embeddinggemma-300M
EMBEDDING_DIMENSION=384
LLM_MODEL_NAME=gemini-1.5-pro

# RAG settings
RAG_TOP_K=5
```

## Architecture

### Multi-Agent System (LangGraph)

```
User Query
    ↓
Intent Detection (rule-based, no LLM)
    ↓
┌───────────────┐
│  Orchestrator │
└───────┬───────┘
        ↓
    [Route based on intent]
        ↓
    ┌───┴───┬───────┬────────┐
    ↓       ↓       ↓        ↓
Classification Similarity Prediction RAG
  Agent      Agent     Agent    Agent
    ↓       ↓       ↓        ↓
    └───────┴───────┴────────┘
            ↓
        Response
```

### Intent Routing (Rule-Based)

**NO LLM NEEDED** - Uses keyword matching:

| Intent | Trigger | Agent |
|--------|---------|-------|
| **UPLOAD_AND_CLASSIFY** | `has_file=True` | Classification |
| **PREDICT_OUTCOME** | "predict", "outcome", "chances" | Prediction |
| **SIMILARITY_SEARCH** | "similar", "like this", "related" | Similarity |
| **ROLE_SPECIFIC_QA** | "facts", "reasoning", "decision" | RAG (filtered) |
| **SEARCH_CASES** | "search", "find cases" | Similarity |
| **GENERAL_QA** | Default with active case | RAG (all roles) |

### Role-Aware RAG Pipeline

**Two-Stage Process:**

1. **Stage 1: Classification** (when document is uploaded)
   ```
   Document → Sentences → InLegalBERT → Role Labels → Pinecone (with metadata)
   ```

2. **Stage 2: Retrieval** (when user asks question)
   ```
   Query → EmbeddingGemma → Vector → Pinecone (filter by role) → Context → Gemini → Answer
   ```

**Key Innovation:** Standard RAG returns mixed content; Nyaya returns **only** the requested role.

## Usage Examples

### 1. Upload & Classify Document

```bash
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@judgment.pdf" \
  -F "session_id=session_123" \
  -F "case_id=case_001"
```

Response:
```json
{
  "answer": "✅ Document processed successfully!\n**Case ID:** case_001\n...",
  "intent": "UPLOAD_AND_CLASSIFY",
  "classification_result": {
    "case_id": "case_001",
    "sentence_count": 145,
    "distribution": {
      "Facts": {"count": 42, "percentage": 29.0},
      "Reasoning": {"count": 38, "percentage": 26.2},
      ...
    }
  }
}
```

### 2. Role-Specific Question

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the facts of the case?",
    "session_id": "session_123"
  }'
```

### 3. Find Similar Cases

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cases involving breach of contract",
    "session_id": "session_123",
    "top_k": 5
  }'
```

### 4. Predict Outcome

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "case_description": "Petitioner challenges constitutional validity...",
    "relevant_laws": ["Article 14", "Article 21"],
    "session_id": "session_123"
  }'
```

## Model Requirements

### 1. InLegalBERT Classifier

**Expected file:** `backend/models/inlegalbert_classifier.pt`

Train using the notebooks in `docs/` or provide a pre-trained model.

**Format:** PyTorch state_dict with 7 output classes (7 roles)

### 2. EmbeddingGemma

**Auto-downloaded** from Hugging Face on first run (~1.2GB)

**Requirements:**
- Accept license at https://huggingface.co/google/embeddinggemma-300M
- Set `HF_TOKEN` in `.env`

### 3. Vertex AI Gemini

**Requirements:**
- Google Cloud project with Vertex AI enabled
- Service account with `Vertex AI User` role
- Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`

## Testing

### Test Services Individually

```bash
cd backend

# Test Pinecone integration
python -c "from app.services.pinecone_service import PineconeService; from app.core.config import Settings; s = Settings(); p = PineconeService(s); print(p.get_index_stats())"

# Test embedding service
python -c "from app.services.embedding_service import EmbeddingService; from app.core.config import Settings; s = Settings(); e = EmbeddingService(s); print(e.encode_query('test'))"
```

### Test API Endpoints

```bash
# Start server
uvicorn app.main:app --reload

# Test health check
curl http://localhost:8000/api/v1/health

# Test stats
curl http://localhost:8000/api/v1/stats
```

## Performance Notes

- **Startup time:** 15-30 seconds (model loading)
- **Classification:** ~5-10 sentences/second (GPU), ~1-2 sentences/second (CPU)
- **Embedding:** ~100 sentences/second
- **RAG query:** 2-5 seconds (depends on Gemini API)

## Troubleshooting

### "Model file not found"

```bash
# Check path in .env
echo $CLASSIFIER_MODEL_PATH

# Make sure model.pt exists
ls -la backend/models/inlegalbert_classifier.pt
```

### "Invalid authentication token" (Hugging Face)

```bash
# Accept license and set token
export HF_TOKEN=hf_xxxxx...
python -c "from huggingface_hub import login; login()"
```

### "Pinecone index not found"

```bash
# Check Pinecone dashboard: https://app.pinecone.io/
# Or create index automatically (first run creates it)
```

### "Google Cloud credentials not found"

```bash
# Set service account path
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Verify
gcloud auth application-default print-access-token
```

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install uv && uv pip install --system -r pyproject.toml

COPY backend/app ./app
COPY backend/models ./models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment

- Set proper CORS origins in `app/main.py`
- Use production-grade secret management
- Enable API rate limiting
- Add authentication middleware
- Configure proper logging

## License

See root repository LICENSE file.

## References

- **Paper:** "NYAYAANUMANA and INLEGALLLAMA" (Nigam et al., COLING 2025)
- **InLegalBERT:** "Pre-trained Language Models for the Legal Domain" (Paul et al., 2023)
- **EmbeddingGemma:** Google Gemma Cookbook
- **LangGraph:** LangChain documentation
