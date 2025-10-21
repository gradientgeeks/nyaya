# Nyaya Backend-Frontend Integration Guide

Complete guide for running the integrated Nyaya Legal RAG System with both backend and frontend.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Nyaya Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐              ┌──────────────────────┐   │
│  │   Frontend   │◄─────────────┤   Backend API        │   │
│  │  React + Vite│  REST API    │   FastAPI            │   │
│  │  Port: 5173  │              │   Port: 8000         │   │
│  └──────────────┘              └──────────────────────┘   │
│        │                                │                  │
│        │                                │                  │
│        ▼                                ▼                  │
│  ┌──────────────┐              ┌──────────────────────┐   │
│  │  Mock Data   │              │   LangGraph          │   │
│  │  (Fallback)  │              │   Orchestrator       │   │
│  └──────────────┘              └──────────────────────┘   │
│                                          │                 │
│                                          ▼                 │
│                                 ┌────────────────────┐    │
│                                 │  Services          │    │
│                                 ├────────────────────┤    │
│                                 │ - Pinecone Vector  │    │
│                                 │ - EmbeddingGemma   │    │
│                                 │ - InLegalBERT      │    │
│                                 │ - Vertex AI Gemini │    │
│                                 └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Backend Requirements
- Python 3.12+
- Pinecone account and API key
- Hugging Face account and token
- Google Cloud account (for Vertex AI)

### Frontend Requirements
- Node.js 18+
- npm or yarn

## Setup Instructions

### Step 1: Clone Repository

```bash
git clone https://github.com/gradientgeeks/nyaya.git
cd nyaya
```

### Step 2: Backend Setup

#### 2.1 Install Python Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

#### 2.2 Configure Environment Variables

```bash
cd backend
cp .env.example .env
```

Edit `backend/.env` with your API keys:

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

# Models
CLASSIFIER_MODEL_PATH=./models/inlegalbert_classifier.pt
EMBEDDING_MODEL=google/embeddinggemma-300M
EMBEDDING_DIMENSION=384
LLM_MODEL_NAME=gemini-1.5-pro

# RAG Configuration
RAG_TOP_K=5
SIMILARITY_ROLE_WEIGHTS={"Facts": 0.25, "Issue": 0.25, "Reasoning": 0.30, "Decision": 0.20}
```

#### 2.3 Accept Hugging Face Licenses

Visit and accept licenses for:
- https://huggingface.co/google/embeddinggemma-300M

Then login:
```bash
export HF_TOKEN=hf_xxxxx...
python -c "from huggingface_hub import login; login()"
```

#### 2.4 Setup Pinecone Index

If you don't have a Pinecone index yet:

```bash
python setup_pinecone.sh
```

Or create manually with:
- Dimension: 384
- Metric: cosine
- Cloud: AWS (recommended)
- Region: us-east-1

#### 2.5 Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/api/v1/health

### Step 3: Frontend Setup

#### 3.1 Install Node Dependencies

```bash
cd client
npm install
```

#### 3.2 Configure Environment

```bash
cp .env.example .env
```

Edit `client/.env`:

```bash
# Backend API URL (default is fine for local development)
VITE_API_URL=http://localhost:8000
```

#### 3.3 Start Frontend Development Server

```bash
npm run dev
```

Frontend will be available at: http://localhost:5173

## Running the Integrated System

### Development Mode (Recommended)

Run both backend and frontend simultaneously:

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd client
npm run dev
```

Then open http://localhost:5173 in your browser.

### Production Build

**Build Frontend:**
```bash
cd client
npm run build
```

**Serve Production Build:**
```bash
npm run preview
```

**Run Backend in Production Mode:**
```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Testing the Integration

### 1. Health Check

Open http://localhost:8000/api/v1/health in your browser or:

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Nyaya Legal RAG API",
  "version": "1.0.0"
}
```

### 2. Frontend Connection

1. Open http://localhost:5173
2. You should see the Nyaya interface
3. If backend is running, no warning banner appears
4. If backend is not running, you'll see: "⚠️ Backend not connected - using mock data"

### 3. Document Upload Test

1. Click the upload button (📎 icon) in the chat input
2. Select a legal document (PDF or TXT)
3. Backend will:
   - Classify sentences by rhetorical role
   - Store in Pinecone with role metadata
   - Return classification results
4. Frontend displays the analysis

### 4. Query Test

Try these example queries:

```
"What were the facts of the case?"
"Show me the reasoning"
"What was the final decision?"
"Predict the outcome" (for pending cases)
"Find similar cases"
```

### 5. API Documentation

Explore the interactive API docs at http://localhost:8000/docs to test endpoints directly.

## API Endpoints

### Session Management
```
POST /api/v1/sessions
```
Creates a new conversation session.

### Document Operations
```
POST /api/v1/upload
```
Upload and classify a legal document.

```
POST /api/v1/query
```
Ask questions with role-aware RAG.

```
POST /api/v1/search
```
Search for similar cases.

```
POST /api/v1/predict
```
Predict case outcomes.

### System Monitoring
```
GET /api/v1/health
```
Health check endpoint.

```
GET /api/v1/stats
```
Get Pinecone and session statistics.

## Data Flow

### Upload & Classification Flow

```
User uploads PDF → Frontend → Backend → Extract text
                                      ↓
                                InLegalBERT classifies roles
                                      ↓
                                EmbeddingGemma creates embeddings
                                      ↓
                                Pinecone stores vectors with role metadata
                                      ↓
                                Frontend ← Classification results
```

### Query Flow (Role-Aware RAG)

```
User asks "What were the facts?" → Frontend → Backend
                                             ↓
                                    Intent Detection
                                             ↓
                                    LangGraph Orchestrator
                                             ↓
                                    RAG Agent
                                             ↓
                         Pinecone query with role filter (Facts only)
                                             ↓
                         Vertex AI Gemini generates answer
                                             ↓
                         Frontend ← Answer with sources
```

## Troubleshooting

### Backend Issues

**Problem:** "Orchestrator not initialized"
**Solution:** 
- Check that all required environment variables are set
- Verify Pinecone API key is valid
- Check Hugging Face token is accepted

**Problem:** "Pinecone index not found"
**Solution:**
```bash
python backend/check_pinecone_status.py
```

**Problem:** "HuggingFace authentication error"
**Solution:**
```bash
export HF_TOKEN=hf_xxxxx...
python -c "from huggingface_hub import login; login()"
```

### Frontend Issues

**Problem:** "Backend not connected" warning
**Solution:**
- Ensure backend is running at http://localhost:8000
- Check `client/.env` has correct API URL
- Verify CORS is enabled (already configured in backend)

**Problem:** Build errors
**Solution:**
```bash
cd client
rm -rf node_modules dist
npm install
npm run build
```

### Integration Issues

**Problem:** CORS errors in browser console
**Solution:**
Backend already includes CORS middleware for localhost. If using different ports, update `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Add your port
        "*"  # Or use wildcard (not recommended for production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Problem:** Session not persisting
**Solution:**
- Check browser console for errors
- Verify session_id is being stored in frontend state
- Check backend logs for session creation

## Development Tips

### Hot Reloading

Both backend and frontend support hot reloading:
- **Backend:** Changes to Python files auto-reload with `--reload` flag
- **Frontend:** Vite's HMR updates UI instantly on save

### API Testing

Use the interactive docs at http://localhost:8000/docs to test API endpoints before implementing in frontend.

### Mock Data Fallback

The frontend works with mock data when backend is unavailable, allowing:
- UI development without backend
- Testing frontend features independently
- Graceful degradation in case of backend issues

### Debugging

**Backend Logs:**
```bash
# Backend logs appear in terminal where uvicorn is running
# Look for ✅ success and ❌ error indicators
```

**Frontend Logs:**
```bash
# Open browser console (F12)
# Look for API calls and errors
```

### Code Organization

**Backend:**
- Routes: `backend/app/api/routes.py`
- Agents: `backend/app/agents/`
- Services: `backend/app/services/`

**Frontend:**
- Components: `client/src/components/`
- API Client: `client/src/services/api.ts`
- Types: `client/src/types/index.ts`

## Performance Considerations

### First Run
- Backend: ~30s to load models (EmbeddingGemma is 1.2GB)
- Frontend: Instant with Vite

### Subsequent Runs
- Backend: <5s (models cached)
- Frontend: Instant

### API Response Times
- Session creation: <100ms
- Document upload: 2-10s (depending on size)
- Query: 1-3s (with RAG)
- Search: 1-2s

## Security Best Practices

1. **Never commit `.env` files** - Already gitignored
2. **Use environment variables** for all secrets
3. **Rotate API keys** regularly
4. **Limit CORS origins** in production
5. **Use HTTPS** in production

## Next Steps

1. Upload sample legal documents
2. Train InLegalBERT classifier on your domain data
3. Tune RAG parameters (top_k, role_weights)
4. Add more documents to Pinecone
5. Customize UI theme and branding

## Support

For issues:
- Backend: See `backend/README.md`
- Frontend: See `client/README.md`
- Integration: This document

For questions, open an issue on GitHub.
