# Nyaya: Role-Aware Legal RAG System

**Nyaya** (Sanskrit: न्याय, "justice") is a legal document analysis system using **Rhetorical Role Classification** with **Role-Aware RAG** to provide precise, structured answers from Indian legal judgments.

## 🎯 Core Innovation: Role-Aware Retrieval

Unlike standard RAG (chunks by size), Nyaya chunks by **semantic role** and retrieves by **role + similarity**:
- Standard RAG: "What were the facts?" → Returns Facts + Reasoning + Decision (mixed)
- **Nyaya RAG**: "What were the facts?" → Returns **only Facts-labeled sentences**

This role-filtering is the key differentiator from generic RAG.

## 📋 The 7 Rhetorical Roles

Every sentence is classified into one of:
1. **Facts** - Background and case events  
2. **Issue** - Legal questions to resolve  
3. **Arguments of Petitioner** - Petitioner's claims  
4. **Arguments of Respondent** - Respondent's counter-arguments  
5. **Reasoning** - Court's legal analysis  
6. **Decision** - Final judgment  
7. **None** - Other content

## 📂 Project Structure

```
nyaya/
├── backend/                    # FastAPI backend with LangGraph agents
│   ├── app/
│   │   ├── main.py            # FastAPI application entry
│   │   ├── agents/            # 4 specialized agents + orchestrator
│   │   │   ├── orchestrator.py     # LangGraph coordinator
│   │   │   ├── classification_agent.py  # Upload → Classify → Store
│   │   │   ├── similarity_agent.py      # Find similar cases
│   │   │   ├── prediction_agent.py      # Predict outcomes
│   │   │   └── rag_agent.py            # Role-aware Q&A
│   │   ├── api/routes.py      # REST endpoints (5 routes)
│   │   ├── services/          # Business logic
│   │   │   ├── intent_detection.py     # Rule-based routing
│   │   │   ├── embedding_service.py    # EmbeddingGemma wrapper
│   │   │   ├── pinecone_service.py     # Vector ops
│   │   │   ├── classification_service.py # InLegalBERT wrapper
│   │   │   └── context_manager.py      # Session tracking
│   │   ├── core/config.py     # Pydantic settings
│   │   └── models/schemas.py  # 26 data models
│   ├── test_pinecone_embedding.py  # Integration test
│   ├── official_rag_pattern.py     # Reference implementation
│   └── .env                   # API keys (not committed)
│
├── client/                    # React 19 + Vite + Tailwind v4
│   ├── src/
│   │   ├── components/        # 8 UI components
│   │   ├── types/index.ts     # TypeScript interfaces
│   │   ├── data/mockData.ts   # Mock data (7 roles)
│   │   └── contexts/ThemeContext.tsx
│   └── package.json
│
├── docs/                      # System documentation
│   ├── SYSTEM_WORKFLOW_EXPLANATION.md
│   ├── QUICK_START_GUIDE.md
│   └── REMOTE_GPU_TRAINING_GUIDE.md
│
└── pyproject.toml             # Python dependencies (uv)
```

## 🔑 Critical Development Patterns

### 1. Intent Routing: Rule-Based (No LLM!)

**Speed > Complexity** - Use keyword matching instead of LLM for routing:

```python
# backend/app/services/intent_detection.py
UPLOAD_KEYWORDS = ["upload", "analyze", "classify", "my case"]
PREDICTION_KEYWORDS = ["predict", "outcome", "what will happen", "chances"]
SIMILARITY_KEYWORDS = ["similar", "like this", "related cases"]

# Intent detection is instant and free
intent = intent_detector.detect_intent(query, has_file=True)
# → Returns Intent.UPLOAD_AND_CLASSIFY in <1ms
```

**Why rule-based?** Faster (no API), cheaper (no tokens), deterministic (easier debugging).

### 2. Role-Aware Metadata (THE KEY INNOVATION)

Each vector in Pinecone stores role metadata for filtered retrieval:

```python
# Upload with role metadata
vectors = [{
    "id": f"{case_id}_sent_{i}",
    "values": embedding.tolist(),  # 384-dim from EmbeddingGemma
    "metadata": {
        "text": "The petition is allowed.",
        "role": "Decision",           # One of 7 roles
        "confidence": 0.95,
        "case_id": "case_12345",
        "sentence_index": i
    }
}]
index.upsert(vectors=vectors, namespace="user_documents")

# Query with role filtering
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"role": {"$eq": "Reasoning"}},  # Only Reasoning!
    include_metadata=True
)
```

### 3. Asymmetric Embedding (EmbeddingGemma Pattern)

**Different prompts for documents vs queries** (+16.7% accuracy):

```python
# backend/app/services/embedding_service.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)

# Documents: Use "Retrieval-document" prompt
doc_embeddings = model.encode(
    [f"title: {case_id} | text: {text}" for text in docs],
    prompt_name="Retrieval-document",
    normalize_embeddings=True  # Always normalize!
)

# Queries: Use "Retrieval-query" prompt (DIFFERENT!)
query_embedding = model.encode(
    user_query,
    prompt_name="Retrieval-query",
    normalize_embeddings=True
)
```

### 4. LangGraph Multi-Agent Orchestration

See `backend/app/agents/orchestrator.py`:

```python
from langgraph.graph import StateGraph

# Define shared state
class AgentState(TypedDict):
    user_query: str
    intent: Intent
    search_results: list
    final_answer: str

# Build graph with conditional routing
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("classification_agent", classification_agent_node)
workflow.add_node("similarity_agent", similarity_agent_node)
workflow.add_node("prediction_agent", prediction_agent_node)
workflow.add_node("rag_agent", rag_agent_node)

workflow.add_conditional_edges(
    "detect_intent",
    route_to_agent,  # Function returns agent name
    {
        "classification_agent": "classification_agent",
        "similarity_agent": "similarity_agent",
        # ...
    }
)
```

### 5. Frontend Type Safety (React + TypeScript)

Components use centralized interfaces (`client/src/types/index.ts`):

```typescript
export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  type?: 'text' | 'analysis' | 'prediction' | 'greeting';
  data?: CaseSummary | CasePrediction;
  timestamp: Date;
}

// Usage in components
export function ChatMessage({ message }: { message: ChatMessage }) {
  if (message.type === 'analysis' && message.data) {
    return <AnalysisView data={message.data as CaseSummary} />;
  }
  // ...
}
```

## ⚡ Quick Development Commands

### Running the System

**Backend:**
```bash
cd backend

# First time setup
cp .env.example .env  # Add your API keys
export HF_TOKEN=your_token
python -c "from huggingface_hub import login; login()"

# Start server
uvicorn app.main:app --reload
# → API: http://localhost:8000
# → Docs: http://localhost:8000/docs
```

**Frontend:**
```bash
cd client
npm install
npm run dev
# → http://localhost:5173
```

**Testing:**
```bash
# Backend Pinecone integration
cd backend && python test_pinecone_embedding.py

# Frontend build
cd client && npm run build
```

### Common Tasks

**Add a Python dependency:**
```bash
# Edit pyproject.toml [project.dependencies]
# Then install
cd /home/uttam/B.Tech\ Major\ Project/nyaya
uv pip install package-name
```

**Add frontend component:**
```typescript
// 1. Create in client/src/components/NewComponent.tsx
// 2. Add interface to client/src/types/index.ts
// 3. Export from client/src/components/index.ts
// 4. Use: import { NewComponent } from '@/components'
```

**Test API endpoint:**
```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions

# Query with session
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were the facts?", "session_id": "xxx"}'
```

## 🚨 Common Pitfalls & Solutions

### ❌ Don't use `server/` directory
**It doesn't exist.** All backend code is in `backend/`.

### ❌ Don't mix embedding prompts
```python
# Wrong:
doc_emb = model.encode(doc, prompt_name="Retrieval-query")
query_emb = model.encode(query, prompt_name="Retrieval-query")

# Right:
doc_emb = model.encode(doc, prompt_name="Retrieval-document")
query_emb = model.encode(query, prompt_name="Retrieval-query")
```

### ❌ Don't forget to normalize embeddings
```python
# Wrong:
embeddings = model.encode(texts)

# Right:
embeddings = model.encode(texts, normalize_embeddings=True)
```

### ❌ Don't query Pinecone without role filtering for role-specific questions
```python
# Wrong (standard RAG):
results = index.query(vector=emb, top_k=5)

# Right (role-aware RAG):
results = index.query(
    vector=emb, 
    top_k=5,
    filter={"role": {"$eq": "Reasoning"}}
)
```

### ❌ Don't hardcode secrets
Always use `.env`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
```

## 🔍 Troubleshooting Guide

**"Invalid authentication token" (Hugging Face)**
```bash
# 1. Accept license: https://huggingface.co/google/embeddinggemma-300M
# 2. Get token: https://huggingface.co/settings/tokens
# 3. Login:
export HF_TOKEN=hf_xxxxx...
python -c "from huggingface_hub import login; login()"
```

**"PineconeException: Index not found"**
```bash
# Check Pinecone dashboard or run:
python backend/check_pinecone_status.py
```

**"Dimension mismatch"**
```python
# Check model dimension matches Pinecone index (384)
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
print(model.get_sentence_embedding_dimension())  # Should be 384
```

**First run is slow**
- Expected: Downloads ~1.2GB model from Hugging Face
- Subsequent runs use cached model from `~/.cache/huggingface/`

## 📚 Key Files Reference

- **Architecture:** `backend/docs/ARCHITECTURE.md` - System design
- **API Docs:** `backend/README.md` - Complete API documentation  
- **Implementation Status:** `backend/IMPLEMENTATION_SUMMARY.md` - What's done vs planned
- **Frontend Types:** `client/src/types/index.ts` - All TypeScript interfaces
- **Mock Data:** `client/src/data/mockData.ts` - Example case with all 7 roles
- **Orchestrator:** `backend/app/agents/orchestrator.py` - Multi-agent coordination
- **Intent Detection:** `backend/app/services/intent_detection.py` - Routing logic

## 🎓 Domain Knowledge

**Legal Document Structure (Indian Judgments):**
1. Case details (parties, court, date)
2. Facts → Issue → Arguments (Petitioner + Respondent) → Reasoning → Decision

**Research Foundation:**
- Paper: "NYAYAANUMANA and INLEGALLLAMA" (Nigam et al., COLING 2025)
- Model: InLegalBERT (Paul et al., 2023)

**Performance Expectations:**
- InLegalBERT: 85-90% accuracy on legal documents
- Custom trained: 90-95% on domain-specific data
- Training: 2-4 hours on RTX 5000 GPU (50k sentences)

## 🔐 Naming Conventions

- **Vector IDs:** `{case_id}_sent_{index}` (e.g., `case_12345_sent_0`)
- **Pinecone namespaces:** `user_documents`, `training_data`
- **React components:** PascalCase (`ChatMessage.tsx`)
- **Python modules:** snake_case (`intent_detection.py`)
- **Training data:** Tab-separated, blank lines between documents

## 📊 Implementation Status

✅ **Production Ready:**
- FastAPI backend with 5 REST endpoints
- LangGraph multi-agent orchestration (4 agents)
- Pinecone integration with role metadata
- React 19 frontend with 8 components
- Rule-based intent routing
- EmbeddingGemma integration (384-dim)

⏳ **Needs Setup:**
- Trained InLegalBERT classifier model
- Google Cloud credentials (for Vertex AI Gemini)
- Environment variables in `.env`
- Frontend-backend API connection

## 🎯 When Working on This Codebase

1. **Understand role-aware RAG is the differentiator** - Not just similarity, but role-filtered similarity
2. **Use rule-based intent detection** - No LLM needed for routing
3. **Always use asymmetric encoding** - Different prompts for docs vs queries
4. **Follow LangGraph patterns** - See `orchestrator.py` for state management
5. **Keep types centralized** - All interfaces in `client/src/types/index.ts`

**Before committing:**
- [ ] Backend scripts run: `python test_pinecone_embedding.py`
- [ ] Frontend builds: `cd client && npm run build`
- [ ] No hardcoded secrets
- [ ] Dependencies added to `pyproject.toml` / `package.json`
- [ ] Types updated if data models changed
