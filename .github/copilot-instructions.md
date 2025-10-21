# Nyaya: Role-Aware Legal RAG System

**Nyaya** (Sanskrit: न्याय, "justice") is a research-to-production legal document analysis system using **Rhetorical Role Classification** with **Role-Aware RAG** to provide precise, structured answers from Indian legal judgments.

## The Core Innovation: Role-Aware Retrieval

Unlike standard RAG systems that chunk by size, Nyaya chunks by **semantic role** and retrieves by **role + similarity**:
- User asks "What were the facts?" → System retrieves **only Facts-labeled sentences**
- User asks "What was the reasoning?" → System retrieves **only Reasoning-labeled sentences**

This is the key differentiator from generic RAG implementations.

## The 7 Rhetorical Roles

Every sentence is classified into one of:
1. **Facts** - Background and case events  
2. **Issue** - Legal questions to resolve  
3. **Arguments of Petitioner** - Petitioner's claims  
4. **Arguments of Respondent** - Respondent's counter-arguments  
5. **Reasoning** - Court's legal analysis  
6. **Decision** - Final judgment  
7. **None** - Other content

## Current Project Structure (Actual)

```
nyaya/
├── backend/                    # Python backend (not "server/")
│   ├── nyaya_multi_agent.py   # LangGraph multi-agent orchestrator (prototype)
│   ├── official_rag_pattern.py # Reference implementation from Gemma Cookbook
│   ├── test_pinecone_embedding.py # Production Pinecone + EmbeddingGemma setup
│   ├── .env                    # Pinecone API keys, HF tokens
│   ├── ARCHITECTURE.md         # Detailed system design diagrams
│   ├── PINECONE_SETUP.md       # Setup guide for embeddings
│   └── embeddings_output/      # Generated embeddings from experiments
│
├── client/                     # React 19 + Vite + TailwindCSS v4
│   ├── src/
│   │   ├── components/         # ChatInput, ChatMessage, DocumentUpload, etc.
│   │   ├── contexts/           # ThemeContext for dark/light mode
│   │   └── data/mockData.ts    # Mock case data for frontend demo
│   └── package.json
│
├── docs/                       # Comprehensive documentation
│   ├── SYSTEM_WORKFLOW_EXPLANATION.md
│   ├── ARCHITECTURE.md → see backend/ARCHITECTURE.md (duplicate)
│   ├── QUICK_START_GUIDE.md
│   └── REMOTE_GPU_TRAINING_GUIDE.md
│
├── pyproject.toml              # Root Python dependencies (uv-based)
└── main.py                     # Placeholder entry point
```

**Note:** The `server/` directory mentioned in older docs **does not exist**. All backend code is in `backend/`.

## Key Technologies Stack

### Backend (Python)
- **LangGraph** - Multi-agent orchestration (see `nyaya_multi_agent.py`)
- **Pinecone** - Serverless vector database (us-east-1)
- **EmbeddingGemma** - `google/embeddinggemma-300M` for 384-dim embeddings
- **Google VertexAI Gemini** - LLM for answer generation
- **FastAPI** - API server (planned, not yet implemented)
- **InLegalBERT** - `law-ai/InLegalBERT` for role classification (planned)

### Frontend (TypeScript)
- **React 19** with TypeScript
- **Vite** - Build tooling with HMR
- **TailwindCSS v4** - Styling (using @tailwindcss/vite plugin)
- **Lucide React** - Icon library

### Current Implementation Status
- ✅ Pinecone vector DB setup with 384-dim EmbeddingGemma
- ✅ Frontend UI with mock data (7 components, theme toggle)
- ✅ Multi-agent pattern designed (prototype in `nyaya_multi_agent.py`)
- ⏳ Role classifier training (InLegalBERT) - referenced but not integrated
- ⏳ FastAPI backend - planned but not implemented
- ⏳ End-to-end RAG pipeline - components exist separately

## Critical Development Patterns

### 1. Pinecone + EmbeddingGemma Integration (PRODUCTION READY)

**The system uses the official Gemma Cookbook pattern** (`backend/official_rag_pattern.py`):

```python
from sentence_transformers import SentenceTransformer

# Load model with 384-dim truncation (Matryoshka Representation Learning)
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)

# CRITICAL: Use asymmetric encoding for better retrieval (+16.7% accuracy)
# Documents: "Retrieval-document" prompt
doc_embeddings = model.encode(
    [f"title: {title} | text: {content}" for title, content in docs],
    normalize_embeddings=True  # Always normalize for cosine similarity
)

# Queries: "Retrieval-query" prompt (different from documents!)
query_embedding = model.encode(
    user_question,
    prompt_name="Retrieval-query",
    normalize_embeddings=True
)
```

**Hugging Face Authentication Required:**
```bash
# Must accept license at: https://huggingface.co/google/embeddinggemma-300M
export HF_TOKEN=hf_xxxxxxxxxxxxx  # Or add to backend/.env
python -c "from huggingface_hub import login; login()"
```

### 2. Role-Aware Metadata Schema (THE KEY INNOVATION)

Each vector in Pinecone stores role metadata for filtered retrieval:

```python
# Upload to Pinecone with role metadata
vectors = [{
    "id": f"{case_id}_sent_{i}",
    "values": embedding.tolist(),  # 384-dim float array
    "metadata": {
        "text": "The petition is allowed.",
        "role": "Decision",           # One of 7 roles
        "confidence": 0.95,            # Classifier confidence
        "case_id": "case_12345",
        "sentence_index": i,
        "user_uploaded": True
    }
}]
index.upsert(vectors=vectors, namespace="user_documents")

# Query with role filtering (NYAYA'S DIFFERENTIATOR)
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"role": {"$eq": "Reasoning"}},  # Only Reasoning sentences!
    include_metadata=True
)
```

**Why this matters:** Standard RAG returns mixed content; Nyaya returns precisely what the user asked for.

### 3. Multi-Agent Architecture (LangGraph Pattern)

See `backend/nyaya_multi_agent.py` for the complete pattern:

```python
from langgraph.graph import StateGraph, END

# Define shared state
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    user_query: str
    intent: str  # UPLOAD_AND_CLASSIFY | SIMILARITY_SEARCH | PREDICT_OUTCOME | QUESTION_ANSWERING
    search_results: list
    final_answer: str

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("detect_intent", detect_intent_node)
workflow.add_node("classify_agent", classification_agent)
workflow.add_node("similarity_agent", similarity_search_agent)
workflow.add_node("prediction_agent", prediction_agent)
workflow.add_node("rag_agent", rag_agent)

# Conditional routing based on intent
workflow.add_conditional_edges(
    "detect_intent",
    route_to_agent,
    {
        "classify_agent": "classify_agent",
        "similarity_agent": "similarity_agent",
        "prediction_agent": "prediction_agent",
        "rag_agent": "rag_agent"
    }
)
```

**Intent Detection Keywords:**
- Upload: "upload", "analyze this case", "classify"
- Similarity: "similar", "like this", "find cases", "related"
- Prediction: "predict", "outcome", "what will happen", "chances"
- Role-specific QA: "facts", "issue", "reasoning", "decision", "arguments"

### 4. Frontend Component Pattern (React 19 + TypeScript)

Components follow a consistent pattern with TypeScript interfaces:

```typescript
// types/index.ts
export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  type?: 'text' | 'analysis' | 'prediction' | 'greeting';
  text?: string;
  data?: CaseSummary | PredictionData;
  timestamp: Date;
}

// components/ChatMessage.tsx
import type { ChatMessage } from '../types';

export function ChatMessage({ message }: { message: ChatMessage }) {
  // Role-specific rendering based on message.type
  if (message.type === 'analysis' && message.data) {
    return <AnalysisView data={message.data} />;
  }
  // ...
}
```

**Mock Data Structure:** See `client/src/data/mockData.ts` for complete case structure with all 7 roles.

### 5. Training Data Format (For Future Integration)

When integrating InLegalBERT classifier, training files use **tab-separated** format:

```
The petitioner filed a writ petition.	Facts
The main issue is constitutional validity.	Issue

The respondent filed an appeal.	Facts
The court analyzed Article 14.	Reasoning
```

**CRITICAL:** 
- Use **tabs** (not spaces) between sentence and role
- Blank lines separate documents
- Role labels must match exactly: `Facts`, `Issue`, `Arguments of Petitioner`, etc.

## Common Development Tasks

### Running the System

**Frontend:**
```bash
cd client
npm install
npm run dev  # http://localhost:5173
```

**Backend (Pinecone Integration):**
```bash
cd backend

# Setup (first time only)
export HF_TOKEN=your_token  # Or add to .env
python -c "from huggingface_hub import login; login()"

# Install dependencies
pip install sentence-transformers python-dotenv pinecone
pip install "git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview"

# Test the integration
python test_pinecone_embedding.py  # Full Pinecone integration test
python official_rag_pattern.py     # Minimal RAG pattern demo

# Multi-agent demo (prototype)
python nyaya_multi_agent.py
```

### Testing Pinecone Integration

```bash
cd backend

# Check index status
python check_pinecone_status.py

# List all vectors
python list_all_vectors.py

# Test embedding + upload + query
python test_pinecone_embedding.py
```

**Expected output from test_pinecone_embedding.py:**
```
🏛️  NYAYA: Legal RAG with EmbeddingGemma (384-dim) + Pinecone
✅ Index 'nyaya-legal-rag' exists
✅ Model loaded! Embedding dimension: 384
⬆️  Upserted 6 vectors successfully!
🔍 Testing query: 'What are the main facts?'
📊 Top result (filtered by role='Facts'):
   Score: 0.8234
   Text: The petitioner filed a writ petition...
```

### Working with the Frontend

**Key Components:**
```bash
client/src/components/
├── ChatInput.tsx       # Message input + file upload
├── ChatMessage.tsx     # Message display with role-specific formatting
├── AnalysisView.tsx    # Renders structured case analysis
├── PredictionView.tsx  # Renders outcome predictions
├── DocumentUpload.tsx  # File upload interface
├── Sidebar.tsx         # Document list navigation
├── ThemeToggle.tsx     # Dark/light mode switch
└── LoadingIndicator.tsx
```

**To add a new role display:**
1. Update `types/index.ts` with the interface
2. Add rendering logic in `AnalysisView.tsx`
3. Update `mockData.ts` with sample data
4. Test with "show me [role name]" query

### Dependency Management

**Python (uses `uv` for fast installs):**
```bash
# Install all dependencies
uv pip install -r requirements.txt  # Or just: pip install

# Add new dependency
# 1. Edit pyproject.toml [project.dependencies]
# 2. Install with uv
uv pip install package-name

# IMPORTANT: pyproject.toml is source of truth
```

**Node/Frontend:**
```bash
cd client
npm install  # Install from package.json
npm add package-name  # Add new dependency
```

## Important Conventions

### File Organization
```
backend/
├── *.py files are standalone scripts (test, demo, utilities)
├── .env - Secrets (Pinecone API key, HF token) - NEVER commit
├── embeddings_output/ - Generated embeddings from experiments
└── *.md - Documentation for specific features

client/
├── src/components/ - Reusable UI components
├── src/types/index.ts - Centralized TypeScript interfaces
├── src/data/mockData.ts - Mock data for development
└── src/contexts/ - React contexts (theme, auth, etc.)

docs/
└── *.md - System-wide documentation
```

### Naming Conventions
- **Vector IDs:** `{case_id}_sent_{index}` (e.g., `case_12345_sent_0`)
- **Pinecone namespaces:** `user_documents`, `training_data`, `demo`
- **React components:** PascalCase (e.g., `ChatMessage.tsx`)
- **Python scripts:** snake_case (e.g., `test_pinecone_embedding.py`)

### Environment Variables Pattern
```bash
# backend/.env (REQUIRED)
PINECONE_API_KEY=pcsk_xxxxx...
PINECONE_INDEX_NAME=nyaya-legal-rag
PINECONE_ENVIRONMENT=us-east-1
HF_TOKEN=hf_xxxxx...
```

**Never hardcode secrets** - always use `python-dotenv`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
```

## Testing & Validation

### Expected Model Performance
- **InLegalBERT:** 85-90% accuracy on legal documents
- **Custom trained:** 90-95% on domain-specific data
- **Training time:** 2-4 hours on RTX 5000 GPU (50k sentences, 15 epochs)

### Validation Checklist
- [ ] Training data is tab-separated (not space-separated)
- [ ] Blank lines separate documents in training files
- [ ] Role labels match exactly: `Facts`, `Issue`, `Arguments of Petitioner`, etc.
- [ ] Model loads without errors
- [ ] Test predictions are reasonable (spot check with legal text)
- [ ] GPU utilization >80% during training (if using GPU)

## Domain-Specific Knowledge

### Legal Document Structure
Indian legal judgments typically follow this structure:
1. Case details (parties, court, date)
2. **Facts** - What happened
3. **Issue** - Legal questions
4. **Arguments of Petitioner/Respondent** - Both sides' positions
5. **Reasoning** - Court's analysis
6. **Decision** - Final ruling

The role classifier is trained to recognize this structure.

### Citation Format
The system is built on research published at COLING 2025:
- **Paper:** "NYAYAANUMANA and INLEGALLLAMA" (Nigam et al., 2025)
- **InLegalBERT:** "Pre-trained Language Models for the Legal Domain" (Paul et al., 2023)

Reference these when discussing the system's capabilities.

### Data Privacy
- Legal documents often contain sensitive information
- The system processes documents in-memory when possible
- Vector embeddings should be stored with appropriate access controls
- Session data should be ephemeral or encrypted

## Implementation Status & Roadmap

### ✅ What's Implemented (Production-Ready)

**Backend:**
- `test_pinecone_embedding.py` - Complete Pinecone integration with role metadata
- `official_rag_pattern.py` - Reference RAG implementation from Gemma Cookbook
- `nyaya_multi_agent.py` - LangGraph multi-agent pattern (prototype, not integrated)
- `.env` configuration - Pinecone API keys, HF token management

**Frontend:**
- Full React 19 + TypeScript UI with 8 components
- Mock data system (`mockData.ts`) demonstrating all 7 roles
- Theme toggle (dark/light mode)
- Responsive design (mobile + desktop)
- Type-safe interfaces for all data structures

**Infrastructure:**
- Pinecone serverless index configured (us-east-1, 384-dim, cosine)
- EmbeddingGemma model integration with MRL (Matryoshka) truncation
- Comprehensive documentation (10+ MD files)

### ⏳ What's In Progress / Planned

**Backend:**
- [ ] FastAPI server integration (mentioned in docs, not implemented)
- [ ] InLegalBERT role classifier integration (training notebooks exist, not in pipeline)
- [ ] PDF text extraction utilities
- [ ] LangChain RAG chain with VertexAI Gemini
- [ ] Context manager for conversation state
- [ ] Batch document processing

**Integration:**
- [ ] Frontend ↔ Backend API connection (currently uses mock data)
- [ ] File upload → Classification → Embedding → Storage pipeline
- [ ] Real-time query → RAG → Response flow
- [ ] Session management for multi-turn conversations

**Models:**
- [ ] Trained InLegalBERT classifier deployment
- [ ] Fine-tuned embedding model for legal domain
- [ ] Outcome prediction model integration

### 🎯 Critical Next Steps (Priority Order)

1. **Create FastAPI server** in `backend/` with endpoints:
   - `POST /upload` - Document upload + classification
   - `POST /query` - RAG query with role filtering
   - `POST /similar` - Find similar cases

2. **Integrate role classifier:**
   - Load trained InLegalBERT model
   - Create classification service
   - Add to document processing pipeline

3. **Connect frontend to backend:**
   - Replace mock data with API calls
   - Add error handling and loading states
   - Implement file upload

4. **Implement end-to-end RAG:**
   - Use `official_rag_pattern.py` as template
   - Add LangChain with VertexAI Gemini
   - Add conversation context tracking

## Troubleshooting

### Common Issues

**"Invalid authentication token" (Hugging Face)**
```bash
# Solution: Accept license and login
# 1. Go to https://huggingface.co/google/embeddinggemma-300M and accept license
# 2. Get token from https://huggingface.co/settings/tokens
# 3. Login:
python -c "from huggingface_hub import login; login()"
# Or set environment variable:
export HF_TOKEN=hf_xxxxx...
```

**"PineconeException: Index not found"**
```bash
# Check index exists:
python backend/check_pinecone_status.py

# If not, create it (384 dimensions):
# Edit test_pinecone_embedding.py and run it - it creates the index automatically
```

**"Dimension mismatch" in Pinecone**
```python
# Cause: Model embedding size != index dimension
# Check model dimension:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
print(model.get_sentence_embedding_dimension())  # Should be 384

# If wrong, recreate Pinecone index with correct dimensions
```

**Frontend: "Cannot find module '@/components'"**
```bash
# This is normal - frontend uses mock data for now
# Just run: cd client && npm install && npm run dev
```

**"Module 'sentence_transformers' not found"**
```bash
cd backend
pip install sentence-transformers
pip install "git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview"
```

**Slow first run / Model downloading**
- **Expected:** First run downloads ~1.2GB model
- **Solution:** Wait once, subsequent runs use cached model
- **Cache location:** `~/.cache/huggingface/`

## Additional Resources

- **Full system explanation:** `docs/SYSTEM_WORKFLOW_EXPLANATION.md`
- **Quick start:** `docs/QUICK_START_GUIDE.md`
- **Remote training:** `docs/REMOTE_GPU_TRAINING_GUIDE.md`
- **Example workflows:** `docs/EXAMPLE_FLOW_1.md`, `docs/EXAMPLE_FLOW_2.md`
- **API documentation:** `docs/DOCUMENT_QUERY_API.md`

## When Working on This Codebase

### Architectural Principles

1. **Understand the role-aware paradigm** - This isn't just RAG, it's role-aware RAG
   - Standard RAG: "What was the reasoning?" → Returns Facts + Arguments + Reasoning mixed
   - Nyaya RAG: "What was the reasoning?" → Returns **only** Reasoning sentences
   
2. **Respect the two-stage pipeline:**
   - Stage 1: Classification (InLegalBERT assigns roles to sentences)
   - Stage 2: Retrieval (Vector search filtered by role metadata)

3. **Use the official patterns:**
   - Embeddings: Follow `official_rag_pattern.py` from Gemma Cookbook
   - Multi-agent: Follow `nyaya_multi_agent.py` LangGraph pattern
   - Frontend: Follow existing component patterns in `client/src/components/`

### Development Workflow

1. **Backend changes:**
   ```bash
   cd backend
   # Edit .py files
   python your_script.py  # Test immediately (scripts are standalone)
   # No build step needed for Python
   ```

2. **Frontend changes:**
   ```bash
   cd client
   npm run dev  # Starts dev server with HMR
   # Edit .tsx files - changes appear instantly
   ```

3. **Testing integration:**
   ```bash
   # Backend
   cd backend && python test_pinecone_embedding.py
   
   # Frontend  
   cd client && npm run dev
   # Test with mock data - simulate backend responses
   ```

### Key Files to Reference

- **Architecture:** `backend/ARCHITECTURE.md` - Detailed system design with diagrams
- **Setup:** `backend/PINECONE_SETUP.md` - Complete Pinecone + EmbeddingGemma guide
- **Embeddings:** `backend/README_EMBEDDINGS.md` - Official patterns and best practices
- **Frontend types:** `client/src/types/index.ts` - All TypeScript interfaces
- **Mock data:** `client/src/data/mockData.ts` - Example of complete case structure

### Common Mistakes to Avoid

❌ **Don't use `server/` directory** - It doesn't exist. Use `backend/`.

❌ **Don't mix embedding prompts:**
```python
# Wrong:
doc_emb = model.encode(doc, prompt_name="Retrieval-query")
query_emb = model.encode(query, prompt_name="Retrieval-query")

# Right:
doc_emb = model.encode(doc, prompt_name="Retrieval-document")
query_emb = model.encode(query, prompt_name="Retrieval-query")
```

❌ **Don't forget to normalize embeddings:**
```python
# Wrong:
embeddings = model.encode(texts)

# Right:
embeddings = model.encode(texts, normalize_embeddings=True)
```

❌ **Don't query Pinecone without role filtering for role-specific questions:**
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

❌ **Don't add secrets to version control** - Use `.env` file

### Testing Checklist

Before committing:
- [ ] Backend scripts run without errors: `python test_pinecone_embedding.py`
- [ ] Frontend builds: `cd client && npm run build`
- [ ] Environment variables are in `.env`, not hardcoded
- [ ] New dependencies added to `pyproject.toml` (backend) or `package.json` (frontend)
- [ ] TypeScript types updated if data structures changed
- [ ] Mock data updated if API contract changed

## Additional Resources

- **Full system explanation:** `docs/SYSTEM_WORKFLOW_EXPLANATION.md`
- **Quick start:** `docs/QUICK_START_GUIDE.md`
- **Remote training:** `docs/REMOTE_GPU_TRAINING_GUIDE.md`
- **Example workflows:** `docs/EXAMPLE_FLOW_1.md`, `docs/EXAMPLE_FLOW_2.md`
- **API documentation:** `docs/DOCUMENT_QUERY_API.md`

## Key Differences from Standard RAG

| Aspect | Standard RAG | Nyaya RAG |
|--------|-------------|-----------|
| **Chunking** | By size (512 tokens) | By semantic role (sentence-level) |
| **Retrieval** | Similarity only | Similarity + Role filter |
| **User query:** "What were the facts?" | Returns mixed content | Returns **only** Facts |
| **User query:** "Show reasoning" | Returns full document | Returns **only** Reasoning |
| **Metadata** | Optional, generic | Required, role-specific |
| **Use case** | General Q&A | Structured legal analysis |

This role-awareness is the core innovation - maintain it in all features.
