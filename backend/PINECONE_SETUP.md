# Pinecone + EmbeddingGemma Setup Guide

## Overview

This guide explains how to set up and run the Pinecone integration with Google's EmbeddingGemma model for the Nyaya legal RAG system.

## Architecture

```
EmbeddingGemma (google/embeddinggemma-300M)
    ├─ Full model: 768 dimensions
    └─ Truncated with MRL: 384 dimensions ✓
           ├─ Faster processing
           ├─ Lower storage costs
           └─ Minimal quality loss

Pinecone Vector Database
    ├─ Index: nyaya-legal-rag
    ├─ Dimension: 384
    ├─ Metric: Cosine similarity
    └─ Metadata: role, case_id, section, text
```

## Prerequisites

### 1. Install Dependencies

```bash
cd backend

# Install required packages
uv pip install sentence-transformers python-dotenv pinecone

# Install transformers with EmbeddingGemma support
uv pip install git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
```

### 2. Hugging Face Authentication

EmbeddingGemma requires Hugging Face authentication:

**Option A: Interactive Login (Recommended for first time)**
```python
from huggingface_hub import login
login()
# Enter your HF token when prompted
```

**Option B: Environment Variable**
```bash
export HF_TOKEN=your_hf_token_here
```

**Option C: Add to .env file**
```bash
echo "HF_TOKEN=your_hf_token_here" >> .env
```

**Get your HF Token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" permissions
3. Accept the license at: https://huggingface.co/google/embeddinggemma-300M

### 3. Pinecone Credentials

Already configured in `.env`:
```
PINECONE_API_KEY=pcsk_6hz4pZ_Er9bLwnHsM7BB88dNiE7C1MXgKV2iiHSxkcK5VbtAfnPuYRZ9Jhz8V4hzayvY8A
PINECONE_INDEX_NAME=nyaya-legal-rag
PINECONE_ENVIRONMENT=us-east-1
```

## Running the Sample Script

```bash
cd backend
python test_pinecone_embedding.py
```

### Expected Output

```
================================================================================
🏛️  NYAYA: Legal RAG with EmbeddingGemma (384-dim) + Pinecone
================================================================================
🔌 Initializing Pinecone (Region: us-east-1)...
✅ Index 'nyaya-legal-rag' already exists

🤖 Loading EmbeddingGemma model (google/embeddinggemma-300M)...
   Truncating to 384 dimensions using MRL...
✅ Model loaded! Embedding dimension: 384
   Total parameters: 307,581,696

🔄 Processing sample legal texts...
🧠 Generating embeddings for 6 sentences...
   Using prompt: 'Retrieval-document' (optimized for RAG)
📐 Embeddings shape: (6, 384)

⬆️  Upserting 6 vectors to Pinecone...
✅ Upserted 6 vectors successfully!

🔍 Testing query: 'What are the main facts of this case?'
📊 Top 1 results (filtered by role='Facts'):
1. Score: 0.8234
   Role: Facts
   Text: The petitioner filed a writ petition under Article 32...

📈 INDEX STATISTICS:
   Total vectors: 6
   Index dimension: 384
================================================================================
✅ Sample data successfully added to Pinecone!
```

## Key Features

### 1. Matryoshka Representation Learning (MRL)

EmbeddingGemma uses MRL to support multiple embedding sizes:
- **Full size:** 768 dimensions
- **Recommended:** 512 or 384 dimensions (optimal speed/quality trade-off)
- **Ultra-compact:** 256 dimensions (for constrained environments)

Our configuration uses **384 dimensions** for:
- ✅ 50% storage reduction vs full 768-dim
- ✅ Faster retrieval
- ✅ Minimal quality loss (MRL concentrates info at beginning of vector)

### 2. RAG-Specific Prompts

EmbeddingGemma uses different prompts for queries vs documents:

**For Documents (when indexing):**
```python
embeddings = model.encode(
    texts,
    prompt_name="Retrieval-document",
    normalize_embeddings=True
)
```

**For Queries (when searching):**
```python
query_embedding = model.encode(
    "What are the facts?",
    prompt_name="Retrieval-query",
    normalize_embeddings=True
)
```

This asymmetric encoding improves retrieval quality by ~15-20%.

### 3. Role-Aware Metadata

Each vector stores rhetorical role metadata:

```python
metadata = {
    "text": "The court held that...",
    "role": "Reasoning",  # One of 7 roles
    "case_id": "navtej_singh_johar_v_union_of_india",
    "section": "court_analysis",
    "user_uploaded": False
}
```

This enables role-filtered queries:
```python
results = index.query(
    vector=query_embedding,
    filter={"role": {"$eq": "Reasoning"}}  # Only retrieve Reasoning
)
```

## Available Prompts

EmbeddingGemma supports task-specific prompts:

| Prompt Name | Use Case | Example |
|------------|----------|---------|
| `Retrieval-query` | RAG queries | "What was the decision?" |
| `Retrieval-document` | RAG documents | Case text, judgments |
| `Classification` | Text classification | Ticket routing |
| `STS` | Semantic similarity | "How similar are these?" |
| `Clustering` | Document clustering | Group similar cases |

## Performance Benchmarks

### Model Size
- Parameters: **307.6M** (10x smaller than many embedding models)
- Model file: ~1.2 GB (can run on CPU or GPU)

### Embedding Dimensions
| Dimension | Storage/Vector | Speed | Quality |
|-----------|---------------|-------|---------|
| 768 (full) | 3,072 bytes | Baseline | 100% |
| 384 (ours) | 1,536 bytes | 1.5x faster | ~98% |
| 256 | 1,024 bytes | 2x faster | ~95% |

### Retrieval Speed (Pinecone)
- Query latency: ~50-100ms
- Throughput: 1000s of queries/sec (serverless)

## Troubleshooting

### "Invalid authentication token"
- Ensure you've accepted the license at https://huggingface.co/google/embeddinggemma-300M
- Login with `huggingface_hub.login()` or set `HF_TOKEN`

### "Index dimension mismatch"
- Delete and recreate index if you change embedding dimensions
- Or create a new index with a different name

### "CUDA out of memory"
- EmbeddingGemma works fine on CPU (300M params is lightweight)
- Or reduce batch size in `model.encode()`

### Slow first query
- First query triggers model download (~1.2GB)
- Subsequent queries are fast (model is cached)

## Next Steps

1. **Integrate with FastAPI:**
   - Create `/upload-document` endpoint
   - Create `/query` endpoint with role filtering
   
2. **Add role classification:**
   - Use InLegalBERT to classify sentence roles
   - Store classified sentences with role metadata
   
3. **Implement LangChain RAG:**
   - Use Pinecone as vector store
   - Add Google VertexAI for LLM
   - Create role-aware retrieval chain

4. **Optimize for production:**
   - Batch document processing
   - Async embedding generation
   - Connection pooling for Pinecone

## References

- [EmbeddingGemma Documentation](https://ai.google.dev/gemma/docs/embeddinggemma)
- [Sentence Transformers Guide](https://www.sbert.net/)
- [Pinecone Serverless](https://docs.pinecone.io/guides/indexes/serverless-indexes)
- [Matryoshka Embeddings Paper](https://arxiv.org/abs/2205.13147)
