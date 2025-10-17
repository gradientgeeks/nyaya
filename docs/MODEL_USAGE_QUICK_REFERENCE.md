# Model Usage Summary: Quick Reference

## 🎯 One-Sentence Summary

**InLegalBERT classifies roles (labels to metadata), text-embedding-005 enables RAG (vectors to Pinecone).**

---

## Model Comparison Table

| Feature | InLegalBERT | text-embedding-005 |
|---------|-------------|-------------------|
| **Purpose** | Role Classification | Semantic Search (RAG) |
| **Used During** | Training + Document Upload | Document Upload + Query |
| **Output** | Role label + confidence | 768-dim vector |
| **Stored in Pinecone?** | ❌ No (only label in metadata) | ✅ Yes (actual vector) |
| **Model Type** | BERT fine-tuned on legal corpus | Google's embedding model |
| **Dimension** | 768 (intermediate, not stored) | 768 (stored) |
| **Training Required?** | ✅ Yes (supervised) | ❌ No (pre-trained) |
| **Hosted Where?** | Your server/GPU | Vertex AI (cloud) |
| **Speed** | ~50ms/sentence | ~10ms/sentence |
| **Best For** | Understanding legal roles | Finding similar documents |

---

## What Goes Into Pinecone?

### ✅ Stored (Actual Data):
- **Vector values**: From `text-embedding-005` (768 dimensions)
- **Text**: Original sentence content
- **Metadata fields**: Document ID, sentence index, etc.

### 📋 Stored (Metadata Only):
- **Role label**: From `InLegalBERT` classifier (e.g., "Facts", "Issue")
- **Confidence score**: From `InLegalBERT` softmax probabilities
- **Cluster ID**: From clustering (if using unsupervised approach)

### ❌ NOT Stored:
- InLegalBERT embeddings (ephemeral, discarded after classification)
- Clustering features (ephemeral, discarded after clustering)

---

## Code Examples

### ❌ WRONG: Using InLegalBERT for RAG
```python
# This is WRONG!
classifier = InLegalBERTClassifier()
embedding = classifier.get_embedding(sentence)  # InLegalBERT embedding
pinecone_index.upsert(vectors=[(id, embedding)])  # Wrong model!

# Query
query_embedding = classifier.get_embedding(query)  # InLegalBERT
results = pinecone_index.query(query_embedding)  # Will work but suboptimal
```

**Problems**:
- Not optimized for retrieval
- Slower inference
- Need to host model
- Less consistent results

---

### ✅ CORRECT: Two-Pipeline Approach
```python
# Pipeline 1: Classification (InLegalBERT)
classifier = InLegalBERTClassifier()
role, confidence = classifier.classify_sentence(sentence)
# ↑ InLegalBERT embeddings used internally, then discarded

# Pipeline 2: RAG Embedding (text-embedding-005)
embeddings_model = VertexAIEmbeddings(model_name="text-embedding-005")
rag_embedding = embeddings_model.embed_query(sentence)
# ↑ This is what gets stored!

# Store in Pinecone
pinecone_index.upsert(vectors=[{
    "id": "sent_123",
    "values": rag_embedding,  # ← text-embedding-005
    "metadata": {
        "text": sentence,
        "role": role,  # ← InLegalBERT result (label only)
        "confidence": confidence  # ← InLegalBERT confidence
    }
}])

# Query
query_embedding = embeddings_model.embed_query(query)  # text-embedding-005
results = pinecone_index.query(query_embedding)  # Correct!
```

**Benefits**:
- Best model for each task
- Optimized performance
- Consistent embeddings
- Easy deployment

---

## Workflow Diagrams

### Document Upload
```
Document
   ↓
Split into sentences
   ↓
For each sentence:
   ├─→ InLegalBERT.classify() → role="Facts", conf=0.92
   │   (embedding used internally, then discarded)
   │
   └─→ VertexAI.embed() → vector=[0.123, -0.456, ...]
       (embedding stored in Pinecone)
   ↓
Combine into Pinecone record:
{
  values: [0.123, ...],      ← text-embedding-005
  metadata: {
    role: "Facts",           ← InLegalBERT (label only)
    confidence: 0.92         ← InLegalBERT
  }
}
```

### Query Processing
```
User Query
   ↓
VertexAI.embed(query) → query_vector=[0.234, ...]
   ↓                     (text-embedding-005)
Pinecone.query(query_vector)
   ↓
Find similar vectors (cosine similarity)
   ↓
Retrieve documents with metadata:
   - Text content
   - Role label (from InLegalBERT)
   - Similarity score (from text-embedding-005)
   ↓
LLM generates answer
```

**Note**: InLegalBERT is NOT used during query processing!

---

## Decision Tree: Which Model to Use?

```
Need to...
├─ Classify sentence role?
│  └─→ Use InLegalBERT classifier
│      - Training: Fine-tune on labeled data
│      - Inference: Get role label + confidence
│      - Result: Metadata only (not stored as vector)
│
└─ Find similar documents?
   └─→ Use text-embedding-005
       - Indexing: Embed all documents
       - Query: Embed user question
       - Search: Cosine similarity in Pinecone
       - Result: Vectors stored + retrieved
```

---

## Common Mistakes to Avoid

### ❌ Mistake 1: Storing InLegalBERT embeddings
```python
# WRONG!
embedding = inlegal_bert.encode(sentence)
pinecone.upsert([(id, embedding)])
```
**Fix**: Use text-embedding-005 for storage, InLegalBERT only for classification

### ❌ Mistake 2: Using different models for index and query
```python
# WRONG!
# At index time:
doc_embedding = model_A.embed(document)

# At query time:
query_embedding = model_B.embed(query)  # Different model!
```
**Fix**: Use text-embedding-005 for BOTH indexing and querying

### ❌ Mistake 3: Using InLegalBERT for query embedding
```python
# WRONG!
query_embedding = inlegal_bert.encode(query)
results = pinecone.query(query_embedding)
```
**Fix**: Use text-embedding-005 for query (same as documents)

### ❌ Mistake 4: Thinking you need to store both embeddings
```python
# WRONG!
{
  "inlegal_bert_embedding": [...],  # Don't store this!
  "rag_embedding": [...]  # Only need this!
}
```
**Fix**: Only store text-embedding-005 embeddings

---

## Performance Characteristics

### InLegalBERT Classifier
```
Purpose: Role classification
Speed: ~50ms per sentence (GPU)
Accuracy: 85-90% (with good training data)
Memory: ~500MB model size
Output: Role label + confidence (2 fields)
Storage: ~20 bytes (just metadata)
```

### text-embedding-005
```
Purpose: Semantic similarity
Speed: ~10ms per sentence (cloud API)
Accuracy: High for retrieval tasks
Memory: 0 (cloud-hosted)
Output: 768-dimensional vector
Storage: ~3KB per sentence (vector + metadata)
```

---

## Clustering Approach: Which Model?

### Option 1: Sentence-Transformers (Current)
```python
# For clustering
clustering_model = SentenceTransformer('all-MiniLM-L6-v2')
features = clustering_model.encode(sentences)  # 384-dim
clusters = KMeans(n_clusters=7).fit_predict(features)

# For RAG (separate!)
rag_model = VertexAIEmbeddings("text-embedding-005")
rag_embeddings = rag_model.embed_documents(sentences)  # 768-dim

# Store only RAG embeddings
pinecone.upsert(vectors=[(id, rag_emb, {"cluster_id": cluster})])
```

### Option 2: InLegalBERT for Clustering (Recommended)
```python
# For clustering (better for legal domain)
inlegal_bert = AutoModel.from_pretrained("law-ai/InLegalBERT")
features = inlegal_bert.encode(sentences)  # 768-dim
clusters = KMeans(n_clusters=7).fit_predict(features)

# For RAG (separate!)
rag_model = VertexAIEmbeddings("text-embedding-005")
rag_embeddings = rag_model.embed_documents(sentences)  # 768-dim

# Store only RAG embeddings
pinecone.upsert(vectors=[(id, rag_emb, {"cluster_id": cluster})])
```

**Key Point**: Even in clustering, only text-embedding-005 embeddings are stored!

---

## Summary: The Golden Rules

### Rule 1: Classification ≠ Storage
- InLegalBERT classifies → role label goes to metadata
- text-embedding-005 embeds → vector goes to Pinecone
- Never store classification embeddings

### Rule 2: Consistency is Key
- Index with text-embedding-005
- Query with text-embedding-005
- SAME model for both = valid similarity scores

### Rule 3: Right Tool for the Job
- Legal understanding → InLegalBERT
- Semantic search → text-embedding-005
- Don't use classification model for retrieval

### Rule 4: Metadata vs Vectors
- Vectors: From embedding model (text-embedding-005)
- Metadata: From classifier (InLegalBERT) or clusterer
- Only vectors are used for similarity search

---

## Quick Verification Checklist

When implementing, verify:

- [ ] Documents embedded with `text-embedding-005`
- [ ] Queries embedded with `text-embedding-005` (same model!)
- [ ] InLegalBERT used only for classification
- [ ] InLegalBERT embeddings NOT stored in Pinecone
- [ ] Only role labels from InLegalBERT in metadata
- [ ] Vector dimension is 768 (text-embedding-005)
- [ ] Cosine similarity used for search

---

## Your Statement: Confirmed ✅

> "But for training the model classifier obviously we need InLegalBERT Embedding
> So classifier is used for role classification else everything handled by rag text embedding"

**This is 100% correct!**

- ✅ InLegalBERT: Training + Classification (role labels)
- ✅ text-embedding-005: RAG (indexing + querying + retrieval)
- ✅ Two models, two purposes, no interference
- ✅ Only RAG embeddings stored in Pinecone

You understand the architecture perfectly! 🎉
