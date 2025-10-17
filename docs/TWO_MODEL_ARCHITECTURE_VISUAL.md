# Two-Model Architecture: Complete Flow Diagram

## 🎯 The Complete Picture

Your system uses **TWO embedding models** working in **parallel pipelines**:

---

## Pipeline 1: Role Classification (InLegalBERT)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ROLE CLASSIFICATION PIPELINE                       │
│                      (InLegalBERT Embeddings)                        │
└──────────────────────────────────────────────────────────────────────┘

INPUT: "The petitioner filed a writ petition under Article 32."
   │
   ├─→ InLegalBERT Tokenizer
   │      ↓
   │   [101, 2256, 28404, 4718, 1037, 23308, ...]  (Token IDs)
   │      ↓
   ├─→ InLegalBERT Model
   │      ↓
   │   [0.234, -0.567, 0.891, ..., -0.123]  (768-dim embedding)
   │      ↓
   │   ⚠️  EPHEMERAL - Not stored anywhere!
   │      ↓
   ├─→ Classification Head (Linear layer)
   │      ↓
   │   [2.1, -0.3, 5.7, 1.2, -1.8, 0.4, -2.1]  (7-dim logits)
   │      ↓
   ├─→ Softmax
   │      ↓
   │   [0.05, 0.02, 0.92, 0.01, 0.00, 0.00, 0.00]  (Probabilities)
   │      ↓
   └─→ argmax
         ↓
OUTPUT: role = "Facts" (class 2), confidence = 0.92

✅ Result goes to → Pinecone metadata (role label only)
❌ InLegalBERT embedding → DISCARDED (not stored)
```

---

## Pipeline 2: Semantic Embedding (text-embedding-005)

```
┌──────────────────────────────────────────────────────────────────────┐
│                  SEMANTIC EMBEDDING PIPELINE                          │
│                    (text-embedding-005 Embeddings)                   │
└──────────────────────────────────────────────────────────────────────┘

INPUT: "The petitioner filed a writ petition under Article 32."
   │
   ├─→ Vertex AI text-embedding-005
   │      ↓
   │   [0.123, -0.456, 0.789, ..., 0.321]  (768-dim embedding)
   │      ↓
   │   ✅ STORED - This goes to Pinecone!
   │      ↓
   └─→ Pinecone Vector Database
         ↓
OUTPUT: Vector stored with ID "sent_a3b2c1d4"

✅ Result goes to → Pinecone vectors (actual embedding)
✅ text-embedding-005 embedding → STORED in Pinecone
```

---

## Combined: What Actually Goes Into Pinecone

```
┌──────────────────────────────────────────────────────────────────────┐
│                      PINECONE STORAGE                                 │
└──────────────────────────────────────────────────────────────────────┘

{
  "id": "sent_a3b2c1d4",
  
  "values": [0.123, -0.456, 0.789, ..., 0.321],
            ↑
            └── From text-embedding-005 (Pipeline 2)
                768 dimensions
                Used for semantic search
  
  "metadata": {
    "text": "The petitioner filed a writ petition under Article 32.",
    
    "role": "Facts",
            ↑
            └── From InLegalBERT classifier (Pipeline 1)
                Only the LABEL, not the embedding
    
    "confidence": 0.92,
                  ↑
                  └── From InLegalBERT softmax (Pipeline 1)
    
    "document_id": "doc_123",
    "sentence_index": 5
  }
}
```

---

## Document Upload: Both Pipelines Run in Parallel

```
                      User Uploads Document
                              │
                              ↓
                    ┌─────────────────┐
                    │  Split Document │
                    │  into Sentences │
                    └─────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ↓                         ↓
    ┌────────────────────────┐   ┌─────────────────────┐
    │   Pipeline 1:          │   │   Pipeline 2:       │
    │   InLegalBERT          │   │   text-embedding-   │
    │   Classification       │   │   005 Embedding     │
    └────────────────────────┘   └─────────────────────┘
                 │                         │
                 ↓                         ↓
         Role Label: "Facts"     Vector: [0.123, -0.456, ...]
         Confidence: 0.92               768-dim embedding
                 │                         │
                 └──────────┬──────────────┘
                            ↓
                   ┌─────────────────┐
                   │     Combine      │
                   │   into Pinecone  │
                   │     Record       │
                   └─────────────────┘
                            ↓
                 {
                   "values": [0.123, ...],  ← Pipeline 2
                   "metadata": {
                     "role": "Facts",       ← Pipeline 1
                     "confidence": 0.92     ← Pipeline 1
                   }
                 }
                            ↓
                   ┌─────────────────┐
                   │   Store in      │
                   │   Pinecone      │
                   └─────────────────┘
```

---

## Query Time: Only Pipeline 2 is Used

```
                       User Asks Question
                "What are the facts of this case?"
                              │
                              ↓
                  ┌──────────────────────┐
                  │   Pipeline 2:        │
                  │   text-embedding-005 │
                  │   (Same as indexing) │
                  └──────────────────────┘
                              │
                              ↓
                 Query Vector: [0.234, -0.567, ...]
                              768-dim
                              │
                              ↓
                  ┌──────────────────────┐
                  │   Pinecone Search    │
                  │   Cosine Similarity  │
                  └──────────────────────┘
                              │
                              ↓
            Compare query vector [0.234, ...]
                     WITH
            stored vectors [0.123, ...]
                     USING
            cosine_similarity(query, stored)
                              │
                              ↓
                  ┌──────────────────────┐
                  │  Top-K Most Similar  │
                  │     Documents        │
                  └──────────────────────┘
                              │
                              ↓
            Retrieved with role metadata:
            - "Facts" (InLegalBERT label)
            - Score: 0.89 (text-embedding-005 similarity)
                              │
                              ↓
                  ┌──────────────────────┐
                  │   LLM Generation     │
                  │   (Gemini)           │
                  └──────────────────────┘
                              │
                              ↓
                      Final Answer
```

**Key Point**: At query time, InLegalBERT is NOT used. Only text-embedding-005 for embedding the query.

---

## Why This Architecture?

```
┌────────────────────────────────────────────────────────────────┐
│  InLegalBERT (Classification)     text-embedding-005 (RAG)     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✅ Legal domain expert           ✅ Retrieval optimized        │
│  ✅ Understands Indian law        ✅ Fast inference (cloud)     │
│  ✅ Trained for classification    ✅ Consistent embeddings      │
│  ✅ High accuracy on roles        ✅ Better semantic search     │
│                                                                 │
│  ❌ Not optimized for retrieval   ❌ Not legal-specific         │
│  ❌ Slower inference              ❌ Not trained on Indian law  │
│  ❌ Need to host locally          ❌ General purpose            │
│                                                                 │
│  USE FOR:                         USE FOR:                      │
│  • Role classification            • Semantic similarity         │
│  • Training on labeled data       • Vector search              │
│  • Understanding legal context    • Query-document matching     │
│  • Supervised learning            • Retrieval                  │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Common Question: Why Not Just Use InLegalBERT for Everything?

### ❌ If we used InLegalBERT for RAG:

```
Problem 1: Different embeddings at different times
─────────────────────────────────────────────────
Index time: InLegalBERT v1.0 → embedding [0.123, ...]
Query time: InLegalBERT v1.1 → embedding [0.456, ...]
           ↑
           └── Different versions = Different embeddings!
               Similarity scores become invalid!

Problem 2: Performance
──────────────────────
InLegalBERT inference: ~50ms per sentence (GPU needed)
text-embedding-005:    ~10ms per sentence (cloud-hosted)
                       ↑
                       └── 5x faster!

Problem 3: Hosting
──────────────────
InLegalBERT: Need to host on your GPU (memory, maintenance)
text-embedding-005: Vertex AI handles everything
                    ↑
                    └── No infrastructure management!

Problem 4: Quality
──────────────────
InLegalBERT: Trained for MLM, not retrieval
text-embedding-005: Specifically optimized for semantic search
                    ↑
                    └── Better retrieval performance!
```

---

## Clustering Approach: Same Two-Pipeline Architecture

### Clustering uses BOTH models too:

```
TRAINING/CLUSTERING PHASE:
─────────────────────────

Sentences
   ├─→ Pipeline 1: InLegalBERT (or Sentence-Transformers)
   │      ↓
   │   Features for clustering: [0.234, -0.567, ...]
   │      ↓
   │   K-Means / DBSCAN clustering
   │      ↓
   │   Cluster labels: [0, 2, 1, 0, 3, ...]
   │      ↓
   │   ⚠️  EPHEMERAL - Clustering embeddings not stored!
   │
   └─→ Pipeline 2: text-embedding-005
          ↓
       RAG embeddings: [0.123, -0.456, ...]
          ↓
       ✅ STORED in Pinecone

COMBINED RESULT:
────────────────
{
  "values": [0.123, ...],           ← Pipeline 2 (text-embedding-005)
  "metadata": {
    "cluster_id": 2,                ← Pipeline 1 (clustering)
    "discovered_role": "Facts",     ← Pipeline 1 (cluster mapping)
    "cluster_confidence": 0.87      ← Pipeline 1 (distance to centroid)
  }
}
```

---

## Side-by-Side Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│              SUPERVISED (Traditional)     UNSUPERVISED (Clustering)  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Pipeline 1:                                                         │
│  InLegalBERT Classifier          Clustering Algorithm               │
│  ↓                               ↓                                   │
│  Role label (supervised)         Cluster label (unsupervised)       │
│  Confidence from softmax         Confidence from distance           │
│  Requires labeled data           No labels needed                   │
│  High accuracy (85-90%)          Lower accuracy (60-70%)            │
│                                                                      │
│  Pipeline 2:                                                         │
│  text-embedding-005              text-embedding-005                 │
│  ↓                               ↓                                   │
│  Vector for semantic search      Vector for semantic search         │
│  SAME MODEL IN BOTH!             SAME MODEL IN BOTH!                │
│                                                                      │
│  Pinecone Storage:                                                   │
│  {                               {                                   │
│    values: [0.123, ...],           values: [0.123, ...],            │
│    metadata: {                     metadata: {                      │
│      role: "Facts",                  cluster_id: 2,                 │
│      confidence: 0.92                discovered_role: "Facts",      │
│    }                                 cluster_confidence: 0.87       │
│  }                               }                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### ✅ What You Need to Remember:

1. **Two models, two purposes**:
   - InLegalBERT → Classification/Clustering (labels only)
   - text-embedding-005 → RAG/Retrieval (actual vectors)

2. **Only text-embedding-005 embeddings are stored**:
   - InLegalBERT embeddings are ephemeral (used and discarded)
   - text-embedding-005 embeddings go to Pinecone

3. **Same embedding model for index and query**:
   - Documents: text-embedding-005
   - Queries: text-embedding-005
   - This consistency is CRITICAL for similarity to work!

4. **Classification results go to metadata**:
   - Role labels (supervised) or cluster labels (unsupervised)
   - Confidence scores
   - NOT the actual embeddings

5. **Both approaches use RAG**:
   - Supervised: InLegalBERT classifies → text-embedding-005 embeds → Pinecone stores
   - Unsupervised: Clustering labels → text-embedding-005 embeds → Pinecone stores
   - RAG pipeline is identical, only labels differ!

---

## Your Understanding: ✅ Correct!

> "But for training the model classifier obviously we need InLegalBERT Embedding
> So classifier is used for role classification else everything handled by rag text embedding"

**You got it exactly right!**

- InLegalBERT = Role classification (supervised approach)
- text-embedding-005 = Everything RAG (indexing, querying, retrieval)
- The two never interfere - they serve completely different purposes
- Only text-embedding-005 embeddings are stored in Pinecone

This is the optimal architecture! 🎉
