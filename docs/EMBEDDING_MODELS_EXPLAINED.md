# Embedding Models: Classifier vs RAG

## 🎯 Key Point: TWO DIFFERENT EMBEDDING MODELS

Your system uses **TWO separate embedding models** for **TWO different purposes**:

---

## Model 1: InLegalBERT (for Role Classification)

### Purpose
**Role Classification ONLY** - Classifying sentences into 7 rhetorical roles

### Model Details
- **Name**: `law-ai/InLegalBERT`
- **Type**: BERT-based transformer fine-tuned on Indian legal corpus
- **Usage**: Training and inference of role classifier
- **Output Dimension**: 768 (BERT hidden size)
- **Where Used**: `role_classifier.py`

### Training Process
```python
# In train_role_classifier.ipynb

# 1. Load InLegalBERT base model
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# 2. Add classification head
model.classifier = nn.Linear(768, 7)  # 7 roles

# 3. Train on labeled data
for sentence, role_label in training_data:
    # Tokenize with InLegalBERT tokenizer
    inputs = tokenizer(sentence, return_tensors="pt")
    
    # Get InLegalBERT embeddings
    embeddings = model(**inputs)  # [768-dim vector]
    
    # Classify
    logits = classifier(embeddings)  # [7-dim vector]
    loss = criterion(logits, role_label)
    
    # Backprop
    loss.backward()

# 4. Save trained classifier
torch.save(model.state_dict(), "role_classifier_final.pt")
```

### Inference Process
```python
# In role_classifier.py

class InLegalBERTClassifier:
    def __init__(self):
        # Load InLegalBERT + trained classification head
        self.model = AutoModel.from_pretrained("law-ai/InLegalBERT")
        self.model.load_state_dict(torch.load("role_classifier_final.pt"))
    
    def classify_sentence(self, sentence):
        # Tokenize with InLegalBERT tokenizer
        inputs = self.tokenizer(sentence, return_tensors="pt")
        
        # Get InLegalBERT embeddings
        embeddings = self.model(**inputs)  # [768-dim]
        
        # Classify to get role
        logits = self.classifier(embeddings)  # [7-dim]
        predicted_role = torch.argmax(logits)
        confidence = torch.softmax(logits, dim=-1).max()
        
        return predicted_role, confidence
```

**Important**: These InLegalBERT embeddings are **NOT stored** anywhere. They're only used during classification and then discarded.

---

## Model 2: text-embedding-005 (for RAG)

### Purpose
**Semantic Search ONLY** - Finding similar documents in vector database

### Model Details
- **Name**: `text-embedding-005` (Google Vertex AI)
- **Type**: Optimized embedding model for semantic similarity
- **Usage**: Generating embeddings for Pinecone vector database
- **Output Dimension**: 768
- **Where Used**: `legal_rag.py`, `document_processor.py`

### Indexing Process
```python
# In legal_rag.py or document_processor.py

# 1. Initialize embedding model
embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

# 2. First, classify sentence with InLegalBERT classifier
role, confidence = role_classifier.classify_sentence(sentence)

# 3. Then, generate embedding for RAG with text-embedding-005
embedding = embeddings.embed_query(sentence)  # [768-dim vector]

# 4. Store in Pinecone
pinecone_index.upsert(
    vectors=[{
        "id": "sent_123",
        "values": embedding,  # ← text-embedding-005 embedding stored here
        "metadata": {
            "text": sentence,
            "role": role,  # ← InLegalBERT classification result stored here
            "confidence": confidence
        }
    }]
)
```

### Query Process
```python
# In legal_rag.py

def query(self, user_query):
    # 1. Embed query with text-embedding-005
    query_embedding = self.embeddings.embed_query(user_query)  # [768-dim]
    
    # 2. Search in Pinecone using cosine similarity
    results = self.pinecone_index.query(
        vector=query_embedding,  # ← text-embedding-005 embedding
        top_k=10
    )
    
    # 3. Retrieved documents were also embedded with text-embedding-005
    # So similarity calculation is valid!
    
    return results
```

---

## Why Two Different Models?

### InLegalBERT is better for Classification because:
- ✅ Fine-tuned on **legal domain** (Indian case law)
- ✅ Understands **legal terminology** and **context**
- ✅ Better at **distinguishing rhetorical roles** in legal text
- ✅ Trained specifically for **supervised learning** tasks

### text-embedding-005 is better for RAG because:
- ✅ Optimized for **semantic similarity** and **retrieval**
- ✅ **Faster inference** (hosted on Vertex AI)
- ✅ **Better generalization** across different query types
- ✅ **Consistent embeddings** (important for vector search)
- ✅ No need to host model locally

---

## Complete Workflow with Both Models

### 1️⃣ Training Phase (One-Time)

```python
# Train role classifier using InLegalBERT
# File: train_role_classifier.ipynb

from transformers import AutoTokenizer, AutoModel

# Load InLegalBERT
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
base_model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# Add classification head
classifier_head = nn.Linear(768, 7)

# Train on labeled dataset
for sentence, role_label in training_data:
    inputs = tokenizer(sentence, return_tensors="pt")
    embeddings = base_model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
    logits = classifier_head(embeddings)
    
    loss = criterion(logits, role_label)
    loss.backward()
    optimizer.step()

# Save trained model
torch.save({
    'model_state_dict': base_model.state_dict(),
    'classifier_state_dict': classifier_head.state_dict()
}, 'role_classifier_final.pt')
```

**Result**: `role_classifier_final.pt` containing trained InLegalBERT classifier

---

### 2️⃣ Document Upload Phase (Runtime)

```python
# File: agent_orchestrator.py or document_processor.py

def process_document(document_text):
    # Step 1: Split into sentences
    sentences = split_into_sentences(document_text)
    
    # Step 2: Classify each sentence with InLegalBERT
    role_classifier = InLegalBERTClassifier()  # Uses InLegalBERT
    
    classified_sentences = []
    for sentence in sentences:
        role, confidence = role_classifier.classify_sentence(sentence)
        # ↑ Uses InLegalBERT embeddings internally (not stored)
        
        classified_sentences.append({
            "text": sentence,
            "role": role,  # Facts, Issue, Arguments, etc.
            "confidence": confidence
        })
    
    # Step 3: Generate embeddings for RAG with text-embedding-005
    embeddings_model = VertexAIEmbeddings(model_name="text-embedding-005")
    
    vectors_to_store = []
    for sent_data in classified_sentences:
        # Generate RAG embedding (DIFFERENT from InLegalBERT embedding)
        rag_embedding = embeddings_model.embed_query(sent_data["text"])
        # ↑ This is text-embedding-005, NOT InLegalBERT!
        
        vectors_to_store.append({
            "id": f"sent_{uuid.uuid4()}",
            "values": rag_embedding,  # ← text-embedding-005 embedding
            "metadata": {
                "text": sent_data["text"],
                "role": sent_data["role"],  # ← InLegalBERT classification
                "confidence": sent_data["confidence"]
            }
        })
    
    # Step 4: Store in Pinecone
    pinecone_index.upsert(vectors=vectors_to_store)
    
    return {
        "sentences_processed": len(sentences),
        "embeddings_stored": len(vectors_to_store)
    }
```

**Summary**:
- InLegalBERT → Used to **classify** → Role label goes in metadata
- text-embedding-005 → Used to **embed** → Embedding vector goes in Pinecone
- InLegalBERT embeddings are **discarded** after classification

---

### 3️⃣ Query Phase (Runtime)

```python
# File: legal_rag.py

def query(user_query, role_filter=None):
    # Step 1: Embed query with text-embedding-005 (SAME as indexed docs)
    embeddings_model = VertexAIEmbeddings(model_name="text-embedding-005")
    query_embedding = embeddings_model.embed_query(user_query)
    # ↑ NOT using InLegalBERT here!
    
    # Step 2: Search in Pinecone
    # This works because both query and documents use text-embedding-005
    search_results = pinecone_index.query(
        vector=query_embedding,  # ← text-embedding-005
        filter={"role": role_filter} if role_filter else None,
        top_k=10
    )
    
    # Step 3: Extract retrieved documents
    retrieved_docs = []
    for match in search_results['matches']:
        retrieved_docs.append({
            "text": match['metadata']['text'],
            "role": match['metadata']['role'],  # ← InLegalBERT classification
            "score": match['score'],  # ← Similarity with text-embedding-005
            "confidence": match['metadata']['confidence']
        })
    
    # Step 4: Generate answer with LLM
    context = "\n".join([doc['text'] for doc in retrieved_docs])
    answer = llm.generate(f"Context: {context}\n\nQuestion: {user_query}")
    
    return {
        "answer": answer,
        "sources": retrieved_docs
    }
```

**Summary**:
- Query embedded with **text-embedding-005** (for semantic search)
- Retrieved documents also embedded with **text-embedding-005**
- Role labels (from InLegalBERT) used only for **filtering**
- InLegalBERT **NOT used** during query phase

---

## Visual Representation

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                            │
└─────────────────────────────────────────────────────────────────┘

Labeled Data → InLegalBERT → Classification Head → Trained Classifier
(sentence, role)     ↓                ↓
                 [768-dim]        [7-dim logits]
                 embeddings
                    ↓
           USED FOR TRAINING ONLY
           (Not stored anywhere)

Result: role_classifier_final.pt


┌─────────────────────────────────────────────────────────────────┐
│                     DOCUMENT UPLOAD PHASE                        │
└─────────────────────────────────────────────────────────────────┘

Document → Sentences
             ↓
             ├─→ InLegalBERT Classifier → Role Label ──┐
             │   (Uses InLegalBERT embeddings)          │
             │   (Embeddings discarded after)            │
             │                                           ↓
             └─→ text-embedding-005 → RAG Embedding → Pinecone
                 (768-dim vector STORED)              (vector + metadata)


┌─────────────────────────────────────────────────────────────────┐
│                        QUERY PHASE                               │
└─────────────────────────────────────────────────────────────────┘

User Query → text-embedding-005 → Query Embedding (768-dim)
                                        ↓
                                   Pinecone Search
                                   (cosine similarity)
                                        ↓
                            Retrieved Documents (with role labels)
                                        ↓
                                   LLM Generation
                                        ↓
                                    Final Answer
```

---

## What's Stored in Pinecone?

```json
{
  "id": "sent_a3b2c1d4",
  "values": [0.123, -0.456, 0.789, ...],  // ← text-embedding-005 (768-dim)
  "metadata": {
    "text": "The petitioner filed a writ petition under Article 32.",
    "role": "Facts",  // ← InLegalBERT classification result
    "confidence": 0.92,  // ← InLegalBERT confidence
    "document_id": "doc_123",
    "sentence_index": 5
  }
}
```

**Key Insight**: 
- **Vectors** = text-embedding-005 embeddings (for semantic search)
- **Metadata** = InLegalBERT classification results (for filtering/organization)

---

## Common Misconceptions ❌

### ❌ Misconception 1: "We use InLegalBERT embeddings for RAG"
**✅ Reality**: InLegalBERT is only used for classification. RAG uses text-embedding-005.

### ❌ Misconception 2: "We need to store InLegalBERT embeddings in Pinecone"
**✅ Reality**: InLegalBERT embeddings are ephemeral - used only during classification, then discarded.

### ❌ Misconception 3: "We should use the same model for both tasks"
**✅ Reality**: Different models are optimized for different tasks. InLegalBERT for classification, text-embedding-005 for retrieval.

### ❌ Misconception 4: "Query should be embedded with InLegalBERT"
**✅ Reality**: Query must be embedded with text-embedding-005 (same as indexed documents) for similarity to work.

---

## Why Not Use InLegalBERT for Everything?

### Problems with using InLegalBERT for RAG:

1. **Not optimized for retrieval**: InLegalBERT is trained for masked language modeling, not similarity search
2. **Slower inference**: Would need to host model locally or on your GPU
3. **Worse retrieval quality**: text-embedding-005 is specifically designed for semantic similarity
4. **No vector DB optimization**: Vertex AI embeddings integrate better with Google Cloud infrastructure
5. **Maintenance burden**: Need to keep InLegalBERT model updated and hosted

### Benefits of current two-model approach:

1. ✅ **Best of both worlds**: Legal domain knowledge for classification, optimized retrieval for RAG
2. ✅ **Faster inference**: text-embedding-005 is cloud-hosted and optimized
3. ✅ **Better retrieval**: Purpose-built embedding model for semantic search
4. ✅ **Easier deployment**: No need to host InLegalBERT for RAG
5. ✅ **Scalability**: Vertex AI handles load and scaling

---

## Clustering Approach: Which Model?

In the clustering approach (`clustering_role_classifier.py`):

### Option 1: Use Sentence-Transformers (Current Implementation)
```python
from sentence_transformers import SentenceTransformer

# For clustering feature extraction
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim

# Generate features for clustering
features = model.encode(sentences)

# Cluster
kmeans = KMeans(n_clusters=7)
cluster_labels = kmeans.fit_predict(features)

# But for RAG, still use text-embedding-005!
rag_embeddings = vertex_ai_embeddings.embed_documents(sentences)
```

### Option 2: Use InLegalBERT for Clustering (Better for Legal)
```python
from transformers import AutoTokenizer, AutoModel

# Load InLegalBERT for feature extraction
tokenizer = AutoTokenizer.from_pretrained("law-ai/InLegalBERT")
model = AutoModel.from_pretrained("law-ai/InLegalBERT")

# Generate features for clustering
features = []
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    embeddings = model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
    features.append(embeddings.detach().numpy())

# Cluster
kmeans = KMeans(n_clusters=7)
cluster_labels = kmeans.fit_predict(features)

# But for RAG, still use text-embedding-005!
rag_embeddings = vertex_ai_embeddings.embed_documents(sentences)
```

**Recommendation**: Use InLegalBERT for clustering features (better legal understanding), but text-embedding-005 for RAG embeddings.

---

## Summary Table

| Model | Purpose | When Used | Output Dimension | Stored? |
|-------|---------|-----------|-----------------|---------|
| **InLegalBERT** | Role classification | Training + Document upload | 768 | ❌ No (ephemeral) |
| **text-embedding-005** | Semantic search (RAG) | Document upload + Query | 768 | ✅ Yes (in Pinecone) |
| **Sentence-Transformers** | Clustering features (optional) | Clustering approach | 384 | ❌ No (ephemeral) |

---

## Key Takeaways

1. **Two models, two purposes**: InLegalBERT classifies, text-embedding-005 retrieves
2. **InLegalBERT embeddings are NOT stored**: Only used during classification, then discarded
3. **Only text-embedding-005 embeddings go to Pinecone**: For semantic search
4. **Role labels from InLegalBERT stored as metadata**: For filtering and organization
5. **Query uses text-embedding-005**: Must match indexed documents' embedding model
6. **This is the correct architecture**: Each model does what it's best at

---

## Your Understanding is Correct! ✅

You said:
> "But for training the model classifier obviously we need InLegalBERT Embedding So classifier is used for role classification else everything handled by rag text embedding"

**This is 100% correct!**

- InLegalBERT → Classifier training and inference (role classification only)
- text-embedding-005 → Everything RAG-related (indexing, querying, retrieval)
- The two models serve completely different purposes and never interfere with each other
