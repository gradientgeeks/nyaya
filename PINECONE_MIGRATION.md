# Pinecone Migration Summary

## Overview

Successfully migrated the Nyaya Legal Document Analysis System from ChromaDB to Pinecone with **role-based indexes** for better organization and performance.

## Key Changes

### Architecture

**Before (ChromaDB):**
- Single collection for all documents
- Metadata filtering for roles
- Local storage

**After (Pinecone):**
- **Separate index per rhetorical role** (facts, issue, arguments_petitioner, arguments_respondent, reasoning, decision, none)
- **Main index** for cross-role queries
- **case_id metadata** for document mapping
- Cloud-based, serverless, auto-scaling

### Files Modified

1. **`server/pyproject.toml`**
   - Removed: `chromadb==1.0.20`, `langchain-chroma>=0.2.5`
   - Added: `pinecone[grpc]==5.0.0`, `langchain-pinecone==0.3.0`

2. **`server/.env`**
   - Added: `PINECONE_INDEX_NAME=nyaya-legal-rag`
   - Added: `PINECONE_ENVIRONMENT=us-east-1`
   - Kept: `PINECONE_API_KEY=pcsk_...`

3. **`server/src/config/pinecone_setup.py`** ✨ NEW
   - `PineconeIndexManager`: Manages role-based indexes
   - Lazy initialization support (Kaggle-friendly)
   - CLI for index management
   - Batch operations (create/delete/reset all indexes)

4. **`server/src/core/legal_rag.py`** 🔄 MIGRATED
   - Uses `PineconeVectorStore` instead of `Chroma`
   - Role-specific vectorstores (`self.role_vectorstores`)
   - Lazy index creation on first use
   - case_id filtering in retrieval

5. **`server/src/config/vector_db_config.py`** ⚠️ DEPRECATED
   - Marked as deprecated
   - Use `pinecone_setup.py` and `legal_rag.py` instead

## Role-Based Index Structure

Each role gets its own Pinecone index:

```
nyaya-legal-rag-facts               # Facts role
nyaya-legal-rag-issue               # Issue role
nyaya-legal-rag-arguments_petitioner # Arguments of Petitioner
nyaya-legal-rag-arguments_respondent # Arguments of Respondent
nyaya-legal-rag-reasoning           # Reasoning role
nyaya-legal-rag-decision            # Decision role
nyaya-legal-rag-none                # Unclassified
nyaya-legal-rag                     # Main index (cross-role)
```

## Usage

### 1. Install Dependencies

```bash
cd server
uv sync
```

### 2. Configure Environment

Ensure `.env` has:
```env
PINECONE_API_KEY=your_api_key_here
PINECONE_INDEX_NAME=nyaya-legal-rag
PINECONE_ENVIRONMENT=us-east-1
```

### 3. Initialize Indexes

**Option A: Create all indexes immediately**
```bash
python -m src.config.pinecone_setup --action create
```

**Option B: Use lazy initialization (recommended for Kaggle)**
```python
from src.core.legal_rag import LegalRAGSystem

# Indexes created automatically on first use
rag_system = LegalRAGSystem(lazy_init=True)
```

### 4. Using the System

```python
from src.core.legal_rag import LegalRAGSystem

# Initialize with lazy loading (Kaggle-friendly)
rag_system = LegalRAGSystem(lazy_init=True)

# Process a legal document
tagged_docs = rag_system.process_legal_document(
    document_text,
    doc_metadata={"case_id": "case_12345", "court": "Supreme Court"}
)

# Add to role-specific Pinecone indexes
rag_system.add_documents_to_store(tagged_docs)

# Query with role filtering
response = rag_system.query_legal_rag(
    "What were the facts of the case?",
    specific_roles=["Facts", "Issue"],
    k=5
)

# Query specific case by case_id
docs = rag_system.retrieve_by_role(
    query="court's reasoning",
    roles=["Reasoning"],
    case_id="case_12345"
)
```

## CLI Management

```bash
# Check index stats
python -m src.config.pinecone_setup --action stats

# Create all indexes
python -m src.config.pinecone_setup --action create

# List all indexes
python -m src.config.pinecone_setup --action list

# Clear all vectors (keep indexes)
python -m src.config.pinecone_setup --action clear

# Delete all indexes (WARNING: permanent)
python -m src.config.pinecone_setup --action delete

# Reset all indexes (delete + recreate)
python -m src.config.pinecone_setup --action reset
```

## Benefits

### 1. **Role Isolation**
- Each role has dedicated index
- Better organization and querying
- Clearer separation of legal document structure

### 2. **Case-Specific Retrieval**
- Filter by `case_id` for document-specific queries
- Multi-case support with proper isolation

### 3. **Scalability**
- Serverless Pinecone auto-scales
- No local storage concerns
- Handles large document collections

### 4. **Resource-Friendly**
- Lazy initialization for Kaggle/limited environments
- Create indexes only when needed
- Reduced upfront resource usage

### 5. **Performance**
- Optimized role-specific queries
- Faster retrieval with smaller, focused indexes
- Better relevance with role-aware search

## Kaggle-Specific Optimizations

### Lazy Initialization

```python
# Don't create indexes until first document is added
rag_system = LegalRAGSystem(lazy_init=True)
```

### Create One Index at a Time

```python
# Only create index for specific role when needed
rag_system.pinecone_manager.ensure_role_index_exists("facts")
```

### Batch Processing

```python
# Process documents in batches to avoid memory issues
for batch in document_batches:
    tagged_docs = rag_system.process_legal_document(batch)
    rag_system.add_documents_to_store(tagged_docs)
```

## Testing

### 1. Test Index Creation

```bash
python -m src.config.pinecone_setup --action create
python -m src.config.pinecone_setup --action stats
```

### 2. Test Document Processing

```python
from src.core.legal_rag import LegalRAGSystem

rag = LegalRAGSystem(lazy_init=True)

# Test document
test_doc = """
The petitioner filed a writ petition under Article 32.
The facts are that the arrest was made without warrant.
The main issue is whether the arrest was constitutional.
The petitioner argues that fundamental rights were violated.
The court reasoned that the arrest procedure was flawed.
The court held that the arrest was unconstitutional.
"""

# Process and store
tagged_docs = rag.process_legal_document(test_doc, {"case_id": "test_001"})
rag.add_documents_to_store(tagged_docs)

# Query
response = rag.query_legal_rag("What were the facts?")
print(response["answer"])
```

### 3. Test Role-Specific Retrieval

```python
# Query specific roles
facts_docs = rag.retrieve_by_role(
    query="what happened",
    roles=["Facts"],
    k=3
)

reasoning_docs = rag.retrieve_by_role(
    query="why did the court decide",
    roles=["Reasoning"],
    k=3
)

print(f"Retrieved {len(facts_docs)} facts documents")
print(f"Retrieved {len(reasoning_docs)} reasoning documents")
```

## Troubleshooting

### Issue: "Index does not exist"

**Solution:** Indexes are created lazily. Add a document first, or create manually:
```bash
python -m src.config.pinecone_setup --action create
```

### Issue: "No module named 'pinecone'"

**Solution:** Install dependencies:
```bash
cd server
uv sync
```

### Issue: "API key not found"

**Solution:** Set environment variable:
```bash
export PINECONE_API_KEY=your_key_here
# Or add to server/.env file
```

### Issue: Memory/Resource limits on Kaggle

**Solution:** Use lazy initialization:
```python
rag_system = LegalRAGSystem(lazy_init=True)
```

## Migration Checklist

- [x] Update dependencies (pyproject.toml)
- [x] Configure environment variables (.env)
- [x] Create Pinecone index manager
- [x] Migrate primary RAG system
- [x] Add role-based index support
- [x] Add case_id filtering
- [x] Add lazy initialization
- [x] Deprecate old ChromaDB code
- [ ] Run full system test
- [ ] Test with real legal documents
- [ ] Verify all API endpoints work

## Next Steps

1. **Install dependencies:** `cd server && uv sync`
2. **Create indexes:** `python -m src.config.pinecone_setup --action create`
3. **Run server:** `python main.py`
4. **Test API:** Check `http://localhost:8000/docs`
5. **Upload test document** via `/api/document-query/upload-and-ask`

## Support

For issues or questions:
1. Check Pinecone console: https://app.pinecone.io
2. View index stats: `python -m src.config.pinecone_setup --action stats`
3. Check logs in server output
4. Verify environment variables are set correctly

---

**Migration completed:** ✅
**System ready for testing:** 🚀
**Kaggle-optimized:** ✅
