# Document Upload Guide

This guide explains how to upload legal judgment documents to Pinecone without requiring the classification model.

## Overview

Since the classification model (InLegalBERT) requires training first, we provide scripts to upload documents directly to Pinecone for:
1. **Testing RAG functionality** without classification
2. **Uploading existing judgments** as-is
3. **Populating the vector database** before model training

## Single Document Upload

### Basic Usage

```bash
python upload_documents.py <file_path> <case_id>
```

### Examples

**Upload a judgment with automatic role (None):**
```bash
python upload_documents.py judgments/supreme_court_case_1.pdf case_001
```

**Upload with specific role:**
```bash
python upload_documents.py facts_only.txt case_002 --role Facts
python upload_documents.py reasoning.txt case_003 --role Reasoning
```

**Upload to different namespace:**
```bash
python upload_documents.py doc.txt case_004 --namespace training_data
```

### Available Roles

- `Facts` - Background and case events
- `Issue` - Legal questions to resolve
- `Arguments_Petitioner` - Petitioner's claims
- `Arguments_Respondent` - Respondent's counter-arguments
- `Reasoning` - Court's legal analysis
- `Decision` - Final judgment
- `None` - Other content (default)

### Output Example

```
2025-10-21 11:45:00,123 - __main__ - INFO - 🚀 Starting document upload...
2025-10-21 11:45:00,124 - __main__ - INFO -   File: judgments/case_001.pdf
2025-10-21 11:45:00,124 - __main__ - INFO -   Case ID: case_001
2025-10-21 11:45:00,124 - __main__ - INFO -   Role: None
2025-10-21 11:45:00,500 - __main__ - INFO - 📄 Extracting text...
2025-10-21 11:45:02,345 - __main__ - INFO - ✅ Extracted 45231 characters
2025-10-21 11:45:02,346 - __main__ - INFO - ✂️  Splitting into sentences...
2025-10-21 11:45:02,789 - __main__ - INFO - ✅ Split into 523 sentences
2025-10-21 11:45:02,790 - __main__ - INFO - 🔄 Generating embeddings...
2025-10-21 11:45:15,234 - __main__ - INFO - ✅ Generated 523 embeddings
2025-10-21 11:45:15,235 - __main__ - INFO - ⬆️  Uploading to Pinecone...
2025-10-21 11:45:18,456 - __main__ - INFO - ✅ Upload complete!
2025-10-21 11:45:18,456 - __main__ - INFO -   Vectors uploaded: 523
2025-10-21 11:45:18,456 - __main__ - INFO -   Namespace: user_documents

============================================================
✅ SUCCESS!
============================================================
Case ID: case_001
Vectors: 523
Namespace: user_documents
============================================================
```

## Batch Upload (Multiple Documents)

Use the batch upload script to process entire directories:

```bash
python batch_upload_documents.py <directory> [--namespace NAMESPACE]
```

### Examples

**Upload all documents in a directory:**
```bash
python batch_upload_documents.py judgments/
```

**Upload with specific namespace:**
```bash
python batch_upload_documents.py training_data/ --namespace training_data
```

### Supported Formats

- **PDF** (.pdf) - Extracted using PyPDF2
- **TXT** (.txt) - Plain text files

## Metadata Structure

Each uploaded sentence vector includes:

```python
{
    "id": "case_001_sent_0",
    "values": [0.123, -0.456, ...],  # 384-dimensional embedding
    "metadata": {
        "text": "The petition is allowed.",
        "role": "None",                    # Role (or specific role if provided)
        "confidence": 0.0,                 # 0.0 = not classified
        "classified": False,               # False = uploaded without model
        "case_id": "case_001",
        "sentence_index": 0
    }
}
```

## Querying Uploaded Documents

Once uploaded, you can query using the API:

```bash
# Start the server
uvicorn app.main:app --reload

# Query documents
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What were the facts of case_001?",
    "session_id": "your_session_id",
    "case_id": "case_001"
  }'
```

## Difference from Classified Upload

| Feature | Manual Upload | Classified Upload (with model) |
|---------|---------------|-------------------------------|
| **Requires model** | ❌ No | ✅ Yes |
| **Role accuracy** | Manual only | 85-95% automatic |
| **Confidence** | 0.0 | 0.0-1.0 |
| **Speed** | Fast | Slower (model inference) |
| **Use case** | Testing, pre-training | Production use |

## Next Steps

After uploading documents manually:

1. **Test RAG queries** - Verify retrieval works
2. **Train classification model** - See `docs/REMOTE_GPU_TRAINING_GUIDE.md`
3. **Re-upload with classification** - Get role-specific retrieval

## Troubleshooting

### "Index not found"
```bash
# Check Pinecone status
python check_pinecone_status.py

# Setup index if needed
bash setup_pinecone.sh
```

### "Dimension mismatch"
Ensure EmbeddingGemma is configured with `truncate_dim=384`:
```python
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
```

### "File not found"
Use absolute paths or ensure you're in the `backend/` directory:
```bash
cd /home/uttam/B.Tech\ Major\ Project/nyaya/backend
python upload_documents.py ../path/to/file.pdf case_001
```

## API Limitations

⚠️ **Note:** The `/upload` endpoint in the API still requires the classification model. Use these scripts for manual uploads without the model.

When the model is available, you can use the API:
```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@judgment.pdf" \
  -F "session_id=xxx" \
  -F "case_id=case_001"
```
