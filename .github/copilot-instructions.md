# Nyaya: Legal Document Analysis with Role Classification & RAG

**Nyaya** (Sanskrit: न्याय, "justice") is an intelligent legal document analysis system that uses **Rhetorical Role Classification** with **Role-Aware RAG** to provide precise, structured answers from legal judgments.

## Core Architecture

### The 7 Rhetorical Roles

The system classifies every sentence in legal documents into:
1. **Facts** - Background information and case events
2. **Issue** - Legal questions to be resolved
3. **Arguments of Petitioner** - Petitioner's claims
4. **Arguments of Respondent** - Respondent's counter-arguments
5. **Reasoning** - Court's legal analysis and rationale
6. **Decision** - Final judgment
7. **None** - Other content

This classification drives **role-aware retrieval** - when a user asks "What were the facts?", only Facts-labeled sentences are retrieved, not the entire document.

### System Flow

```
Document Upload → Role Classification → Vector Storage → Query Processing
                   (InLegalBERT)        (ChromaDB/FAISS)  (Role-Aware RAG)
```

## Project Structure

### Monorepo Layout
```
nyaya/
├── client/              # React + Vite + TailwindCSS frontend
│   ├── src/
│   │   ├── components/  # UI components (ChatInput, DocumentUpload, etc.)
│   │   └── contexts/    # ThemeContext for dark/light mode
│   └── package.json     # Node dependencies
│
├── server/              # Python FastAPI backend
│   ├── pyproject.toml   # Python dependencies (uv-based)
│   ├── requirements.txt # Generated from pyproject.toml
│   ├── train_model.py   # Main training script for role classifier
│   └── dataset/         # Training data in tab-separated format
│       └── Hier_BiLSTM_CRF/
│           ├── train/   # sentence\trole format
│           ├── val/     # validation data
│           └── test/    # test data
│
└── docs/                # Comprehensive documentation
    ├── SYSTEM_WORKFLOW_EXPLANATION.md  # Deep technical dive
    ├── QUICK_START_GUIDE.md            # Get started quickly
    ├── REMOTE_GPU_TRAINING_GUIDE.md    # Train on RTX 5000
    └── TRAINING_QUICK_CHECKLIST.md     # Step-by-step checklist
```

### Key Technologies

**Backend:**
- **FastAPI** - API server
- **LangChain** - RAG orchestration with Google VertexAI
- **Transformers + PyTorch** - Role classification models
- **ChromaDB/FAISS** - Vector storage
- **spaCy** - Text processing
- **InLegalBERT** - Pre-trained legal language model

**Frontend:**
- **React 19** with TypeScript
- **Vite** - Build tooling
- **TailwindCSS v4** - Styling

**Models:**
- `law-ai/InLegalBERT` - Transformer-based classifier
- BiLSTM-CRF - Sequential model with CRF layer

## Critical Development Patterns

### 1. Training Data Format

Training files use **tab-separated** format with blank lines separating documents:

```
The petitioner filed a writ petition.	Facts
The main issue is constitutional validity.	Issue

The respondent filed an appeal.	Facts
The court analyzed Article 14.	Reasoning
```

**Never use spaces instead of tabs** - this will break the data loader.

### 2. Context Modes for Classification

When classifying sentences, the model can use different context:
- `single`: Only current sentence
- `prev`: Current + previous sentence (recommended)
- `prev_two`: Current + two previous sentences
- `surrounding`: Previous + current + next sentence

Example:
```python
results = classifier.classify_document(text, context_mode="prev")
```

### 3. Role-Aware RAG Pattern

The system stores embeddings **with role metadata** and filters during retrieval:

```python
# Store with role
tagged_doc = {
    "content": "The petition is allowed.",
    "metadata": {"role": "Decision", "confidence": 0.95}
}

# Retrieve by role
docs = vectorstore.similarity_search(
    query, 
    filter={"role": {"$in": ["Reasoning", "Decision"]}}
)
```

This is the **key differentiator** - most RAG systems don't have role-aware retrieval.

### 4. Model Training Workflow

**Local development:**
```bash
cd server
python train_model.py  # Uses config in the file
```

**Remote GPU training (recommended for production):**
1. Prepare training package locally
2. Transfer to remote GPU (RTX 5000)
3. Run in `screen`/`tmux` session:
   ```bash
   screen -S nyaya_training
   python train.py --batch_size 32 --num_epochs 15
   # Detach: Ctrl+A, D
   ```
4. Download trained model
5. Integrate into server

See `docs/REMOTE_GPU_TRAINING_GUIDE.md` for full details.

### 5. Dependency Management

**Python:** Uses `uv` for fast dependency management
```bash
# Install dependencies
uv pip install -r requirements.txt

# Add new dependency
# Edit pyproject.toml, then:
uv pip compile pyproject.toml -o requirements.txt
```

**Important:** `pyproject.toml` is source of truth. `requirements.txt` is generated.

### 6. Document Processing Pipeline

```python
# Step 1: Extract text
text = processor.extract_text_from_pdf(pdf_bytes)

# Step 2: Classify roles
results = classifier.classify_document(text, context_mode="prev")

# Step 3: Create embeddings
for result in results:
    doc = {
        "content": result["sentence"],
        "metadata": {"role": result["role"], "confidence": result["confidence"]}
    }
    vectorstore.add_documents([doc])

# Step 4: Query with role filtering
answer = rag_system.query(
    "What are the facts?",
    role_filter=["Facts"]
)
```

### 7. API Endpoint Pattern (When Implemented)

The system supports document upload with simultaneous querying:

```bash
# Upload and ask in one call
curl -X POST "http://localhost:8000/api/document-query/upload-and-ask" \
  -F "file=@case.pdf" \
  -F "query=What are the main facts?" \
  -F "role_filter=[\"Facts\", \"Issue\"]"

# Follow-up questions use session_id
curl -X POST "http://localhost:8000/api/document-query/ask-followup" \
  -F "query=What was the decision?" \
  -F "session_id=session-uuid"
```

## Common Development Tasks

### Running the System

**Frontend:**
```bash
cd client
npm install
npm run dev  # http://localhost:5173
```

**Backend:**
```bash
cd server
uv pip install -r requirements.txt
python -m spacy download en_core_web_sm
# Start FastAPI server (when main.py exists)
python main.py  # http://localhost:8000
```

### Training a New Model

```bash
cd server

# Quick test (2 epochs, 80% of data)
python train_model.py  # Takes ~15-30 min on GPU

# Full training (edit config in train_model.py first)
# Recommended: batch_size=32, num_epochs=15, context_mode="prev"
python train_model.py
```

**Output location:** `server/src/models/trained_models/{model_type}_{timestamp}/`

### Testing Role Classification

```python
from src.models.role_classifier import RoleClassifier

classifier = RoleClassifier(model_type="inlegalbert", device="cpu")

text = """
The petitioner filed a writ petition challenging Section 377.
The main issue is whether Section 377 violates fundamental rights.
The petitioner argues it is discriminatory.
The court finds it infringes privacy rights.
Therefore, Section 377 is declared unconstitutional.
"""

results = classifier.classify_document(text, context_mode="prev")
for r in results:
    print(f"{r['role']}: {r['sentence'][:50]}...")
```

## Important Conventions

### File Naming
- Training data: `file_####.txt` (e.g., `file_6409.txt`)
- Models: `{model_type}_{timestamp}` (e.g., `inlegalbert_20250119_143022`)
- Checkpoints: `best_model.pt` (highest validation F1), `final_model.pt` (last epoch)

### Code Organization
- Model training code: `server/src/models/training/`
- Trained models: `server/src/models/trained_models/`
- Dataset: `server/dataset/Hier_BiLSTM_CRF/`
- API routes: To be implemented in `server/src/api/`
- Core logic: To be implemented in `server/src/core/`

### Configuration Management
Models and system configs should be centralized:
```python
# Example pattern for server/src/config/model_config.py
MODEL_CONFIG = {
    "role_classifier": {
        "model_type": "inlegalbert",
        "model_path": "./src/models/trained_models/best_model.pt",
        "context_mode": "prev",
        "device": "cuda"  # or "cpu"
    }
}
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

## Troubleshooting

### Common Issues

**"CUDA Out of Memory"** during training:
- Reduce `batch_size` (try 16 instead of 32)
- Reduce `max_length` (try 256 instead of 512)
- Clear CUDA cache: `torch.cuda.empty_cache()`

**"No module named 'src'"** errors:
- Ensure `sys.path` includes project root
- Check you're running from correct directory

**Training data not loading:**
- Verify tab separation (not spaces)
- Check blank lines between documents
- Ensure UTF-8 encoding

**Poor classification accuracy:**
- Try different `context_mode` (usually "prev" is best)
- Increase training epochs
- Check data quality and label distribution
- Consider fine-tuning for specific legal domain

## Additional Resources

- **Full system explanation:** `docs/SYSTEM_WORKFLOW_EXPLANATION.md`
- **Quick start:** `docs/QUICK_START_GUIDE.md`
- **Remote training:** `docs/REMOTE_GPU_TRAINING_GUIDE.md`
- **Example workflows:** `docs/EXAMPLE_FLOW_1.md`, `docs/EXAMPLE_FLOW_2.md`
- **API documentation:** `docs/DOCUMENT_QUERY_API.md`

## When Working on This Codebase

1. **Understand the role-aware paradigm** - This isn't just RAG, it's role-aware RAG
2. **Respect the training data format** - Tab-separated, blank-line delimited
3. **Use context modes** - Don't classify sentences in isolation
4. **Preserve metadata** - Role labels, confidence scores, document IDs
5. **Test incrementally** - Train with small data first (set `dataset_sample_ratio=0.1`)
6. **Monitor GPU usage** - Should be >80% during training
7. **Document changes** - This is research code moving toward production

## Key Differences from Standard RAG

- **Standard RAG:** Chunks by size, retrieves by similarity alone
- **Nyaya RAG:** Chunks by semantic role, retrieves by role + similarity
- **Benefit:** "What was the reasoning?" returns only Reasoning, not Facts or Arguments

This role-awareness is the core innovation - maintain it in all features.
