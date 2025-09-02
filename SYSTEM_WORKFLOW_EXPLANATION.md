# Nyaya Legal Document Analysis System - Complete Workflow Explanation

## System Overview

The Nyaya system is a sophisticated legal document analysis platform that combines **Rhetorical Role Classification** with **Role-Aware RAG (Retrieval-Augmented Generation)** to provide intelligent analysis of legal documents. The system processes legal documents, classifies sentences into rhetorical roles, and then uses this classification to enhance retrieval and generation.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NYAYA SYSTEM ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 Document Input → 🔍 Role Classification →📊Role-Aware RAG   │
│                                                                 │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │ Document        │   │ Role Classifier │   │ RAG System      │ │
│  │ Processor       │ → │ (InLegalBERT/   │ → │ (ChromaDB +     │ │
│  │ (PDF/TXT)       │   │  BiLSTM-CRF)    │   │  LLM)           │ │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘ │
│                                                                  │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐ │
│  │ Query Router    │   │ Conversation    │   │ Agent           │ │
│  │ (Intent         │ ← │ Manager         │ ← │ Orchestrator    │ │
│  │  Detection)     │   │ (Multi-turn)    │   │ (Main Control)  │ │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📝 Rhetorical Roles

The system classifies legal document sentences into these 7 rhetorical roles:

1. **Facts** - Background information and events of the case
2. **Issue** - Legal questions and problems to be resolved  
3. **Arguments of Petitioner** - Claims and arguments made by the petitioner
4. **Arguments of Respondent** - Claims and arguments made by the respondent
5. **Reasoning** - Court's legal analysis and rationale
6. **Decision** - Final judgment and court's decision
7. **None** - Other content that doesn't fit the above categories

## 🔄 Complete System Workflow

### Phase 1: Document Processing Pipeline
📄 Document Upload (PDF/TXT)
         ↓
📋 Document Processor (document_processor.py)
    - Extract text from PDF/TXT
    - Clean and normalize text
    - Extract metadata (case name, court, parties)
    - Identify document sections
         ↓
📝 Sentence Segmentation
    - Split document into sentences using spaCy
    - Preserve sentence order and context
         ↓
🏷️ Role Classification (role_classifier.py)
    - Apply InLegalBERT or BiLSTM-CRF model
    - Classify each sentence into rhetorical roles
    - Consider context (previous sentences)
         ↓
💾 Store in Vector Database
    - Create embeddings for each classified sentence
    - Store with role metadata in ChromaDB
    - Enable role-based retrieval
```

### Phase 2: Query Processing Pipeline

```
❓ User Query Input
         ↓
🎯 Query Router (agent_orchestrator.py)
    - Analyze query intent
    - Detect relevant rhetorical roles
    - Classify query type
         ↓
🔍 Role-Aware Retrieval (legal_rag.py)
    - Retrieve relevant sentences based on roles
    - Apply role-specific filtering
    - Rank by relevance and role importance
         ↓
🧠 LLM Generation
    - Generate response using retrieved content
    - Structure answer by rhetorical roles
    - Provide sources and confidence scores
         ↓
💬 Conversation Management
    - Store query and response in session
    - Maintain context for follow-up questions
    - Update conversation state
```

## 🤖 Role Classifier Deep Dive

### Model Architecture Options

#### 1. InLegalBERT Classifier (`InLegalBERTClassifier`)
```python
Input: Legal text + context
    ↓
InLegalBERT Encoder (law-ai/InLegalBERT)
    ↓
Pooler Output (768 dimensions)
    ↓
Dropout (0.1)
    ↓
Linear Classifier (768 → 7 classes)
    ↓
Softmax → Role Probabilities
```

#### 2. BiLSTM-CRF Classifier (`BiLSTMCRFClassifier`)
```python
Sentence Embeddings
    ↓
Bidirectional LSTM (512 → 256 hidden)
    ↓
Linear Layer (256 → 7 tags)
    ↓
CRF Layer (considers sequence dependencies)
    ↓
Viterbi Decoding → Best Role Sequence
```

### Context Modes for Classification

1. **Single**: Only current sentence
2. **Prev**: Current + previous sentence
3. **Prev_two**: Current + two previous sentences  
4. **Surrounding**: Previous + current + next sentence

### Training Process

The system provides a complete training pipeline:

```python
# Training Script Usage
python train.py \
    --train_data /path/to/train/data \
    --val_data /path/to/val/data \
    --model_type inlegalbert \
    --context_mode prev \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5
```

**Training Data Format:**
```
The petitioner filed a writ petition.	Facts
The main issue is constitutional validity.	Issue
The petitioner argues violation of rights.	Arguments of Petitioner
The respondent contends it is valid.	Arguments of Respondent
The court analyzed Article 14.	Reasoning
The petition is allowed.	Decision
```

## 🔍 Role-Aware RAG System Deep Dive

### How Role-Aware RAG Works

#### 1. Document Ingestion
```python
# Example from legal_rag.py
def process_legal_document(self, document_text, context_mode="prev"):
    # Step 1: Classify roles for each sentence
    role_results = self.role_classifier.classify_document(
        document_text, context_mode=context_mode
    )
    
    # Step 2: Create role-tagged documents
    tagged_docs = []
    for result in role_results:
        tagged_doc = RoleTaggedDocument(
            content=result["sentence"],
            role=result["role"],
            confidence=result["confidence"],
            metadata={"role": result["role"], ...}
        )
        tagged_docs.append(tagged_doc)
    
    return tagged_docs
```

#### 2. Vector Storage with Role Metadata
```python
# Documents stored with role information
{
    "content": "The petitioner filed a writ petition...",
    "metadata": {
        "role": "Facts",
        "confidence": 0.95,
        "doc_id": "uuid-123",
        "sentence_index": 0
    },
    "embedding": [0.1, 0.2, 0.3, ...]  # Vector embedding
}
```

#### 3. Role-Based Retrieval
```python
def retrieve_by_role(self, query, roles=None, k=5):
    if roles:
        # Filter by specific roles
        role_filter = {"role": {"$in": roles}}
        docs = self.vectorstore.similarity_search(
            query, k=k, filter=role_filter
        )
    else:
        # Search all roles
        docs = self.retriever.invoke(query, k=k)
    
    return docs
```

#### 4. Intelligent Query Intent Detection
```python
def analyze_query_intent(self, query):
    """
    Determines which rhetorical roles are relevant for a query
    """
    role_keywords = {
        "Facts": ["facts", "background", "what happened"],
        "Issue": ["issue", "question", "problem"],
        "Reasoning": ["reasoning", "why", "analysis"],
        "Decision": ["decision", "judgment", "ruling"]
    }
    
    relevant_roles = []
    for role, keywords in role_keywords.items():
        if any(keyword in query.lower() for keyword in keywords):
            relevant_roles.append(role)
    
    return relevant_roles
```

## 📡 API Endpoints and Usage

### 1. Document Upload and Query
```bash
# Upload document and ask question simultaneously
curl -X POST "http://localhost:8000/api/document-query/upload-and-ask" \
  -F "file=@legal_document.pdf" \
  -F "query=What are the main facts of this case?" \
  -F "role_filter=[\"Facts\", \"Issue\"]"
```

### 2. Role Classification
```bash
# Classify rhetorical roles in text
curl -X POST "http://localhost:8000/api/classify/roles" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The petitioner filed a writ petition...",
    "context_mode": "prev"
  }'
```

### 3. General Query Processing
```bash
# Ask questions about uploaded documents
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the court reasoning?",
    "session_id": "session-123",
    "role_filter": ["Reasoning", "Decision"]
  }'
```

## 🔧 Key Python Files Explanation

### 1. `role_classifier.py` - Core Classification Engine
- **Purpose**: Implements both InLegalBERT and BiLSTM-CRF models
- **Key Classes**:
  - `InLegalBERTClassifier`: BERT-based transformer model
  - `BiLSTMCRFClassifier`: Sequential model with CRF layer
  - `RoleClassifier`: Main interface for classification
- **Usage**: Classifies legal text into 7 rhetorical roles

### 2. `legal_rag.py` - Role-Aware RAG System
- **Purpose**: Implements retrieval system that considers rhetorical roles
- **Key Features**:
  - Role-tagged document storage
  - Role-based filtering during retrieval
  - Intent-aware query processing
  - Structured response generation

### 3. `agent_orchestrator.py` - Main Controller
- **Purpose**: Coordinates all system components
- **Key Functions**:
  - Query routing and classification
  - Component orchestration
  - Response generation and formatting
  - Error handling and logging

### 4. `document_processor.py` - Document Handling
- **Purpose**: Processes various document formats
- **Features**:
  - PDF and text extraction
  - Metadata extraction (case names, courts)
  - Text cleaning and normalization
  - Section identification

### 5. `conversation_manager.py` - Multi-turn Conversations
- **Purpose**: Manages conversation state and context
- **Features**:
  - Session management
  - Message history storage
  - Context preservation
  - Database persistence

## 🎓 Training Your Own Model

### Do You Need to Train a Model?

**Short Answer**: The system comes with pre-configured models (InLegalBERT), but you can train your own for better performance on your specific legal domain.

### Training Data Requirements

Your existing training data (in `Hier_BiLSTM_CRF/train/`, `test/`, `val/`) needs to be in this format:

```
sentence1	role1
sentence2	role2
sentence3	role3

sentence1	role1  # New document
sentence2	role2
```

### Creating a Training Jupyter Notebook

Here's a comprehensive training notebook you can create:

```python
# training_notebook.ipynb

# Cell 1: Setup and Imports
import sys
sys.path.append('/path/to/your/server/src/models/training')

from train import RoleClassifierTrainer
from data_loader import create_data_loaders
from evaluate import ModelEvaluator
import torch

# Cell 2: Configure Training
config = {
    "model_type": "inlegalbert",  # or "bilstm_crf"
    "model_name": "law-ai/InLegalBERT",
    "train_data": "/path/to/your/Hier_BiLSTM_CRF/train/",
    "val_data": "/path/to/your/Hier_BiLSTM_CRF/val/", 
    "test_data": "/path/to/your/Hier_BiLSTM_CRF/test/",
    "context_mode": "prev",
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 2e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Cell 3: Initialize Trainer
trainer = RoleClassifierTrainer(
    model_type=config["model_type"],
    model_name=config["model_name"],
    device=config["device"],
    output_dir="./trained_models"
)

# Cell 4: Start Training
trainer.train(
    train_data_path=config["train_data"],
    val_data_path=config["val_data"],
    test_data_path=config["test_data"],
    context_mode=config["context_mode"],
    batch_size=config["batch_size"],
    num_epochs=config["num_epochs"],
    learning_rate=config["learning_rate"]
)

# Cell 5: Evaluate Model
evaluator = ModelEvaluator(
    model_path="./trained_models/best_model.pt",
    device=config["device"]
)

metrics = evaluator.evaluate_dataset(
    test_data_path=config["test_data"],
    context_mode=config["context_mode"]
)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Test F1 Score: {metrics['weighted_f1']:.4f}")

# Cell 6: Test Single Prediction
result = evaluator.predict_single(
    "The petitioner filed a writ petition challenging the validity.",
    context_mode="prev"
)
print(f"Predicted Role: {result['predicted_role']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## 🚀 Example Complete Workflow

### Scenario: Analyzing a Legal Judgment

#### Step 1: Upload Document
```python
# Document gets uploaded via API
POST /api/document-query/upload-and-ask
{
    "file": "judgment.pdf",
    "query": "What are the main facts of this case?"
}
```

#### Step 2: Document Processing
```python
# document_processor.py processes the PDF
extracted_text = processor.extract_text_from_pdf(pdf_content)
cleaned_text = processor.clean_text(extracted_text)
metadata = processor.extract_case_metadata(cleaned_text)
```

#### Step 3: Role Classification
```python
# role_classifier.py classifies each sentence
sentences = classifier.preprocess_document(cleaned_text)
results = []

for i, sentence in enumerate(sentences):
    inputs = classifier.prepare_input(sentences, i, "prev")
    logits = classifier.model(**inputs)
    predicted_role = classifier.id_to_role[torch.argmax(logits).item()]
    
    results.append({
        "sentence": sentence,
        "role": predicted_role,
        "confidence": torch.softmax(logits, dim=-1).max().item()
    })
```

#### Step 4: Vector Storage
```python
# legal_rag.py stores role-tagged sentences
tagged_docs = rag_system.process_legal_document(cleaned_text)
rag_system.add_documents_to_store(tagged_docs)
```

#### Step 5: Query Processing
```python
# agent_orchestrator.py processes user query
query = "What are the main facts?"
classification = router.classify_query(query)  # Detects "Facts" role
relevant_docs = rag_system.retrieve_by_role(query, roles=["Facts"])
response = llm.generate_response(query, relevant_docs)
```

#### Step 6: Response Generation
```json
{
    "answer": "The main facts of the case are: 1) The petitioner filed a writ petition...",
    "sources": [
        {
            "content": "The petitioner filed a writ petition...",
            "role": "Facts",
            "confidence": 0.95
        }
    ],
    "session_id": "session-123",
    "tools_used": ["role_classifier", "vector_retrieval", "llm_generation"]
}
```

## 🎯 Key Advantages of Role-Aware RAG

1. **Precision**: Retrieves only relevant role-specific information
2. **Structure**: Organizes responses by legal document structure  
3. **Context**: Maintains legal reasoning flow
4. **Efficiency**: Reduces irrelevant information in responses
5. **Explainability**: Shows which parts of documents were used

## 🔧 Configuration and Customization

### Model Configuration
```python
# In role_classifier.py
classifier = RoleClassifier(
    model_type="inlegalbert",  # or "bilstm_crf"
    device="cuda"  # or "cpu"
)

# Load custom trained weights
classifier.load_pretrained_weights("path/to/your/model.pt")
```

### RAG System Configuration
```python
# In legal_rag.py
rag_system = LegalRAGSystem(
    embedding_model="text-embedding-005",
    role_classifier_type="inlegalbert",
    collection_name="legal_rag",
    device="cpu"
)
```

This comprehensive system provides an end-to-end solution for legal document analysis with state-of-the-art role classification and intelligent retrieval capabilities.