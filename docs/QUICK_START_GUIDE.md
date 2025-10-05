# Quick Start Guide - Nyaya Legal Document Analysis System

## 🚀 Immediate Setup and Usage

### Prerequisites
```bash
cd /home/uttam/B.Tech\ Major\ Project/nyaya/server
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 1. Using Pre-trained Models (Recommended for Quick Start)

The system comes with InLegalBERT pre-trained model. You can start using it immediately:

```python
from src.models.role_classifier import RoleClassifier

# Initialize with pre-trained InLegalBERT
classifier = RoleClassifier(
    model_type="inlegalbert",
    device="cpu"  # or "cuda" if available
)

# Classify a legal document
legal_text = """
The petitioner filed a writ petition challenging the constitutional validity of Section 377.
The main issue in this case is whether Section 377 violates fundamental rights.
The petitioner argues that Section 377 is discriminatory and violates Article 14.
The respondent contends that Section 377 is constitutionally valid.
The court finds that Section 377 infringes upon the right to privacy and equality.
Therefore, Section 377 is hereby declared unconstitutional.
"""

results = classifier.classify_document(legal_text, context_mode="prev")

for result in results:
    print(f"Sentence: {result['sentence'][:50]}...")
    print(f"Role: {result['role']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 50)
```

### 2. Running the Complete System

```bash
# Start the FastAPI server
cd /home/uttam/B.Tech\ Major\ Project/nyaya/server
python main.py
```

The API will be available at `http://localhost:8000` with documentation at `http://localhost:8000/docs`

### 3. Using Your Custom Trained Model

If you've trained your own model using the Jupyter notebook:

```python
from src.models.role_classifier import RoleClassifier

# Initialize classifier
classifier = RoleClassifier(model_type="inlegalbert", device="cpu")

# Load your custom trained weights
classifier.load_pretrained_weights("path/to/your/best_model.pt")

# Use for classification
results = classifier.classify_document(your_document_text)
```

## 📊 Training Your Own Model - Quick Steps

### Step 1: Prepare Your Data
Your data should be in format:
```
sentence1	role1
sentence2	role2

sentence1	role1  # New document
sentence2	role2
```

### Step 2: Use the Training Notebook
Open and run: `ROLE_CLASSIFIER_TRAINING.ipynb`

Update the paths in the configuration:
```python
config = {
    "train_data": "/path/to/your/train/data",
    "val_data": "/path/to/your/val/data", 
    "test_data": "/path/to/your/test/data",
    # ... other settings
}
```

### Step 3: Run Training
Execute all cells in the notebook. The training will:
1. Load and validate your data
2. Train the model
3. Evaluate performance
4. Save the best model

## 🔧 System Components Quick Reference

### Role Classifier (`role_classifier.py`)
- **Input**: Legal document text
- **Output**: Sentence-level role classifications
- **Models**: InLegalBERT, BiLSTM-CRF
- **Roles**: Facts, Issue, Arguments (Petitioner/Respondent), Reasoning, Decision, None

### Legal RAG System (`legal_rag.py`)
- **Input**: Classified legal documents + user queries
- **Output**: Role-aware retrieval and generation
- **Features**: Role-based filtering, intelligent query routing

### Agent Orchestrator (`agent_orchestrator.py`)
- **Purpose**: Coordinates all system components
- **Features**: Query classification, component routing, response generation

### Document Processor (`document_processor.py`)
- **Input**: PDF/TXT files
- **Output**: Cleaned text + metadata
- **Features**: Text extraction, cleaning, metadata extraction

## 📡 API Usage Examples

### Upload and Query Document
```bash
curl -X POST "http://localhost:8000/api/document-query/upload-and-ask" \
  -F "file=@document.pdf" \
  -F "query=What are the main facts?" \
  -F "role_filter=[\"Facts\", \"Issue\"]"
```

### Classify Text Roles
```bash
curl -X POST "http://localhost:8000/api/classify/roles" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The petitioner filed a writ petition...",
    "context_mode": "prev"
  }'
```

### Ask Questions
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the court reasoning?",
    "role_filter": ["Reasoning", "Decision"]
  }'
```

## 🎯 Model Performance Tips

### For Better Classification:
1. **Context Mode**: Use "prev" for better context awareness
2. **Training Data**: Ensure balanced role distribution
3. **Fine-tuning**: Train on your specific legal domain

### For Better RAG:
1. **Role Filtering**: Use specific roles for targeted queries
2. **Query Intent**: System auto-detects relevant roles
3. **Multi-turn**: Leverage conversation context

## 🔍 Understanding the Workflow

### Document Processing Flow:
```
PDF/TXT → Text Extraction → Sentence Segmentation → Role Classification → Vector Storage → Query Processing
```

### Query Processing Flow:
```
User Query → Intent Detection → Role-based Retrieval → LLM Generation → Structured Response
```

## 🛠️ Customization Options

### Model Configuration:
```python
# Different context modes
classifier = RoleClassifier(context_mode="surrounding")  # prev, prev_two, single

# Different models
classifier = RoleClassifier(model_type="bilstm_crf")  # or "inlegalbert"
```

### RAG Configuration:
```python
# Custom role weights for retrieval
rag_system = LegalRAGSystem(
    embedding_model="text-embedding-005",
    role_classifier_type="inlegalbert"
)
```

## 🐛 Troubleshooting

### Common Issues:
1. **Import Errors**: Check Python path and dependencies
2. **CUDA Errors**: Use `device="cpu"` if GPU issues
3. **Memory Issues**: Reduce batch size or sequence length
4. **Low Accuracy**: Try different context modes or more training data

### Data Format Issues:
- Ensure UTF-8 encoding
- Use tab separation (not spaces)
- Include empty lines between documents
- Validate role names match expected categories

## 📈 Performance Expectations

### Pre-trained InLegalBERT:
- **Accuracy**: ~85-90% on legal documents
- **Best for**: General legal text classification
- **Speed**: Fast inference

### Custom Trained Models:
- **Accuracy**: 90-95% on domain-specific data
- **Best for**: Specific legal domains or jurisdictions
- **Training Time**: 2-4 hours on GPU

## 🔄 Integration with Existing Systems

### Standalone Usage:
```python
# Just classification
from src.models.role_classifier import RoleClassifier
classifier = RoleClassifier()
results = classifier.classify_document(text)
```

### Full System Usage:
```python
# Complete workflow
from src.core.agent_orchestrator import AgentOrchestrator
orchestrator = AgentOrchestrator()
response = orchestrator.process_query(query, session_id)
```

## 📚 Next Steps

1. **Start with pre-trained models** for immediate results
2. **Train custom models** for your specific use case
3. **Integrate with your applications** via API
4. **Fine-tune performance** based on your requirements
5. **Expand functionality** with additional legal tools

For detailed technical explanations, see `SYSTEM_WORKFLOW_EXPLANATION.md`
For training your own models, use `ROLE_CLASSIFIER_TRAINING.ipynb`