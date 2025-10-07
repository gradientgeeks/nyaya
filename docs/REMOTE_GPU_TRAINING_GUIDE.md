# Remote GPU Training Guide for Nyaya Role Classifier

## 🎯 Overview

This guide explains how to train your legal document role classifier on a **remote RTX 5000 GPU** and integrate the trained model back into your Nyaya system.

## 📋 Table of Contents

1. [Training Options: Notebook vs Script](#training-options)
2. [Step-by-Step Training Process](#step-by-step-training)
3. [Remote GPU Setup](#remote-gpu-setup)
4. [Training Execution](#training-execution)
5. [Model Transfer & Integration](#model-integration)
6. [Best Practices](#best-practices)

---

## 🤔 Training Options: Notebook vs Script

### Option 1: Jupyter Notebook (Recommended for Beginners)
**File**: `ROLE_CLASSIFIER_TRAINING.ipynb`

✅ **Pros**:
- Interactive exploration and visualization
- Step-by-step execution with immediate feedback
- Easy to debug and modify parameters
- Great for understanding the training process
- Built-in data exploration and analysis

❌ **Cons**:
- Requires Jupyter environment on remote GPU
- Connection issues can interrupt training
- Less efficient for batch processing

**Best for**: First-time training, experimentation, parameter tuning

### Option 2: Python Training Scripts (Recommended for Production)
**Files**: `server/src/models/training/train.py`

✅ **Pros**:
- Can run in background (nohup, screen, tmux)
- More robust for long training sessions
- Better for automation and batch jobs
- Connection-independent training
- Easier to schedule and monitor

❌ **Cons**:
- Less interactive
- Requires more terminal knowledge
- Harder to debug interactively

**Best for**: Production training, long training runs, automated workflows

---

## 🚀 Step-by-Step Training Process

### Phase 1: Preparation (Local Machine)

#### Step 1: Verify Dataset Structure
```bash
cd /home/nyaya/server

# Check your dataset structure
ls -la dataset/Hier_BiLSTM_CRF/train/
ls -la dataset/Hier_BiLSTM_CRF/val/
ls -la dataset/Hier_BiLSTM_CRF/test/

# Verify file format (should be: sentence\trole)
head -20 dataset/Hier_BiLSTM_CRF/train/file_6409.txt
```

#### Step 2: Prepare Training Package
```bash
# Create a training package to transfer
mkdir -p ~/training_package
cd ~/training_package

# Copy necessary files
cp -r /home/nyaya/server/src/models/ ./
cp -r /home/nyaya/server/dataset/ ./
cp /home/nyaya/ROLE_CLASSIFIER_TRAINING.ipynb ./
cp /home/nyaya/server/requirements.txt ./
```

#### Step 3: Create Remote Training Script
Create `remote_train.sh`:
```bash
#!/bin/bash
# Remote training script for background execution

cd /path/to/training_package/models/training

python train.py \
    --train_data ../../dataset/Hier_BiLSTM_CRF/train \
    --val_data ../../dataset/Hier_BiLSTM_CRF/val \
    --test_data ../../dataset/Hier_BiLSTM_CRF/test \
    --model_type inlegalbert \
    --model_name law-ai/InLegalBERT \
    --context_mode prev \
    --batch_size 32 \
    --num_epochs 15 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --max_length 512 \
    --output_dir ./trained_models \
    --device cuda

echo "Training completed at $(date)"
```

---

### Phase 2: Remote GPU Setup

#### Step 1: Connect to Remote GPU
```bash
# SSH into your remote GPU machine
ssh user@remote-gpu-server

# Or if using JarvisLabs/vast.ai/runpod
ssh -p PORT user@hostname
```

#### Step 2: Setup Python Environment
```bash
# Create virtual environment
python3 -m venv nyaya_training_env
source nyaya_training_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (RTX 5000 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install transformers==4.35.0
pip install scikit-learn pandas matplotlib seaborn tqdm
pip install tensorboard
pip install spacy
python -m spacy download en_core_web_sm

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA RTX 5000
```

#### Step 3: Transfer Training Package
```bash
# On local machine - transfer files to remote
scp -r ~/training_package user@remote-gpu:/home/user/

# Or using rsync (better for large datasets)
rsync -avz --progress ~/training_package/ user@remote-gpu:/home/user/training_package/

# Verify transfer
ssh user@remote-gpu "ls -lh training_package/"
```

---

### Phase 3: Training Execution

#### Method A: Using Jupyter Notebook (Interactive)

**Step 1: Setup Jupyter on Remote GPU**
```bash
# On remote machine
source nyaya_training_env/bin/activate
pip install jupyter jupyterlab ipywidgets

# Start Jupyter with no-browser
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

**Step 2: Port Forwarding (Local Machine)**
```bash
# Forward remote Jupyter to local browser
ssh -N -L 8888:localhost:8888 user@remote-gpu
```

**Step 3: Access and Run Notebook**
1. Open browser: `http://localhost:8888`
2. Open `ROLE_CLASSIFIER_TRAINING.ipynb`
3. Update paths in configuration cell:
```python
config = {
    "train_data": "/home/user/training_package/dataset/Hier_BiLSTM_CRF/train",
    "val_data": "/home/user/training_package/dataset/Hier_BiLSTM_CRF/val",
    "test_data": "/home/user/training_package/dataset/Hier_BiLSTM_CRF/test",
    "model_type": "inlegalbert",
    "batch_size": 32,  # Increase for RTX 5000
    "num_epochs": 15,
    "device": "cuda",
    # ... other settings
}
```
4. Run all cells sequentially
5. Monitor training progress in real-time

#### Method B: Using Training Script (Recommended)

**Step 1: Prepare for Background Training**
```bash
# On remote machine
cd /home/user/training_package/models/training

# Make script executable
chmod +x remote_train.sh

# Test run (foreground) - verify everything works
bash remote_train.sh
```

**Step 2: Start Background Training with Screen/Tmux**

**Using Screen (Simpler)**:
```bash
# Create new screen session
screen -S nyaya_training

# Run training
bash remote_train.sh

# Detach from screen: Press Ctrl+A, then D
# Reattach later: screen -r nyaya_training
# List sessions: screen -ls
```

**Using Tmux (More Features)**:
```bash
# Create new tmux session
tmux new -s nyaya_training

# Run training
bash remote_train.sh

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t nyaya_training
# List sessions: tmux ls
```

**Using Nohup (Simplest)**:
```bash
# Run in background with nohup
nohup bash remote_train.sh > training.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f training.log

# Check if still running
ps aux | grep train.py
```

**Step 3: Monitor Training Progress**
```bash
# Check logs
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check TensorBoard (if running)
tensorboard --logdir ./trained_models/tensorboard_logs --port 6006

# Port forward TensorBoard to local machine
# On local: ssh -N -L 6006:localhost:6006 user@remote-gpu
# Open: http://localhost:6006
```

---

### Phase 4: Model Transfer & Integration

#### Step 1: Download Trained Model
```bash
# On local machine - download trained model
scp user@remote-gpu:/home/user/training_package/models/training/trained_models/best_model.pt \
    /home/nyaya/server/trained_models/

# Download all training outputs
rsync -avz --progress \
    user@remote-gpu:/home/user/training_package/models/training/trained_models/ \
    /home/nyaya/server/trained_models/

# Files you should have:
# - best_model.pt (main model file)
# - final_model.pt (last epoch)
# - training_history.json (metrics)
# - training_curves.png (visualization)
# - evaluation_metrics.json (test results)
```

#### Step 2: Integrate Model into Nyaya Server

**Option A: Update role_classifier.py to use trained model**
```python
# In server/src/models/role_classifier.py

class RoleClassifier:
    def __init__(self, model_type: str = "inlegalbert", 
                 device: str = "cpu",
                 pretrained_path: str = None):
        self.model_type = model_type
        self.device = device
        self.nlp = spacy.load("en_core_web_sm")
        self.model = None
        self.role_to_id = {role.value: i for i, role in enumerate(RhetoricalRole)}
        self.id_to_role = {i: role.value for i, role in enumerate(RhetoricalRole)}
        
        self._load_model()
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    # ... rest of the class
```

**Option B: Create production configuration**
```python
# In server/src/config/model_config.py (create this file)

MODEL_CONFIG = {
    "role_classifier": {
        "model_type": "inlegalbert",
        "model_path": "/home/nyaya/server/trained_models/best_model.pt",
        "context_mode": "prev",
        "device": "cpu",  # Use "cuda" if deploying on GPU
    }
}
```

**Option C: Update main.py to use trained model**
```python
# In server/main.py

from src.models.role_classifier import RoleClassifier
from src.config.model_config import MODEL_CONFIG

# Initialize role classifier with trained weights
role_classifier = RoleClassifier(
    model_type=MODEL_CONFIG["role_classifier"]["model_type"],
    device=MODEL_CONFIG["role_classifier"]["device"]
)

# Load trained weights
role_classifier.load_pretrained_weights(
    MODEL_CONFIG["role_classifier"]["model_path"]
)
```

#### Step 3: Test Integrated Model
```python
# Test script: server/test_trained_model.py
from src.models.role_classifier import RoleClassifier

# Initialize with trained model
classifier = RoleClassifier(model_type="inlegalbert", device="cpu")
classifier.load_pretrained_weights("./trained_models/best_model.pt")

# Test with sample legal text
sample_text = """
The petitioner filed a writ petition challenging the constitutional validity of Section 377.
The main issue in this case is whether Section 377 violates fundamental rights.
The petitioner argues that Section 377 is discriminatory and violates Article 14.
The respondent contends that Section 377 is constitutionally valid and necessary.
The court finds that Section 377 infringes upon the right to privacy and equality.
Therefore, Section 377 is hereby declared unconstitutional and is struck down.
"""

results = classifier.classify_document(sample_text, context_mode="prev")

print("\n🎯 Classification Results:")
print("=" * 80)
for result in results:
    print(f"\nSentence: {result['sentence'][:60]}...")
    print(f"Predicted Role: {result['role']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print("-" * 80)
```

Run test:
```bash
cd /home/nyaya/server
python test_trained_model.py
```

#### Step 4: Update API to Use Trained Model
```python
# In server/src/api/classification.py

from fastapi import APIRouter
from src.models.role_classifier import RoleClassifier

router = APIRouter()

# Initialize with trained model
classifier = RoleClassifier(model_type="inlegalbert")
classifier.load_pretrained_weights("./trained_models/best_model.pt")

@router.post("/classify/roles")
async def classify_roles(text: str, context_mode: str = "prev"):
    results = classifier.classify_document(text, context_mode=context_mode)
    return {"results": results}
```

#### Step 5: Deploy and Verify
```bash
# Start server
cd /home/nyaya/server
python main.py

# Test API
curl -X POST "http://localhost:8000/api/classify/roles" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The petitioner filed a writ petition...",
    "context_mode": "prev"
  }'
```

---

## 🎨 Best Practices

### For Training on Remote GPU

1. **Use Background Processes**
   - Always use screen/tmux/nohup for long training
   - Never rely on SSH connection staying alive

2. **Monitor GPU Usage**
   ```bash
   # Install gpustat for better monitoring
   pip install gpustat
   gpustat -i 1  # Update every second
   ```

3. **Save Checkpoints Frequently**
   - Training script saves checkpoints every epoch
   - Keep best model separate from final model

4. **Use Adequate Batch Size**
   - RTX 5000 has 16GB VRAM
   - Recommended batch sizes:
     - InLegalBERT: 32-48
     - With longer sequences (512 tokens): 24-32
     - Monitor GPU memory: `nvidia-smi`

5. **Log Everything**
   ```bash
   # Comprehensive logging
   nohup bash remote_train.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

### For Model Integration

1. **Version Control**
   ```bash
   # Tag your trained models
   mkdir -p /home/nyaya/server/trained_models/v1.0
   cp best_model.pt trained_models/v1.0/
   
   # Keep training metadata
   cp training_history.json trained_models/v1.0/
   cp evaluation_metrics.json trained_models/v1.0/
   ```

2. **Model Registry**
   Create `server/trained_models/MODEL_REGISTRY.json`:
   ```json
   {
     "models": [
       {
         "version": "1.0",
         "date": "2025-10-07",
         "model_type": "inlegalbert",
         "context_mode": "prev",
         "training_config": {
           "batch_size": 32,
           "num_epochs": 15,
           "learning_rate": 2e-5
         },
         "performance": {
           "test_accuracy": 0.89,
           "test_f1": 0.87
         },
         "path": "./trained_models/v1.0/best_model.pt"
       }
     ],
     "active_model": "1.0"
   }
   ```

3. **A/B Testing**
   - Keep old model while testing new one
   - Compare performance on validation set
   - Gradual rollout

### Training Time Estimates (RTX 5000)

| Configuration | Approx. Time | Notes |
|--------------|--------------|-------|
| 10 epochs, batch 32, ~50k sentences | 2-3 hours | Standard training |
| 15 epochs, batch 32, ~50k sentences | 3-4.5 hours | Better convergence |
| 20 epochs, batch 48, ~100k sentences | 6-8 hours | Large dataset |

### Cost Optimization

1. **Data Transfer**
   - Compress dataset before transfer
   ```bash
   tar -czf dataset.tar.gz dataset/
   scp dataset.tar.gz user@remote-gpu:/home/user/
   ```

2. **Efficient Training**
   - Use mixed precision training for faster training
   - Implement early stopping
   - Use learning rate finder

3. **Resource Management**
   - Stop GPU instance when not training
   - Download models immediately after training
   - Clean up unnecessary checkpoints

---

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory
```bash
# Solution: Reduce batch size
--batch_size 16  # Instead of 32

# Or reduce max sequence length
--max_length 256  # Instead of 512
```

### Issue: Training Too Slow
```bash
# Check GPU utilization
nvidia-smi

# If utilization < 80%, increase batch size
# If GPU memory allows, increase batch_size
```

### Issue: Model Not Converging
```bash
# Try different learning rates
--learning_rate 1e-5  # Lower
--learning_rate 3e-5  # Higher

# Increase epochs
--num_epochs 20

# Try different context mode
--context_mode prev_two
```

### Issue: SSH Connection Lost
```bash
# Reattach to screen session
screen -r nyaya_training

# Or check if process still running
ps aux | grep python
```

### Issue: Model File Corrupted During Transfer
```bash
# Use rsync with checksum verification
rsync -avz --checksum best_model.pt user@local:/path/

# Or use SCP with verification
scp best_model.pt user@local:/path/
ssh user@local "md5sum /path/best_model.pt"
```

---

## 📊 Recommended Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Local Preparation                                        │
│    - Verify dataset                                         │
│    - Create training package                                │
│    - Prepare scripts                                        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Remote GPU Setup                                         │
│    - Setup environment                                      │
│    - Transfer files                                         │
│    - Verify CUDA                                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Training (Choose One)                                    │
│    A. Jupyter Notebook (Interactive)                        │
│    B. Python Script (Background - RECOMMENDED)              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Monitor Training                                         │
│    - Watch logs                                             │
│    - Check GPU usage                                        │
│    - View TensorBoard                                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Transfer Model                                           │
│    - Download best_model.pt                                 │
│    - Download metrics and logs                              │
│    - Verify file integrity                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Integration                                              │
│    - Test model locally                                     │
│    - Update server code                                     │
│    - Deploy and verify                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Final Recommendations

### For Your RTX 5000 Setup:

1. **Use Python Script (train.py)** - More reliable for remote training
2. **Run in Background** - Use screen or tmux
3. **Optimal Settings**:
   ```bash
   --batch_size 32
   --num_epochs 15
   --context_mode prev
   --max_length 512
   ```
4. **Expected Training Time**: 3-4 hours for ~50k sentences
5. **Download immediately** after training completes

### Next Steps:

1. ✅ Review this guide
2. ✅ Prepare training package
3. ✅ Setup remote GPU environment
4. ✅ Start with small test run (2 epochs)
5. ✅ Run full training
6. ✅ Integrate trained model
7. ✅ Test with real documents
8. ✅ Deploy to production

---

**Good luck with your training! 🚀**
