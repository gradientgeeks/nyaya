# Training Quick Reference Checklist

## 🎯 Pre-Training Checklist

### Local Machine Preparation
- [ ] Dataset verified and properly formatted
  ```bash
  cd /home/nyaya/server
  head dataset/Hier_BiLSTM_CRF/train/file_6409.txt
  ```
- [ ] Training package created
  ```bash
  mkdir ~/training_package
  cp -r server/src/models ~/training_package/
  cp -r server/dataset ~/training_package/
  ```
- [ ] Remote training script prepared (`remote_train.sh`)

### Remote GPU Setup
- [ ] SSH connection established
  ```bash
  ssh user@remote-gpu
  ```
- [ ] Python environment created
  ```bash
  python3 -m venv nyaya_training_env
  source nyaya_training_env/bin/activate
  ```
- [ ] CUDA verified
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] Dependencies installed
  ```bash
  pip install torch transformers scikit-learn pandas matplotlib seaborn tqdm tensorboard spacy
  python -m spacy download en_core_web_sm
  ```
- [ ] Files transferred
  ```bash
  rsync -avz ~/training_package/ user@remote-gpu:/home/user/training_package/
  ```

## 🚀 Training Execution

### Method A: Jupyter Notebook
- [ ] Jupyter started on remote
  ```bash
  jupyter lab --no-browser --port=8888
  ```
- [ ] Port forwarding active
  ```bash
  ssh -N -L 8888:localhost:8888 user@remote-gpu
  ```
- [ ] Configuration updated in notebook
- [ ] Training cells executed sequentially

### Method B: Python Script (Recommended)
- [ ] Screen/tmux session created
  ```bash
  screen -S nyaya_training
  ```
- [ ] Training script started
  ```bash
  cd training_package/models/training
  bash remote_train.sh
  ```
- [ ] Session detached (Ctrl+A, D)
- [ ] Monitoring setup
  ```bash
  # Reattach: screen -r nyaya_training
  # Or watch logs: tail -f training.log
  ```

## 📊 During Training

- [ ] GPU usage monitored
  ```bash
  watch -n 1 nvidia-smi
  ```
- [ ] Training logs checked
  ```bash
  tail -f training.log
  ```
- [ ] TensorBoard running (optional)
  ```bash
  tensorboard --logdir ./trained_models/tensorboard_logs
  ```

## 📥 Post-Training

### Model Download
- [ ] Training completed successfully
- [ ] Best model downloaded
  ```bash
  scp user@remote-gpu:/home/user/training_package/models/training/trained_models/best_model.pt \
      /home/nyaya/server/trained_models/
  ```
- [ ] Metrics downloaded
  ```bash
  rsync -avz user@remote-gpu:/home/user/training_package/models/training/trained_models/ \
      /home/nyaya/server/trained_models/
  ```
- [ ] Files verified
  ```bash
  ls -lh /home/nyaya/server/trained_models/
  ```

### Model Integration
- [ ] Test script created
  ```bash
  cd /home/nyaya/server
  python test_trained_model.py
  ```
- [ ] Model tested successfully
- [ ] Server code updated to use new model
- [ ] API endpoints tested
  ```bash
  curl -X POST "http://localhost:8000/api/classify/roles" \
    -H "Content-Type: application/json" \
    -d '{"text": "Sample legal text...", "context_mode": "prev"}'
  ```

## 🎯 Training Parameters (RTX 5000)

### Recommended Settings
```bash
--model_type inlegalbert
--batch_size 32
--num_epochs 15
--learning_rate 2e-5
--context_mode prev
--max_length 512
--device cuda
```

### Memory Management
| Batch Size | Sequence Length | GPU Memory | Throughput |
|------------|----------------|------------|------------|
| 16         | 512           | ~8 GB      | Slower     |
| 32         | 512           | ~12 GB     | Optimal    |
| 48         | 512           | ~15 GB     | Faster     |
| 32         | 256           | ~8 GB      | Faster     |

## ⏱️ Expected Timelines

| Dataset Size | Epochs | Batch Size | Estimated Time |
|-------------|--------|------------|----------------|
| 20k sentences | 10 | 32 | 1.5-2 hours |
| 50k sentences | 15 | 32 | 3-4 hours |
| 100k sentences | 15 | 32 | 6-8 hours |

## 🔧 Common Commands

### Screen Management
```bash
# Create session
screen -S nyaya_training

# List sessions
screen -ls

# Reattach
screen -r nyaya_training

# Detach: Ctrl+A, then D

# Kill session
screen -X -S nyaya_training quit
```

### Tmux Management
```bash
# Create session
tmux new -s nyaya_training

# List sessions
tmux ls

# Reattach
tmux attach -t nyaya_training

# Detach: Ctrl+B, then D

# Kill session
tmux kill-session -t nyaya_training
```

### GPU Monitoring
```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed stats
nvidia-smi -l 1

# Process-specific
nvidia-smi pmon -i 0 -c 1

# Install gpustat (better visualization)
pip install gpustat
gpustat -i 1
```

### File Transfer
```bash
# Single file
scp best_model.pt user@remote:/path/

# Directory
rsync -avz --progress trained_models/ user@remote:/path/

# With compression
tar -czf models.tar.gz trained_models/
scp models.tar.gz user@remote:/path/
```

## 🚨 Troubleshooting Quick Fixes

### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 16

# Reduce sequence length
--max_length 256

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Training Stalled
```bash
# Check process
ps aux | grep python

# Check GPU
nvidia-smi

# Check logs
tail -f training.log

# Reattach to screen
screen -r nyaya_training
```

### Connection Lost
```bash
# Check if training still running
ssh user@remote-gpu "ps aux | grep python"

# Reattach to screen
ssh user@remote-gpu
screen -r nyaya_training
```

### Model Not Converging
```bash
# Lower learning rate
--learning_rate 1e-5

# Increase epochs
--num_epochs 20

# Try different context
--context_mode prev_two
```

## 📝 Training Command Templates

### Full Training
```bash
python train.py \
    --train_data /path/to/train \
    --val_data /path/to/val \
    --test_data /path/to/test \
    --model_type inlegalbert \
    --batch_size 32 \
    --num_epochs 15 \
    --learning_rate 2e-5 \
    --output_dir ./trained_models
```

### Quick Test Run
```bash
python train.py \
    --train_data /path/to/train \
    --val_data /path/to/val \
    --model_type inlegalbert \
    --batch_size 16 \
    --num_epochs 2 \
    --output_dir ./test_run
```

### Background Training
```bash
nohup python train.py \
    --train_data /path/to/train \
    --val_data /path/to/val \
    --test_data /path/to/test \
    --model_type inlegalbert \
    --batch_size 32 \
    --num_epochs 15 \
    --output_dir ./trained_models \
    > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## ✅ Success Criteria

### Training Complete When:
- [ ] All epochs finished without errors
- [ ] Validation F1 score > 0.80
- [ ] best_model.pt file created
- [ ] Training curves show convergence
- [ ] Test evaluation completed

### Integration Complete When:
- [ ] Model loads without errors
- [ ] Test predictions are accurate
- [ ] API responds correctly
- [ ] Performance metrics documented
- [ ] Production deployment ready

## 📚 Key Files to Keep

```
trained_models/
├── best_model.pt                 # ⭐ Main model file
├── training_history.json         # Training metrics
├── training_curves.png           # Visual progress
├── evaluation_metrics.json       # Test performance
├── classification_report_*.json  # Detailed per-class metrics
└── production_usage.py          # Integration example
```

## 🎓 Pro Tips

1. **Always test with 2 epochs first** to verify everything works
2. **Use screen/tmux** - never rely on SSH staying connected
3. **Monitor GPU utilization** - should be >80% during training
4. **Save model immediately** after training completes
5. **Keep training logs** for debugging and analysis
6. **Document your configuration** for reproducibility
7. **Version your models** with clear naming
8. **Test locally** before deploying to production

---

**Ready to train? Follow the checklist step by step! 🚀**
