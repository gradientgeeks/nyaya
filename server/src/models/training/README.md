# Training Scripts for Legal Document Role Classification

This directory contains comprehensive training and evaluation scripts for the legal document role classifier.

## Overview

The training pipeline supports:
- **InLegalBERT** model (transformer-based)
- **BiLSTM-CRF** model (sequence labeling)
- Multiple context modes (single, previous, surrounding)
- Custom dataset formats
- Comprehensive evaluation and analysis

## Files

### 1. `data_loader.py`
Data loading and preprocessing utilities:
- Supports multiple input formats (text files, CSV, JSON)
- Handles different context modes
- Creates PyTorch DataLoaders
- Automatic data validation and role mapping

### 2. `train.py`
Main training script:
- Configurable model training
- TensorBoard logging
- Learning rate scheduling
- Model checkpointing
- Training curve visualization

### 3. `evaluate.py`
Comprehensive evaluation:
- Detailed metrics calculation
- Confusion matrix generation
- Per-role analysis
- Error analysis and visualization
- Single text prediction

### 4. `dataset_preprocessor.py`
Dataset conversion and preprocessing:
- Convert text files to structured formats
- Train/validation/test splitting
- Dataset analysis and statistics
- Format conversion utilities

## Quick Start

### 1. Prepare Your Dataset

Your dataset should be in one of these formats:

**Option A: Text files (like your current format)**
```
Each file contains:
sentence1<TAB>role1
sentence2<TAB>role2
...
```

**Option B: CSV format**
```csv
document_id,sentence_index,sentence,role
doc1,0,"This is sentence 1","Facts"
doc1,1,"This is sentence 2","Issue"
...
```

### 2. Convert and Split Dataset

```bash
# Convert text files to CSV
python dataset_preprocessor.py \
    --task convert \
    --input_dir /path/to/your/train/directory \
    --output_file dataset.csv

# Split into train/val/test
python dataset_preprocessor.py \
    --task split \
    --csv_file dataset.csv \
    --output_dir ./data_splits \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### 3. Train the Model

```bash
# Train InLegalBERT model
python train.py \
    --train_data ./data_splits/train.csv \
    --val_data ./data_splits/val.csv \
    --test_data ./data_splits/test.csv \
    --model_type inlegalbert \
    --context_mode prev \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 2e-5 \
    --output_dir ./outputs

# Train with your existing directory structure
python train.py \
    --train_data /path/to/Hier_BiLSTM_CRF/train \
    --test_data /path/to/Hier_BiLSTM_CRF/test \
    --model_type inlegalbert \
    --context_mode prev \
    --batch_size 16 \
    --num_epochs 10 \
    --output_dir ./outputs
```

### 4. Evaluate the Model

```bash
python evaluate.py \
    --model_path ./outputs/best_model.pt \
    --test_data ./data_splits/test.csv \
    --context_mode prev \
    --output_dir ./evaluation_results
```

## Configuration Options

### Context Modes
- `single`: Use only the current sentence
- `prev`: Use current + previous sentence
- `prev_two`: Use current + two previous sentences  
- `surrounding`: Use previous + current + next sentence

### Model Types
- `inlegalbert`: Transformer-based model using InLegalBERT
- `bilstm_crf`: BiLSTM with CRF layer (requires additional implementation)

### Training Parameters
- `batch_size`: Batch size (default: 16)
- `learning_rate`: Learning rate (default: 2e-5)
- `num_epochs`: Number of epochs (default: 10)
- `weight_decay`: L2 regularization (default: 0.01)
- `max_length`: Maximum sequence length (default: 512)

## Working with Your Existing Dataset

Since you have the Hier_BiLSTM_CRF dataset structure, you can use it directly:

```bash
# Analyze your dataset
python dataset_preprocessor.py \
    --task analyze \
    --data_path /path/to/Hier_BiLSTM_CRF/train

# Train directly on your structure
python train.py \
    --train_data /path/to/Hier_BiLSTM_CRF/train \
    --test_data /path/to/Hier_BiLSTM_CRF/test \
    --model_type inlegalbert \
    --context_mode prev \
    --batch_size 16 \
    --num_epochs 10 \
    --output_dir ./outputs
```

## Output Structure

After training, you'll get:
```
outputs/
├── best_model.pt              # Best model checkpoint
├── final_model.pt             # Final model
├── training_history.json      # Training metrics
├── training_curves.png        # Loss/accuracy plots
├── tensorboard_logs/          # TensorBoard logs
└── checkpoint_epoch_*.pt      # Epoch checkpoints
```

After evaluation:
```
evaluation_results/
├── evaluation_metrics.json    # Overall metrics
├── classification_report.txt  # Detailed classification report
├── confusion_matrix.png       # Confusion matrix plot
├── per_role_analysis.csv      # Per-role performance
├── error_analysis.csv         # Detailed error analysis
└── error_statistics.json      # Error statistics
```

## Advanced Usage

### Custom Role Mapping

If your dataset uses different role names, update the `role_mapping` in `dataset_preprocessor.py`:

```python
self.role_mapping = {
    "your_fact_role": "Facts",
    "your_issue_role": "Issue",
    # ... other mappings
}
```

### Hyperparameter Tuning

Create a script to test different configurations:

```bash
# Test different learning rates
for lr in 1e-5 2e-5 3e-5; do
    python train.py \
        --train_data ./data \
        --learning_rate $lr \
        --output_dir ./outputs_lr_$lr
done
```

### Integration with Existing Role Classifier

To integrate with your existing `role_classifier.py`:

1. Train a model using these scripts
2. Save the model weights
3. Load the weights in your `RoleClassifier` class:

```python
classifier = RoleClassifier(model_type="inlegalbert")
classifier.load_pretrained_weights("./outputs/best_model.pt")
```

## Requirements

Install required packages:
```bash
pip install torch transformers scikit-learn pandas matplotlib seaborn tqdm tensorboard
```

## Tips for Better Performance

1. **Use document-level splitting** to avoid data leakage
2. **Experiment with context modes** - `prev` often works well for legal documents
3. **Monitor validation metrics** to avoid overfitting
4. **Use class weights** if you have imbalanced data
5. **Fine-tune learning rate** - start with 2e-5 for InLegalBERT
6. **Increase epochs gradually** - legal documents often need more training

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or max_length
2. **Poor Performance**: Try different context modes or increase epochs
3. **Data Loading Errors**: Check file formats and role names
4. **Slow Training**: Use GPU if available, reduce sequence length

### Getting Help

Check the logs for detailed error messages:
```bash
python train.py --help
python evaluate.py --help
python dataset_preprocessor.py --help
```