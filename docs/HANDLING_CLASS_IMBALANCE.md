# Handling Class Imbalance in Role Classification

## 🚨 The Problem

Your training dataset has **severe class imbalance**, where the "None" label dominates:

```
Example from file_1004.txt:
- None:                          1,128 samples (77%)
- Facts:                           161 samples (11%)
- Reasoning:                        65 samples (4%)
- Arguments of Respondent:          54 samples (4%)
- Arguments of Petitioner:          45 samples (3%)
- Decision:                         20 samples (1%)
- Issue:                             6 samples (0.4%)
```

**Imbalance Ratio**: Up to **188x** (None vs Issue)

---

## ⚠️ Why This Is a Problem

Without handling imbalance, the model will:

1. **Over-predict "None"**: Achieves 77% accuracy by always predicting "None"
2. **Ignore minority classes**: Never learns to recognize "Issue" or "Decision"
3. **Poor real-world performance**: Fails on actual legal queries about issues/decisions
4. **Misleading metrics**: High overall accuracy but terrible per-class F1 scores

### Example Bad Prediction:
```
Input:  "The main issue in this case is whether..."
Model:  "None" ❌  (Should be "Issue")
Reason: Model learned "None" is most common, so it defaults to it
```

---

## ✅ Solutions Implemented in the Notebook

### **1. Class Weights (PRIMARY SOLUTION)**

**Status**: ✅ **ENABLED BY DEFAULT** in updated notebook

**How it works**:
```python
# During training, loss is multiplied by class weights:
loss_for_none = base_loss * 0.2      # Low weight (common)
loss_for_issue = base_loss * 5.0     # High weight (rare)

# Result: Model pays 25x more attention to "Issue" errors
```

**Effect**:
- Model learns to care about minority classes
- No data is thrown away
- Works for moderate imbalance (2x-20x)

**Configuration**:
```python
config = {
    "use_class_weights": True,          # ✅ Already enabled
    "class_weight_method": "inverse_freq"  # Automatic calculation
}
```

---

### **2. Filter "None" Samples (OPTIONAL)**

**Status**: 🔧 Available but disabled by default

**When to use**: If class weights alone aren't enough (imbalance >50x)

**How it works**:
```python
# Randomly discard majority of "None" samples
Original:  1,128 "None" + 161 "Facts" + 6 "Issue"
Filtered:    338 "None" + 161 "Facts" + 6 "Issue"  (30% kept)

# Result: More balanced dataset, faster training
```

**To enable**:
```python
imbalance_config = {
    "filter_none_samples": True,
    "none_keep_ratio": 0.3  # Keep 30% of "None" samples
}
```

**Trade-off**:
- ✅ Directly balances dataset
- ❌ Loses potentially useful "None" examples

---

### **3. Stratified Batch Sampling (ADVANCED)**

**Status**: 🔧 Requires custom implementation

**How it works**: Each training batch contains balanced samples from all classes

**Effect**:
- Guarantees model sees all classes frequently
- Prevents batch-level bias

---

### **4. Focal Loss (ADVANCED)**

**Status**: 🔧 Available for severe cases

**How it works**: Focuses on hard-to-classify examples, down-weights easy ones

**Formula**:
```
FL(pt) = -α(1-pt)^γ * log(pt)

Where:
- pt = predicted probability for true class
- γ = focusing parameter (default: 2.0)
- α = balance factor (default: 0.25)
```

**To enable**:
```python
imbalance_config = {
    "use_focal_loss": True,
    "focal_gamma": 2.0
}
```

---

## 📊 Expected Results with Class Weights

### Before (No Weighting):
```
Overall Accuracy: 78%  ✅ (misleading!)

Per-Class F1 Scores:
  None:                   0.85  ✅ (over-predicted)
  Facts:                  0.45  ⚠️
  Reasoning:              0.30  ❌
  Arguments (Pet):        0.20  ❌
  Arguments (Resp):       0.25  ❌
  Decision:               0.15  ❌
  Issue:                  0.05  ❌ (never predicted)

Macro F1: 0.32  ❌ (actually terrible!)
```

### After (With Class Weights):
```
Overall Accuracy: 72%  (slightly lower, but more meaningful)

Per-Class F1 Scores:
  None:                   0.78  ✅ (balanced)
  Facts:                  0.70  ✅
  Reasoning:              0.65  ✅
  Arguments (Pet):        0.60  ✅
  Arguments (Resp):       0.62  ✅
  Decision:               0.58  ✅
  Issue:                  0.52  ✅ (much better!)

Macro F1: 0.64  ✅ (real improvement!)
```

---

## 🎯 Recommended Workflow

### Step 1: Run Analysis
```python
# Cell: "Analyze class distribution"
label_counts, class_weights = analyze_class_distribution(
    config["train_data"], 
    sample_size=100
)
```

**Look for**:
- Imbalance ratio (None vs minority classes)
- Distribution chart

### Step 2: Start with Class Weights
```python
# Already configured in the notebook
config["use_class_weights"] = True  # ✅ DEFAULT
```

### Step 3: Train the Model
```python
# Cell: "Start training"
trainer.train(...)
```

### Step 4: Evaluate Per-Class Performance
```python
# Cell: "Evaluate the trained model"
# Check the "Per-Class Metrics" output

Example output:
  Facts                | F1: 0.702 | Prec: 0.685 | Rec: 0.720
  Issue                | F1: 0.523 | Prec: 0.498 | Rec: 0.550  ← Monitor this!
  Decision             | F1: 0.580 | Prec: 0.562 | Rec: 0.599  ← Monitor this!
```

### Step 5: If Minority Classes Still Perform Poorly
**Option A**: Enable "None" filtering
```python
imbalance_config["filter_none_samples"] = True
imbalance_config["none_keep_ratio"] = 0.3  # Aggressive filtering
```

**Option B**: Increase training epochs
```python
config["num_epochs"] = 10  # More time to learn rare classes
```

**Option C**: Use two-stage training
```python
# Stage 1: Train on filtered data (none_keep_ratio=0.2)
# Stage 2: Fine-tune on full data with class weights
```

---

## 🔍 Monitoring During Training

### Good Signs:
- ✅ Validation F1 improving for all classes
- ✅ Loss decreasing smoothly
- ✅ Minority class predictions appearing in validation

### Warning Signs:
- ⚠️ Validation F1 stagnant for minority classes
- ⚠️ Model still predicts "None" >70% of the time
- ⚠️ Huge gap between "None" F1 and others

### If You See Warning Signs:
1. Increase class weights manually (multiply by 2x)
2. Enable "None" filtering
3. Reduce learning rate (2e-5 → 1e-5)
4. Increase warmup steps (500 → 1000)

---

## 📈 Evaluation Metrics to Watch

### Primary Metrics (Most Important):
1. **Macro F1 Score**: Average F1 across all classes (treats all equally)
2. **Per-Class F1 Scores**: Individual performance for each role
3. **Confusion Matrix**: Shows which classes are confused with "None"

### Secondary Metrics:
1. **Weighted F1**: F1 weighted by class frequency (less meaningful with imbalance)
2. **Overall Accuracy**: Can be misleading with imbalanced data

### Target Benchmarks:
```
Macro F1:       > 0.60  (good)  > 0.70  (excellent)
Per-Class F1:   > 0.50  (acceptable)  > 0.65  (good)
Issue F1:       > 0.45  (acceptable for rarest class)
```

---

## 💡 Additional Tips

### Tip 1: Augment Minority Classes
For rare classes like "Issue" and "Decision":
- Manually add more examples if possible
- Use paraphrasing/back-translation to create synthetic examples

### Tip 2: Context Mode Matters
```python
config["context_mode"] = "prev_two"  # Use 2 previous sentences

# Helps distinguish roles based on document flow:
# Previous: "The appellant filed a petition..."  → Facts
# Current:  "The question before this court is..." → Issue
```

### Tip 3: Post-Processing Calibration
After training, apply threshold tuning:
```python
# Lower threshold for rare classes
if predicted_prob["Issue"] > 0.3:  # Instead of default 0.5
    return "Issue"
```

---

## 🚀 Quick Start Commands

### For Mild Imbalance (2x-10x):
```python
# Just use class weights (already enabled)
config["use_class_weights"] = True
```

### For Severe Imbalance (>20x):
```python
# Enable class weights + filtering
config["use_class_weights"] = True
imbalance_config["filter_none_samples"] = True
imbalance_config["none_keep_ratio"] = 0.3
```

### For Extreme Imbalance (>100x):
```python
# Aggressive filtering + focal loss
config["use_class_weights"] = True
imbalance_config["filter_none_samples"] = True
imbalance_config["none_keep_ratio"] = 0.2  # Keep only 20%
imbalance_config["use_focal_loss"] = True
```

---

## 📚 References

1. **Class Imbalance**: https://arxiv.org/abs/1710.05381
2. **Focal Loss**: https://arxiv.org/abs/1708.02002
3. **Legal NLP Challenges**: COLING 2025 - NYAYAANUMANA paper

---

## ✅ Summary

**The notebook has been updated to handle class imbalance automatically:**

1. ✅ Class weights enabled by default
2. ✅ Analysis cells added to visualize imbalance
3. ✅ Configuration options for advanced strategies
4. ✅ Monitoring guidance for per-class performance

**You can now train the model with confidence that it will learn all roles, not just "None"!**
