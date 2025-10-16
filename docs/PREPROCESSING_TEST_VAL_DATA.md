# Preprocessing Raw Test/Val Data for Evaluation

## 🚨 Problem Identified

Your **test** and **validation** datasets contain **raw legal documents** (plain text) without labels:

```
# Training data format (✅ Labeled):
sentence1\tFacts
sentence2\tIssue
sentence3\tReasoning

# Test/Val data format (❌ Unlabeled - Raw text):
Dr Dhananjaya Y Chandrachud, J 1.
This appeal arises from a judgment dated 24 June 2020...
The NCLAT upheld the interim order...
```

**This creates a mismatch**: The training pipeline expects `sentence\trole` format, but test/val files are raw text.

---

## 🎯 Solution Options

### **Option 1: Preprocess Raw Files (RECOMMENDED)**

Convert raw text to `sentence\trole` format by:
1. Sentence segmentation
2. Initial labeling as "None"
3. Let the model predict actual labels

#### When to use:
- ✅ You want to evaluate model performance
- ✅ You have NO gold labels for test/val
- ✅ You need to bootstrap evaluation data

#### How to do it:
```python
# In the notebook, run cells 19-21 (new preprocessing cells)

# Enable preprocessing
preprocess_data = True

# This will:
# 1. Split raw text into sentences
# 2. Create sentence\trole format (all labeled as 'None' initially)
# 3. Save to test_preprocessed/ and val_preprocessed/
```

---

### **Option 2: Manual Annotation (GOLD STANDARD)**

If you want **true evaluation**, you need gold labels:

#### Steps:
1. **Export raw test files** to annotation tool
2. **Manually label** each sentence with correct role
3. **Create test/val files** in `sentence\trole` format
4. **Place them** in the correct directories

#### Tools for annotation:
- **Label Studio**: https://labelstud.io/
- **Prodigy**: https://prodi.gy/
- **Doccano**: https://github.com/doccano/doccano
- **Excel/CSV**: Simple but works for small datasets

#### Example workflow:
```bash
# 1. Extract sentences from raw files
python preprocess_for_annotation.py --input test/ --output test_to_annotate.csv

# 2. Annotate in tool (e.g., Label Studio)
# 3. Export labeled data

# 4. Convert back to sentence\trole format
python convert_annotations.py --input annotations.csv --output test_labeled/
```

---

### **Option 3: Use Model for Inference Only (NO EVALUATION)**

If you don't need evaluation and just want predictions:

#### When to use:
- ✅ No gold labels available
- ✅ Just need to classify new documents
- ✅ Manual review is acceptable

#### How to do it:
```python
# In the notebook, run cell 22 (inference cell)

# Enable inference mode (AFTER training)
inference_mode = True

# This will:
# 1. Load raw document
# 2. Predict role for each sentence
# 3. Save predictions to file
# 4. Display distribution and samples
```

---

## 📋 Detailed Workflow

### **Workflow A: Preprocess for Evaluation**

```
┌─────────────────────────────────────────────────┐
│ Step 1: Preprocess Raw Files                   │
├─────────────────────────────────────────────────┤
│ Input:  test/file_6409.txt (raw text)         │
│ Output: test_preprocessed/file_6409.txt        │
│         (sentence\tNone format)                 │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 2: Train Model on Training Data           │
├─────────────────────────────────────────────────┤
│ Uses: train/ dataset (labeled)                 │
│ Output: trained_models/best_model.pt           │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 3: Evaluate on Preprocessed Test Data     │
├─────────────────────────────────────────────────┤
│ Uses: test_preprocessed/ (now has labels)      │
│ Note: Initial labels are 'None', but model     │
│       predictions replace them during eval      │
│ Output: Metrics (accuracy, F1, etc.)           │
└─────────────────────────────────────────────────┘
```

### **Workflow B: Manual Gold Labeling**

```
┌─────────────────────────────────────────────────┐
│ Step 1: Export Raw Files to Annotation Format  │
├─────────────────────────────────────────────────┤
│ Script: preprocess_for_annotation.py           │
│ Input:  test/*.txt (raw)                        │
│ Output: test_to_annotate.csv                    │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 2: Manual Annotation                      │
├─────────────────────────────────────────────────┤
│ Tool: Label Studio / Prodigy / Doccano         │
│ Task: Assign role to each sentence             │
│ Output: annotations.csv (with gold labels)      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 3: Convert Back to Training Format        │
├─────────────────────────────────────────────────┤
│ Script: convert_annotations.py                 │
│ Input:  annotations.csv                         │
│ Output: test_labeled/*.txt (sentence\trole)     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 4: Train and Evaluate with Gold Labels    │
├─────────────────────────────────────────────────┤
│ Now you have TRUE evaluation metrics!          │
└─────────────────────────────────────────────────┘
```

---

## 🔧 Implementation in Notebook

### **New Cells Added (19-22)**

#### **Cell 19**: Documentation
- Explains the problem and solutions

#### **Cell 20**: Single File Preprocessing
```python
# Test preprocessing on one file
sample_test_file = Path(config["test_data"]) / "file_6409.txt"
result = preprocess_raw_document(sample_test_file)
```

#### **Cell 21**: Batch Preprocessing
```python
# Set to True to preprocess all files
preprocess_data = True

# Processes all test and val files
# Creates test_preprocessed/ and val_preprocessed/
```

#### **Cell 22**: Inference Mode
```python
# Set to True AFTER training
inference_mode = True

# Predicts on raw documents
# Saves predictions
# No gold labels needed
```

---

## 📊 Expected Output

### **After Preprocessing (Cell 21)**:
```
📂 Processing 500 files from server/dataset/Hier_BiLSTM_CRF/test
📂 Output directory: server/dataset/Hier_BiLSTM_CRF/test_preprocessed

✅ Preprocessing complete!
   Total files: 500
   Total sentences: 45,234
   Average sentences per file: 90.5
```

### **Preprocessed File Format**:
```
# test_preprocessed/file_6409.txt
Dr Dhananjaya Y Chandrachud, J 1.	None
This appeal arises from a judgment dated 24 June 2020...	None
The NCLAT upheld the interim order...	None
```

### **After Inference (Cell 22)**:
```
📊 Prediction Summary:
   Total sentences: 87

🏷️ Predicted Role Distribution:
   Facts                           32 ( 36.8%)
   Reasoning                       18 ( 20.7%)
   Arguments of Petitioner         12 ( 13.8%)
   Issue                            8 (  9.2%)
   Decision                         7 (  8.0%)
   Arguments of Respondent          6 (  6.9%)
   None                             4 (  4.6%)
```

---

## ⚠️ Important Considerations

### **1. Evaluation Without Gold Labels**

When you preprocess raw files to `sentence\tNone` format:

**What happens during evaluation?**
- The model predicts labels for each sentence
- Evaluation compares predictions against "None" labels
- **This is NOT a true evaluation!**
- Metrics will be meaningless

**Solution**: Use this approach to:
- ✅ Generate predictions for manual review
- ✅ Bootstrap annotation process
- ✅ Test model inference pipeline
- ❌ NOT for reporting final evaluation metrics

### **2. True Evaluation Requires Gold Labels**

For **meaningful metrics** (accuracy, F1, precision, recall):
- You MUST have manually labeled test data
- Each sentence needs a correct gold label
- Compare model predictions against gold labels

### **3. Sentence Segmentation Quality**

The preprocessing uses **spaCy** for sentence splitting:
- ✅ Works well for standard legal text
- ⚠️ May split incorrectly on:
  - Section numbers (e.g., "Section 377. The court held...")
  - Abbreviations (e.g., "Dr. Smith said...")
  - Citations (e.g., "In ABC v. DEF (2020) 1 SCC 123...")

**Solution**: Manually review a sample of preprocessed files

---

## 🎯 Recommended Approach

### **For Your Use Case:**

Since your test/val are raw documents, I recommend:

#### **Phase 1: Bootstrap (NOW)**
1. ✅ **Enable preprocessing** (Cell 21: `preprocess_data = True`)
2. ✅ **Train model** on training data
3. ✅ **Run inference** on preprocessed test data
4. ✅ **Manually review** ~50-100 predictions

#### **Phase 2: Gold Labeling (NEXT)**
1. 🔄 **Select 100-200 sentences** from test set
2. 🔄 **Manually label** them with correct roles
3. 🔄 **Create small gold test set**
4. 🔄 **Evaluate on gold set** for true metrics

#### **Phase 3: Production (LATER)**
1. 🚀 **Use trained model** for inference
2. 🚀 **Predict on new documents**
3. 🚀 **Integrate into Nyaya API**

---

## 📝 Code Templates

### **Template 1: Preprocess Single File**
```python
from pathlib import Path
import spacy

def preprocess_file(input_file, output_file):
    nlp = spacy.load("en_core_web_sm")
    
    with open(input_file, 'r') as f:
        raw_text = f.read()
    
    doc = nlp(raw_text)
    
    with open(output_file, 'w') as f:
        for sent in doc.sents:
            if len(sent.text.strip()) > 10:
                f.write(f"{sent.text.strip()}\tNone\n")

# Usage
preprocess_file("test/file_6409.txt", "test_preprocessed/file_6409.txt")
```

### **Template 2: Predict and Save**
```python
def predict_document(evaluator, input_file, output_file):
    nlp = spacy.load("en_core_web_sm")
    
    with open(input_file, 'r') as f:
        raw_text = f.read()
    
    doc = nlp(raw_text)
    
    with open(output_file, 'w') as f:
        for sent in doc.sents:
            sentence = sent.text.strip()
            if len(sentence) > 10:
                result = evaluator.predict_single(sentence)
                role = result['predicted_role']
                conf = result['confidence']
                f.write(f"{sentence}\t{role}\t{conf:.3f}\n")

# Usage (after training)
predict_document(evaluator, "test/file_6409.txt", "predictions/file_6409.txt")
```

### **Template 3: Compare Predictions vs Gold**
```python
def evaluate_predictions(pred_file, gold_file):
    from sklearn.metrics import classification_report
    
    pred_labels = []
    gold_labels = []
    
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pred_labels.append(parts[1])
    
    with open(gold_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                gold_labels.append(parts[1])
    
    print(classification_report(gold_labels, pred_labels))

# Usage (when you have gold labels)
evaluate_predictions("predictions/file_6409.txt", "gold_test/file_6409.txt")
```

---

## 🚀 Quick Start Commands

### **Enable Preprocessing**
```python
# In Cell 21
preprocess_data = True  # Change from False to True
```

### **Run Inference After Training**
```python
# In Cell 22 (AFTER Cell 14-18 training is complete)
inference_mode = True  # Change from False to True
```

### **Check Preprocessed Output**
```bash
# In terminal
head -20 server/dataset/Hier_BiLSTM_CRF/test_preprocessed/file_6409.txt
```

---

## ✅ Summary

| Approach | Gold Labels | Evaluation Metrics | Use Case |
|----------|-------------|-------------------|----------|
| **Preprocess Raw** | ❌ No | ❌ Not meaningful | Bootstrapping, inference testing |
| **Manual Annotation** | ✅ Yes | ✅ True metrics | Research, production validation |
| **Inference Only** | ❌ No | ❌ None | Production inference, new documents |

**Recommendation**: Start with preprocessing to test the pipeline, then create a small gold labeled set for true evaluation.

---

## 📚 Additional Resources

1. **Sentence Segmentation**: https://spacy.io/usage/linguistic-features#sbd
2. **Annotation Tools**: https://github.com/heartexlabs/awesome-data-labeling
3. **Inter-Annotator Agreement**: https://en.wikipedia.org/wiki/Inter-rater_reliability
4. **Active Learning**: https://modal-python.readthedocs.io/en/latest/

---

**You're now ready to handle raw test/val data!** 🎉
