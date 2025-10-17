# Confidence Scores and Clustering Approach Guide

## Table of Contents
1. [How Confidence Works](#how-confidence-works)
2. [Clustering as Unsupervised Approach](#clustering-as-unsupervised-approach)
3. [Comparison Framework](#comparison-framework)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)

---

## 1. How Confidence Works

### 1.1 Supervised Classification Confidence

In the **supervised InLegalBERT classifier**, confidence comes from **softmax probabilities**:

```python
# Forward pass through model
logits = model(input_ids, attention_mask)  # Raw scores

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=-1)
# Output: [0.05, 0.12, 0.71, 0.03, 0.06, 0.02, 0.01]
#         [None, Facts, Issue, AoP, AoR, Reasoning, Decision]

# Get prediction and confidence
predicted_id = torch.argmax(probabilities, dim=-1)  # Index of max prob
confidence = probabilities[predicted_id].item()      # Max probability value

# Example:
# If predicted_id = 2 (Issue), confidence = 0.71 (71%)
```

#### Confidence Interpretation

| Confidence Range | Interpretation | Action |
|------------------|----------------|---------|
| 0.90 - 1.00 | Very High | Trust the prediction |
| 0.70 - 0.89 | High | Generally reliable |
| 0.50 - 0.69 | Moderate | Review manually |
| 0.30 - 0.49 | Low | High uncertainty |
| 0.00 - 0.29 | Very Low | Likely incorrect |

### 1.2 Unsupervised Clustering Confidence

In the **clustering approach**, confidence is calculated from **distance to cluster center**:

```python
# Calculate distance to assigned cluster center
distance = np.linalg.norm(feature_vector - cluster_center)

# Convert to confidence (inverse distance, normalized)
confidence = 1.0 / (1.0 + distance)

# Interpretation:
# - distance = 0.1 → confidence = 0.91 (very close to center)
# - distance = 1.0 → confidence = 0.50 (moderate distance)
# - distance = 5.0 → confidence = 0.17 (far from center)
```

### 1.3 Comparison: Confidence Calculation

```python
# SUPERVISED: Softmax probability distribution
def supervised_confidence(logits):
    probs = softmax(logits)
    # Confidence = probability of predicted class
    return max(probs)

# UNSUPERVISED: Distance-based confidence
def unsupervised_confidence(point, cluster_center):
    distance = euclidean_distance(point, cluster_center)
    # Confidence = inverse distance (normalized)
    return 1.0 / (1.0 + distance)
```

**Key Differences:**
- **Supervised**: Based on model's learned decision boundaries
- **Unsupervised**: Based on proximity to cluster centroid
- **Supervised**: Generally more calibrated (if well-trained)
- **Unsupervised**: More interpretable (geometric distance)

---

## 2. Clustering as Unsupervised Approach

### 2.1 Why Use Clustering?

**Advantages:**
1. ✅ **No labeled data required** for training
2. ✅ **Discovers natural patterns** in document structure
3. ✅ **Interpretable** through cluster visualization
4. ✅ **Fast inference** (no neural network forward pass)
5. ✅ **Works across domains** (transferable)

**Disadvantages:**
1. ❌ **Lower accuracy** than supervised methods
2. ❌ **Requires manual cluster-to-role mapping**
3. ❌ **Sensitive to hyperparameters** (k, distance metric)
4. ❌ **No context modeling** (unlike transformers)

### 2.2 Clustering Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              INPUT: LEGAL DOCUMENT                       │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │ 1. SENTENCE SPLITTING│
          │  - spaCy or regex    │
          └──────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │ 2. EMBEDDING         │
          │  - SentenceTransformer│
          │  - Output: 384-dim   │
          └──────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │ 3. CLUSTERING        │
          │  - K-Means (k=7)     │
          │  - Hierarchical      │
          │  - DBSCAN/HDBSCAN    │
          └──────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │ 4. CLUSTER LABELING  │
          │  - Keyword matching  │
          │  - Position analysis │
          └──────────┬───────────┘
                     │
          ┌──────────▼──────────┐
          │ 5. ROLE ASSIGNMENT   │
          │  - Cluster → Role    │
          │  - Confidence calc.  │
          └─────────────────────┘
```

### 2.3 Clustering Algorithms Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **K-Means** | Balanced clusters | Fast, simple | Requires k, spherical clusters |
| **Hierarchical** | Varying cluster sizes | No k needed, dendrogram | Slow for large data |
| **DBSCAN** | Arbitrary shapes | Finds outliers, no k | Sensitive to density |
| **HDBSCAN** | Varying densities | Robust, hierarchical | Complex parameters |

### 2.4 Cluster-to-Role Mapping

**Heuristic-based mapping:**

```python
# Example: Cluster analysis
Cluster 0 sentences:
- "The petitioner filed a writ petition..."
- "The appellant is a citizen of India..."
- "The parties entered into an agreement..."

Keywords detected: ["petitioner", "filed", "appellant", "parties"]
→ Mapped to: "Facts"

Cluster 3 sentences:
- "The court examined the constitutional provisions..."
- "In our view, the right to privacy is fundamental..."
- "We find that the impugned provision violates..."

Keywords detected: ["court", "view", "find", "examined"]
→ Mapped to: "Reasoning"
```

**Statistical mapping:**

```python
# Calculate keyword scores for each cluster
for cluster_id in range(k):
    cluster_sentences = sentences[labels == cluster_id]
    
    role_scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        score = count_keyword_matches(cluster_sentences, keywords)
        role_scores[role] = score
    
    # Assign role with highest score
    cluster_to_role[cluster_id] = max(role_scores, key=role_scores.get)
```

---

## 3. Comparison Framework

### 3.1 Evaluation Metrics

#### **Accuracy**
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

#### **F1-Score (Macro)**
```
F1_macro = Average of F1 scores across all roles
F1_per_role = 2 × (Precision × Recall) / (Precision + Recall)
```

#### **F1-Score (Weighted)**
```
F1_weighted = Σ (F1_per_role × support_per_role) / total_samples
```

#### **Confidence Metrics**
```
Mean Confidence = Average confidence across all predictions
Confidence Std = Standard deviation of confidence scores
```

#### **Inference Time**
```
Avg Inference Time = Total time / Number of documents
```

### 3.2 Expected Performance Comparison

| Metric | Supervised | Unsupervised | Winner |
|--------|------------|--------------|--------|
| **Accuracy** | 85-90% | 60-70% | Supervised |
| **F1 (Macro)** | 0.85-0.88 | 0.55-0.65 | Supervised |
| **Confidence** | 0.75-0.85 | 0.60-0.75 | Supervised |
| **Speed** | Slower | Faster | Unsupervised |
| **Training** | Requires labels | Label-free | Unsupervised |
| **Interpretability** | Black box | Transparent | Unsupervised |

### 3.3 When to Use Each Approach

**Use Supervised (InLegalBERT) when:**
- ✅ You have labeled training data
- ✅ High accuracy is critical
- ✅ Domain-specific patterns matter
- ✅ Computational resources available

**Use Unsupervised (Clustering) when:**
- ✅ No labeled data available
- ✅ Quick prototyping needed
- ✅ Interpretability is important
- ✅ Cross-domain application
- ✅ Resource constraints (no GPU)

**Use Hybrid Approach when:**
- ✅ Limited labeled data
- ✅ Need both accuracy and interpretability
- ✅ Active learning scenarios
- ✅ Ensemble methods

---

## 4. Implementation Details

### 4.1 Supervised Classifier

```python
from src.models.role_classifier import RoleClassifier

# Initialize classifier
classifier = RoleClassifier(model_type="inlegalbert", device="cpu")

# Load trained weights
classifier.load_pretrained_weights("path/to/model.pt")

# Classify document
results = classifier.classify_document(document_text, context_mode="prev")

# Access results
for result in results:
    print(f"Sentence: {result['sentence']}")
    print(f"Role: {result['role']}")
    print(f"Confidence: {result['confidence']:.3f}")  # ← Softmax probability
```

### 4.2 Unsupervised Clustering Classifier

```python
from src.models.clustering_role_classifier import (
    ClusteringRoleClassifier, 
    ClusteringConfig
)

# Configure clustering
config = ClusteringConfig(
    num_clusters=7,
    clustering_algorithm="kmeans",  # or "hierarchical", "dbscan"
    embedding_model="all-MiniLM-L6-v2"
)

# Initialize classifier
classifier = ClusteringRoleClassifier(config)

# Fit on data (can use unlabeled data!)
sentences = extract_sentences(document_text)
classifier.fit(sentences)

# Classify new document
results = classifier.classify_document(document_text)

# Access results
for result in results:
    print(f"Sentence: {result.sentence}")
    print(f"Role: {result.predicted_role}")
    print(f"Confidence: {result.confidence:.3f}")  # ← Distance-based
    print(f"Cluster: {result.cluster_id}")
```

### 4.3 Comparison Framework

```python
from compare_approaches import ApproachComparison

# Initialize comparison
comparison = ApproachComparison(
    data_dir="dataset/Hier_BiLSTM_CRF/test",
    output_dir="comparison_results"
)

# Load test data
comparison.load_test_data(max_files=50)

# Train unsupervised model
comparison.train_unsupervised(use_train_data=True)

# Evaluate both approaches
comparison.evaluate_supervised()
comparison.evaluate_unsupervised()

# Generate comprehensive report
comparison.generate_comparison_report()

# Create visualizations
comparison.plot_comparison()

# Print summary
comparison.print_summary()
```

---

## 5. Usage Examples

### 5.1 Compare on Your Dataset

```bash
# Run comparison
python compare_approaches.py \
    --test-dir dataset/Hier_BiLSTM_CRF/test \
    --max-files 50 \
    --output-dir comparison_results
```

**Output:**
```
comparison_results/
├── comparison_report.json      # Detailed metrics
├── comparison_plots.png        # Visualizations
└── insights.txt               # Key findings
```

### 5.2 Visualize Clusters

```python
import umap
import matplotlib.pyplot as plt

# Get embeddings
embeddings = classifier.embedding_model.encode(sentences)

# Reduce to 2D for visualization
reducer = umap.UMAP(n_components=2, random_state=42)
embeddings_2d = reducer.fit_transform(embeddings)

# Plot with role colors
plt.figure(figsize=(12, 8))
for role in ROLES:
    mask = predictions == role
    plt.scatter(
        embeddings_2d[mask, 0], 
        embeddings_2d[mask, 1],
        label=role,
        alpha=0.6
    )
plt.legend()
plt.title("Sentence Embeddings Colored by Role")
plt.show()
```

### 5.3 Confidence Analysis

```python
import numpy as np
import matplotlib.pyplot as plt

# Collect confidences from both approaches
sup_confidences = [r['confidence'] for r in supervised_results]
unsup_confidences = [r.confidence for r in unsupervised_results]

# Plot distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(sup_confidences, bins=50, alpha=0.7, color='blue')
plt.title("Supervised: Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(unsup_confidences, bins=50, alpha=0.7, color='red')
plt.title("Unsupervised: Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Statistics
print(f"Supervised - Mean: {np.mean(sup_confidences):.3f}, Std: {np.std(sup_confidences):.3f}")
print(f"Unsupervised - Mean: {np.mean(unsup_confidences):.3f}, Std: {np.std(unsup_confidences):.3f}")
```

---

## 6. Key Takeaways

### Confidence:
1. **Supervised**: Uses softmax probabilities (model learned)
2. **Unsupervised**: Uses distance to cluster center (geometric)
3. Both provide 0-1 scores, but interpretation differs

### Clustering Approach:
1. **No labels needed** - great for bootstrapping
2. **Lower accuracy** than supervised - expect 60-70% vs 85-90%
3. **More interpretable** - can visualize clusters
4. **Useful for** data exploration and active learning

### Comparison:
1. **Supervised wins** on accuracy and F1-score
2. **Unsupervised wins** on speed and no-label requirement
3. **Best practice**: Use both and compare on your data
4. **Consider hybrid**: Use clustering to suggest labels, then fine-tune

---

## 7. Next Steps

1. **Run the comparison** on your test set
2. **Analyze the results** - which roles are hard to cluster?
3. **Try different algorithms** - K-Means vs Hierarchical vs HDBSCAN
4. **Tune hyperparameters** - k, embedding model, distance metric
5. **Consider ensemble** - combine both approaches
6. **Active learning** - use clustering to find uncertain samples for labeling

---

## References

- InLegalBERT: [https://huggingface.co/law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT)
- Sentence-Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- Scikit-learn Clustering: [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)
- UMAP: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/)
