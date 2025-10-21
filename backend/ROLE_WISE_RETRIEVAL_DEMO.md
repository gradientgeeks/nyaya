# Role-Wise Retrieval: How It Works

## Overview

The upload script creates a data structure in Pinecone that enables **role-aware retrieval** - the key innovation of Nyaya that makes it different from standard RAG systems.

---

## Data Structure in Pinecone

### After Upload, Each Sentence Becomes a Vector:

```
Vector ID: file_1_sent_5
├── Values: [0.123, 0.456, ..., 0.789]  (384 dimensions)
└── Metadata:
    ├── text: "The petitioner filed a writ petition under Article 32..."
    ├── role: "Facts"                    # ← KEY: Role label
    ├── case_id: "file_1"
    ├── sentence_index: 5
    ├── confidence: 1.0
    ├── user_uploaded: false
    ├── court: "Indian Courts"
    └── category: "Legal Training Data"
```

### Namespace Organization:

```
nyaya-legal-rag (Pinecone Index)
│
├── training_data (Namespace)           # ← Your uploaded training files
│   ├── file_1_sent_0 (Facts)
│   ├── file_1_sent_1 (Facts)
│   ├── file_1_sent_2 (Issue)
│   ├── file_1_sent_3 (Arguments of Petitioner)
│   ├── ...
│   ├── file_2_sent_0 (Facts)
│   └── ...
│
├── user_documents (Namespace)          # ← User-uploaded cases (via FastAPI)
│   ├── case_12345_sent_0 (Facts)
│   └── ...
│
└── demo (Namespace)                    # ← Demo/test data
    └── ...
```

---

## How Role-Wise Retrieval Works

### Standard RAG (Without Roles):

```python
# User asks: "What were the facts?"
query_embedding = embed("What were the facts?")

# Standard RAG retrieves ANY similar text
results = index.query(vector=query_embedding, top_k=5)

# Returns MIXED content:
# - Match 1: "The petitioner filed..." (Facts) ✓
# - Match 2: "The court reasoned that..." (Reasoning) ✗
# - Match 3: "The petition is dismissed." (Decision) ✗
# - Match 4: "Article 21 guarantees..." (Reasoning) ✗
# - Match 5: "The facts are as follows..." (Facts) ✓
```

**Problem:** User asked for facts, but got reasoning and decision mixed in.

---

### Nyaya RAG (With Role Filtering):

```python
# User asks: "What were the facts?"
query_embedding = embed("What were the facts?")

# Nyaya detects intent: user wants "Facts" role
# Query with role filter
results = index.query(
    vector=query_embedding,
    top_k=5,
    filter={"role": {"$eq": "Facts"}},  # ← FILTER BY ROLE
    namespace="training_data"
)

# Returns ONLY Facts:
# - Match 1: "The petitioner filed..." (Facts) ✓
# - Match 2: "The case arose from..." (Facts) ✓
# - Match 3: "On 15th March 2020..." (Facts) ✓
# - Match 4: "The appellant contends..." (Facts) ✓
# - Match 5: "Background of the case..." (Facts) ✓
```

**Result:** User gets EXACTLY what they asked for - only Facts!

---

## Example Queries with Role Filtering

### 1. Get Facts from a Specific Case

```python
results = index.query(
    vector=query_embedding,
    filter={
        "case_id": {"$eq": "file_42"},
        "role": {"$eq": "Facts"}
    },
    namespace="training_data",
    top_k=10
)
```

**Returns:** All Facts sentences from file_42.txt

---

### 2. Get Court's Reasoning

```python
results = index.query(
    vector=embed("What was the court's reasoning?"),
    filter={"role": {"$eq": "Reasoning"}},
    namespace="training_data",
    top_k=5
)
```

**Returns:** Only Reasoning sentences from all cases.

---

### 3. Get Final Decisions

```python
results = index.query(
    vector=embed("What was the decision?"),
    filter={"role": {"$eq": "Decision"}},
    namespace="training_data",
    top_k=3
)
```

**Returns:** Only Decision sentences.

---

### 4. Multi-Role Query

```python
# Get both Facts and Issues
results = index.query(
    vector=query_embedding,
    filter={"role": {"$in": ["Facts", "Issue"]}},
    namespace="training_data",
    top_k=10
)
```

**Returns:** Facts and Issue sentences only (no Reasoning/Decision).

---

## Role-Wise Similarity Search (Advanced)

Following the architecture (line 600-698 in ARCHITECTURE.md):

### Algorithm:

```python
def find_similar_cases(uploaded_case_id: str):
    """Find cases similar to uploaded case using role-wise comparison."""

    # Step 1: Get all vectors for the uploaded case
    uploaded_vectors = index.query(
        filter={"case_id": uploaded_case_id},
        top_k=1000,
        namespace="user_documents"
    )

    # Step 2: Group by role
    vectors_by_role = {
        "Facts": [],
        "Issue": [],
        "Reasoning": [],
        "Decision": []
    }

    for vector in uploaded_vectors:
        role = vector.metadata.role
        if role in vectors_by_role:
            vectors_by_role[role].append(vector.values)

    # Step 3: For each role, search similar cases
    case_scores = {}

    for role, vectors in vectors_by_role.items():
        # Average embedding for this role
        avg_embedding = mean(vectors, axis=0)

        # Search training_data with role filter
        results = index.query(
            vector=avg_embedding,
            filter={
                "role": {"$eq": role},
                "case_id": {"$ne": uploaded_case_id}  # Exclude self
            },
            namespace="training_data",
            top_k=50
        )

        # Accumulate scores by case_id
        for match in results.matches:
            case_id = match.metadata.case_id

            if case_id not in case_scores:
                case_scores[case_id] = {}

            case_scores[case_id][role] = match.score

    # Step 4: Calculate weighted similarity
    similar_cases = []

    for case_id, role_scores in case_scores.items():
        overall = (
            role_scores.get("Facts", 0) * 0.40 +      # 40% weight
            role_scores.get("Reasoning", 0) * 0.30 +  # 30% weight
            role_scores.get("Issue", 0) * 0.30        # 30% weight
        )

        similar_cases.append({
            "case_id": case_id,
            "overall_similarity": overall,
            "role_scores": role_scores
        })

    # Step 5: Sort by overall similarity
    similar_cases.sort(key=lambda x: x["overall_similarity"], reverse=True)

    return similar_cases[:10]
```

### Why This is Better:

**Standard RAG:**
- Compares entire document embeddings
- No understanding of document structure
- Facts might match with Reasoning (semantically similar)

**Nyaya Role-Wise:**
- Compares Facts with Facts, Reasoning with Reasoning
- Understands document structure
- More precise similarity (comparing apples to apples)

---

## Prediction Using Role-Filtered Decisions

Following ARCHITECTURE.md (line 217-291):

```python
def predict_outcome(uploaded_case_id: str):
    """Predict case outcome based on similar cases."""

    # Step 1: Find similar cases (role-wise)
    similar_cases = find_similar_cases(uploaded_case_id)

    # Step 2: Get Decision role from each similar case
    outcomes = []

    for similar_case in similar_cases[:20]:
        # Get ONLY Decision sentences
        decisions = index.query(
            filter={
                "case_id": similar_case["case_id"],
                "role": {"$eq": "Decision"}
            },
            namespace="training_data",
            top_k=5,
            include_metadata=True
        )

        for decision in decisions.matches:
            text = decision.metadata.text.lower()

            # Parse outcome
            if any(word in text for word in ["allowed", "granted", "upheld"]):
                outcomes.append("Favorable")
            elif any(word in text for word in ["dismissed", "rejected"]):
                outcomes.append("Unfavorable")
            elif any(word in text for word in ["remanded", "partly"]):
                outcomes.append("Neutral")

    # Step 3: Calculate statistics
    total = len(outcomes)
    favorable = outcomes.count("Favorable")
    unfavorable = outcomes.count("Unfavorable")
    neutral = outcomes.count("Neutral")

    return {
        "prediction": max(set(outcomes), key=outcomes.count),
        "confidence": max(favorable, unfavorable, neutral) / total,
        "distribution": {
            "Favorable": f"{favorable}/{total} ({favorable/total*100:.1f}%)",
            "Unfavorable": f"{unfavorable}/{total} ({unfavorable/total*100:.1f}%)",
            "Neutral": f"{neutral}/{total} ({neutral/total*100:.1f}%)"
        }
    }
```

**Key:** We only look at Decision role to extract outcomes, not mixed content.

---

## Testing Role-Wise Retrieval

After running the upload script, test with these queries:

### Test 1: Facts Only

```python
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("nyaya-legal-rag")
model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)

query = "What are the facts of this case?"
query_emb = model.encode(query, prompt_name="Retrieval-query", normalize_embeddings=True)

results = index.query(
    vector=query_emb.tolist(),
    filter={"role": {"$eq": "Facts"}},
    namespace="training_data",
    top_k=5,
    include_metadata=True
)

for match in results.matches:
    print(f"Score: {match.score:.4f} | Role: {match.metadata.role}")
    print(f"Text: {match.metadata.text[:100]}...\n")
```

**Expected:** All results have `role: "Facts"`

---

### Test 2: Compare Different Roles

```python
roles = ["Facts", "Issue", "Reasoning", "Decision"]

for role in roles:
    results = index.query(
        vector=query_emb.tolist(),
        filter={"role": {"$eq": role}},
        namespace="training_data",
        top_k=3,
        include_metadata=True
    )

    print(f"\n{'='*60}")
    print(f"Role: {role}")
    print(f"{'='*60}")

    for match in results.matches:
        print(f"Case: {match.metadata.case_id} | Score: {match.score:.4f}")
        print(f"Text: {match.metadata.text[:80]}...\n")
```

**Expected:** Different results for each role, all matching the queried role.

---

## Summary

### ✅ What the Upload Script Enables:

1. **Role-based filtering** in queries
2. **Namespace separation** (training vs. user data)
3. **Role-wise similarity** comparisons
4. **Precise retrieval** (Facts vs. Reasoning vs. Decision)
5. **Outcome prediction** based on Decision-role analysis

### ✅ Matches ARCHITECTURE.md:

- ✅ Vector metadata schema (line 411-426)
- ✅ Namespace strategy (line 428-445)
- ✅ Role-wise similarity (line 600-698)
- ✅ Prediction logic (line 217-291)
- ✅ RAG with role filtering (line 295-352)

### 🎯 Key Differentiator:

**Standard RAG:** "What were the facts?" → Mixed content
**Nyaya RAG:** "What were the facts?" → **Only Facts**

This is the core innovation that makes Nyaya unique!
