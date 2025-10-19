# **Legal Document Processing and RAG Pipeline**

## **1. Objective**

The goal is to design a system where a user uploads a legal document, which is then:

1. Preprocessed and split into sentences.
2. Classified into legal sections (Facts, Arguments of Petitioner, Arguments of Respondent, etc.) using a **machine learning classifier** (no LLM used for classification).
3. Displayed back to the user in **UI as grouped plain text**.
4. Converted to **high-quality vector embeddings** using a dedicated embedding model for semantic search.
5. Stored in **Pinecone vector database** to enable **retrieval-augmented generation (RAG)** with an LLM for answering queries like “Find similar cases”.

---

## **2. Architecture Overview**

```
           ┌───────────────────────┐
           │ 1. Document Upload    │
           │   (PDF / DOCX / Text) │
           └─────────┬─────────────┘
                     ↓
           ┌────────────────────────────┐
           │ 2. Text Extraction         │
           │   - Clean formatting       │
           │   - Remove headers/footer  │
           │   - OCR if required        │
           └─────────┬──────────────────┘
                     ↓
           ┌───────────────────────────┐
           │ 3. Sentence Splitting     │
           │   - Split document into   │
           │     individual sentences  │
           │   - Remove noise          │
           └─────────┬─────────────────┘
                     ↓
           ┌─────────────────────────────────────────────┐
           │ 4. Classification (No LLM)                  │
           │   - Convert sentences to small embeddings   │
           │     (MiniLM / TF-IDF) for ML classifier    │
           │   - Classifier (SVM / Logistic Regression) │
           │     predicts section labels: Facts, AoP,   │
           │     AoR, Judgment, etc.                    │
           └──────────┬──────────────────────────────────┘
                      ↓
          ┌────────────────────────────┐
          │ 5. UI Display              │
          │   - Group sentences by label│
          │   - Only **plain text** is │
          │     sent to UI             │
          └──────────┬─────────────────┘
                     ↓
          ┌─────────────────────────────────────────┐
          │ 6. Embeddings for RAG                  │
          │   - Convert each sentence to high-      │
          │     quality embedding (bge-m3 or       │
          │     EmbeddingGemma)                     │
          └──────────┬──────────────────────────────┘
                     ↓
          ┌──────────────────────────────────────────────┐
          │ 7. Store in Pinecone                          │
          │   - id: unique ID                              │
          │   - values: embedding vector                   │
          │   - metadata: {text, section, case_id, user} │
          └──────────────────────────────────────────────┘
                     ↓
          ┌───────────────────────────────────┐
          │ 8. LLM Query / RAG                │
          │   - User queries about cases      │
          │   - Retrieve similar sentences    │
          │   - LLM generates answers         │
          └───────────────────────────────────┘
```

---

## **3. Detailed Steps**

### **3.1 Document Upload and Text Extraction**

* Accept files: PDF, DOCX, or plain text.
* Clean text by removing headers, footers, or irrelevant content.
* If scanned PDFs are uploaded, perform OCR to extract text.
* Ensure **newline characters** and other formatting do not break sentences.

---

### **3.2 Sentence Splitting**

* Split document text into individual sentences.
* Libraries: `nltk.sent_tokenize`, `spacy`, or regex-based splitting.
* Optional: Remove sentences below a length threshold to filter out noise.

**Example:**

```python
sentences = [
    "The petitioner filed the case in 2021.",
    "The dispute arose over property ownership.",
    "Petitioner claims the contract is invalid.",
    "Petitioner seeks compensation.",
    "Respondent argues contract is legally binding.",
    "Respondent denies claims for damages."
]
```

---

### **3.3 Classification (No LLM)**

* Purpose: Assign each sentence to a **legal section**.
* **Input to classifier**: embeddings generated from each sentence using a lightweight model (MiniLM or TF-IDF).
* **Classifier**: SVM, Logistic Regression, or similar ML model.
* **Output**: Section label for each sentence.

**Labels example**:

* Facts
* Arguments of Petitioner (AoP)
* Arguments of Respondent (AoR)
* Judgement / Decision

**Classification embeddings are temporary**, only for ML input.

---

### **3.4 UI Display**

* After classification, **group sentences by label**.
* Only **plain text** is displayed to user; **no vectors** are exposed.
* **JSON structure for UI**:

```json
{
    "Facts": [
        "The petitioner filed the case in 2021.",
        "The dispute arose over property ownership."
    ],
    "AoP": [
        "Petitioner claims the contract is invalid.",
        "Petitioner seeks compensation."
    ],
    "AoR": [
        "Respondent argues contract is legally binding.",
        "Respondent denies claims for damages."
    ]
}
```

---

### **3.5 Embeddings for Pinecone (RAG)**

* Use a **high-quality embedding model** like:

  * **bge-m3** (open-source, high accuracy)
  * **EmbeddingGemma** (Google, multilingual, lightweight)
* Each sentence is converted to a vector for semantic search.
* Store **with metadata** in Pinecone.

**Example:**

```python
from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("BAAI/bge-m3")
rag_embeddings = embed_model.encode(sentences, normalize_embeddings=True)
```

---

### **3.6 Pinecone Storage**

* Store each sentence as a **vector** with associated metadata:

| Field    | Example                                                                                                               |
| -------- | --------------------------------------------------------------------------------------------------------------------- |
| id       | "case123_fact_1"                                                                                                      |
| values   | [0.012, -0.045, …]                                                                                                    |
| metadata | { "text": "The petitioner filed the case in 2021.", "section": "Facts", "case_id": "case123", "user_uploaded": true } |

* This allows **retrieval by section, case, or semantic similarity**.

---

### **3.7 Query / Retrieval with LLM (RAG)**

* User can ask questions like:

  * “Find similar cases regarding property disputes”
  * “What are the key facts in this case?”
* The system:

  1. Embeds the query using the **same RAG embedding model**.
  2. Retrieves **top-K similar vectors** from Pinecone (user + existing cases).
  3. Feeds retrieved sentences into **LLM** (Gemma / Llama / GPT) to generate human-readable answers.

---

## **4. Key Points / Best Practices**

1. **Separate embedding models**:

   * **Small embedding** for classification (MiniLM, TF-IDF)
   * **High-quality embedding** for RAG retrieval (bge-m3 / EmbeddingGemma)

2. **Vectors are never shown to users**; only plain text is displayed.

3. **Metadata in Pinecone** is crucial**:

   * `section`, `case_id`, `user_uploaded`, `text`
   * Enables **filtering** and hybrid retrieval (user docs + existing cases).

4. **Chunking strategy**:

   * Sentences or small paragraphs (~300–500 tokens)
   * Optional overlap to preserve context

5. **Scalability**:

   * Use Pinecone vector DB for efficient similarity search.
   * Embeddings can be updated or added as new user uploads arrive.

---

## **5. Example Python Flow Snippets**

### **5.1 Classification**

```python
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC

# Step 1: Embeddings for classifier
clf_model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_embeddings = clf_model.encode(sentences)

# Step 2: ML classifier
classifier = SVC()
labels = classifier.predict(sentence_embeddings)
```

### **5.2 Group Sentences for UI**

```python
from collections import defaultdict

grouped_sentences = defaultdict(list)
for sentence, label in zip(sentences, labels):
    grouped_sentences[label].append(sentence)

ui_output = dict(grouped_sentences)
```

### **5.3 Embedding for Pinecone**

```python
embed_model = SentenceTransformer("BAAI/bge-m3")
rag_embeddings = embed_model.encode(sentences, normalize_embeddings=True)
```

### **5.4 Store in Pinecone**

```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")
index = pinecone.Index("legal-cases")

for i, (sentence, embedding, label) in enumerate(zip(sentences, rag_embeddings, labels)):
    index.upsert([
        {
            "id": f"case123_{label}_{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": sentence,
                "section": label,
                "case_id": "case123",
                "user_uploaded": True
            }
        }
    ])
```

---

## **6. Summary**

1. **Upload → Text Extraction → Sentence Split**
2. **ML Classification using small embeddings → Labels**
3. **Display grouped plain text to user**
4. **Generate high-quality embeddings for RAG**
5. **Store embeddings + metadata in Pinecone**
6. **Use LLM for user queries on top of Pinecone retrieval**

> **Vectors are used internally only**. UI always displays plain text grouped by legal section. Classification and retrieval use **separate embeddings** optimized for their respective tasks.

---

This document captures the **complete architecture, flow, and best practices** for your legal RAG system.

---

If you want, I can also create a **visual diagram** of this pipeline with arrows showing **classification vectors vs RAG vectors** and **UI flow** for documentation purposes.

Do you want me to create that diagram?
