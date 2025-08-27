## **1. Input Stage (Document Upload)**

* The raw PDF / text is uploaded.
* **Document Processor** extracts text, cleans it, and splits it into sentences/clauses.
* Metadata extracted:

  * Case: *TCS vs SK Wheels Pvt Ltd*
  * Court: *Supreme Court of India*
  * Judge: *Dr. D.Y. Chandrachud*
  * Date: *2020*

---

## **2. Role Classification Stage (InLegalBERT / BiLSTM-CRF)**

* Each sentence is classified into **rhetorical roles**:

  * Example:

    * *“This appeal arises from a judgment dated 24 June 2020…”* → **Facts**
    * *“Based on the appeal, two issues have arisen for consideration…”* → **Issue**
    * *“Ms Fereshte D Sethna, learned counsel… has made the following submissions”* → **Arguments of Petitioner (AoP)**
    * *“Ms Udhita Singh, learned counsel… urged that”* → **Arguments of Respondent (AoR)**
    * *“Thus, we are of the view that the NCLT does not have any residuary jurisdiction…”* → **Reasoning**
    * *“We accordingly set aside the judgment of the NCLAT…”* → **Decision**

* Output is a **role-tagged sequence**, e.g.:

  ```
  Sentence: "This appeal arises from a judgment dated 24 June 2020..."
  Role: Facts
  ```

---

## **3. Embedding + Storage (Vector DB)**

* Each sentence is **embedded** (using Sentence Transformers).
* Stored in **FAISS / Pinecone**, along with **role labels**.

Schema:

```json
{
  "sentence": "Based on the appeal, two issues have arisen for consideration...",
  "role": "Issue",
  "embedding": [0.12, -0.45, ...],
  "metadata": {"case": "TCS v SK Wheels", "court": "Supreme Court"}
}
```

---

## **4. Query Handling (User Interaction)**

Now, suppose a lawyer asks:

> “What were the issues considered by the Court in this case?”

### **a. Query Router**

* Detects: **ROLE\_SPECIFIC\_QUERY → Issues**
* Passes control to **RAG Engine** with role filter = “Issue”.

### **b. Role-Aware RAG Retrieval**

* Vector search retrieves only **Issue-labeled embeddings**.
* Example retrieved chunks:

  * *“Based on the appeal, two issues have arisen…”*
  * *“(i) Whether the NCLT can exercise residuary jurisdiction…”*
  * *“(ii) Whether in such jurisdiction it can impose stay on termination…”*

### **c. LLM Answer Generation**

* LLM (GPT-4 / InLegalLLaMA) is prompted:

  * Context = retrieved Issue chunks
  * Query = user’s question
* Answer generated:

  > *The Court considered two issues: (i) whether NCLT has residuary jurisdiction under Section 60(5)(c) of IBC to adjudicate contractual disputes, and (ii) whether it can impose an ad-interim stay on termination of the Facilities Agreement.*

---

## **5. Conversation Manager**

* If user then asks:

  > “What was the Court’s reasoning for dismissing NCLT jurisdiction?”

* The **short-term memory** recalls last query = “Issues”.

* The **long-term memory** recalls case metadata.

* Orchestrator routes query → **Reasoning role filter**.

* Retrieved reasoning chunks (e.g., para 27, “there is nothing to indicate termination was due to insolvency…”).

* LLM summarizes reasoning.

---

## **6. Prediction Module (if pending case)**

* If this were a **pending appeal**, prediction module would:

  * Retrieve similar precedents (e.g., *Gujarat Urja v Amit Gupta*).
  * Analyze outcomes.
  * Return **probability distribution of likely outcome**.

Here, since case is already decided, prediction module is **skipped**.

---

# 🔹 Final Flow Diagram (for this case)

1. **Document Upload** → Preprocessing & Metadata Extraction
2. **Role Classifier (InLegalBERT)** → Assigns Facts, Issue, AoP, AoR, Reasoning, Decision
3. **Embeddings + Vector DB** → Store role-aware chunks
4. **Query Router + Orchestrator** → Detects query type (facts/issues/arguments)
5. **Role-Aware RAG Retrieval** → Get relevant role chunks
6. **LLM Answer Generation** → Grounded, role-specific response
7. **Conversation Manager** → Maintains multi-turn dialogue context
8. **Prediction Module** (only if case undecided)

---

✅ In summary:
For this Chandrachud J. judgment, your system would:

* Split into **roles** (Facts → Issues → AoP/AoR → Reasoning → Decision).
* Enable precise **role-aware retrieval**.
* Support **multi-turn legal queries** with conversational memory.
* Provide **explainable answers** (retrieved snippets shown to user).

---
