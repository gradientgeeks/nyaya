# Intelligent Agent for Legal Document Analysis using RAG, Role Classifier, and Multi-turn Conversation Support

---

## 1. Introduction

Legal judgments are often **long, complex, and unstructured**, making it difficult for lawyers, judges, and researchers to quickly extract reasoning, arguments, or decisions.
Traditional keyword-based search retrieves entire documents, lacking **precision** at the rhetorical-role level (facts, arguments, reasoning, decision).

This project develops an **intelligent agent system** that integrates:

1. **Role Classifier** – Segments legal documents into rhetorical roles (*Facts, Issues, Arguments of Petitioner (AoP), Arguments of Respondent (AoR), Reasoning, Decision*).
2. **Retrieval-Augmented Generation (RAG)** – Retrieves **role-specific embeddings** and enhances Large Language Model (LLM) responses.
3. **Agent Orchestrator** – Directs user queries to the right retrievers (facts, reasoning, decision).
4. **Conversation Manager** – Maintains **multi-turn dialogue** for context-aware conversations.
5. **Predictive Extension** – Provides **probable judgments** for pending cases by analyzing similar precedents.

---

## 2. Objectives

* Build a **Role Classifier** for sentence-level segmentation of judgments.
* Implement **Role-Aware RAG** to improve legal Q\&A precision.
* Develop a **multi-turn conversational agent** with context memory.
* Enable **structured answers** (facts, arguments, reasoning, decision).
* Extend system to handle **pending cases** and **predict probable outcomes**.

---

## 3. System Architecture

### 3.1 High-Level Components

1. **Frontend (Client Layer)**

   * Chat interface (React/Angular).
   * Functions: document upload, structured summaries, multi-turn queries.

2. **Backend API (Application Layer)**

   * Framework: FastAPI / Django.
   * Modules:

     * **Document Processor** – Cleans & segments uploaded judgments.
     * **Role Classifier** – Labels sentences by rhetorical role.
     * **RAG Engine** – Retrieves role-aware segments.
     * **Conversation Manager** – Maintains context across turns.
     * **Agent Orchestrator** – Routes queries to tools (retrievers, predictor).
     * **Prediction Module** – Provides outcome probabilities for pending cases.

3. **Data Layer**

   * **Relational DB (Postgres/MySQL)** → Metadata (case name, court, citations).
   * **Vector Store (FAISS / Pinecone)** → Embeddings + role labels + conversation memory.

4. **Model Layer (AI Services)**

   * **Role Classifier** → InLegalBERT / BiLSTM-CRF.
   * **Embeddings** → Sentence Transformers, OpenAI embeddings.
   * **LLM** → GPT-4 / LLaMA-2 (fine-tuned for legal domain).

5. **Deployment & Infra**

   * **Dockerized microservices**.
   * Orchestration: Kubernetes.
   * Monitoring: ELK Stack, Prometheus.

---

### 3.2 Textual Architecture Diagram

```
 ┌──────────────────────┐
 │      Frontend        │
 │ (Chat UI + Upload)   │
 └─────────▲────────────┘
           │
 ┌─────────┴────────────┐
 │   Backend Server     │
 │ - Doc Processor      │
 │ - Role Classifier    │
 │ - RAG Engine         │
 │ - Conv. Manager      │
 │ - Agent Orchestrator │
 │ - Prediction Module  │
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │     Data Layer       │
 │ - Relational DB      │
 │ - Vector Store (FAISS│
 │   with role metadata)│
 └─────────┬────────────┘
           │
 ┌─────────▼────────────┐
 │   Model Layer (AI)   │
 │ - Classifier         │
 │ - Embeddings         │
 │ - LLM (GPT/LLaMA-2)  │
 └──────────────────────┘
```

---

## 4. Workflow

### Case Already Judged

* User query → Role-specific retrieval → LLM generates structured answer.

### Pending Case (No Judgment Yet)

* Extracts available roles (*Facts, AoP, AoR*).
* System clarifies missing roles (*Reasoning/Decision not available*).

### Predictive Extension

* For pending cases, retrieve **similar past cases**.
* Compute **probable outcomes** + supporting precedents.

---

## 5. Role Classifier

* Input: Sentences from judgments.
* Output: Labels ∈ {Facts, Issues, AoP, AoR, Reasoning, Decision, None}.
* Models: InLegalBERT baseline, BiLSTM-CRF, Role-aware Transformer.
* Evaluation: Macro-F1, per-class precision & recall.

---

## 6. RAG Integration

* Store role-tagged embeddings:

```json
{
  "sentence": "The writ petition is dismissed.",
  "role": "Decision",
  "embedding": [...]
}
```

* Retrieval happens **per role**.
* Multi-role queries → **parallel retrieval**, aggregated into structured JSON → formatted LLM response.

---

## 7. Conversation Manager

* **Short-term memory** → last N turns.
* **Long-term memory** → vector DB of past interactions.
* Maintains context for:

  * Pronouns (*“Why did the court dismiss it?”*).
  * Multi-turn refinements (*“Now show petitioner’s arguments too”*).

---

## 8. Multi-role Query Handling

* Detect roles in query (facts, AoP, AoR, reasoning, decision).
* Retrieve each role’s segments in parallel.
* Aggregate into structured summary.
* Output formatted with role headings.

---

## 9. Pending / Non-Judgment Case Handling

* If **no reasoning/decision**:

  * System explicitly states: *“Case still pending, no final judgment available.”*
* If user asks for complete summary:

  * Present available roles.
  * Mark unavailable roles.

---

## 10. Probable Judgment Prediction (Extension)

1. Retrieve **similar past cases** (via facts/issues embeddings).
2. Extract outcomes of those precedents.
3. Compute **outcome probabilities**.
4. Present with precedents and disclaimer.

**Example Output**:

```
Probable Judgment:
- 70%: Petition dismissed
- 25%: Petition partly allowed
- 5%: Petition allowed

Based on precedents:
- Case A vs B (2018): Dismissed
- Case C vs D (2021): Partly allowed
```

---

## 11. User Upload Workflow

* **Judged case** → Full role-aware RAG summary.
* **Pending case** → Role extraction + “No judgment available” message.
* **Pending case (with prediction enabled)** → Outcome probabilities + precedents.

---

## 12. Deployment

* **Dockerized services**: Backend, classifier, DB, LLM.
* **Kubernetes**: For load balancing and scaling.
* **Monitoring**: API latency, classifier accuracy, user analytics.

---

## 13. Benefits

* **Precision**: Retrieves specific roles, not whole documents.
* **Explainability**: Shows retrieved text chunks and labels.
* **Interactive**: Supports multi-turn, conversational queries.
* **Scalable**: Modular microservice design.
* **Extensible**: Predictive module for ongoing cases.

---

## 14. Future Enhancements

* Cross-lingual support (regional Indian judgments).
* RhetoricLLaMA fine-tuning for legal discourse.
* Citation graph linking across multiple cases.
* Interactive visual summaries of arguments vs. outcomes.

---

## Citation

```
@inproceedings{nigam-etal-2025-nyayaanumana,
    title = "{NYAYAANUMANA} and {INLEGALLLAMA}: The Largest {I}ndian Legal Judgment Prediction Dataset and Specialized Language Model for Enhanced Decision Analysis",
    author = "Nigam, Shubham Kumar  and
      Balaramamahanthi, Deepak Patnaik  and
      Mishra, Shivam  and
      Shallum, Noel  and
      Ghosh, Kripabandhu  and
      Bhattacharya, Arnab",
    editor = "Rambow, Owen  and
      Wanner, Leo  and
      Apidianaki, Marianna  and
      Al-Khalifa, Hend  and
      Eugenio, Barbara Di  and
      Schockaert, Steven",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.738/",
    pages = "11135--11160",
    abstract = "The integration of artificial intelligence (AI) in legal judgment prediction (LJP) has the potential to transform the legal landscape, particularly in jurisdictions like India, where a significant backlog of cases burdens the legal system. This paper introduces NyayaAnumana, the largest and most diverse corpus of Indian legal cases compiled for LJP, encompassing a total of 7,02,945 preprocessed cases. NyayaAnumana, which combines the words {\textquotedblleft}Nyaya{\textquotedblright} and {\textquotedblleft}Anumana{\textquotedblright} that means {\textquotedblleft}judgment{\textquotedblright} and {\textquotedblleft}inference{\textquotedblright} respectively for most major Indian languages, includes a wide range of cases from the Supreme Court, High Courts, Tribunal Courts, District Courts, and Daily Orders and, thus, provides unparalleled diversity and coverage. Our dataset surpasses existing datasets like PredEx and ILDC, offering a comprehensive foundation for advanced AI research in the legal domain. In addition to the dataset, we present INLegalLlama, a domain-specific generative large language model (LLM) tailored to the intricacies of the Indian legal system. It is developed through a two-phase training approach over a base LLaMa model. First, Indian legal documents are injected using continual pretraining. Second, task-specific supervised finetuning is done. This method allows the model to achieve a deeper understanding of legal contexts. Our experiments demonstrate that incorporating diverse court data significantly boosts model accuracy, achieving approximately 90{\%} F1-score in prediction tasks. INLegalLlama not only improves prediction accuracy but also offers comprehensible explanations, addressing the need for explainability in AI-assisted legal decisions."



}
  @inproceedings{paul-2022-pretraining,
  url = {https://arxiv.org/abs/2209.06049},
  author = {Paul, Shounak and Mandal, Arpan and Goyal, Pawan and Ghosh, Saptarshi},
  title = {Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law},
  booktitle = {Proceedings of 19th International Conference on Artificial Intelligence and Law - ICAIL 2023}
  year = {2023},
}


