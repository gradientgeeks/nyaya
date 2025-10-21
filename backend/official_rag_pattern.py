"""
Official Gemma Cookbook RAG Pattern with EmbeddingGemma

This demonstrates the EXACT pattern from Google's Gemma Cookbook:
https://github.com/google-gemini/gemma-cookbook/blob/main/Gemma/[Gemma_3]RAG_with_EmbeddingGemma.ipynb

Key patterns from the official example:
1. Load EmbeddingGemma: google/embeddinggemma-300M
2. Use task-specific prompts for asymmetric encoding:
   - Documents: prompt_name="Retrieval-document" or prompt="title: ... | text: ..."
   - Queries: prompt_name="Retrieval-query"
3. Calculate similarity using model.similarity()
4. Generate answers with Gemma 3 LLM using retrieved context

This is the canonical way to implement RAG with EmbeddingGemma.
"""

import torch
from sentence_transformers import SentenceTransformer

# ========================================
# 1. LOAD EMBEDDINGGEMMA MODEL
# ========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id).to(device=device)

print(f"Device: {model.device}")
print(f"Total parameters: {sum([p.numel() for _, p in model.named_parameters()]):,}")

# ========================================
# 2. SAMPLE KNOWLEDGE BASE
# ========================================

# Following the cookbook's corporate knowledge base pattern
legal_knowledge_base = [
    {
        "category": "Constitutional Law",
        "documents": [
            {
                "title": "Right to Privacy - Navtej Singh Johar",
                "content": "The Court held that the right to privacy is an intrinsic part of the right to life and personal liberty under Article 21."
            },
            {
                "title": "Section 377 Ruling",
                "content": "Section 377 of the Indian Penal Code, insofar as it criminalizes consensual sexual conduct between adults in private, is declared unconstitutional."
            }
        ]
    },
    {
        "category": "Criminal Law",
        "documents": [
            {
                "title": "Burden of Proof",
                "content": "In criminal cases, the burden of proof lies with the prosecution to establish guilt beyond reasonable doubt."
            }
        ]
    }
]

# ========================================
# 3. ENCODE DOCUMENTS WITH PROPER PROMPTS
# ========================================

# Extract all documents
all_documents = []
for category in legal_knowledge_base:
    for doc in category["documents"]:
        all_documents.append(doc)

# Method 1: Using prompt_name (simpler)
print("\n📊 Encoding documents with 'Retrieval-document' prompt...")
doc_texts = [doc["content"] for doc in all_documents]
doc_embeddings_simple = model.encode(
    doc_texts,
    prompt_name="Retrieval-document",
    normalize_embeddings=True
)

# Method 2: Using prompt with title (better quality)
print("📊 Encoding documents with title prefix...")
doc_texts_with_title = [
    f"title: {doc['title']} | text: {doc['content']}"
    for doc in all_documents
]
doc_embeddings_with_title = model.encode(
    doc_texts_with_title,
    normalize_embeddings=True  # Title prompt is already included in text
)

print(f"✅ Encoded {len(all_documents)} documents")
print(f"   Embedding shape: {doc_embeddings_simple.shape}")

# ========================================
# 4. QUERY WITH RETRIEVAL-QUERY PROMPT
# ========================================

user_question = "What was the ruling on Section 377?"

print(f"\n🔍 Query: {user_question}")

# Encode query with Retrieval-query prompt (asymmetric encoding)
query_embedding = model.encode(
    user_question,
    prompt_name="Retrieval-query",
    normalize_embeddings=True
)

print(f"✅ Query encoded with shape: {query_embedding.shape}")

# ========================================
# 5. FIND BEST MATCHING DOCUMENT
# ========================================

# Calculate similarities (following cookbook pattern)
similarities = model.similarity(query_embedding, doc_embeddings_with_title)

print(f"\n📈 Similarity scores:")
for i, (doc, score) in enumerate(zip(all_documents, similarities[0])):
    print(f"   {i+1}. {doc['title']}: {score:.4f}")

# Get best match
best_idx = similarities.argmax().item()
best_document = all_documents[best_idx]
best_score = similarities[0][best_idx].item()

print(f"\n🎯 Best match: {best_document['title']} (score: {best_score:.4f})")

# ========================================
# 6. GENERATE ANSWER WITH GEMMA 3 (Optional)
# ========================================

# If you have Gemma 3 LLM loaded, use this pattern:
"""
from transformers import pipeline

llm_pipeline = pipeline(
    task="text-generation",
    model="google/gemma-3-4b-it",
    device_map="auto",
    dtype="auto"
)

qa_prompt_template = \"\"\"Answer the user's question using ONLY the information from the context below.

---
CONTEXT:
{context}
---
QUESTION:
{question}
\"\"\"

if best_document:
    context = best_document["content"]
    prompt = qa_prompt_template.format(context=context, question=user_question)
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        },
    ]
    
    answer = llm_pipeline(messages, max_new_tokens=256, disable_compile=True)[0]["generated_text"][1]["content"]
    print(f"\n💡 Answer: {answer}")
"""

print("\n" + "="*80)
print("✅ RAG PATTERN DEMONSTRATION COMPLETE")
print("="*80)
print("\n📚 Key Takeaways:")
print("   1. Use google/embeddinggemma-300M model")
print("   2. Documents: prompt_name='Retrieval-document' or 'title: X | text: Y'")
print("   3. Queries: prompt_name='Retrieval-query'")
print("   4. Normalize embeddings for cosine similarity")
print("   5. Use model.similarity() to find best matches")
print("   6. Pass retrieved context to Gemma 3 LLM for answer generation")
