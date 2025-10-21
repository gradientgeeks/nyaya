"""
Sample script to test Pinecone integration with EmbeddingGemma (384-dim).

This demonstrates:
1. Loading Google's EmbeddingGemma model via sentence-transformers
2. Creating 384-dimensional embeddings for legal text using MRL (Matryoshka Representation Learning)
3. Storing in Pinecone with role-aware metadata
4. Using RAG-specific prompts (Retrieval-query vs Retrieval-document)

Based on: https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers

PREREQUISITES:
1. Accept EmbeddingGemma license at: https://huggingface.co/google/embeddinggemma-300M
2. Login to Hugging Face Hub (first time only):
   >>> from huggingface_hub import login
   >>> login()
   
   Or set HF_TOKEN environment variable:
   >>> export HF_TOKEN=your_hf_token_here
"""

import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Sample legal text with rhetorical roles
SAMPLE_LEGAL_TEXTS = [
    {
        "text": "The petitioner filed a writ petition under Article 32 of the Constitution challenging the constitutional validity of Section 377 of the Indian Penal Code.",
        "role": "Facts",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "background"
    },
    {
        "text": "The principal issue for consideration is whether Section 377 of the Indian Penal Code violates the fundamental rights guaranteed under Articles 14, 15, 19, and 21 of the Constitution.",
        "role": "Issue",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "legal_questions"
    },
    {
        "text": "The petitioner contends that Section 377 criminalizes consensual sexual acts between adults in private and thus infringes upon the right to privacy and dignity.",
        "role": "Arguments of Petitioner",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "petitioner_arguments"
    },
    {
        "text": "The respondent argued that Section 377 is necessary to protect public morality and maintain social order, as recognized by established traditions.",
        "role": "Arguments of Respondent",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "respondent_arguments"
    },
    {
        "text": "The Court held that the right to privacy is an intrinsic part of the right to life and personal liberty under Article 21, and consensual sexual acts between adults in private are protected by this fundamental right.",
        "role": "Reasoning",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "court_analysis"
    },
    {
        "text": "Section 377 of the Indian Penal Code, insofar as it criminalizes consensual sexual conduct between adults in private, is declared unconstitutional.",
        "role": "Decision",
        "case_id": "navtej_singh_johar_v_union_of_india",
        "section": "final_judgment"
    },
]


def initialize_pinecone():
    """Initialize Pinecone client and ensure index exists."""
    print(f"🔌 Initializing Pinecone (Region: {PINECONE_ENVIRONMENT})...")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists, create if not
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"📊 Creating index '{PINECONE_INDEX_NAME}' with 384 dimensions...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # EmbeddingGemma dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully!")
    else:
        print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")
    
    return pc.Index(PINECONE_INDEX_NAME)


def load_embedding_model():
    """
    Load Google's EmbeddingGemma model with 384 dimensions.
    
    Using the official google/embeddinggemma-300M model with Matryoshka
    Representation Learning (MRL) to truncate to 384 dimensions.
    
    Reference: https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
    """
    print("🤖 Loading EmbeddingGemma model (google/embeddinggemma-300M)...")
    print("   Truncating to 384 dimensions using MRL...")
    
    # Load the official EmbeddingGemma model
    # The full model outputs 768-dim embeddings, but MRL allows us to truncate to 384
    # for faster processing with minimal quality loss
    model = SentenceTransformer(
        "google/embeddinggemma-300M",
        truncate_dim=384,  # Use MRL to truncate to 384 dimensions
        trust_remote_code=True
    )
    
    print(f"✅ Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print(f"   Total parameters: {sum([p.numel() for _, p in model.named_parameters()]):,}")
    return model


def create_embeddings_and_upsert(model, index):
    """Create embeddings for sample legal texts and upsert to Pinecone."""
    print("\n🔄 Processing sample legal texts...")
    
    # Extract text for embedding with title prefix (improves RAG quality)
    # Format: "title: <case_id> | text: <content>"
    doc_texts = [
        f"title: {item['case_id']} | text: {item['text']}"
        for item in SAMPLE_LEGAL_TEXTS
    ]
    
    # Create embeddings (batch processing)
    # Using the "Retrieval-document" prompt for RAG use case
    print(f"🧠 Generating embeddings for {len(doc_texts)} sentences...")
    print("   Using prompt: 'Retrieval-document' (optimized for RAG)")
    embeddings = model.encode(
        doc_texts,
        prompt_name="Retrieval-document",  # Use RAG-optimized prompt
        normalize_embeddings=True,  # Normalize for cosine similarity
        show_progress_bar=True
    )
    
    print(f"📐 Embeddings shape: {embeddings.shape}")
    print(f"   - Number of texts: {embeddings.shape[0]}")
    print(f"   - Embedding dimension: {embeddings.shape[1]}")
    
    # Prepare vectors for Pinecone upsert
    vectors = []
    for i, (text_data, embedding) in enumerate(zip(SAMPLE_LEGAL_TEXTS, embeddings)):
        vector_id = f"{text_data['case_id']}_chunk_{i}"
        
        metadata = {
            "text": text_data["text"],
            "role": text_data["role"],
            "case_id": text_data["case_id"],
            "section": text_data["section"],
            "user_uploaded": False,  # System sample data
        }
        
        vectors.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": metadata
        })
    
    # Upsert to Pinecone
    print(f"\n⬆️  Upserting {len(vectors)} vectors to Pinecone...")
    upsert_response = index.upsert(vectors=vectors)
    print(f"✅ Upserted {upsert_response.upserted_count} vectors successfully!")
    
    # CRITICAL: Wait for serverless index to sync
    print("\n⏳ Waiting for Pinecone index to sync (this can take 10-30 seconds for serverless)...")
    time.sleep(5)  # Initial wait
    
    # Check index stats with retry
    max_retries = 6
    for attempt in range(max_retries):
        stats = index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"✅ Index synced! Total vectors: {stats.total_vector_count}")
            break
        else:
            if attempt < max_retries - 1:
                print(f"   ⏳ Attempt {attempt + 1}/{max_retries}: Still syncing... (waiting 5s)")
                time.sleep(5)
            else:
                print(f"   ⚠️  Index still showing 0 vectors after {max_retries * 5}s. Continuing anyway...")
    
    return vectors


def test_query(model, index):
    """Test querying the index with a sample question."""
    print("\n" + "=" * 80)
    print("🔍 TESTING QUERIES")
    print("=" * 80)
    
    # Query 1: Facts
    print("\n📋 Query 1: 'What are the main facts of this case?'")
    query_text = "What are the main facts of this case?"
    
    # Use "Retrieval-query" prompt for query embeddings in RAG
    query_embedding = model.encode(
        query_text,
        prompt_name="Retrieval-query",  # Query-specific prompt
        normalize_embeddings=True
    )
    
    # Query with role filter for Facts
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_metadata=True,
            filter={"role": {"$eq": "Facts"}}
        )
        
        print(f"\n📊 Retrieved {len(results['matches'])} results (filtered by role='Facts'):")
        if results['matches']:
            for i, match in enumerate(results['matches'], 1):
                print(f"\n{i}. Score: {match['score']:.4f}")
                print(f"   Role: {match['metadata']['role']}")
                print(f"   Text: {match['metadata']['text'][:100]}...")
        else:
            print("   ⚠️  No matches found. Index might still be syncing.")
    except Exception as e:
        print(f"   ❌ Error querying: {e}")
    
    # Query 2: Reasoning
    print("\n� Query 2: 'What was the court's reasoning?'")
    query_text = "What was the court's reasoning?"
    query_embedding = model.encode(
        query_text,
        prompt_name="Retrieval-query",
        normalize_embeddings=True
    )
    
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=2,
            include_metadata=True,
            filter={"role": {"$eq": "Reasoning"}}
        )
        
        print(f"\n📊 Retrieved {len(results['matches'])} results (filtered by role='Reasoning'):")
        if results['matches']:
            for i, match in enumerate(results['matches'], 1):
                print(f"\n{i}. Score: {match['score']:.4f}")
                print(f"   Role: {match['metadata']['role']}")
                print(f"   Text: {match['metadata']['text'][:150]}...")
        else:
            print("   ⚠️  No matches found. Index might still be syncing.")
    except Exception as e:
        print(f"   ❌ Error querying: {e}")
    
    # Query 3: No filter (get all)
    print("\n📋 Query 3: 'What is this case about?' (no role filter)")
    query_text = "What is this case about?"
    query_embedding = model.encode(
        query_text,
        prompt_name="Retrieval-query",
        normalize_embeddings=True
    )
    
    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_metadata=True
            # No filter - retrieve all roles
        )
        
        print(f"\n📊 Retrieved {len(results['matches'])} results (all roles):")
        if results['matches']:
            for i, match in enumerate(results['matches'], 1):
                print(f"\n{i}. Score: {match['score']:.4f}")
                print(f"   Role: {match['metadata']['role']}")
                print(f"   Text: {match['metadata']['text'][:100]}...")
        else:
            print("   ⚠️  No matches found. Index might still be syncing.")
    except Exception as e:
        print(f"   ❌ Error querying: {e}")


def main():
    """Main execution flow."""
    print("=" * 80)
    print("🏛️  NYAYA: Legal RAG with EmbeddingGemma (384-dim) + Pinecone")
    print("=" * 80)
    
    # Step 1: Initialize Pinecone
    index = initialize_pinecone()
    
    # Step 2: Load embedding model
    model = load_embedding_model()
    
    # Step 3: Create embeddings and upsert to Pinecone
    vectors = create_embeddings_and_upsert(model, index)
    
    # Step 4: Test queries
    test_query(model, index)
    
    # Step 5: Display stats
    print("\n" + "=" * 80)
    print("📈 FINAL INDEX STATISTICS:")
    stats = index.describe_index_stats()
    print(f"   Total vectors: {stats.total_vector_count}")
    print(f"   Index dimension: {stats.dimension}")
    print("=" * 80)
    
    if stats.total_vector_count == 0:
        print("\n⚠️  WARNING: Index still shows 0 vectors!")
        print("   This is common with Pinecone serverless indexes immediately after creation.")
        print("\n   Troubleshooting:")
        print("   1. Wait 1-2 minutes and run this script again")
        print("   2. Check Pinecone console: https://app.pinecone.io/")
        print("   3. The data IS uploaded, it just needs time to index")
        print("\n   Run this to check later:")
        print("   python -c \"from pinecone import Pinecone; import os; from dotenv import load_dotenv;\"")
        print("   python -c \"load_dotenv(); pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'));\"")
        print("   python -c \"print(pc.Index('nyaya-legal-rag').describe_index_stats())\"")
    else:
        print("\n✅ Sample data successfully added to Pinecone!")
    
    print("\n💡 Next steps:")
    print("   - View data in Pinecone console: https://app.pinecone.io/")
    print("   - Integrate with FastAPI endpoints")
    print("   - Implement role-aware RAG queries")
    print("   - Add InLegalBERT for automatic role classification")


if __name__ == "__main__":
    main()
