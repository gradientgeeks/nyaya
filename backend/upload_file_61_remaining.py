"""
Upload Remaining Sentences from file_61.txt

This script uploads only the remaining sentences from file_61.txt:
- Starts from sentence 527 (skips first 526)
- Uploads remaining 54 sentences
- Quick targeted upload for completing this specific file

Usage:
    python upload_file_61_remaining.py

Current Status:
    - Total uploaded: 25,000 vectors
    - file_61.txt: 526/580 sentences uploaded
    - Remaining: 54 sentences (527-580)
"""

import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")

# Configuration - file_61.txt only
DATA_DIR = Path(__file__).parent / "pinecone_data_insert"
TARGET_FILE = "file_61.txt"
SKIP_SENTENCES = 526  # Skip first 526 already uploaded
BATCH_SIZE = 100  # Pinecone upsert batch size
EMBEDDING_DIM = 384


def parse_legal_file(file_path: Path, skip_lines: int = 0) -> List[Dict]:
    """
    Parse a tab-separated legal case file, skipping first N lines.

    Format: <sentence text><TAB><role label>

    Args:
        file_path: Path to the .txt file
        skip_lines: Number of lines to skip from start

    Returns:
        List of dicts with 'text' and 'role' keys
    """
    sentences = []
    skipped_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines (don't count towards skip)
                if not line:
                    continue

                # Skip already processed lines
                if skipped_count < skip_lines:
                    skipped_count += 1
                    continue

                # Split by tab
                parts = line.split('\t')

                if len(parts) != 2:
                    print(f"   ⚠️  Warning: Line {line_num} has {len(parts)} parts (expected 2), skipping")
                    continue

                text, role = parts

                # Validate role
                valid_roles = [
                    "Facts", "Issue", "Arguments of Petitioner",
                    "Arguments of Respondent", "Reasoning", "Decision", "None"
                ]

                if role not in valid_roles:
                    print(f"   ⚠️  Warning: Invalid role '{role}' at line {line_num}, keeping as-is")

                sentences.append({
                    "text": text,
                    "role": role,
                    "file": file_path.stem,  # "file_61"
                    "line_num": line_num
                })

    except Exception as e:
        print(f"   ❌ Error reading {file_path}: {e}")
        return []

    return sentences


def initialize_pinecone():
    """Initialize Pinecone and return index."""
    print(f"🔌 Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    # Check current status
    stats = index.describe_index_stats()
    training_data_count = stats.namespaces.get('training_data', {}).get('vector_count', 0)

    print(f"✅ Connected! Current vectors in 'training_data' namespace: {training_data_count:,}")

    return index


def load_embedding_model():
    """Load EmbeddingGemma model with 384 dimensions."""
    print("🤖 Loading EmbeddingGemma model (google/embeddinggemma-300M)...")
    print("   Truncating to 384 dimensions using MRL...")

    model = SentenceTransformer(
        "google/embeddinggemma-300M",
        truncate_dim=EMBEDDING_DIM,
        trust_remote_code=True
    )

    print(f"✅ Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def create_embeddings_batch(model, sentences: List[Dict]) -> List:
    """Create embeddings for a batch of sentences."""
    # Format: "title: <case_id> | text: <content>"
    doc_texts = [
        f"title: {s['file']} | text: {s['text']}"
        for s in sentences
    ]

    # Create embeddings with document prompt
    embeddings = model.encode(
        doc_texts,
        prompt_name="Retrieval-document",
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings


def upload_to_pinecone(index, model, sentences: List[Dict], start_vector_num: int):
    """
    Upload sentences to Pinecone.

    Args:
        index: Pinecone index
        model: SentenceTransformer model
        sentences: List of sentences to upload
        start_vector_num: Starting vector number for display
    """
    total_sentences = len(sentences)
    print(f"\n📊 Sentences to upload: {total_sentences}")
    print(f"📍 Vector range: #{start_vector_num:,} - #{start_vector_num + total_sentences - 1:,}")

    # Create embeddings for all sentences (small batch, no need to split)
    print(f"\n🧠 Generating embeddings for {total_sentences} sentences...")
    embeddings = create_embeddings_batch(model, sentences)

    # Prepare vectors for Pinecone
    vectors = []
    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        vector_id = f"{sentence['file']}_sent_{sentence['line_num']}"

        # Metadata following ARCHITECTURE.md schema
        metadata = {
            "text": sentence["text"][:1000],  # Pinecone metadata limit
            "role": sentence["role"],
            "case_id": sentence["file"],
            "sentence_index": sentence["line_num"],
            "user_uploaded": False,  # Training data
            "confidence": 1.0,
            "court": "Indian Courts",
            "category": "Legal Training Data",
        }

        vectors.append({
            "id": vector_id,
            "values": embedding.tolist(),
            "metadata": metadata
        })

    # Upsert to Pinecone
    print(f"⬆️  Upserting {len(vectors)} vectors to 'training_data' namespace...")
    try:
        upsert_response = index.upsert(
            vectors=vectors,
            namespace="training_data"
        )
        print(f"✅ Upserted {upsert_response.upserted_count} vectors successfully!")
        return upsert_response.upserted_count
    except Exception as e:
        print(f"❌ Error upserting: {e}")
        return 0


def main():
    """Main execution flow."""
    print("=" * 80)
    print("🏛️  NYAYA: Upload Remaining Sentences from file_61.txt")
    print("=" * 80)
    print(f"📄 Target file: {TARGET_FILE}")
    print(f"📍 Skipping first {SKIP_SENTENCES} sentences (already uploaded)")
    print(f"🎯 Uploading sentences {SKIP_SENTENCES + 1} onwards")

    # Step 1: Check file exists
    file_path = DATA_DIR / TARGET_FILE

    if not file_path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return

    # Step 2: Parse file (skip first 526 sentences)
    print(f"\n{'='*80}")
    print("📖 PARSING FILE")
    print(f"{'='*80}")

    sentences = parse_legal_file(file_path, skip_lines=SKIP_SENTENCES)

    if not sentences:
        print("\n❌ No sentences to upload (all already processed or file empty)")
        return

    print(f"✅ Parsed {len(sentences)} remaining sentences")

    # Role distribution
    role_counts = {}
    for s in sentences:
        role = s['role']
        role_counts[role] = role_counts.get(role, 0) + 1

    print(f"\nRole distribution (remaining sentences):")
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(sentences)) * 100
        print(f"   {role:30s}: {count:3} ({percentage:5.1f}%)")

    # Step 3: Initialize Pinecone
    print(f"\n{'='*80}")
    print("🔌 INITIALIZING PINECONE")
    print(f"{'='*80}")

    index = initialize_pinecone()

    # Step 4: Load embedding model
    print(f"\n{'='*80}")
    print("🤖 LOADING EMBEDDING MODEL")
    print(f"{'='*80}")

    model = load_embedding_model()

    # Step 5: Upload to Pinecone
    print(f"\n{'='*80}")
    print("⬆️  UPLOADING TO PINECONE")
    print(f"{'='*80}")

    start_time = time.time()
    current_vector_count = 25000  # Current total in Pinecone
    uploaded_count = upload_to_pinecone(index, model, sentences, start_vector_num=current_vector_count + 1)
    elapsed_time = time.time() - start_time

    if uploaded_count == 0:
        print("\n❌ Upload failed!")
        return

    # Step 6: Verify upload
    print(f"\n{'='*80}")
    print("⏳ VERIFYING UPLOAD")
    print(f"{'='*80}")
    print("⏳ Waiting for Pinecone index to sync...")
    time.sleep(5)

    stats = index.describe_index_stats()
    training_data_count = stats.namespaces.get('training_data', {}).get('vector_count', 0)

    print(f"✅ Vectors in 'training_data' namespace: {training_data_count:,}")

    # Final statistics
    print(f"\n{'='*80}")
    print("✅ UPLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"File: {TARGET_FILE}")
    print(f"Sentences uploaded: {uploaded_count}")
    print(f"Upload time: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"Average speed: {uploaded_count/elapsed_time:.2f} sentences/second")
    print(f"Total vectors in Pinecone: {training_data_count:,}")
    print(f"\n🎉 file_61.txt is now COMPLETE!")

    # Test query
    print(f"\n{'='*80}")
    print("🔍 TESTING QUERY ON NEW DATA")
    print(f"{'='*80}")

    test_query_text = "What were the facts presented?"
    print(f"Query: '{test_query_text}'")

    query_embedding = model.encode(
        test_query_text,
        prompt_name="Retrieval-query",
        normalize_embeddings=True
    )

    try:
        results = index.query(
            vector=query_embedding.tolist(),
            top_k=3,
            include_metadata=True,
            filter={"case_id": {"$eq": "file_61"}},  # Only from file_61
            namespace="training_data"
        )

        print(f"\n📊 Retrieved {len(results['matches'])} results from file_61:")
        for i, match in enumerate(results['matches'], 1):
            print(f"\n{i}. Score: {match['score']:.4f}")
            print(f"   Case: {match['metadata']['case_id']}")
            print(f"   Role: {match['metadata']['role']}")
            print(f"   Sentence: {match['metadata']['sentence_index']}")
            print(f"   Text: {match['metadata']['text'][:120]}...")
    except Exception as e:
        print(f"❌ Error querying: {e}")

    print(f"\n{'='*80}")
    print("💡 NEXT STEPS")
    print(f"{'='*80}")
    print("✅ file_61.txt complete!")
    print("📂 Next file to process: file_62.txt")
    print(f"📊 Progress: {training_data_count:,} / 51,505 vectors")
    print(f"⏭️  Remaining: {51505 - training_data_count:,} vectors")
    print("\n🔄 To continue with remaining files (62-100), run:")
    print("   python resume_upload_from_checkpoint.py")
    print("   (Update START_FILE_NUM = 62, START_SENTENCE = 0)")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
