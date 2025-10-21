"""
Upload Legal Cases to Pinecone from Training Data Files

This script:
1. Reads tab-separated txt files from pinecone_data_insert/ directory
2. Each line format: <sentence text><TAB><role label>
3. Creates embeddings using EmbeddingGemma (384-dim)
4. Uploads to Pinecone with role-aware metadata
5. Processes first 100 files by default

Usage:
    python upload_legal_cases_to_pinecone.py

Prerequisites:
    - .env file with PINECONE_API_KEY and HF_TOKEN
    - EmbeddingGemma license accepted
    - Pinecone index 'nyaya-legal-rag' exists (384 dimensions)
"""

import os
import time
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Configuration
DATA_DIR = Path(__file__).parent / "pinecone_data_insert"
MAX_FILES = 100  # Process first 100 files
BATCH_SIZE = 100  # Pinecone upsert batch size
EMBEDDING_DIM = 384


def parse_legal_file(file_path: Path) -> List[Dict]:
    """
    Parse a tab-separated legal case file.

    Format: <sentence text><TAB><role label>

    Args:
        file_path: Path to the .txt file

    Returns:
        List of dicts with 'text' and 'role' keys
    """
    sentences = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Split by tab
                parts = line.split('\t')

                if len(parts) != 2:
                    print(f"   ⚠️  Warning: Line {line_num} in {file_path.name} has {len(parts)} parts (expected 2), skipping")
                    continue

                text, role = parts

                # Validate role (must be one of the 7 valid roles)
                valid_roles = [
                    "Facts", "Issue", "Arguments of Petitioner",
                    "Arguments of Respondent", "Reasoning", "Decision", "None"
                ]

                if role not in valid_roles:
                    print(f"   ⚠️  Warning: Invalid role '{role}' in {file_path.name}:{line_num}, keeping as-is")

                sentences.append({
                    "text": text,
                    "role": role,
                    "file": file_path.stem,  # e.g., "file_1"
                    "line_num": line_num
                })

    except Exception as e:
        print(f"   ❌ Error reading {file_path}: {e}")
        return []

    return sentences


def initialize_pinecone():
    """Initialize Pinecone and return index."""
    print(f"🔌 Initializing Pinecone (Region: {PINECONE_ENVIRONMENT})...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"📊 Creating index '{PINECONE_INDEX_NAME}' with {EMBEDDING_DIM} dimensions...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENVIRONMENT
            )
        )
        print(f"✅ Index '{PINECONE_INDEX_NAME}' created!")
    else:
        print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists")

    return pc.Index(PINECONE_INDEX_NAME)


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
    """
    Create embeddings for a batch of sentences.

    Args:
        model: SentenceTransformer model
        sentences: List of sentence dicts

    Returns:
        Embeddings array
    """
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


def upload_to_pinecone(index, model, all_sentences: List[Dict]):
    """
    Upload all sentences to Pinecone in batches with proper namespace and metadata.

    Args:
        index: Pinecone index
        model: SentenceTransformer model
        all_sentences: List of all sentences to upload
    """
    total_sentences = len(all_sentences)
    print(f"\n📊 Total sentences to upload: {total_sentences:,}")

    # Process in batches
    uploaded_count = 0

    for batch_start in range(0, total_sentences, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_sentences)
        batch = all_sentences[batch_start:batch_end]

        print(f"\n🔄 Processing batch {batch_start//BATCH_SIZE + 1} ({batch_start+1}-{batch_end} of {total_sentences})...")

        # Create embeddings for this batch
        print(f"   🧠 Generating embeddings...")
        embeddings = create_embeddings_batch(model, batch)

        # Prepare vectors for Pinecone
        vectors = []
        for i, (sentence, embedding) in enumerate(zip(batch, embeddings)):
            vector_id = f"{sentence['file']}_sent_{sentence['line_num']}"

            # Metadata following ARCHITECTURE.md schema (line 411-426)
            metadata = {
                "text": sentence["text"][:1000],  # Pinecone metadata limit (~40KB)
                "role": sentence["role"],  # One of 7 roles
                "case_id": sentence["file"],
                "sentence_index": sentence["line_num"],
                "user_uploaded": False,  # Training data (not user upload)
                "confidence": 1.0,  # Assumed high confidence for training data
                "court": "Indian Courts",  # Generic (we don't have this info)
                "category": "Legal Training Data",  # Generic category
            }

            vectors.append({
                "id": vector_id,
                "values": embedding.tolist(),
                "metadata": metadata
            })

        # Upsert to Pinecone with namespace for training data
        print(f"   ⬆️  Upserting {len(vectors)} vectors to 'training_data' namespace...")
        try:
            upsert_response = index.upsert(
                vectors=vectors,
                namespace="training_data"  # Separate from user uploads
            )
            uploaded_count += upsert_response.upserted_count
            print(f"   ✅ Upserted {upsert_response.upserted_count} vectors (Total: {uploaded_count:,})")
        except Exception as e:
            print(f"   ❌ Error upserting batch: {e}")
            continue

    return uploaded_count


def main():
    """Main execution flow."""
    print("=" * 80)
    print("🏛️  NYAYA: Upload Legal Cases to Pinecone")
    print("=" * 80)

    # Step 1: Check data directory
    if not DATA_DIR.exists():
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        return

    # Get first 100 .txt files (sorted numerically)
    all_files = sorted(DATA_DIR.glob("*.txt"), key=lambda f: int(f.stem.split('_')[1]))
    files_to_process = all_files[:MAX_FILES]

    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"📂 Total files available: {len(all_files)}")
    print(f"📋 Files to process: {len(files_to_process)}")

    if not files_to_process:
        print("❌ No .txt files found in directory")
        return

    # Step 2: Parse all files
    print(f"\n{'='*80}")
    print("📖 PARSING FILES")
    print(f"{'='*80}")

    all_sentences = []
    file_stats = {}

    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n[{i}/{len(files_to_process)}] Parsing {file_path.name}...")

        sentences = parse_legal_file(file_path)

        if sentences:
            all_sentences.extend(sentences)
            file_stats[file_path.name] = {
                "sentence_count": len(sentences),
                "role_distribution": {}
            }

            # Count roles
            for s in sentences:
                role = s['role']
                file_stats[file_path.name]['role_distribution'][role] = \
                    file_stats[file_path.name]['role_distribution'].get(role, 0) + 1

            print(f"   ✅ Parsed {len(sentences)} sentences")
            print(f"      Roles: {dict(file_stats[file_path.name]['role_distribution'])}")
        else:
            print(f"   ⚠️  No sentences parsed")

    if not all_sentences:
        print("\n❌ No sentences parsed from any files")
        return

    # Display statistics
    print(f"\n{'='*80}")
    print("📊 PARSING STATISTICS")
    print(f"{'='*80}")
    print(f"Total files parsed: {len(file_stats)}")
    print(f"Total sentences: {len(all_sentences):,}")

    # Overall role distribution
    overall_roles = {}
    for s in all_sentences:
        role = s['role']
        overall_roles[role] = overall_roles.get(role, 0) + 1

    print(f"\nOverall role distribution:")
    for role, count in sorted(overall_roles.items(), key=lambda x: -x[1]):
        percentage = (count / len(all_sentences)) * 100
        print(f"   {role:30s}: {count:6,} ({percentage:5.2f}%)")

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
    uploaded_count = upload_to_pinecone(index, model, all_sentences)
    elapsed_time = time.time() - start_time

    # Step 6: Wait for index sync
    print(f"\n{'='*80}")
    print("⏳ WAITING FOR INDEX SYNC")
    print(f"{'='*80}")
    print("⏳ Waiting for Pinecone serverless index to sync (10-30 seconds)...")
    time.sleep(5)

    max_retries = 6
    for attempt in range(max_retries):
        stats = index.describe_index_stats()

        # Check namespace-specific count
        training_data_count = stats.namespaces.get('training_data', {}).get('vector_count', 0)

        if training_data_count > 0:
            print(f"✅ Index synced! Vectors in 'training_data' namespace: {training_data_count:,}")
            print(f"   Total vectors across all namespaces: {stats.total_vector_count:,}")
            break
        else:
            if attempt < max_retries - 1:
                print(f"   ⏳ Attempt {attempt + 1}/{max_retries}: Still syncing... (waiting 5s)")
                time.sleep(5)
            else:
                print(f"   ⚠️  Index still showing 0 vectors after {max_retries * 5}s")

    # Final statistics
    print(f"\n{'='*80}")
    print("✅ UPLOAD COMPLETE")
    print(f"{'='*80}")
    print(f"Files processed: {len(file_stats)}")
    print(f"Sentences uploaded: {uploaded_count:,}")
    print(f"Upload time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {uploaded_count/elapsed_time:.2f} sentences/second")
    print(f"Pinecone index: {PINECONE_INDEX_NAME}")
    print(f"Index dimension: {EMBEDDING_DIM}")

    # Test query
    print(f"\n{'='*80}")
    print("🔍 TESTING QUERY")
    print(f"{'='*80}")

    test_query_text = "What are the facts of the case?"
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
            filter={"role": {"$eq": "Facts"}},
            namespace="training_data"  # Query from training_data namespace
        )

        print(f"\n📊 Retrieved {len(results['matches'])} results (filtered by role='Facts'):")
        for i, match in enumerate(results['matches'], 1):
            print(f"\n{i}. Score: {match['score']:.4f}")
            print(f"   Case: {match['metadata']['case_id']}")
            print(f"   Role: {match['metadata']['role']}")
            print(f"   Text: {match['metadata']['text'][:150]}...")
    except Exception as e:
        print(f"❌ Error querying: {e}")

    print(f"\n{'='*80}")
    print("💡 NEXT STEPS")
    print(f"{'='*80}")
    print("1. View data in Pinecone console: https://app.pinecone.io/")
    print("2. Test role-aware queries with different roles")
    print("3. Integrate with FastAPI backend")
    print("4. Process remaining files if needed")
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
