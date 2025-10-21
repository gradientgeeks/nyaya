"""
Resume Upload to Pinecone from Checkpoint

This script resumes the upload from where it was interrupted:
- Starts from file_33.txt, sentence 606
- Continues through remaining files (33-100)
- Uses same metadata structure as original upload
- Skips already uploaded vectors automatically (upsert handles this)

Usage:
    python resume_upload_from_checkpoint.py

Configuration:
    - START_FILE_NUM: 33 (file_33.txt)
    - START_SENTENCE: 606 (within file_33.txt)
    - END_FILE_NUM: 100 (process until file_100.txt)
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
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# Configuration - RESUME FROM CHECKPOINT
DATA_DIR = Path(__file__).parent / "pinecone_data_insert"
START_FILE_NUM = 33  # Resume from file_33.txt
START_SENTENCE = 606  # Skip first 605 sentences in file_33.txt
END_FILE_NUM = 100  # Process through file_100.txt
BATCH_SIZE = 100  # Pinecone upsert batch size
EMBEDDING_DIM = 384


def parse_legal_file(file_path: Path, skip_lines: int = 0) -> List[Dict]:
    """
    Parse a tab-separated legal case file, optionally skipping first N lines.

    Format: <sentence text><TAB><role label>

    Args:
        file_path: Path to the .txt file
        skip_lines: Number of lines to skip from start (for resuming mid-file)

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

                # Skip already processed lines
                if line_num <= skip_lines:
                    continue

                # Split by tab
                parts = line.split('\t')

                if len(parts) != 2:
                    print(f"   ⚠️  Warning: Line {line_num} in {file_path.name} has {len(parts)} parts (expected 2), skipping")
                    continue

                text, role = parts

                # Validate role
                valid_roles = [
                    "Facts", "Issue", "Arguments of Petitioner",
                    "Arguments of Respondent", "Reasoning", "Decision", "None"
                ]

                if role not in valid_roles:
                    print(f"   ⚠️  Warning: Invalid role '{role}' in {file_path.name}:{line_num}, keeping as-is")

                sentences.append({
                    "text": text,
                    "role": role,
                    "file": file_path.stem,  # e.g., "file_33"
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


def upload_to_pinecone(index, model, all_sentences: List[Dict], start_from: int = 0):
    """
    Upload all sentences to Pinecone in batches.

    Args:
        index: Pinecone index
        model: SentenceTransformer model
        all_sentences: List of all sentences to upload
        start_from: Starting index (for display purposes)
    """
    total_sentences = len(all_sentences)
    print(f"\n📊 Total sentences to upload: {total_sentences:,}")
    print(f"📍 Starting from vector #{start_from + 1:,}")

    # Process in batches
    uploaded_count = 0
    overall_batch_num = start_from // BATCH_SIZE + 1

    for batch_start in range(0, total_sentences, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_sentences)
        batch = all_sentences[batch_start:batch_end]

        global_start = start_from + batch_start + 1
        global_end = start_from + batch_end

        print(f"\n🔄 Processing batch {overall_batch_num} ({global_start:,}-{global_end:,})...")

        # Create embeddings for this batch
        print(f"   🧠 Generating embeddings...")
        embeddings = create_embeddings_batch(model, batch)

        # Prepare vectors for Pinecone
        vectors = []
        for i, (sentence, embedding) in enumerate(zip(batch, embeddings)):
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
        print(f"   ⬆️  Upserting {len(vectors)} vectors to 'training_data' namespace...")
        try:
            upsert_response = index.upsert(
                vectors=vectors,
                namespace="training_data"
            )
            uploaded_count += upsert_response.upserted_count
            total_uploaded = start_from + uploaded_count
            print(f"   ✅ Upserted {upsert_response.upserted_count} vectors (Total: {total_uploaded:,})")
        except Exception as e:
            print(f"   ❌ Error upserting batch: {e}")
            continue

        overall_batch_num += 1

    return uploaded_count


def main():
    """Main execution flow."""
    print("=" * 80)
    print("🏛️  NYAYA: Resume Upload to Pinecone from Checkpoint")
    print("=" * 80)
    print(f"📍 Checkpoint: file_{START_FILE_NUM}.txt, sentence {START_SENTENCE + 1}")
    print(f"📂 Processing files: file_{START_FILE_NUM}.txt to file_{END_FILE_NUM}.txt")

    # Step 1: Check data directory
    if not DATA_DIR.exists():
        print(f"❌ Error: Data directory not found: {DATA_DIR}")
        return

    # Get files from START_FILE_NUM to END_FILE_NUM
    all_files = sorted(DATA_DIR.glob("*.txt"), key=lambda f: int(f.stem.split('_')[1]))
    files_to_process = [
        f for f in all_files
        if START_FILE_NUM <= int(f.stem.split('_')[1]) <= END_FILE_NUM
    ]

    print(f"\n📁 Data directory: {DATA_DIR}")
    print(f"📋 Files to process: {len(files_to_process)}")

    if not files_to_process:
        print("❌ No files found in specified range")
        return

    # Step 2: Parse files (with checkpoint for first file)
    print(f"\n{'='*80}")
    print("📖 PARSING FILES")
    print(f"{'='*80}")

    all_sentences = []
    file_stats = {}
    vectors_before_start = 16600  # Already uploaded

    for i, file_path in enumerate(files_to_process, 1):
        file_num = int(file_path.stem.split('_')[1])
        
        # Skip sentences for the first file (resume point)
        skip_lines = START_SENTENCE if file_num == START_FILE_NUM else 0
        
        skip_msg = f" (skipping first {skip_lines} sentences)" if skip_lines > 0 else ""
        print(f"\n[{i}/{len(files_to_process)}] Parsing {file_path.name}{skip_msg}...")

        sentences = parse_legal_file(file_path, skip_lines=skip_lines)

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
    print(f"Files parsed: {len(file_stats)}")
    print(f"New sentences to upload: {len(all_sentences):,}")
    print(f"Previously uploaded: {vectors_before_start:,}")
    print(f"Total after completion: {vectors_before_start + len(all_sentences):,}")

    # Overall role distribution
    overall_roles = {}
    for s in all_sentences:
        role = s['role']
        overall_roles[role] = overall_roles.get(role, 0) + 1

    print(f"\nRole distribution (new sentences):")
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
    uploaded_count = upload_to_pinecone(index, model, all_sentences, start_from=vectors_before_start)
    elapsed_time = time.time() - start_time

    # Step 6: Verify final count
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
    print(f"Files processed: {len(file_stats)}")
    print(f"New sentences uploaded: {uploaded_count:,}")
    print(f"Upload time: {elapsed_time:.2f} seconds")
    print(f"Average speed: {uploaded_count/elapsed_time:.2f} sentences/second")
    print(f"Total vectors in Pinecone: {training_data_count:,}")

    # Test query
    print(f"\n{'='*80}")
    print("🔍 TESTING QUERY")
    print(f"{'='*80}")

    test_query_text = "What was the court's reasoning?"
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
            filter={"role": {"$eq": "Reasoning"}},
            namespace="training_data"
        )

        print(f"\n📊 Retrieved {len(results['matches'])} results (filtered by role='Reasoning'):")
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
    
    if training_data_count >= 51000:
        print("✅ All files processed successfully!")
        print("1. View data in Pinecone console: https://app.pinecone.io/")
        print("2. Test role-aware queries with different roles")
        print("3. Integrate with FastAPI backend")
    else:
        remaining = 51505 - training_data_count
        print(f"⚠️  Still {remaining:,} sentences remaining to reach 51,505 total")
        print("1. Check if all 100 files were processed")
        print("2. Run this script again if interrupted")
        print("3. Or adjust END_FILE_NUM to process more files")
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
