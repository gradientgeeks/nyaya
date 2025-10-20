#!/usr/bin/env python3
"""
List all vectors in Pinecone index with full metadata.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")

print("=" * 80)
print("📋 ALL VECTORS IN PINECONE INDEX")
print("=" * 80)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Get stats
stats = index.describe_index_stats()
print(f"\n📊 Index: {PINECONE_INDEX_NAME}")
print(f"   Total vectors: {stats.total_vector_count}")
print(f"   Dimension: {stats.dimension}")

if stats.total_vector_count == 0:
    print("\n❌ No vectors in index")
    exit(1)

# Query with a dummy vector to retrieve all vectors
# (Since Pinecone doesn't have a "list all" API, we query with high top_k)
print(f"\n📋 Fetching all {stats.total_vector_count} vectors...")

try:
    # Create a dummy query vector (all zeros)
    dummy_vector = [0.0] * stats.dimension
    
    # Query with top_k = total count to get all vectors
    results = index.query(
        vector=dummy_vector,
        top_k=stats.total_vector_count,
        include_metadata=True,
        include_values=False  # Don't need the actual embedding values
    )
    
    print(f"✅ Retrieved {len(results['matches'])} vectors\n")
    
    # Display each vector
    for i, match in enumerate(results['matches'], 1):
        print("=" * 80)
        print(f"VECTOR {i}")
        print("=" * 80)
        print(f"ID: {match['id']}")
        print(f"Score: {match['score']:.4f}")
        print(f"\nMetadata:")
        for key, value in match['metadata'].items():
            if key == 'text':
                # Truncate long text for display
                text_preview = value[:200] + "..." if len(value) > 200 else value
                print(f"  {key}: {text_preview}")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Summary by role
    print("=" * 80)
    print("📊 SUMMARY BY ROLE")
    print("=" * 80)
    
    role_counts = {}
    for match in results['matches']:
        role = match['metadata'].get('role', 'Unknown')
        role_counts[role] = role_counts.get(role, 0) + 1
    
    for role, count in sorted(role_counts.items()):
        print(f"  {role}: {count} vector(s)")
    
    print("\n" + "=" * 80)
    print("✅ ALL VECTORS LISTED SUCCESSFULLY")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error fetching vectors: {e}")
    import traceback
    traceback.print_exc()
