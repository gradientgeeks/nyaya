#!/usr/bin/env python3
"""
Quick script to check Pinecone index status.

Run this after test_pinecone_embedding.py to verify vectors are indexed.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")

print("=" * 80)
print("🔍 PINECONE INDEX STATUS CHECK")
print("=" * 80)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Get stats
stats = index.describe_index_stats()

print(f"\n📊 Index: {PINECONE_INDEX_NAME}")
print(f"   Total vectors: {stats.total_vector_count}")
print(f"   Dimension: {stats.dimension}")
print(f"   Namespaces: {stats.namespaces if stats.namespaces else 'default'}")

if stats.total_vector_count == 0:
    print("\n⚠️  No vectors found yet. Possible reasons:")
    print("   1. Index is still syncing (wait 30-60 seconds)")
    print("   2. Vectors were not uploaded successfully")
    print("   3. Index was just created (serverless indexing takes time)")
    print("\n   💡 Try running test_pinecone_embedding.py again")
else:
    print(f"\n✅ Index has {stats.total_vector_count} vectors!")
    
    # Try a sample query
    print("\n🔍 Testing sample query...")
    try:
        # Query without filters to see if anything is there
        results = index.query(
            vector=[0.1] * stats.dimension,  # Dummy vector
            top_k=3,
            include_metadata=True
        )
        
        print(f"   Found {len(results['matches'])} results")
        if results['matches']:
            print("\n   Sample results:")
            for i, match in enumerate(results['matches'][:3], 1):
                role = match['metadata'].get('role', 'N/A')
                text = match['metadata'].get('text', 'N/A')[:50]
                print(f"   {i}. Role: {role}, Text: {text}...")
    except Exception as e:
        print(f"   ❌ Error querying: {e}")

print("\n" + "=" * 80)
