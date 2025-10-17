"""
Populate Pinecone Database with Training Data

This script processes your training .txt files with rhetorical role labels,
generates embeddings using the same model as RAG system, and stores them
in role-specific Pinecone indexes.

Usage:
    python populate_pinecone.py --data-dir dataset/Hier_BiLSTM_CRF/train --split train
    python populate_pinecone.py --data-dir dataset/Hier_BiLSTM_CRF/test --split test
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import time

from dotenv import load_dotenv
from langchain_community.embeddings import VertexAIEmbeddings
from pinecone import Pinecone

from src.config.pinecone_setup import PineconeIndexManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LegalDocumentIndexer:
    """Indexes legal documents with rhetorical roles into Pinecone"""
    
    # Map role labels from files to your system's role names
    ROLE_MAPPING = {
        "Facts": "facts",
        "Issue": "issue",
        "Arguments of Petitioner": "arguments_petitioner",
        "Arguments of Respondent": "arguments_respondent",
        "Reasoning": "reasoning",
        "Decision": "decision",
        "None": "none"
    }
    
    def __init__(self, embedding_model: str = "text-embedding-005", batch_size: int = 100):
        """
        Initialize the indexer
        
        Args:
            embedding_model: Name of embedding model (must match RAG system)
            batch_size: Number of documents to process in each batch
        """
        # Initialize embedding model (SAME as RAG system!)
        self.embeddings = VertexAIEmbeddings(model_name=embedding_model)
        self.batch_size = batch_size
        
        # Initialize Pinecone manager
        self.pinecone_manager = PineconeIndexManager()
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_manager.api_key)
        
        # Ensure all indexes exist
        logger.info("Ensuring all role-based indexes exist...")
        self._ensure_indexes_exist()
        
        # Get index references
        self.indexes = {
            role: self.pc.Index(index_name)
            for role, index_name in self.pinecone_manager.role_indexes.items()
        }
        
        logger.info(f"Initialized indexer with {len(self.indexes)} role-based indexes")
    
    def _ensure_indexes_exist(self):
        """Ensure all role-based Pinecone indexes exist"""
        for role, index_name in self.pinecone_manager.role_indexes.items():
            if not self.pinecone_manager.index_exists(index_name):
                logger.info(f"Creating index for role '{role}': {index_name}")
                self.pinecone_manager.create_index(
                    index_name=index_name,
                    dimension=768,  # Vertex AI text-embedding-005
                    metric="cosine"
                )
                # Wait for index to be ready
                time.sleep(2)
    
    def parse_txt_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Parse a .txt file with tab-separated sentence and role labels
        
        Args:
            file_path: Path to .txt file
            
        Returns:
            List of (sentence, role) tuples
        """
        sentences_with_roles = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Split by tab (sentence\trole)
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0].strip()
                        role = parts[1].strip()
                        
                        if sentence:  # Only add non-empty sentences
                            sentences_with_roles.append((sentence, role))
            
            return sentences_with_roles
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    def process_file(self, file_path: Path, split: str = "train") -> List[Dict[str, Any]]:
        """
        Process a single file and prepare vectors for indexing
        
        Args:
            file_path: Path to the .txt file
            split: Dataset split (train/test/val)
            
        Returns:
            List of vectors with metadata ready for Pinecone
        """
        # Parse file
        sentences_with_roles = self.parse_txt_file(file_path)
        
        if not sentences_with_roles:
            logger.warning(f"No valid sentences found in {file_path}")
            return []
        
        logger.info(f"Processing {file_path.name}: {len(sentences_with_roles)} sentences")
        
        # Prepare data
        vectors = []
        document_id = file_path.stem  # e.g., "file_6409"
        
        # Extract sentences and roles
        sentences = [sent for sent, _ in sentences_with_roles]
        roles = [role for _, role in sentences_with_roles]
        
        # Generate embeddings in batch (CRITICAL: Using same model as RAG!)
        try:
            embeddings = self.embeddings.embed_documents(sentences)
            logger.info(f"Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for {file_path}: {e}")
            return []
        
        # Prepare vectors with metadata
        for i, (sentence, role, embedding) in enumerate(zip(sentences, roles, embeddings)):
            # Normalize role name
            role_key = self.ROLE_MAPPING.get(role, "none")
            
            vector_data = {
                "id": f"{document_id}_sent_{i}",
                "values": embedding,
                "metadata": {
                    "text": sentence,
                    "role": role,  # Original role label
                    "role_key": role_key,  # Normalized role key
                    "document_id": document_id,
                    "sentence_index": i,
                    "split": split,
                    "source_file": file_path.name
                }
            }
            vectors.append(vector_data)
        
        return vectors
    
    def index_vectors(self, vectors: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Index vectors into appropriate role-based Pinecone indexes
        
        Args:
            vectors: List of vector data with metadata
            
        Returns:
            Dictionary with count of vectors indexed per role
        """
        if not vectors:
            return {}
        
        # Group vectors by role
        role_vectors = {}
        for vector in vectors:
            role_key = vector["metadata"]["role_key"]
            if role_key not in role_vectors:
                role_vectors[role_key] = []
            role_vectors[role_key].append(vector)
        
        # Also add to main index for cross-role queries
        role_vectors["main"] = vectors.copy()
        
        # Index into Pinecone
        indexed_counts = {}
        
        for role_key, role_vecs in role_vectors.items():
            if role_key not in self.indexes:
                logger.warning(f"Index not found for role: {role_key}")
                continue
            
            try:
                # Batch upsert
                batch_size = self.batch_size
                for i in range(0, len(role_vecs), batch_size):
                    batch = role_vecs[i:i + batch_size]
                    self.indexes[role_key].upsert(vectors=batch)
                
                indexed_counts[role_key] = len(role_vecs)
                logger.info(f"  Indexed {len(role_vecs)} vectors into '{role_key}' index")
                
            except Exception as e:
                logger.error(f"Failed to index vectors for role '{role_key}': {e}")
                indexed_counts[role_key] = 0
        
        return indexed_counts
    
    def process_directory(self, data_dir: Path, split: str = "train", max_files: int = None):
        """
        Process all .txt files in a directory
        
        Args:
            data_dir: Directory containing .txt files
            split: Dataset split name
            max_files: Maximum number of files to process (None = all)
        """
        # Find all .txt files
        txt_files = sorted(data_dir.glob("*.txt"))
        
        if max_files:
            txt_files = txt_files[:max_files]
        
        logger.info(f"Found {len(txt_files)} files to process in {data_dir}")
        
        total_indexed = {role: 0 for role in self.ROLE_MAPPING.values()}
        total_indexed["main"] = 0
        
        # Process each file
        for file_path in tqdm(txt_files, desc=f"Indexing {split} files"):
            # Process file
            vectors = self.process_file(file_path, split)
            
            if vectors:
                # Index vectors
                indexed_counts = self.index_vectors(vectors)
                
                # Update totals
                for role, count in indexed_counts.items():
                    total_indexed[role] = total_indexed.get(role, 0) + count
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Indexing Summary for {split} split:")
        logger.info(f"{'='*60}")
        for role, count in sorted(total_indexed.items()):
            logger.info(f"  {role:25s}: {count:6d} vectors")
        logger.info(f"{'='*60}\n")
        
        return total_indexed
    
    def get_index_stats(self):
        """Print statistics for all indexes"""
        logger.info("\nPinecone Index Statistics:")
        logger.info(f"{'='*60}")
        
        stats = self.pinecone_manager.get_all_role_index_stats()
        
        total_vectors = 0
        for role, role_stats in sorted(stats.items()):
            if "total_vectors" in role_stats:
                count = role_stats["total_vectors"]
                total_vectors += count
                logger.info(f"  {role:25s}: {count:6d} vectors")
            else:
                logger.info(f"  {role:25s}: {role_stats}")
        
        logger.info(f"{'='*60}")
        logger.info(f"  {'TOTAL':25s}: {total_vectors:6d} vectors")
        logger.info(f"{'='*60}\n")


def main():
    """Main function to populate Pinecone from training data"""
    parser = argparse.ArgumentParser(description="Populate Pinecone with legal document embeddings")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing .txt files with role labels"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "val"],
        help="Dataset split name"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Pinecone upsert operations"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all indexes before indexing"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Initialize indexer
    indexer = LegalDocumentIndexer(batch_size=args.batch_size)
    
    # Clear indexes if requested
    if args.clear:
        confirm = input("Are you sure you want to clear all indexes? (yes/no): ")
        if confirm.lower() == "yes":
            logger.info("Clearing all indexes...")
            indexer.pinecone_manager.clear_all_role_indexes()
            logger.info("All indexes cleared")
        else:
            logger.info("Clear operation cancelled")
            return
    
    # Process directory
    logger.info(f"\nStarting indexing process for {args.split} split...")
    total_indexed = indexer.process_directory(
        data_dir=data_dir,
        split=args.split,
        max_files=args.max_files
    )
    
    # Show final statistics
    indexer.get_index_stats()
    
    logger.info("✅ Indexing completed successfully!")


if __name__ == "__main__":
    main()
