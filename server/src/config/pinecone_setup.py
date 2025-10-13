"""
Pinecone Index Setup and Configuration

This module provides utilities to initialize and manage Pinecone indexes
for the Legal Document Analysis System with role-aware namespaces.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class PineconeIndexManager:
    """Manager for Pinecone index operations"""

    # Rhetorical roles for legal documents
    RHETORICAL_ROLES = [
        "facts",
        "issue",
        "arguments_petitioner",
        "arguments_respondent",
        "reasoning",
        "decision",
        "none"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_index_name: Optional[str] = None,
        environment: Optional[str] = None
    ):
        """
        Initialize Pinecone index manager

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            base_index_name: Base index name (defaults to PINECONE_INDEX_NAME env var)
            environment: Pinecone environment (defaults to PINECONE_ENVIRONMENT env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.base_index_name = base_index_name or os.getenv("PINECONE_INDEX_NAME", "nyaya-legal-rag")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

        if not self.api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Generate index names for each role
        self.role_indexes = {
            role: f"{self.base_index_name}-{role}"
            for role in self.RHETORICAL_ROLES
        }
        # Add main index for cross-role queries
        self.role_indexes["main"] = self.base_index_name

        logger.info(f"Pinecone client initialized with base name: {self.base_index_name}")
        logger.info(f"Role-specific indexes: {list(self.role_indexes.values())}")

    def create_index(
        self,
        index_name: str,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: Optional[str] = None
    ) -> bool:
        """
        Create a single Pinecone index with serverless configuration

        Args:
            index_name: Name of the index to create
            dimension: Vector dimension (768 for Vertex AI text-embedding-005)
            metric: Distance metric (cosine, euclidean, or dotproduct)
            cloud: Cloud provider (aws, gcp, or azure)
            region: Cloud region (defaults to self.environment)

        Returns:
            True if index created successfully, False if already exists
        """
        try:
            # Check if index already exists
            existing_indexes = self.pc.list_indexes()
            if index_name in [idx.name for idx in existing_indexes]:
                logger.warning(f"Index '{index_name}' already exists")
                return False

            # Create serverless spec
            region = region or self.environment
            spec = ServerlessSpec(cloud=cloud, region=region)

            # Create the index
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec
            )

            logger.info(f"Successfully created index '{index_name}' with dimension {dimension}")
            logger.info(f"Configuration: {metric} metric, {cloud}/{region}")

            return True

        except Exception as e:
            logger.error(f"Failed to create index '{index_name}': {e}")
            raise

    def create_all_role_indexes(
        self,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Create all role-based indexes (one per rhetorical role + main index)

        Args:
            dimension: Vector dimension (768 for Vertex AI text-embedding-005)
            metric: Distance metric (cosine, euclidean, or dotproduct)
            cloud: Cloud provider (aws, gcp, or azure)
            region: Cloud region (defaults to self.environment)

        Returns:
            Dictionary mapping role to creation status
        """
        results = {}
        logger.info(f"Creating {len(self.role_indexes)} role-based indexes...")

        for role, index_name in self.role_indexes.items():
            try:
                created = self.create_index(
                    index_name=index_name,
                    dimension=dimension,
                    metric=metric,
                    cloud=cloud,
                    region=region
                )
                results[role] = created
                logger.info(f"  [{role}] -> {index_name}: {'created' if created else 'already exists'}")
            except Exception as e:
                logger.error(f"  [{role}] -> {index_name}: failed - {e}")
                results[role] = False

        created_count = sum(1 for v in results.values() if v)
        logger.info(f"Successfully created {created_count}/{len(results)} indexes")

        return results

    def delete_index(self, index_name: str) -> bool:
        """
        Delete a specific Pinecone index

        WARNING: This will permanently delete all data in the index

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index deleted successfully
        """
        try:
            self.pc.delete_index(index_name)
            logger.info(f"Successfully deleted index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete index '{index_name}': {e}")
            raise

    def delete_all_role_indexes(self) -> Dict[str, bool]:
        """
        Delete all role-based indexes

        WARNING: This will permanently delete all data in all role indexes

        Returns:
            Dictionary mapping role to deletion status
        """
        results = {}
        logger.info(f"Deleting {len(self.role_indexes)} role-based indexes...")

        for role, index_name in self.role_indexes.items():
            try:
                deleted = self.delete_index(index_name)
                results[role] = deleted
                logger.info(f"  [{role}] -> {index_name}: deleted")
            except Exception as e:
                logger.error(f"  [{role}] -> {index_name}: failed - {e}")
                results[role] = False

        deleted_count = sum(1 for v in results.values() if v)
        logger.info(f"Successfully deleted {deleted_count}/{len(results)} indexes")

        return results

    def index_exists(self, index_name: str) -> bool:
        """
        Check if a specific index exists

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            existing_indexes = self.pc.list_indexes()
            return index_name in [idx.name for idx in existing_indexes]
        except Exception as e:
            logger.error(f"Failed to check index existence: {e}")
            return False

    def all_role_indexes_exist(self) -> Dict[str, bool]:
        """
        Check which role indexes exist

        Returns:
            Dictionary mapping role to existence status
        """
        return {
            role: self.index_exists(index_name)
            for role, index_name in self.role_indexes.items()
        }

    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific index

        Args:
            index_name: Name of the index

        Returns:
            Dictionary with index statistics
        """
        try:
            index = self.pc.Index(index_name)
            stats = index.describe_index_stats()

            logger.info(f"Index stats for '{index_name}':")
            logger.info(f"  Total vectors: {stats.total_vector_count}")
            logger.info(f"  Dimension: {stats.dimension}")

            return {
                "index_name": index_name,
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension
            }

        except Exception as e:
            logger.error(f"Failed to get stats for index '{index_name}': {e}")
            raise

    def get_all_role_index_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all role-based indexes

        Returns:
            Dictionary mapping role to index statistics
        """
        results = {}
        total_vectors = 0

        logger.info(f"Getting stats for {len(self.role_indexes)} role-based indexes...")

        for role, index_name in self.role_indexes.items():
            try:
                if self.index_exists(index_name):
                    stats = self.get_index_stats(index_name)
                    results[role] = stats
                    total_vectors += stats["total_vectors"]
                    logger.info(f"  [{role}] -> {stats['total_vectors']} vectors")
                else:
                    results[role] = {"index_name": index_name, "total_vectors": 0, "exists": False}
                    logger.info(f"  [{role}] -> index does not exist")
            except Exception as e:
                logger.error(f"  [{role}] -> failed to get stats: {e}")
                results[role] = {"error": str(e)}

        logger.info(f"Total vectors across all indexes: {total_vectors}")

        return results

    def clear_index(self, index_name: str) -> bool:
        """
        Clear all vectors from a specific index

        Args:
            index_name: Index name to clear

        Returns:
            True if index cleared successfully
        """
        try:
            index = self.pc.Index(index_name)
            index.delete(delete_all=True)
            logger.info(f"Successfully cleared index '{index_name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to clear index '{index_name}': {e}")
            raise

    def clear_all_role_indexes(self) -> Dict[str, bool]:
        """
        Clear all vectors from all role-based indexes

        Returns:
            Dictionary mapping role to clear status
        """
        results = {}
        logger.info(f"Clearing {len(self.role_indexes)} role-based indexes...")

        for role, index_name in self.role_indexes.items():
            try:
                if self.index_exists(index_name):
                    cleared = self.clear_index(index_name)
                    results[role] = cleared
                    logger.info(f"  [{role}] -> {index_name}: cleared")
                else:
                    results[role] = False
                    logger.info(f"  [{role}] -> {index_name}: does not exist")
            except Exception as e:
                logger.error(f"  [{role}] -> {index_name}: failed - {e}")
                results[role] = False

        cleared_count = sum(1 for v in results.values() if v)
        logger.info(f"Successfully cleared {cleared_count}/{len(results)} indexes")

        return results

    def reset_all_role_indexes(
        self,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Delete and recreate all role-based indexes (useful for development)

        Args:
            dimension: Vector dimension
            metric: Distance metric
            cloud: Cloud provider
            region: Cloud region

        Returns:
            Dictionary mapping role to reset status
        """
        results = {}
        logger.info(f"Resetting {len(self.role_indexes)} role-based indexes...")

        # Delete all indexes first
        delete_results = self.delete_all_role_indexes()

        # Wait for deletions to propagate
        import time
        logger.info("Waiting for index deletions to complete...")
        time.sleep(5)

        # Create all indexes
        create_results = self.create_all_role_indexes(dimension, metric, cloud, region)

        # Combine results
        for role in self.role_indexes.keys():
            results[role] = delete_results.get(role, False) and create_results.get(role, False)

        reset_count = sum(1 for v in results.values() if v)
        logger.info(f"Successfully reset {reset_count}/{len(results)} indexes")

        return results

    def ensure_role_index_exists(
        self,
        role: str,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: Optional[str] = None
    ) -> bool:
        """
        Ensure a specific role index exists (lazy initialization)

        Args:
            role: Rhetorical role
            dimension: Vector dimension
            metric: Distance metric
            cloud: Cloud provider
            region: Cloud region

        Returns:
            True if index exists or was created successfully
        """
        if role not in self.role_indexes:
            logger.error(f"Unknown role: {role}")
            return False

        index_name = self.role_indexes[role]

        if self.index_exists(index_name):
            logger.info(f"Index for role '{role}' already exists: {index_name}")
            return True

        logger.info(f"Creating index for role '{role}': {index_name}")
        return self.create_index(index_name, dimension, metric, cloud, region)

    def list_indexes(self) -> List[str]:
        """
        List all available indexes

        Returns:
            List of index names
        """
        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            logger.info(f"Available indexes: {index_names}")
            return index_names

        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            raise


def setup_pinecone_indexes(
    api_key: Optional[str] = None,
    base_index_name: Optional[str] = None,
    dimension: int = 768,
    reset: bool = False,
    lazy: bool = True
) -> PineconeIndexManager:
    """
    Convenience function to set up Pinecone role-based indexes

    Args:
        api_key: Pinecone API key
        base_index_name: Base index name
        dimension: Vector dimension (768 for Vertex AI embeddings)
        reset: Whether to reset indexes if they exist
        lazy: If True, don't create indexes immediately (create on first use)

    Returns:
        Configured PineconeIndexManager
    """
    manager = PineconeIndexManager(api_key=api_key, base_index_name=base_index_name)

    if reset:
        logger.info("Resetting all role-based indexes...")
        manager.reset_all_role_indexes(dimension=dimension)
    elif not lazy:
        logger.info("Creating all role-based indexes...")
        results = manager.create_all_role_indexes(dimension=dimension)
        created = sum(1 for v in results.values() if v)
        logger.info(f"Created {created}/{len(results)} indexes (others may already exist)")
    else:
        logger.info("Lazy mode: Indexes will be created on first use")
        # Just check what exists
        existing = manager.all_role_indexes_exist()
        existing_count = sum(1 for v in existing.values() if v)
        logger.info(f"{existing_count}/{len(existing)} indexes already exist")

    # Display stats for existing indexes
    try:
        stats = manager.get_all_role_index_stats()
        total = sum(s.get("total_vectors", 0) for s in stats.values())
        logger.info(f"Total vectors across all indexes: {total}")
    except:
        logger.info("Indexes ready for use")

    return manager


# CLI interface for easy setup
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pinecone Index Setup for Nyaya Legal RAG")
    parser.add_argument(
        "--action",
        choices=["create", "delete", "reset", "stats", "list", "clear"],
        default="create",
        help="Action to perform on role-based indexes"
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Vector dimension (default: 768 for Vertex AI embeddings)"
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "euclidean", "dotproduct"],
        default="cosine",
        help="Distance metric"
    )
    parser.add_argument(
        "--base-index-name",
        type=str,
        help="Base index name (defaults to PINECONE_INDEX_NAME env var)"
    )
    parser.add_argument(
        "--lazy",
        action="store_true",
        help="Use lazy initialization (create indexes on first use)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize manager
    manager = PineconeIndexManager(base_index_name=args.base_index_name)

    # Perform action
    if args.action == "create":
        print(f"Creating role-based indexes for '{manager.base_index_name}'...")
        results = manager.create_all_role_indexes(dimension=args.dimension, metric=args.metric)
        created = sum(1 for v in results.values() if v)
        print(f"Successfully created {created}/{len(results)} indexes")
        for role, created in results.items():
            status = "created" if created else "already exists"
            print(f"  [{role}] -> {manager.role_indexes[role]}: {status}")

    elif args.action == "delete":
        confirm = input(f"Are you sure you want to delete ALL role-based indexes? (yes/no): ")
        if confirm.lower() == "yes":
            results = manager.delete_all_role_indexes()
            deleted = sum(1 for v in results.values() if v)
            print(f"Successfully deleted {deleted}/{len(results)} indexes")
        else:
            print("Deletion cancelled")

    elif args.action == "reset":
        confirm = input(f"Are you sure you want to reset ALL role-based indexes? (yes/no): ")
        if confirm.lower() == "yes":
            results = manager.reset_all_role_indexes(dimension=args.dimension, metric=args.metric)
            reset = sum(1 for v in results.values() if v)
            print(f"Successfully reset {reset}/{len(results)} indexes")
        else:
            print("Reset cancelled")

    elif args.action == "clear":
        confirm = input(f"Are you sure you want to clear ALL vectors from role-based indexes? (yes/no): ")
        if confirm.lower() == "yes":
            results = manager.clear_all_role_indexes()
            cleared = sum(1 for v in results.values() if v)
            print(f"Successfully cleared {cleared}/{len(results)} indexes")
        else:
            print("Clear cancelled")

    elif args.action == "stats":
        print(f"\nRole-Based Index Statistics:")
        stats = manager.get_all_role_index_stats()
        total_vectors = 0
        for role, role_stats in stats.items():
            index_name = manager.role_indexes[role]
            if role_stats.get("exists") == False:
                print(f"  [{role}] -> {index_name}: does not exist")
            elif "error" in role_stats:
                print(f"  [{role}] -> {index_name}: error - {role_stats['error']}")
            else:
                vectors = role_stats.get("total_vectors", 0)
                dimension = role_stats.get("dimension", "unknown")
                total_vectors += vectors
                print(f"  [{role}] -> {index_name}: {vectors} vectors (dim: {dimension})")
        print(f"\nTotal vectors across all indexes: {total_vectors}")

    elif args.action == "list":
        indexes = manager.list_indexes()
        print(f"\nAvailable Pinecone indexes: {len(indexes)}")
        for idx in indexes:
            is_role_index = idx in manager.role_indexes.values()
            marker = " (role index)" if is_role_index else ""
            print(f"  - {idx}{marker}")
