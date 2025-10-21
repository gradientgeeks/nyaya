"""
Pinecone Service Wrapper

Handles all Pinecone vector database operations with role-aware metadata.
This is the storage layer for Nyaya's role-aware RAG system.
"""

import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from ..core.config import Settings
from ..models.schemas import RhetoricalRole

logger = logging.getLogger(__name__)


class PineconeService:
    """
    Wrapper for Pinecone operations with role-aware metadata handling.
    
    Key responsibilities:
    - Upsert vectors with role metadata
    - Query with role filtering (the core of role-aware RAG)
    - Namespace management (user_documents, training_data, demo)
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        
        # Initialize or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist (idempotent)."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"📝 Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_environment
                )
            )
            logger.info(f"✅ Index created successfully")
        else:
            logger.info(f"✅ Index already exists: {self.index_name}")
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        roles: List[RhetoricalRole],
        case_id: str,
        namespace: str = "user_documents",
        additional_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Upsert vectors to Pinecone with role metadata.
        
        Args:
            vectors: List of embedding vectors (384-dim each)
            texts: List of sentence texts
            roles: List of RhetoricalRole enums
            case_id: Unique case identifier
            namespace: Pinecone namespace (default: user_documents)
            additional_metadata: Optional list of dicts with extra metadata per vector
        
        Returns:
            Dict with upsert status and count
        """
        if not (len(vectors) == len(texts) == len(roles)):
            raise ValueError("vectors, texts, and roles must have same length")
        
        # Format vectors for Pinecone
        formatted_vectors = []
        for i, (vector, text, role) in enumerate(zip(vectors, texts, roles)):
            metadata = {
                "text": text,
                "role": role.value,
                "case_id": case_id,
                "sentence_index": i,
                "user_uploaded": namespace == "user_documents"
            }
            
            # Add any additional metadata
            if additional_metadata and i < len(additional_metadata):
                metadata.update(additional_metadata[i])
            
            formatted_vectors.append({
                "id": f"{case_id}_sent_{i}",
                "values": vector,
                "metadata": metadata
            })
        
        # Upsert in batches of 100 (Pinecone limit)
        batch_size = 100
        total_upserted = 0
        
        for i in range(0, len(formatted_vectors), batch_size):
            batch = formatted_vectors[i:i + batch_size]
            result = self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += result.upserted_count
        
        logger.info(
            f"⬆️  Upserted {total_upserted} vectors for case {case_id} "
            f"to namespace '{namespace}'"
        )
        
        return {
            "status": "success",
            "upserted_count": total_upserted,
            "case_id": case_id,
            "namespace": namespace
        }
    
    def query_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        role_filter: Optional[RhetoricalRole] = None,
        case_id_filter: Optional[str] = None,
        namespace: str = "user_documents",
        include_metadata: bool = True,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Query Pinecone with optional role filtering (NYAYA'S KEY DIFFERENTIATOR).
        
        Args:
            query_vector: Query embedding (384-dim)
            top_k: Number of results to return
            role_filter: Optional RhetoricalRole to filter by
            case_id_filter: Optional case_id to filter by
            namespace: Pinecone namespace to query
            include_metadata: Whether to include metadata in results
            min_score: Minimum similarity score threshold
        
        Returns:
            List of matches with text, role, score, and metadata
        """
        # Build filter dict
        filter_dict = {}
        if role_filter:
            filter_dict["role"] = {"$eq": role_filter.value}
        if case_id_filter:
            filter_dict["case_id"] = {"$eq": case_id_filter}
        
        # Query Pinecone
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            namespace=namespace,
            include_metadata=include_metadata
        )
        
        # Format results
        matches = []
        for match in results.matches:
            if match.score >= min_score:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata.get("text", ""),
                    "role": match.metadata.get("role", ""),
                    "case_id": match.metadata.get("case_id", ""),
                    "sentence_index": match.metadata.get("sentence_index", -1)
                }
                
                # Include full metadata if requested
                if include_metadata:
                    result["metadata"] = match.metadata
                
                matches.append(result)
        
        logger.info(
            f"🔍 Query returned {len(matches)} results "
            f"(role_filter={role_filter.value if role_filter else 'None'})"
        )
        
        return matches
    
    def delete_case(
        self,
        case_id: str,
        namespace: str = "user_documents"
    ) -> Dict[str, Any]:
        """
        Delete all vectors for a specific case.
        
        Args:
            case_id: Case identifier
            namespace: Pinecone namespace
        
        Returns:
            Dict with deletion status
        """
        # Pinecone doesn't support delete by metadata filter directly
        # So we need to query first to get all IDs, then delete
        
        # Query to get all vector IDs for this case
        results = self.index.query(
            vector=[0.0] * self.settings.embedding_dimension,  # Dummy vector
            top_k=10000,  # Large number to get all
            filter={"case_id": {"$eq": case_id}},
            namespace=namespace,
            include_metadata=False
        )
        
        vector_ids = [match.id for match in results.matches]
        
        if vector_ids:
            self.index.delete(ids=vector_ids, namespace=namespace)
            logger.info(f"🗑️  Deleted {len(vector_ids)} vectors for case {case_id}")
        else:
            logger.info(f"ℹ️  No vectors found for case {case_id}")
        
        return {
            "status": "success",
            "deleted_count": len(vector_ids),
            "case_id": case_id
        }
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": {
                ns: info.vector_count
                for ns, info in stats.namespaces.items()
            }
        }
