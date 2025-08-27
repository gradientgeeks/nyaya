"""
Vector Database Configuration for Legal Document Analysis System

This module provides configuration and setup utilities for vector databases
including ChromaDB and FAISS with role-aware metadata storage.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
import faiss
import numpy as np
import pickle
import json

logger = logging.getLogger(__name__)

class VectorDBConfig:
    """Configuration for vector database systems"""
    
    def __init__(self, data_dir: str = "data/vector_db"):
        """
        Initialize vector database configuration
        
        Args:
            data_dir: Directory for storing vector database files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB configuration
        self.chroma_config = {
            "persist_directory": str(self.data_dir / "chroma"),
            "collection_name": "legal_documents",
            "embedding_dimension": 768,  # For Google Vertex AI embeddings
            "metadata_schema": self._get_metadata_schema()
        }
        
        # FAISS configuration
        self.faiss_config = {
            "index_path": str(self.data_dir / "faiss"),
            "index_type": "IVF",  # Inverted File Index
            "nlist": 100,  # Number of clusters
            "embedding_dimension": 768,
            "metric": "METRIC_INNER_PRODUCT"
        }
        
        # Role-specific configurations
        self.role_configs = self._setup_role_configurations()
    
    def _get_metadata_schema(self) -> Dict[str, str]:
        """Define metadata schema for legal documents"""
        return {
            "document_id": "string",
            "document_type": "string",
            "case_name": "string",
            "court": "string",
            "year": "integer",
            "citation": "string",
            "rhetorical_role": "string",
            "role_confidence": "float",
            "sentence_index": "integer",
            "chunk_index": "integer",
            "case_type": "string",
            "outcome": "string",
            "date_added": "string",
            "processing_version": "string"
        }
    
    def _setup_role_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Setup configurations for different rhetorical roles"""
        roles = [
            "Facts", "Issue", "Arguments of Petitioner", 
            "Arguments of Respondent", "Reasoning", "Decision", "None"
        ]
        
        role_configs = {}
        for role in roles:
            role_configs[role] = {
                "collection_name": f"legal_documents_{role.lower().replace(' ', '_')}",
                "weight": self._get_role_weight(role),
                "retrieval_count": self._get_role_retrieval_count(role),
                "similarity_threshold": self._get_role_similarity_threshold(role)
            }
        
        return role_configs
    
    def _get_role_weight(self, role: str) -> float:
        """Get importance weight for different roles in retrieval"""
        weights = {
            "Facts": 1.2,
            "Issue": 1.5,
            "Arguments of Petitioner": 1.0,
            "Arguments of Respondent": 1.0,
            "Reasoning": 1.3,
            "Decision": 1.4,
            "None": 0.5
        }
        return weights.get(role, 1.0)
    
    def _get_role_retrieval_count(self, role: str) -> int:
        """Get number of documents to retrieve for each role"""
        counts = {
            "Facts": 3,
            "Issue": 5,
            "Arguments of Petitioner": 2,
            "Arguments of Respondent": 2,
            "Reasoning": 4,
            "Decision": 3,
            "None": 1
        }
        return counts.get(role, 2)
    
    def _get_role_similarity_threshold(self, role: str) -> float:
        """Get similarity threshold for each role"""
        thresholds = {
            "Facts": 0.7,
            "Issue": 0.8,
            "Arguments of Petitioner": 0.6,
            "Arguments of Respondent": 0.6,
            "Reasoning": 0.75,
            "Decision": 0.8,
            "None": 0.5
        }
        return thresholds.get(role, 0.7)

class ChromaDBManager:
    """Manager for ChromaDB vector database operations"""
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize ChromaDB manager
        
        Args:
            config: Vector database configuration
        """
        self.config = config
        self.chroma_config = config.chroma_config
        self.role_configs = config.role_configs
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        self.collections = self._initialize_collections()
    
    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client with persistent storage"""
        try:
            # Create persist directory
            persist_dir = Path(self.chroma_config["persist_directory"])
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize client with settings
            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            logger.info(f"ChromaDB client initialized with persist directory: {persist_dir}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise
    
    def _initialize_collections(self) -> Dict[str, Any]:
        """Initialize collections for different rhetorical roles"""
        collections = {}
        
        try:
            # Main collection for all documents
            main_collection_name = self.chroma_config["collection_name"]
            collections["main"] = self.client.get_or_create_collection(
                name=main_collection_name,
                metadata={"description": "Main collection for legal documents"}
            )
            
            # Role-specific collections
            for role, role_config in self.role_configs.items():
                collection_name = role_config["collection_name"]
                collections[role] = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "description": f"Collection for {role} role documents",
                        "role": role,
                        "weight": role_config["weight"]
                    }
                )
            
            logger.info(f"Initialized {len(collections)} ChromaDB collections")
            return collections
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collections: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Add documents to appropriate collections based on rhetorical roles
        
        Args:
            documents: List of document texts
            embeddings: List of document embeddings
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        try:
            # Add to main collection
            self.collections["main"].add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            # Add to role-specific collections
            for i, metadata in enumerate(metadatas):
                role = metadata.get("rhetorical_role", "None")
                if role in self.collections:
                    self.collections[role].add(
                        documents=[documents[i]],
                        embeddings=[embeddings[i]],
                        metadatas=[metadata],
                        ids=[ids[i]]
                    )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def query_by_role(
        self,
        query_embedding: List[float],
        role: str,
        n_results: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Query documents by specific rhetorical role
        
        Args:
            query_embedding: Query embedding vector
            role: Rhetorical role to query
            n_results: Number of results to return
            
        Returns:
            Query results with documents and metadata
        """
        try:
            if role not in self.collections:
                logger.warning(f"Role '{role}' not found in collections")
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Get role-specific configuration
            role_config = self.role_configs.get(role, {})
            if n_results is None:
                n_results = role_config.get("retrieval_count", 3)
            
            # Query the role-specific collection
            results = self.collections[role].query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Apply similarity threshold filtering
            threshold = role_config.get("similarity_threshold", 0.7)
            filtered_results = self._filter_by_threshold(results, threshold)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to query by role '{role}': {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def _filter_by_threshold(
        self,
        results: Dict[str, Any],
        threshold: float
    ) -> Dict[str, Any]:
        """Filter results by similarity threshold"""
        filtered_results = {
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        if not results.get("distances") or not results["distances"][0]:
            return filtered_results
        
        distances = results["distances"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        
        for i, distance in enumerate(distances):
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - distance
            if similarity >= threshold:
                filtered_results["documents"].append(documents[i] if i < len(documents) else "")
                filtered_results["metadatas"].append(metadatas[i] if i < len(metadatas) else {})
                filtered_results["distances"].append(distance)
        
        return filtered_results

class FAISSManager:
    """Manager for FAISS vector database operations"""
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize FAISS manager
        
        Args:
            config: Vector database configuration
        """
        self.config = config
        self.faiss_config = config.faiss_config
        
        # Initialize FAISS components
        self.index_path = Path(self.faiss_config["index_path"])
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = self.faiss_config["embedding_dimension"]
        self.index = self._initialize_index()
        self.metadata_store = {}
        self.id_mapping = {}
        
        # Load existing data if available
        self._load_existing_data()
    
    def _initialize_index(self) -> faiss.Index:
        """Initialize FAISS index"""
        try:
            nlist = self.faiss_config["nlist"]
            
            # Create quantizer
            quantizer = faiss.IndexFlatIP(self.dimension)
            
            # Create IVF index
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
            logger.info(f"Initialized FAISS IVF index with dimension {self.dimension}")
            return index
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _load_existing_data(self):
        """Load existing FAISS index and metadata"""
        try:
            index_file = self.index_path / "index.faiss"
            metadata_file = self.index_path / "metadata.pkl"
            mapping_file = self.index_path / "id_mapping.json"
            
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata_store)} documents")
            
            if mapping_file.exists():
                with open(mapping_file, 'r') as f:
                    self.id_mapping = json.load(f)
                logger.info(f"Loaded ID mappings for {len(self.id_mapping)} documents")
                    
        except Exception as e:
            logger.warning(f"Could not load existing FAISS data: {e}")
    
    def add_vectors(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """
        Add vectors to FAISS index
        
        Args:
            embeddings: Embedding vectors as numpy array
            metadatas: List of metadata dictionaries
            ids: List of document IDs
        """
        try:
            # Normalize embeddings for inner product similarity
            faiss.normalize_L2(embeddings)
            
            # Train index if not already trained
            if not self.index.is_trained:
                self.index.train(embeddings)
                logger.info("FAISS index training completed")
            
            # Add vectors to index
            start_id = self.index.ntotal
            self.index.add(embeddings)
            
            # Store metadata and ID mappings
            for i, (metadata, doc_id) in enumerate(zip(metadatas, ids)):
                faiss_id = start_id + i
                self.metadata_store[faiss_id] = metadata
                self.id_mapping[doc_id] = faiss_id
            
            logger.info(f"Added {len(embeddings)} vectors to FAISS index")
            
            # Save updated index and metadata
            self._save_data()
            
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {e}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        role_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors in FAISS index
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar vectors to retrieve
            role_filter: Optional rhetorical role filter
            
        Returns:
            List of search results with metadata
        """
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            scores, indices = self.index.search(query_embedding, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:  # Valid index
                    metadata = self.metadata_store.get(idx, {})
                    
                    # Apply role filter if specified
                    if role_filter and metadata.get("rhetorical_role") != role_filter:
                        continue
                    
                    result = {
                        "index": int(idx),
                        "score": float(score),
                        "metadata": metadata
                    }
                    results.append(result)
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {e}")
            return []
    
    def _save_data(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save index
            index_file = self.index_path / "index.faiss"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            # Save ID mappings
            mapping_file = self.index_path / "id_mapping.json"
            with open(mapping_file, 'w') as f:
                json.dump(self.id_mapping, f)
            
            logger.info("FAISS data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS data: {e}")

class VectorDBOrchestrator:
    """Orchestrator for managing multiple vector database systems"""
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize vector database orchestrator
        
        Args:
            config: Vector database configuration
        """
        self.config = config
        
        # Initialize database managers
        self.chroma_manager = ChromaDBManager(config)
        self.faiss_manager = FAISSManager(config)
        
        # Default primary database
        self.primary_db = "chromadb"  # or "faiss"
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ):
        """Add documents to all vector databases"""
        try:
            # Add to ChromaDB
            self.chroma_manager.add_documents(documents, embeddings, metadatas, ids)
            
            # Add to FAISS
            embeddings_array = np.array(embeddings).astype('float32')
            self.faiss_manager.add_vectors(embeddings_array, metadatas, ids)
            
            logger.info(f"Added {len(documents)} documents to all vector databases")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector databases: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        role: Optional[str] = None,
        n_results: int = 5,
        use_faiss: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Query vector databases for similar documents
        
        Args:
            query_embedding: Query embedding vector
            role: Optional rhetorical role filter
            n_results: Number of results to return
            use_faiss: Whether to use FAISS instead of ChromaDB
            
        Returns:
            List of similar documents with metadata
        """
        try:
            if use_faiss:
                # Use FAISS for search
                query_array = np.array([query_embedding]).astype('float32')
                results = self.faiss_manager.search(
                    query_array[0], 
                    k=n_results, 
                    role_filter=role
                )
                return results
            else:
                # Use ChromaDB for search
                if role:
                    results = self.chroma_manager.query_by_role(
                        query_embedding, 
                        role, 
                        n_results
                    )
                else:
                    # Query main collection
                    results = self.chroma_manager.collections["main"].query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                
                # Format results
                formatted_results = []
                if results.get("documents") and results["documents"][0]:
                    documents = results["documents"][0]
                    metadatas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0]
                    
                    for i, doc in enumerate(documents):
                        formatted_results.append({
                            "document": doc,
                            "metadata": metadatas[i] if i < len(metadatas) else {},
                            "distance": distances[i] if i < len(distances) else 0.0
                        })
                
                return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to query vector databases: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all vector databases"""
        stats = {
            "chromadb": {
                "total_collections": len(self.chroma_manager.collections),
                "collections": {}
            },
            "faiss": {
                "total_vectors": int(self.faiss_manager.index.ntotal),
                "index_trained": bool(self.faiss_manager.index.is_trained),
                "dimension": self.faiss_manager.dimension
            }
        }
        
        # Get ChromaDB collection stats
        for name, collection in self.chroma_manager.collections.items():
            try:
                count = collection.count()
                stats["chromadb"]["collections"][name] = count
            except:
                stats["chromadb"]["collections"][name] = "Unknown"
        
        return stats

# Factory function for easy initialization
def create_vector_db_orchestrator(data_dir: str = "data/vector_db") -> VectorDBOrchestrator:
    """
    Create and initialize vector database orchestrator
    
    Args:
        data_dir: Directory for storing vector database files
        
    Returns:
        Initialized vector database orchestrator
    """
    config = VectorDBConfig(data_dir)
    orchestrator = VectorDBOrchestrator(config)
    return orchestrator