"""
Clustering-Enhanced RAG System for Legal Documents

This module integrates unsupervised clustering with RAG to provide:
1. Cluster-based document organization in vector DB
2. Hierarchical retrieval (cluster-level then document-level)
3. Automatic role discovery without labels
4. Cluster-aware semantic search

Benefits over traditional RAG:
- No labeled data required
- Discovers natural document structure
- Faster retrieval through cluster pruning
- Better interpretability
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import umap

from langchain_community.embeddings import VertexAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from pinecone import Pinecone
import os
from dotenv import load_dotenv

from ..models.clustering_role_classifier import ClusteringRoleClassifier, ClusteringConfig
from ..config.pinecone_setup import PineconeIndexManager

load_dotenv()

logger = logging.getLogger(__name__)


class ClusteringRAGSystem:
    """
    RAG system enhanced with unsupervised clustering
    
    Architecture:
    1. Documents → Embeddings → Clusters
    2. Store in Pinecone with cluster metadata
    3. Query → Find relevant clusters → Search within clusters
    4. Generate answer using cluster-aware context
    """
    
    def __init__(self,
                 embedding_model: str = "text-embedding-005",
                 clustering_config: Optional[ClusteringConfig] = None,
                 base_index_name: str = None,
                 num_clusters: int = 7,
                 device: str = "cpu"):
        """
        Initialize Clustering-Enhanced RAG System
        
        Args:
            embedding_model: Embedding model for RAG
            clustering_config: Configuration for clustering
            base_index_name: Base name for Pinecone indexes
            num_clusters: Number of clusters to discover
            device: Device for computation
        """
        self.device = device
        self.num_clusters = num_clusters
        
        # Initialize embeddings (for RAG)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embeddings = VertexAIEmbeddings(model_name=embedding_model)
        
        # Initialize clustering classifier (for role discovery)
        if clustering_config is None:
            clustering_config = ClusteringConfig(
                num_clusters=num_clusters,
                clustering_algorithm="kmeans",
                embedding_model="all-MiniLM-L6-v2"  # Separate model for clustering
            )
        
        logger.info("Initializing clustering classifier...")
        self.clustering_classifier = ClusteringRoleClassifier(clustering_config)
        
        # Initialize Pinecone
        self.pinecone_manager = PineconeIndexManager(base_index_name=base_index_name)
        self.pc = Pinecone(api_key=self.pinecone_manager.api_key)
        
        # Use cluster-based index
        self.cluster_index_name = f"{self.pinecone_manager.base_index_name}-clusters"
        self._ensure_cluster_index_exists()
        
        self.cluster_index = self.pc.Index(self.cluster_index_name)
        
        # Initialize LLM for generation
        self.llm = ChatVertexAI(
            temperature=0.1,
            model_name="gemini-2.5-flash",
            max_tokens=2048
        )
        
        # Cluster statistics
        self.cluster_stats = {}
        self.cluster_centroids = None
        
        logger.info("Clustering RAG System initialized successfully")
    
    def _ensure_cluster_index_exists(self):
        """Ensure cluster-based Pinecone index exists"""
        if not self.pinecone_manager.index_exists(self.cluster_index_name):
            logger.info(f"Creating cluster index: {self.cluster_index_name}")
            self.pinecone_manager.create_index(
                index_name=self.cluster_index_name,
                dimension=768,  # Vertex AI embedding dimension
                metric="cosine"
            )
    
    def process_and_index_documents(self, 
                                   documents: List[str],
                                   metadata_list: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Process documents with clustering and index in Pinecone
        
        Steps:
        1. Split into sentences
        2. Cluster sentences to discover roles
        3. Generate embeddings for RAG
        4. Store in Pinecone with cluster metadata
        
        Args:
            documents: List of document texts
            metadata_list: Optional metadata for each document
            
        Returns:
            Indexing statistics
        """
        logger.info(f"Processing {len(documents)} documents with clustering...")
        
        # Step 1: Extract all sentences from all documents
        all_sentences = []
        sentence_metadata = []
        
        for doc_idx, doc_text in enumerate(documents):
            sentences = self.clustering_classifier.preprocess_document(doc_text)
            
            doc_metadata = metadata_list[doc_idx] if metadata_list else {}
            
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                sentence_metadata.append({
                    "document_index": doc_idx,
                    "sentence_index": sent_idx,
                    **doc_metadata
                })
        
        logger.info(f"Extracted {len(all_sentences)} sentences from {len(documents)} documents")
        
        # Step 2: Cluster sentences to discover roles
        logger.info("Clustering sentences to discover rhetorical roles...")
        clustering_results = self.clustering_classifier.fit(all_sentences)
        
        logger.info(f"Discovered {clustering_results['num_clusters']} clusters")
        logger.info(f"Cluster-to-role mapping: {self.clustering_classifier.cluster_to_role}")
        
        # Step 3: Generate embeddings for RAG (using different model)
        logger.info("Generating embeddings for vector search...")
        embeddings = self.embeddings.embed_documents(all_sentences)
        
        # Step 4: Prepare vectors for Pinecone with cluster metadata
        vectors_to_upsert = []
        
        cluster_predictions = self.clustering_classifier.predict(all_sentences)
        
        for i, (sentence, embedding, cluster_result, sent_meta) in enumerate(
            zip(all_sentences, embeddings, cluster_predictions, sentence_metadata)
        ):
            vector_id = f"doc_{sent_meta['document_index']}_sent_{sent_meta['sentence_index']}_{uuid.uuid4().hex[:8]}"
            
            vector_data = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "text": sentence,
                    "cluster_id": cluster_result.cluster_id,
                    "discovered_role": cluster_result.predicted_role,
                    "cluster_confidence": cluster_result.confidence,
                    "sentence_index": sent_meta["sentence_index"],
                    "document_index": sent_meta["document_index"],
                    **sent_meta
                }
            }
            
            vectors_to_upsert.append(vector_data)
        
        # Step 5: Batch upsert to Pinecone
        logger.info(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        batch_size = 100
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.cluster_index.upsert(vectors=batch)
        
        # Step 6: Calculate and store cluster statistics
        self._calculate_cluster_stats(cluster_predictions)
        
        # Step 7: Store cluster centroids for hierarchical search
        self._store_cluster_centroids(embeddings, cluster_predictions)
        
        stats = {
            "total_documents": len(documents),
            "total_sentences": len(all_sentences),
            "num_clusters": clustering_results['num_clusters'],
            "cluster_to_role": self.clustering_classifier.cluster_to_role,
            "cluster_sizes": clustering_results['cluster_sizes'],
            "clustering_metrics": {
                k: v for k, v in clustering_results.items() 
                if k not in ['cluster_sizes', 'num_clusters']
            },
            "vectors_indexed": len(vectors_to_upsert)
        }
        
        logger.info("Indexing completed successfully!")
        return stats
    
    def _calculate_cluster_stats(self, cluster_results: List):
        """Calculate statistics for each cluster"""
        from collections import Counter
        
        cluster_ids = [r.cluster_id for r in cluster_results]
        roles = [r.predicted_role for r in cluster_results]
        confidences = [r.confidence for r in cluster_results]
        
        cluster_counter = Counter(cluster_ids)
        
        self.cluster_stats = {}
        for cluster_id in set(cluster_ids):
            mask = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
            
            self.cluster_stats[cluster_id] = {
                "size": cluster_counter[cluster_id],
                "role": cluster_results[mask[0]].predicted_role,
                "avg_confidence": np.mean([confidences[i] for i in mask]),
                "percentage": cluster_counter[cluster_id] / len(cluster_ids) * 100
            }
        
        logger.info(f"Cluster statistics calculated for {len(self.cluster_stats)} clusters")
    
    def _store_cluster_centroids(self, embeddings: List[List[float]], 
                                 cluster_results: List):
        """Store cluster centroids for hierarchical search"""
        cluster_ids = [r.cluster_id for r in cluster_results]
        embeddings_array = np.array(embeddings)
        
        unique_clusters = set(cluster_ids)
        self.cluster_centroids = {}
        
        for cluster_id in unique_clusters:
            mask = np.array([c == cluster_id for c in cluster_ids])
            cluster_embeddings = embeddings_array[mask]
            centroid = cluster_embeddings.mean(axis=0)
            
            self.cluster_centroids[cluster_id] = centroid.tolist()
        
        logger.info(f"Stored centroids for {len(self.cluster_centroids)} clusters")
    
    def hierarchical_search(self, 
                          query: str,
                          top_k_clusters: int = 3,
                          top_k_per_cluster: int = 5,
                          role_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Hierarchical search: First find relevant clusters, then search within them
        
        This is MORE EFFICIENT than flat search for large datasets
        
        Args:
            query: Query text
            top_k_clusters: Number of most relevant clusters to search
            top_k_per_cluster: Documents to retrieve per cluster
            role_filter: Optional role filter (discovered role name)
            
        Returns:
            Search results with cluster information
        """
        logger.info(f"Hierarchical search for: {query}")
        
        # Step 1: Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Step 2: Find most relevant clusters (using centroids)
        if self.cluster_centroids:
            logger.info("Finding relevant clusters using centroids...")
            cluster_similarities = {}
            
            for cluster_id, centroid in self.cluster_centroids.items():
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, centroid) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(centroid)
                )
                
                # Apply role filter if specified
                if role_filter:
                    cluster_role = self.cluster_stats.get(cluster_id, {}).get("role")
                    if cluster_role != role_filter:
                        continue
                
                cluster_similarities[cluster_id] = similarity
            
            # Get top-k clusters
            sorted_clusters = sorted(
                cluster_similarities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k_clusters]
            
            relevant_cluster_ids = [c[0] for c in sorted_clusters]
            
            logger.info(f"Selected clusters: {relevant_cluster_ids}")
        else:
            # Fallback: search all clusters
            relevant_cluster_ids = None
        
        # Step 3: Search within selected clusters
        results = []
        
        if relevant_cluster_ids:
            for cluster_id in relevant_cluster_ids:
                cluster_results = self.cluster_index.query(
                    vector=query_embedding,
                    filter={"cluster_id": int(cluster_id)},
                    top_k=top_k_per_cluster,
                    include_metadata=True
                )
                
                for match in cluster_results['matches']:
                    results.append({
                        "text": match['metadata']['text'],
                        "score": match['score'],
                        "cluster_id": match['metadata']['cluster_id'],
                        "discovered_role": match['metadata']['discovered_role'],
                        "cluster_confidence": match['metadata']['cluster_confidence'],
                        "metadata": match['metadata']
                    })
        else:
            # Flat search as fallback
            search_results = self.cluster_index.query(
                vector=query_embedding,
                top_k=top_k_clusters * top_k_per_cluster,
                include_metadata=True
            )
            
            for match in search_results['matches']:
                results.append({
                    "text": match['metadata']['text'],
                    "score": match['score'],
                    "cluster_id": match['metadata']['cluster_id'],
                    "discovered_role": match['metadata']['discovered_role'],
                    "cluster_confidence": match['metadata']['cluster_confidence'],
                    "metadata": match['metadata']
                })
        
        # Step 4: Re-rank results globally
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return {
            "results": results[:top_k_clusters * top_k_per_cluster],
            "query": query,
            "clusters_searched": relevant_cluster_ids or "all",
            "total_results": len(results)
        }
    
    def query_with_clustering(self,
                             query: str,
                             role_filter: Optional[str] = None,
                             top_k: int = 10,
                             use_hierarchical: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system with cluster-aware retrieval
        
        Args:
            query: User query
            role_filter: Filter by discovered role (e.g., "Facts", "Reasoning")
            top_k: Number of results to return
            use_hierarchical: Use hierarchical search (faster for large datasets)
            
        Returns:
            Generated answer with sources and cluster information
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        if use_hierarchical and self.cluster_centroids:
            search_results = self.hierarchical_search(
                query=query,
                top_k_clusters=3,
                top_k_per_cluster=max(3, top_k // 3),
                role_filter=role_filter
            )
            results = search_results["results"]
        else:
            # Flat search
            query_embedding = self.embeddings.embed_query(query)
            
            filter_dict = {"discovered_role": role_filter} if role_filter else None
            
            search_results = self.cluster_index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            results = [
                {
                    "text": match['metadata']['text'],
                    "score": match['score'],
                    "cluster_id": match['metadata']['cluster_id'],
                    "discovered_role": match['metadata']['discovered_role'],
                    "cluster_confidence": match['metadata']['cluster_confidence'],
                    "metadata": match['metadata']
                }
                for match in search_results['matches']
            ]
        
        # Step 2: Organize by discovered roles
        role_organized = {}
        for result in results:
            role = result['discovered_role']
            if role not in role_organized:
                role_organized[role] = []
            role_organized[role].append(result)
        
        # Step 3: Build context for LLM
        context_parts = []
        for role, role_results in role_organized.items():
            context_parts.append(f"\n## {role}:")
            for i, result in enumerate(role_results[:3], 1):  # Top 3 per role
                context_parts.append(f"{i}. {result['text']}")
        
        context = "\n".join(context_parts)
        
        # Step 4: Generate answer with cluster-aware prompt
        prompt = f"""
Based on the following legal document excerpts organized by discovered rhetorical roles:

{context}

Please answer the following question:
{query}

Provide a comprehensive answer that:
1. Directly addresses the question
2. References specific roles when relevant
3. Maintains legal accuracy
4. Is clear and concise

Answer:
"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "answer": response.content,
            "query": query,
            "sources": results[:top_k],
            "role_breakdown": {
                role: len(results) for role, results in role_organized.items()
            },
            "clusters_used": list(set(r['cluster_id'] for r in results)),
            "total_sources": len(results),
            "search_method": "hierarchical" if use_hierarchical else "flat"
        }
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of discovered clusters and their roles"""
        summary = {
            "num_clusters": len(self.cluster_stats),
            "clusters": []
        }
        
        for cluster_id, stats in sorted(self.cluster_stats.items()):
            summary["clusters"].append({
                "cluster_id": cluster_id,
                "discovered_role": stats["role"],
                "size": stats["size"],
                "percentage": f"{stats['percentage']:.2f}%",
                "avg_confidence": f"{stats['avg_confidence']:.3f}"
            })
        
        return summary
    
    def visualize_clusters(self, output_path: str = "cluster_visualization.png"):
        """
        Visualize clusters in 2D space
        
        Requires: matplotlib, seaborn
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Get all embeddings and cluster assignments from index
            # Note: This is a simplified version - in production you'd store this separately
            logger.info("Generating cluster visualization...")
            
            # For now, just show the structure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            cluster_ids = list(self.cluster_stats.keys())
            cluster_sizes = [self.cluster_stats[c]["size"] for c in cluster_ids]
            cluster_roles = [self.cluster_stats[c]["role"] for c in cluster_ids]
            
            colors = sns.color_palette("husl", len(cluster_ids))
            
            ax.bar(cluster_ids, cluster_sizes, color=colors)
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Number of Sentences")
            ax.set_title("Discovered Clusters and Their Sizes")
            
            # Add role labels
            for i, (cluster_id, role) in enumerate(zip(cluster_ids, cluster_roles)):
                ax.text(cluster_id, cluster_sizes[i], role, 
                       ha='center', va='bottom', rotation=45, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization")


# Example usage
if __name__ == "__main__":
    # Initialize clustering RAG
    rag_system = ClusteringRAGSystem(
        embedding_model="text-embedding-005",
        num_clusters=7
    )
    
    # Sample documents
    sample_docs = [
        """
        The petitioner filed a writ petition challenging Section 377.
        The appellant contends that the provision violates fundamental rights.
        The main issue is whether Section 377 violates Articles 14, 15, 19 and 21.
        The petitioner argues that Section 377 is discriminatory.
        The respondent contends that Section 377 is constitutionally valid.
        The court has examined the constitutional provisions.
        The court finds that Section 377 infringes upon the right to privacy.
        Therefore, Section 377 is hereby declared unconstitutional.
        """
    ]
    
    # Process and index with clustering
    stats = rag_system.process_and_index_documents(sample_docs)
    
    print("\n" + "="*80)
    print("INDEXING STATISTICS")
    print("="*80)
    print(f"Total sentences indexed: {stats['total_sentences']}")
    print(f"Clusters discovered: {stats['num_clusters']}")
    print(f"Cluster-to-role mapping: {stats['cluster_to_role']}")
    
    # Get cluster summary
    summary = rag_system.get_cluster_summary()
    print("\n" + "="*80)
    print("CLUSTER SUMMARY")
    print("="*80)
    for cluster in summary["clusters"]:
        print(f"Cluster {cluster['cluster_id']}: {cluster['discovered_role']}")
        print(f"  Size: {cluster['size']} ({cluster['percentage']})")
        print(f"  Confidence: {cluster['avg_confidence']}")
    
    # Query the system
    query = "What are the facts of the case?"
    result = rag_system.query_with_clustering(query, use_hierarchical=True)
    
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    print(f"Answer: {result['answer']}")
    print(f"\nRole breakdown: {result['role_breakdown']}")
    print(f"Clusters used: {result['clusters_used']}")
    print(f"Search method: {result['search_method']}")
