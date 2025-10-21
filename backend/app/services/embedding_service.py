"""
Embedding Service

Handles text embedding using EmbeddingGemma with asymmetric encoding.
Follows the official Gemma Cookbook pattern from official_rag_pattern.py.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from ..core.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wrapper for EmbeddingGemma model with asymmetric encoding.
    
    Key features:
    - 384-dim embeddings using Matryoshka Representation Learning (MRL)
    - Asymmetric prompts: "Retrieval-document" for docs, "Retrieval-query" for queries
    - Normalized embeddings for cosine similarity
    - Format: "title: {case_id} | text: {content}"
    
    This is the OFFICIAL PATTERN from Gemma Cookbook.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load EmbeddingGemma model with 384-dim truncation.
        
        CRITICAL: Requires Hugging Face authentication.
        Set HF_TOKEN in .env or run `huggingface-cli login`.
        """
        try:
            logger.info(f"📥 Loading embedding model: {self.settings.embedding_model}")
            
            self.model = SentenceTransformer(
                self.settings.embedding_model,
                truncate_dim=self.settings.embedding_dimension  # 384-dim MRL
            )
            
            # Verify dimension
            actual_dim = self.model.get_sentence_embedding_dimension()
            if actual_dim != self.settings.embedding_dimension:
                raise ValueError(
                    f"Model dimension mismatch: expected {self.settings.embedding_dimension}, "
                    f"got {actual_dim}"
                )
            
            logger.info(
                f"✅ Model loaded! Embedding dimension: {actual_dim}"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            logger.error(
                "💡 Tip: Make sure you've accepted the license at "
                "https://huggingface.co/google/embeddinggemma-300M "
                "and set HF_TOKEN in .env"
            )
            raise
    
    def encode_documents(
        self,
        texts: List[str],
        case_ids: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode documents with "Retrieval-document" prompt.
        
        CRITICAL: Documents use DIFFERENT prompt than queries (asymmetric encoding).
        This improves retrieval accuracy by +16.7% according to Gemma paper.
        
        Args:
            texts: List of sentence texts
            case_ids: List of case IDs (for title field)
            normalize: Whether to normalize embeddings (default: True for cosine similarity)
            show_progress: Show progress bar
        
        Returns:
            numpy array of shape (n_texts, 384)
        """
        if len(texts) != len(case_ids):
            raise ValueError("texts and case_ids must have same length")
        
        # Format: "title: {case_id} | text: {content}"
        formatted_texts = [
            f"title: {case_id} | text: {text}"
            for case_id, text in zip(case_ids, texts)
        ]
        
        logger.info(f"🔄 Encoding {len(texts)} documents...")
        
        embeddings = self.model.encode(
            formatted_texts,
            prompt_name="Retrieval-document",  # Critical: document-specific prompt
            normalize_embeddings=normalize,
            show_progress_bar=show_progress
        )
        
        logger.info(f"✅ Encoded {len(embeddings)} documents")
        
        return embeddings
    
    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode query with "Retrieval-query" prompt.
        
        CRITICAL: Queries use DIFFERENT prompt than documents (asymmetric encoding).
        
        Args:
            query: Query text
            normalize: Whether to normalize embedding (default: True for cosine similarity)
        
        Returns:
            numpy array of shape (384,)
        """
        logger.info(f"🔄 Encoding query: '{query[:50]}...'")
        
        embedding = self.model.encode(
            query,
            prompt_name="Retrieval-query",  # Critical: query-specific prompt
            normalize_embeddings=normalize
        )
        
        logger.info(f"✅ Query encoded")
        
        return embedding
    
    def batch_encode_documents(
        self,
        texts: List[str],
        case_ids: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> List[np.ndarray]:
        """
        Encode documents in batches (for large document collections).
        
        Args:
            texts: List of sentence texts
            case_ids: List of case IDs
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        
        Returns:
            List of numpy arrays (one per batch)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_case_ids = case_ids[i:i + batch_size]
            
            embeddings = self.encode_documents(
                batch_texts,
                batch_case_ids,
                normalize=normalize,
                show_progress=False
            )
            
            all_embeddings.append(embeddings)
            
            logger.info(
                f"📊 Batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} complete"
            )
        
        # Concatenate all batches
        return np.vstack(all_embeddings)
