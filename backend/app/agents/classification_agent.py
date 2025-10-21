"""
Classification Agent

Handles document upload, classification, embedding, and storage.
This agent is triggered when user uploads a document.
"""

import logging
from typing import Dict, Any
from ..models.schemas import AgentState, Intent, RhetoricalRole
from ..services.preprocessing import DocumentPreprocessor
from ..services.classification_service import ClassificationService
from ..services.embedding_service import EmbeddingService
from ..services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


async def classification_agent_node(
    state: AgentState,
    preprocessor: DocumentPreprocessor,
    classifier: ClassificationService,
    embedder: EmbeddingService,
    pinecone: PineconeService
) -> Dict[str, Any]:
    """
    Classification Agent: Upload → Classify → Embed → Store
    
    Flow:
    1. Extract text from uploaded document (PDF or TXT)
    2. Split into sentences
    3. Classify each sentence into rhetorical roles
    4. Generate embeddings for each sentence
    5. Upload to Pinecone with role metadata
    
    Args:
        state: Current agent state
        preprocessor: Document preprocessing service
        classifier: Role classification service
        embedder: Embedding service
        pinecone: Pinecone storage service
    
    Returns:
        Updated state dict
    """
    logger.info("🏛️  Classification Agent: Starting document processing...")
    
    try:
        # Get file path from state
        file_path = state.get("file_path")
        case_id = state.get("case_id")
        
        if not file_path or not case_id:
            raise ValueError("file_path and case_id required for classification")
        
        # Step 1: Extract text
        logger.info(f"📄 Extracting text from: {file_path}")
        
        if file_path.endswith(".pdf"):
            text = preprocessor.extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            # Check if it's training format (sentence\trole)
            try:
                documents = preprocessor.parse_training_format(file_path)
                # If successful, this is pre-labeled data
                if documents:
                    logger.info(f"✅ Detected pre-labeled training format")
                    # Extract just the sentences
                    text = "\n".join([
                        "\n".join([sent for sent, _ in doc])
                        for doc in documents
                    ])
                else:
                    # Not training format, treat as plain text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
            except:
                # Not training format, treat as plain text
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"✅ Extracted {len(text)} characters")
        
        # Step 2: Split into sentences
        logger.info(f"✂️  Splitting into sentences...")
        sentences = preprocessor.split_into_sentences(text)
        logger.info(f"✅ Split into {len(sentences)} sentences")
        
        # Step 3: Classify sentences
        logger.info(f"🔍 Classifying sentences into rhetorical roles...")
        classifications = classifier.classify_document(sentences, batch_size=16)
        
        # Get role distribution
        distribution = classifier.get_role_distribution(classifications)
        logger.info(f"✅ Classification complete!")
        for role, stats in distribution.items():
            if stats['count'] > 0:
                logger.info(f"  - {role}: {stats['count']} ({stats['percentage']:.1f}%)")
        
        # Step 4: Generate embeddings
        logger.info(f"🔄 Generating embeddings...")
        roles = [c.role for c in classifications]
        case_ids = [case_id] * len(sentences)
        embeddings = embedder.encode_documents(sentences, case_ids, normalize=True)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        
        # Step 5: Upload to Pinecone
        logger.info(f"⬆️  Uploading to Pinecone...")
        
        # Convert embeddings to list format
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Add confidence scores to metadata
        additional_metadata = [
            {"confidence": c.confidence}
            for c in classifications
        ]
        
        result = pinecone.upsert_vectors(
            vectors=embeddings_list,
            texts=sentences,
            roles=roles,
            case_id=case_id,
            namespace="user_documents",
            additional_metadata=additional_metadata
        )
        
        logger.info(f"✅ Upload complete: {result['upserted_count']} vectors")
        
        # Update state
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"✅ Document processed successfully!\n\n"
                          f"**Case ID:** `{case_id}`\n"
                          f"**Sentences:** {len(sentences)}\n"
                          f"**Distribution:**\n" +
                          "\n".join([
                              f"- **{role}**: {stats['count']} ({stats['percentage']:.1f}%)"
                              for role, stats in distribution.items()
                              if stats['count'] > 0
                          ])
            }],
            "classification_result": {
                "case_id": case_id,
                "sentence_count": len(sentences),
                "distribution": distribution,
                "pinecone_status": result
            },
            "final_answer": "Document classification and upload complete"
        }
        
    except Exception as e:
        logger.error(f"❌ Classification agent error: {e}", exc_info=True)
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"❌ Error processing document: {str(e)}"
            }],
            "final_answer": f"Error: {str(e)}"
        }
