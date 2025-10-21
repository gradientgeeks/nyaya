"""
Prediction Agent

Predicts case outcomes based on similar precedents.
This is a placeholder for ML-based prediction (future work).
"""

import logging
from typing import Dict, Any
from ..models.schemas import AgentState, PredictionResult
from ..services.embedding_service import EmbeddingService
from ..services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


async def prediction_agent_node(
    state: AgentState,
    embedder: EmbeddingService,
    pinecone: PineconeService
) -> Dict[str, Any]:
    """
    Prediction Agent: Predict case outcome based on precedents.
    
    Current implementation: Rule-based prediction using similar cases.
    Future: ML model trained on historical outcomes.
    
    Flow:
    1. Find similar cases (especially Decision + Reasoning)
    2. Extract outcomes from similar cases
    3. Compute confidence based on precedent alignment
    4. Return prediction with explanation
    
    Args:
        state: Current agent state
        embedder: Embedding service
        pinecone: Pinecone storage service
    
    Returns:
        Updated state dict
    """
    logger.info("🔮 Prediction Agent: Predicting case outcome...")
    
    try:
        query_text = state.get("user_query", "")
        case_id = state.get("case_id")
        
        # Encode query
        query_embedding = embedder.encode_query(query_text, normalize=True)
        
        # Find similar cases focusing on Decision and Reasoning
        logger.info("🔍 Finding similar precedents...")
        
        # Query Decision role (final rulings)
        decision_matches = pinecone.query_vectors(
            query_vector=query_embedding.tolist(),
            top_k=10,
            role_filter="Decision",
            namespace="user_documents",
            min_score=0.6  # Higher threshold for predictions
        )
        
        # Query Reasoning role (legal analysis)
        reasoning_matches = pinecone.query_vectors(
            query_vector=query_embedding.tolist(),
            top_k=10,
            role_filter="Reasoning",
            namespace="user_documents",
            min_score=0.6
        )
        
        logger.info(
            f"  Found {len(decision_matches)} decision matches, "
            f"{len(reasoning_matches)} reasoning matches"
        )
        
        # Simple outcome prediction based on precedents
        # (In production, this would use a trained ML model)
        
        if not decision_matches:
            response_text = (
                "⚠️ Insufficient precedent data for reliable prediction.\n\n"
                "I need more similar cases in the database to make a confident prediction."
            )
            
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": response_text
                }],
                "prediction": None,
                "final_answer": response_text
            }
        
        # Analyze top matches (simple heuristic)
        favorable_count = 0
        unfavorable_count = 0
        
        for match in decision_matches[:5]:
            text = match["text"].lower()
            
            # Simple keyword analysis (placeholder for ML model)
            favorable_keywords = [
                "allowed", "granted", "favour", "accepted", 
                "upheld", "sustained", "success"
            ]
            unfavorable_keywords = [
                "dismissed", "rejected", "denied", "failed",
                "overruled", "quashed"
            ]
            
            if any(kw in text for kw in favorable_keywords):
                favorable_count += 1
            elif any(kw in text for kw in unfavorable_keywords):
                unfavorable_count += 1
        
        # Calculate confidence
        total = favorable_count + unfavorable_count
        if total == 0:
            confidence = 0.5
            outcome = "Uncertain"
        else:
            if favorable_count > unfavorable_count:
                confidence = favorable_count / total
                outcome = "Favorable"
            else:
                confidence = unfavorable_count / total
                outcome = "Unfavorable"
        
        # Adjust confidence based on data quality
        if len(decision_matches) < 3:
            confidence *= 0.7  # Reduce confidence with limited data
        
        # Generate prediction result
        prediction = PredictionResult(
            predicted_outcome=outcome,
            confidence=float(confidence),
            supporting_cases=[m["case_id"] for m in decision_matches[:5]],
            key_factors=[
                f"Based on {len(decision_matches)} similar precedents",
                f"{favorable_count} favorable decisions vs {unfavorable_count} unfavorable",
                "Note: This is a preliminary analysis"
            ]
        )
        
        # Format response
        response_text = (
            f"🔮 **Outcome Prediction**\n\n"
            f"**Predicted Outcome:** {prediction.predicted_outcome}\n"
            f"**Confidence:** {prediction.confidence:.1%}\n\n"
            f"**Analysis:**\n"
            f"- Found {len(decision_matches)} similar precedents\n"
            f"- {favorable_count} cases had favorable outcomes\n"
            f"- {unfavorable_count} cases had unfavorable outcomes\n\n"
            f"**Supporting Cases:** {', '.join(prediction.supporting_cases)}\n\n"
            f"⚠️ **Disclaimer:** This is an AI-generated prediction based on "
            f"similar precedents. It should not be considered legal advice. "
            f"Consult with a qualified legal professional for case-specific guidance."
        )
        
        logger.info(
            f"✅ Prediction complete: {outcome} ({confidence:.1%} confidence)"
        )
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": response_text
            }],
            "prediction": prediction,
            "final_answer": response_text
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction agent error: {e}", exc_info=True)
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"❌ Error predicting outcome: {str(e)}"
            }],
            "final_answer": f"Error: {str(e)}"
        }
