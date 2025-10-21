"""
Similarity Search Agent

Finds similar cases using vector similarity and role-weighted scoring.
"""

import logging
from typing import Dict, Any, List
from ..models.schemas import AgentState, RhetoricalRole, SimilarCase
from ..services.embedding_service import EmbeddingService
from ..services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


async def similarity_agent_node(
    state: AgentState,
    embedder: EmbeddingService,
    pinecone: PineconeService,
    role_weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Similarity Agent: Find similar cases using role-weighted retrieval.
    
    Flow:
    1. Get query case (either from file or case_id)
    2. For each role, query Pinecone with role filter
    3. Combine results with role-specific weights
    4. Rank cases by weighted similarity
    
    Args:
        state: Current agent state
        embedder: Embedding service
        pinecone: Pinecone storage service
        role_weights: Role-specific similarity weights from config
    
    Returns:
        Updated state dict
    """
    logger.info("🔍 Similarity Agent: Finding similar cases...")
    
    try:
        query_text = state.get("user_query", "")
        case_id = state.get("case_id")  # Optional: compare specific case
        top_k = state.get("top_k", 5)
        
        # Encode query
        query_embedding = embedder.encode_query(query_text, normalize=True)
        
        # Query each role separately with role-specific weights
        all_results = {}
        
        for role in RhetoricalRole:
            if role == RhetoricalRole.NONE:
                continue  # Skip "None" role
            
            weight = role_weights.get(role.value, 0.0)
            if weight == 0:
                continue
            
            logger.info(f"🔍 Querying {role.value} (weight: {weight})...")
            
            matches = pinecone.query_vectors(
                query_vector=query_embedding.tolist(),
                top_k=top_k * 2,  # Get more to allow for deduplication
                role_filter=role,
                namespace="user_documents",
                min_score=0.5  # Only decent matches
            )
            
            all_results[role.value] = matches
            logger.info(f"  Found {len(matches)} matches")
        
        # Combine and weight results by case
        case_scores = {}
        case_details = {}
        
        for role, matches in all_results.items():
            weight = role_weights.get(role, 0.0)
            
            for match in matches:
                case_id_match = match["case_id"]
                
                # Skip self-matches if comparing specific case
                if case_id and case_id_match == case_id:
                    continue
                
                # Weighted score
                weighted_score = match["score"] * weight
                
                if case_id_match not in case_scores:
                    case_scores[case_id_match] = 0.0
                    case_details[case_id_match] = {
                        "role_matches": {},
                        "total_matches": 0
                    }
                
                case_scores[case_id_match] += weighted_score
                case_details[case_id_match]["role_matches"][role] = {
                    "score": match["score"],
                    "weighted_score": weighted_score,
                    "sample_text": match["text"][:150] + "..."
                }
                case_details[case_id_match]["total_matches"] += 1
        
        # Rank cases
        ranked_cases = sorted(
            case_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Format results
        similar_cases = []
        for case_id_match, score in ranked_cases:
            details = case_details[case_id_match]
            
            similar_cases.append(SimilarCase(
                case_id=case_id_match,
                similarity_score=float(score),
                matching_roles=list(details["role_matches"].keys()),
                match_count=details["total_matches"],
                role_breakdown=details["role_matches"]
            ))
        
        # Generate response
        if not similar_cases:
            response_text = "❌ No similar cases found matching your criteria."
        else:
            response_text = f"✅ Found {len(similar_cases)} similar cases:\n\n"
            
            for i, case in enumerate(similar_cases, 1):
                response_text += f"**{i}. Case: `{case.case_id}`**\n"
                response_text += f"   - **Similarity Score:** {case.similarity_score:.3f}\n"
                response_text += f"   - **Matching Roles:** {', '.join(case.matching_roles)}\n"
                response_text += f"   - **Total Matches:** {case.match_count}\n\n"
        
        logger.info(f"✅ Similarity search complete: {len(similar_cases)} cases found")
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": response_text
            }],
            "search_results": similar_cases,
            "final_answer": response_text
        }
        
    except Exception as e:
        logger.error(f"❌ Similarity agent error: {e}", exc_info=True)
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"❌ Error finding similar cases: {str(e)}"
            }],
            "final_answer": f"Error: {str(e)}"
        }
