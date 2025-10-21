"""
RAG Agent

Performs role-aware retrieval-augmented generation for question answering.
This is the core of Nyaya's role-aware RAG system.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import ChatPromptTemplate
from ..models.schemas import AgentState, RhetoricalRole, RAGResponse
from ..services.embedding_service import EmbeddingService
from ..services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)


# System prompt for Gemini
SYSTEM_PROMPT = """You are Nyaya (न्याय - Sanskrit for "justice"), an AI legal assistant specialized in Indian legal documents.

Your role is to answer questions about legal cases using retrieved context from judgments.

**Key principles:**
1. **Be precise**: Answer based ONLY on the provided context
2. **Cite sources**: Reference specific roles (Facts, Reasoning, Decision, etc.)
3. **Be honest**: If context is insufficient, say so
4. **Be structured**: Organize answers by rhetorical role when relevant
5. **Be helpful**: Provide legal insights, not just quotes

**Available rhetorical roles:**
- **Facts**: Background and case events
- **Issue**: Legal questions to resolve
- **Arguments of Petitioner**: Petitioner's claims
- **Arguments of Respondent**: Respondent's counter-arguments
- **Reasoning**: Court's legal analysis
- **Decision**: Final judgment

Always maintain professional legal language and cite the role of each piece of information."""


async def rag_agent_node(
    state: AgentState,
    embedder: EmbeddingService,
    pinecone: PineconeService,
    llm: ChatVertexAI,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    RAG Agent: Role-aware question answering.
    
    Flow:
    1. Detect which roles to query (from intent or default to all)
    2. Encode query
    3. Retrieve relevant sentences with role filtering
    4. Build context from retrieved sentences
    5. Generate answer using Gemini with context
    
    Args:
        state: Current agent state
        embedder: Embedding service
        pinecone: Pinecone storage service
        llm: LangChain LLM (Gemini)
        top_k: Number of results per role
    
    Returns:
        Updated state dict
    """
    logger.info("💬 RAG Agent: Answering question with role-aware retrieval...")
    
    try:
        query_text = state.get("user_query", "")
        target_roles = state.get("target_roles", [])  # From intent detection
        case_id = state.get("case_id")  # Optional: restrict to specific case
        
        # If no specific roles, query all (except None)
        if not target_roles:
            target_roles = [
                RhetoricalRole.FACTS,
                RhetoricalRole.ISSUE,
                RhetoricalRole.ARGUMENT_PETITIONER,
                RhetoricalRole.ARGUMENT_RESPONDENT,
                RhetoricalRole.REASONING,
                RhetoricalRole.DECISION
            ]
        
        logger.info(f"🔍 Querying roles: {[r.value for r in target_roles]}")
        
        # Encode query
        query_embedding = embedder.encode_query(query_text, normalize=True)
        
        # Retrieve context for each role
        all_contexts = []
        
        for role in target_roles:
            logger.info(f"🔍 Retrieving {role.value}...")
            
            matches = pinecone.query_vectors(
                query_vector=query_embedding.tolist(),
                top_k=top_k,
                role_filter=role,
                case_id_filter=case_id,
                namespace="user_documents",
                min_score=0.4  # Lower threshold for QA
            )
            
            for match in matches:
                all_contexts.append({
                    "role": match["role"],
                    "text": match["text"],
                    "score": match["score"],
                    "case_id": match["case_id"]
                })
            
            logger.info(f"  Retrieved {len(matches)} matches")
        
        # Sort by relevance score
        all_contexts.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top results overall
        top_contexts = all_contexts[:top_k * 2]  # More context for better answers
        
        if not top_contexts:
            response_text = (
                "❌ I couldn't find relevant information to answer your question.\n\n"
                "This could mean:\n"
                "- No documents have been uploaded yet\n"
                "- Your question is outside the scope of available cases\n"
                "- Try rephrasing your question or uploading relevant documents"
            )
            
            return {
                "messages": state.get("messages", []) + [{
                    "role": "assistant",
                    "content": response_text
                }],
                "final_answer": response_text
            }
        
        # Build context string organized by role
        context_by_role = {}
        for ctx in top_contexts:
            role = ctx["role"]
            if role not in context_by_role:
                context_by_role[role] = []
            context_by_role[role].append(ctx["text"])
        
        context_str = ""
        for role, texts in context_by_role.items():
            context_str += f"\n### {role}:\n"
            for i, text in enumerate(texts, 1):
                context_str += f"{i}. {text}\n"
        
        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", """Based on the following context from legal documents, answer this question:

**Question:** {query}

**Context:**
{context}

**Instructions:**
- Answer directly and concisely
- Cite the rhetorical role (Facts, Reasoning, Decision, etc.) for each point
- If the answer requires information from multiple roles, organize your response accordingly
- If context is insufficient, explain what information is missing

**Answer:**""")
        ])
        
        # Generate answer
        logger.info("🤖 Generating answer with Gemini...")
        
        chain = prompt | llm
        response = await chain.ainvoke({
            "query": query_text,
            "context": context_str
        })
        
        answer_text = response.content
        
        logger.info(f"✅ Answer generated ({len(answer_text)} chars)")
        
        # Build response
        rag_response = RAGResponse(
            answer=answer_text,
            retrieved_contexts=top_contexts,
            roles_used=[r.value for r in target_roles]
        )
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": answer_text
            }],
            "rag_response": rag_response,
            "final_answer": answer_text
        }
        
    except Exception as e:
        logger.error(f"❌ RAG agent error: {e}", exc_info=True)
        
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant",
                "content": f"❌ Error generating answer: {str(e)}"
            }],
            "final_answer": f"Error: {str(e)}"
        }
