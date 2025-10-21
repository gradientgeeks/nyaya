"""
Nyaya Multi-Agent System with LangGraph

This system intelligently routes user queries to different agents:

1. INTENT DETECTION → What does the user want?
   - Upload a case file → Route to Classification Agent
   - Search for similar cases → Route to Similarity Search Agent  
   - Predict outcome → Route to Prediction Agent
   - General Q&A → Route to RAG Agent

2. AGENT EXECUTION → Based on intent
   - Classification Agent: Uses InLegalBERT, uploads to Pinecone with roles
   - Similarity Agent: Semantic search in Pinecone
   - Prediction Agent: Analyzes patterns, predicts outcome
   - RAG Agent: Retrieves context, answers questions

3. RESPONSE GENERATION → Format answer for user

Example Flows:
--------------

User: "I have a case file about privacy rights, can you analyze it?"
→ Intent: UPLOAD_AND_CLASSIFY
→ Agent: Classification Agent
→ Actions: Extract text → Classify roles → Upload to Pinecone
→ Response: "Case analyzed! Found 23 Facts, 5 Issues, 12 Reasoning sentences..."

User: "Find cases similar to Navtej Singh Johar"
→ Intent: SIMILARITY_SEARCH
→ Agent: Similarity Search Agent  
→ Actions: Embed query → Search Pinecone → Rank results
→ Response: "Found 10 similar cases: 1. Section 377 case (95% match)..."

User: "What could be the outcome if I challenge Section 377?"
→ Intent: PREDICT_OUTCOME
→ Agent: Prediction Agent
→ Actions: Search precedents → Analyze patterns → Predict
→ Response: "Based on 15 similar cases, likely outcome: Favorable (78% confidence)..."

User: "What were the facts in the privacy case?"
→ Intent: QUESTION_ANSWERING
→ Agent: RAG Agent (with role filter)
→ Actions: Search role="Facts" → Generate answer
→ Response: "The petitioner filed a writ petition..."
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
import operator


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Shared state across all agents."""
    messages: Annotated[list, operator.add]  # Conversation history
    user_query: str  # Original user input
    intent: str  # Detected intent
    uploaded_file: dict | None  # If user uploaded a file
    search_results: list  # Retrieved documents
    classification_results: dict | None  # Role classification output
    final_answer: str  # Generated response
    metadata: dict  # Additional context


# ============================================================================
# INTENT DETECTION NODE
# ============================================================================

def detect_intent(state: AgentState) -> AgentState:
    """
    Analyze user query to determine intent.
    
    Possible intents:
    - UPLOAD_AND_CLASSIFY: User wants to upload/analyze a case file
    - SIMILARITY_SEARCH: Find similar cases
    - PREDICT_OUTCOME: Predict case outcome
    - QUESTION_ANSWERING: Ask about existing cases
    - ROLE_SPECIFIC_SEARCH: Search specific role (Facts, Reasoning, etc.)
    """
    
    query = state["user_query"].lower()
    
    # Check for file upload
    if state.get("uploaded_file"):
        intent = "UPLOAD_AND_CLASSIFY"
    
    # Keywords for different intents
    elif any(word in query for word in ["upload", "analyze this case", "classify", "my case"]):
        intent = "UPLOAD_AND_CLASSIFY"
    
    elif any(word in query for word in ["similar", "like this", "find cases", "related cases"]):
        intent = "SIMILARITY_SEARCH"
    
    elif any(word in query for word in ["predict", "outcome", "what will happen", "chances", "likely"]):
        intent = "PREDICT_OUTCOME"
    
    elif any(word in query for word in ["facts", "issue", "reasoning", "decision", "arguments"]):
        intent = "ROLE_SPECIFIC_SEARCH"
    
    else:
        intent = "QUESTION_ANSWERING"
    
    print(f"🎯 Detected Intent: {intent}")
    
    state["intent"] = intent
    state["messages"].append(
        AIMessage(content=f"I understand you want to: {intent}")
    )
    
    return state


# ============================================================================
# ROUTER - Decides which agent to call
# ============================================================================

def route_to_agent(state: AgentState) -> Literal[
    "classify_agent",
    "similarity_agent", 
    "prediction_agent",
    "rag_agent"
]:
    """Route to appropriate agent based on intent."""
    
    intent = state["intent"]
    
    if intent == "UPLOAD_AND_CLASSIFY":
        return "classify_agent"
    elif intent == "SIMILARITY_SEARCH":
        return "similarity_agent"
    elif intent == "PREDICT_OUTCOME":
        return "prediction_agent"
    else:  # QUESTION_ANSWERING or ROLE_SPECIFIC_SEARCH
        return "rag_agent"


# ============================================================================
# AGENT 1: CLASSIFICATION AGENT (Uses InLegalBERT)
# ============================================================================

def classification_agent(state: AgentState) -> AgentState:
    """
    Classify case file into rhetorical roles and upload to Pinecone.
    
    Steps:
    1. Extract text from uploaded file
    2. Use InLegalBERT to classify sentences
    3. Generate embeddings for each sentence
    4. Upload to Pinecone with role metadata
    """
    
    print("\n🔬 CLASSIFICATION AGENT ACTIVATED")
    
    # Pseudo-code (actual implementation would call your classifier)
    """
    from src.models.role_classifier import RoleClassifier
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    
    # Load classifier
    classifier = RoleClassifier(model_path="trained_model.pt")
    
    # Extract text
    file_content = state["uploaded_file"]["content"]
    text = extract_text(file_content)
    
    # Classify sentences
    results = classifier.classify_document(text, context_mode="prev")
    
    # Generate embeddings and upload
    model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("nyaya-legal-rag")
    
    vectors = []
    for i, sentence_data in enumerate(results):
        # Create embedding with title prefix
        text_with_title = f"title: {case_id} | text: {sentence_data['sentence']}"
        embedding = model.encode(
            text_with_title,
            prompt_name="Retrieval-document",
            normalize_embeddings=True
        )
        
        # Prepare vector
        vectors.append({
            "id": f"{case_id}_sent_{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": sentence_data["sentence"],
                "role": sentence_data["role"],
                "confidence": sentence_data["confidence"],
                "case_id": case_id
            }
        })
    
    # Upload to Pinecone
    index.upsert(vectors=vectors, namespace="user_documents")
    """
    
    # Simulated output
    state["classification_results"] = {
        "case_id": "case_12345",
        "total_sentences": 156,
        "role_distribution": {
            "Facts": 45,
            "Issue": 12,
            "Arguments of Petitioner": 23,
            "Arguments of Respondent": 18,
            "Reasoning": 38,
            "Decision": 8,
            "None": 12
        },
        "uploaded_to_pinecone": True
    }
    
    state["messages"].append(
        AIMessage(content="✅ Case classified and uploaded to knowledge base!")
    )
    
    print("✅ Classification complete")
    
    return state


# ============================================================================
# AGENT 2: SIMILARITY SEARCH AGENT
# ============================================================================

def similarity_search_agent(state: AgentState) -> AgentState:
    """
    Find similar cases using semantic search.
    
    Steps:
    1. Embed user query
    2. Search Pinecone for similar cases
    3. Rank by relevance
    4. Return top matches
    """
    
    print("\n🔍 SIMILARITY SEARCH AGENT ACTIVATED")
    
    # Pseudo-code
    """
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    
    # Load model
    model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
    
    # Embed query
    query = state["user_query"]
    query_embedding = model.encode(
        query,
        prompt_name="Retrieval-query",
        normalize_embeddings=True
    )
    
    # Search Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("nyaya-legal-rag")
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=10,
        include_metadata=True,
        namespace="user_documents"
    )
    
    # Group by case_id and calculate average similarity
    cases = {}
    for match in results['matches']:
        case_id = match['metadata']['case_id']
        if case_id not in cases:
            cases[case_id] = {
                'case_id': case_id,
                'similarity': match['score'],
                'chunks': []
            }
        cases[case_id]['chunks'].append(match['metadata']['text'])
    
    # Sort by similarity
    similar_cases = sorted(cases.values(), key=lambda x: x['similarity'], reverse=True)
    """
    
    # Simulated output
    state["search_results"] = [
        {
            "case_id": "navtej_singh_johar_v_union_2018",
            "similarity": 0.89,
            "title": "Navtej Singh Johar v. Union of India",
            "summary": "Section 377 challenged on privacy grounds"
        },
        {
            "case_id": "puttaswamy_v_union_2017",
            "similarity": 0.85,
            "title": "K.S. Puttaswamy v. Union of India",
            "summary": "Right to privacy recognized as fundamental right"
        }
    ]
    
    state["messages"].append(
        AIMessage(content="🔍 Found similar cases in our database")
    )
    
    print("✅ Similarity search complete")
    
    return state


# ============================================================================
# AGENT 3: PREDICTION AGENT
# ============================================================================

def prediction_agent(state: AgentState) -> AgentState:
    """
    Predict case outcome based on historical data.
    
    Steps:
    1. Search for similar precedents
    2. Analyze outcome patterns
    3. Calculate probability
    4. Generate prediction with reasoning
    """
    
    print("\n🔮 PREDICTION AGENT ACTIVATED")
    
    # Pseudo-code
    """
    # Step 1: Find similar cases
    similar_cases = similarity_search_agent(state)
    
    # Step 2: Analyze outcomes
    outcomes = []
    for case in similar_cases:
        # Extract decision from role="Decision" chunks
        decision_chunks = search_pinecone(
            case_id=case['case_id'],
            role_filter="Decision"
        )
        outcomes.append({
            'case': case['case_id'],
            'outcome': classify_outcome(decision_chunks),  # Favorable/Unfavorable
            'similarity': case['similarity']
        })
    
    # Step 3: Calculate weighted probability
    favorable = sum(1 for o in outcomes if o['outcome'] == 'Favorable')
    total = len(outcomes)
    confidence = favorable / total
    
    # Step 4: Generate reasoning
    reasoning = analyze_key_factors(similar_cases)
    """
    
    # Simulated output
    state["classification_results"] = {
        "prediction": "Favorable",
        "confidence": 0.78,
        "based_on_cases": 15,
        "key_factors": [
            "Privacy rights precedent strong",
            "Similar constitutional challenge succeeded in 2018",
            "Recent trend towards fundamental rights protection"
        ]
    }
    
    state["messages"].append(
        AIMessage(content="🔮 Outcome prediction generated")
    )
    
    print("✅ Prediction complete")
    
    return state


# ============================================================================
# AGENT 4: RAG AGENT (Question Answering)
# ============================================================================

def rag_agent(state: AgentState) -> AgentState:
    """
    Answer questions using RAG with optional role filtering.
    
    Steps:
    1. Detect if role-specific query
    2. Search Pinecone (with role filter if applicable)
    3. Retrieve relevant chunks
    4. Generate answer using LLM
    """
    
    print("\n💬 RAG AGENT ACTIVATED")
    
    query = state["user_query"]
    
    # Detect role in query
    role_keywords = {
        "facts": "Facts",
        "issue": "Issue",
        "reasoning": "Reasoning",
        "decision": "Decision",
        "petitioner": "Arguments of Petitioner",
        "respondent": "Arguments of Respondent"
    }
    
    role_filter = None
    for keyword, role in role_keywords.items():
        if keyword in query.lower():
            role_filter = role
            break
    
    # Pseudo-code
    """
    from sentence_transformers import SentenceTransformer
    from pinecone import Pinecone
    from langchain_google_vertexai import ChatVertexAI
    
    # Embed query
    model = SentenceTransformer("google/embeddinggemma-300M", truncate_dim=384)
    query_embedding = model.encode(query, prompt_name="Retrieval-query", normalize_embeddings=True)
    
    # Search with role filter
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("nyaya-legal-rag")
    
    filters = {"role": {"$eq": role_filter}} if role_filter else None
    
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        filter=filters
    )
    
    # Build context
    context = "\n\n".join([match['metadata']['text'] for match in results['matches']])
    
    # Generate answer
    llm = ChatVertexAI(model_name="gemini-1.5-pro")
    
    prompt = f'''
    Context from legal cases:
    {context}
    
    Question: {query}
    
    Answer the question based on the context above. Be precise and cite relevant information.
    '''
    
    answer = llm.invoke(prompt)
    """
    
    # Simulated output
    state["search_results"] = [
        {"text": "The petitioner filed a writ petition...", "role": "Facts", "score": 0.91},
        {"text": "Article 32 was invoked...", "role": "Facts", "score": 0.87}
    ]
    
    state["final_answer"] = "Based on the case records, the petitioner filed a writ petition under Article 32..."
    
    state["messages"].append(
        AIMessage(content=state["final_answer"])
    )
    
    print("✅ RAG response generated")
    
    return state


# ============================================================================
# FINAL RESPONSE FORMATTER
# ============================================================================

def format_response(state: AgentState) -> AgentState:
    """Format the final response based on agent outputs."""
    
    intent = state["intent"]
    
    if intent == "UPLOAD_AND_CLASSIFY":
        results = state["classification_results"]
        response = f"""
✅ **Case Analysis Complete**

**Case ID:** {results['case_id']}
**Total Sentences:** {results['total_sentences']}

**Role Distribution:**
- Facts: {results['role_distribution']['Facts']}
- Issue: {results['role_distribution']['Issue']}
- Reasoning: {results['role_distribution']['Reasoning']}
- Decision: {results['role_distribution']['Decision']}
- Arguments (Petitioner): {results['role_distribution']['Arguments of Petitioner']}
- Arguments (Respondent): {results['role_distribution']['Arguments of Respondent']}

Your case has been uploaded to the knowledge base. You can now search for similar cases or ask questions about it!
"""
    
    elif intent == "SIMILARITY_SEARCH":
        results = state["search_results"]
        response = "🔍 **Similar Cases Found:**\n\n"
        for i, case in enumerate(results, 1):
            response += f"{i}. **{case['title']}** (Similarity: {case['similarity']:.0%})\n"
            response += f"   {case['summary']}\n\n"
    
    elif intent == "PREDICT_OUTCOME":
        pred = state["classification_results"]
        response = f"""
🔮 **Outcome Prediction**

**Predicted Outcome:** {pred['prediction']}
**Confidence:** {pred['confidence']:.0%}
**Based on:** {pred['based_on_cases']} similar cases

**Key Factors:**
"""
        for factor in pred['key_factors']:
            response += f"- {factor}\n"
    
    else:  # QUESTION_ANSWERING
        response = state["final_answer"]
    
    state["final_answer"] = response
    
    return state


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_nyaya_agent():
    """Create the LangGraph agent system."""
    
    # Initialize graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("classify_agent", classification_agent)
    workflow.add_node("similarity_agent", similarity_search_agent)
    workflow.add_node("prediction_agent", prediction_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("format_response", format_response)
    
    # Set entry point
    workflow.set_entry_point("detect_intent")
    
    # Add conditional routing from intent detection
    workflow.add_conditional_edges(
        "detect_intent",
        route_to_agent,
        {
            "classify_agent": "classify_agent",
            "similarity_agent": "similarity_agent",
            "prediction_agent": "prediction_agent",
            "rag_agent": "rag_agent"
        }
    )
    
    # All agents route to response formatter
    workflow.add_edge("classify_agent", "format_response")
    workflow.add_edge("similarity_agent", "format_response")
    workflow.add_edge("prediction_agent", "format_response")
    workflow.add_edge("rag_agent", "format_response")
    
    # End after formatting
    workflow.add_edge("format_response", END)
    
    # Compile
    app = workflow.compile()
    
    return app


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demonstrate the multi-agent system."""
    
    print("=" * 80)
    print("🏛️  NYAYA MULTI-AGENT SYSTEM")
    print("=" * 80)
    
    # Create agent
    agent = create_nyaya_agent()
    
    # Test cases
    test_queries = [
        {
            "user_query": "I have a case file about privacy rights, can you analyze it?",
            "uploaded_file": {"filename": "privacy_case.pdf", "content": "..."}
        },
        {
            "user_query": "Find cases similar to Navtej Singh Johar",
            "uploaded_file": None
        },
        {
            "user_query": "What could be the outcome if I challenge Section 377?",
            "uploaded_file": None
        },
        {
            "user_query": "What were the facts in the privacy case?",
            "uploaded_file": None
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {test['user_query']}")
        print('='*80)
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=test["user_query"])],
            "user_query": test["user_query"],
            "intent": "",
            "uploaded_file": test.get("uploaded_file"),
            "search_results": [],
            "classification_results": None,
            "final_answer": "",
            "metadata": {}
        }
        
        # Run agent
        result = agent.invoke(initial_state)
        
        # Print response
        print("\n📤 RESPONSE:")
        print(result["final_answer"])


if __name__ == "__main__":
    demo()
