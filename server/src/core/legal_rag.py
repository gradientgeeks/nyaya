"""
Role-Aware Multi-Modal RAG System for Legal Documents

This module implements the role-aware RAG system that stores and retrieves 
embeddings based on rhetorical roles, extending the basic multi-modal RAG
to handle legal document structure.
"""

import uuid
import base64
import io
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.embeddings import VertexAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatVertexAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from ..models.role_classifier import RoleClassifier, RhetoricalRole

logger = logging.getLogger(__name__)

class RoleTaggedDocument(BaseModel):
    """Document with role-specific metadata"""
    content: str
    role: str
    doc_id: str
    sentence_index: int
    confidence: float
    metadata: Dict[str, Any] = {}

class LegalRAGSystem:
    """
    Role-aware RAG system for legal documents
    Integrates rhetorical role classification with multi-modal retrieval
    """
    
    def __init__(self, 
                 embedding_model: str = "text-embedding-005",
                 role_classifier_type: str = "inlegalbert",
                 collection_name: str = "legal_rag",
                 device: str = "cpu"):
        """
        Initialize the Legal RAG System
        
        Args:
            embedding_model: Name of the embedding model
            role_classifier_type: Type of role classifier to use
            collection_name: Name for the ChromaDB collection
            device: Device for running models
        """
        self.device = device
        self.collection_name = collection_name
        
        # Initialize role classifier
        self.role_classifier = RoleClassifier(
            model_type=role_classifier_type, 
            device=device
        )
        
        # Initialize embeddings
        self.embeddings = VertexAIEmbeddings(model_name=embedding_model)
        
        # Initialize vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        # Initialize storage and retriever
        self.docstore = InMemoryStore()
        self.retriever = None
        self._setup_retriever()
        
        # Initialize LLM for generation
        self.llm = ChatVertexAI(
            temperature=0, 
            model_name="gemini-2.5-flash", 
            max_tokens=1024
        )
        
        logger.info("Legal RAG System initialized successfully")
    
    def _setup_retriever(self):
        """Setup the multi-vector retriever with role awareness"""
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id"
        )
    
    def process_legal_document(self, 
                             document_text: str, 
                             doc_metadata: Dict[str, Any] = None,
                             context_mode: str = "prev") -> List[RoleTaggedDocument]:
        """
        Process a legal document and classify rhetorical roles
        
        Args:
            document_text: The legal document text
            doc_metadata: Additional metadata for the document
            context_mode: Context mode for role classification
            
        Returns:
            List of role-tagged documents
        """
        if doc_metadata is None:
            doc_metadata = {}
        
        # Classify rhetorical roles
        role_results = self.role_classifier.classify_document(
            document_text, context_mode=context_mode
        )
        
        # Create role-tagged documents
        tagged_docs = []
        for result in role_results:
            doc_id = str(uuid.uuid4())
            
            tagged_doc = RoleTaggedDocument(
                content=result["sentence"],
                role=result["role"],
                doc_id=doc_id,
                sentence_index=result["sentence_index"],
                confidence=result["confidence"],
                metadata={
                    **doc_metadata,
                    "role": result["role"],
                    "sentence_index": result["sentence_index"],
                    "confidence": result["confidence"]
                }
            )
            tagged_docs.append(tagged_doc)
        
        return tagged_docs
    
    def add_documents_to_store(self, tagged_docs: List[RoleTaggedDocument]):
        """
        Add role-tagged documents to the vector store
        
        Args:
            tagged_docs: List of role-tagged documents
        """
        # Group documents by role for better organization
        role_groups = {}
        for doc in tagged_docs:
            if doc.role not in role_groups:
                role_groups[doc.role] = []
            role_groups[doc.role].append(doc)
        
        # Add documents to vectorstore and docstore
        for role, docs in role_groups.items():
            doc_ids = [doc.doc_id for doc in docs]
            contents = [doc.content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            
            # Create Document objects with metadata
            documents = [
                Document(
                    page_content=content,
                    metadata={**metadata, "doc_id": doc_id}
                )
                for content, metadata, doc_id in zip(contents, metadatas, doc_ids)
            ]
            
            # Add to vectorstore
            self.retriever.vectorstore.add_documents(documents)
            
            # Add to docstore
            self.retriever.docstore.mset(list(zip(doc_ids, contents)))
        
        logger.info(f"Added {len(tagged_docs)} documents to store across {len(role_groups)} roles")
    
    def retrieve_by_role(self, 
                        query: str, 
                        roles: List[str] = None, 
                        k: int = 5) -> List[Document]:
        """
        Retrieve documents filtered by specific rhetorical roles
        
        Args:
            query: Search query
            roles: List of rhetorical roles to filter by
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if roles is None:
            # If no roles specified, search all
            return self.retriever.invoke(query, k=k)
        
        # Create role-specific filter
        role_filter = {"role": {"$in": roles}}
        
        # Perform similarity search with metadata filter
        docs = self.vectorstore.similarity_search(
            query, 
            k=k, 
            filter=role_filter
        )
        
        # Get full content from docstore
        doc_ids = [doc.metadata.get("doc_id") for doc in docs if doc.metadata.get("doc_id")]
        full_contents = self.retriever.docstore.mget(doc_ids)
        
        # Reconstruct documents with full content
        result_docs = []
        for doc, content in zip(docs, full_contents):
            if content:
                result_docs.append(Document(
                    page_content=content,
                    metadata=doc.metadata
                ))
        
        return result_docs
    
    def analyze_query_intent(self, query: str) -> List[str]:
        """
        Analyze query to determine which rhetorical roles are relevant
        
        Args:
            query: User query
            
        Returns:
            List of relevant rhetorical roles
        """
        query_lower = query.lower()
        relevant_roles = []
        
        # Simple keyword-based intent detection
        role_keywords = {
            RhetoricalRole.FACTS.value: [
                "facts", "background", "what happened", "events", 
                "circumstances", "story", "timeline"
            ],
            RhetoricalRole.ISSUE.value: [
                "issue", "question", "problem", "dispute", 
                "legal question", "main issue"
            ],
            RhetoricalRole.ARGUMENTS_PETITIONER.value: [
                "petitioner", "plaintiff", "appellant", "arguments for",
                "petitioner argues", "plaintiff claims"
            ],
            RhetoricalRole.ARGUMENTS_RESPONDENT.value: [
                "respondent", "defendant", "appellee", "arguments against",
                "respondent argues", "defendant claims"
            ],
            RhetoricalRole.REASONING.value: [
                "reasoning", "rationale", "why", "because", "analysis",
                "court reasoning", "legal analysis", "justification"
            ],
            RhetoricalRole.DECISION.value: [
                "decision", "judgment", "ruling", "verdict", "outcome",
                "final decision", "court decided", "held that"
            ]
        }
        
        for role, keywords in role_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant_roles.append(role)
        
        # If no specific roles detected, include key roles
        if not relevant_roles:
            relevant_roles = [
                RhetoricalRole.FACTS.value,
                RhetoricalRole.REASONING.value,
                RhetoricalRole.DECISION.value
            ]
        
        return relevant_roles
    
    def generate_structured_response(self, 
                                   query: str, 
                                   retrieved_docs: List[Document]) -> Dict[str, Any]:
        """
        Generate a structured response based on retrieved documents
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents with role metadata
            
        Returns:
            Structured response with role-based sections
        """
        # Group documents by role
        role_docs = {}
        for doc in retrieved_docs:
            role = doc.metadata.get("role", "Unknown")
            if role not in role_docs:
                role_docs[role] = []
            role_docs[role].append(doc.page_content)
        
        # Create structured prompt
        structured_context = []
        for role, docs in role_docs.items():
            if docs:
                role_content = "\n".join(docs[:3])  # Limit to top 3 docs per role
                structured_context.append(f"**{role}:**\n{role_content}")
        
        context_text = "\n\n".join(structured_context)
        
        # Create prompt for structured legal response
        prompt_template = PromptTemplate.from_template("""
You are a legal AI assistant specializing in Indian legal documents. 
Based on the provided context organized by rhetorical roles, answer the user's question with accurate legal information.

Query: {query}

Context organized by rhetorical roles:
{context}

Provide a comprehensive answer that:
1. Directly addresses the query
2. References relevant roles when appropriate
3. Maintains legal accuracy
4. Cites specific information from the context

Answer:
""")
        
        # Generate response
        chain = prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context_text
        })
        
        return {
            "answer": response,
            "role_breakdown": {role: len(docs) for role, docs in role_docs.items()},
            "retrieved_roles": list(role_docs.keys()),
            "total_documents": len(retrieved_docs)
        }
    
    def query_legal_rag(self, 
                       query: str, 
                       auto_detect_roles: bool = True,
                       specific_roles: List[str] = None,
                       k: int = 10) -> Dict[str, Any]:
        """
        Main query interface for the legal RAG system
        
        Args:
            query: User query
            auto_detect_roles: Whether to auto-detect relevant roles
            specific_roles: Specific roles to search (overrides auto-detection)
            k: Number of documents to retrieve
            
        Returns:
            Comprehensive response with structured information
        """
        # Determine relevant roles
        if specific_roles:
            search_roles = specific_roles
        elif auto_detect_roles:
            search_roles = self.analyze_query_intent(query)
        else:
            search_roles = None
        
        # Retrieve documents
        retrieved_docs = self.retrieve_by_role(query, roles=search_roles, k=k)
        
        # Generate structured response
        response = self.generate_structured_response(query, retrieved_docs)
        
        # Add metadata about the search
        response["search_metadata"] = {
            "query": query,
            "searched_roles": search_roles,
            "auto_detect_used": auto_detect_roles and not specific_roles,
            "retrieval_count": len(retrieved_docs)
        }
        
        return response
    
    def get_case_summary(self, case_text: str) -> Dict[str, Any]:
        """
        Generate a comprehensive case summary organized by rhetorical roles
        
        Args:
            case_text: Full case text
            
        Returns:
            Structured case summary
        """
        # Process the document
        tagged_docs = self.process_legal_document(case_text)
        
        # Group by roles
        role_summary = {}
        for doc in tagged_docs:
            role = doc.role
            if role not in role_summary:
                role_summary[role] = []
            role_summary[role].append(doc.content)
        
        # Generate summary for each role
        summaries = {}
        for role, sentences in role_summary.items():
            if len(sentences) > 5:  # Only summarize if there are enough sentences
                combined_text = " ".join(sentences)
                summary_prompt = f"""
                Summarize the following {role} section from a legal judgment:
                
                {combined_text}
                
                Provide a concise summary that captures the key points:
                """
                
                summary = self.llm.invoke([HumanMessage(content=summary_prompt)])
                summaries[role] = summary.content
            else:
                summaries[role] = " ".join(sentences)
        
        return {
            "role_summaries": summaries,
            "role_statistics": {role: len(sentences) for role, sentences in role_summary.items()},
            "total_sentences": len(tagged_docs)
        }

# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag_system = LegalRAGSystem()
    
    # Sample legal document
    sample_case = """
    The petitioner filed a writ petition under Article 32 of the Constitution.
    The facts of the case are that the petitioner was arrested without warrant.
    The main issue is whether the arrest was constitutional.
    The petitioner argues that the arrest violated fundamental rights.
    The respondent contends that the arrest was lawful under the applicable law.
    The court analyzed the constitutional provisions and precedents.
    The court found that the arrest was indeed unconstitutional.
    Therefore, the petition is allowed and the arrest is quashed.
    """
    
    # Process and store the document
    tagged_docs = rag_system.process_legal_document(sample_case)
    rag_system.add_documents_to_store(tagged_docs)
    
    # Query the system
    query = "What were the facts of the case?"
    response = rag_system.query_legal_rag(query)
    
    print("Query:", query)
    print("Answer:", response["answer"])
    print("Roles retrieved:", response["retrieved_roles"])