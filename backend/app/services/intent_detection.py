"""
Intent Detection Service - Rule-Based (No LLM Required)

This service uses keyword matching and contextual analysis to determine user intent.
No LLM is needed for intent routing, making it fast and cost-effective.
"""

import re
from typing import Tuple, Optional, List
from app.models.schemas import Intent, RhetoricalRole


class IntentDetector:
    """Rule-based intent detection for routing user queries."""
    
    # Intent detection keywords
    UPLOAD_KEYWORDS = [
        "upload", "analyze", "classify", "my case", "this document",
        "process", "examine", "check this"
    ]
    
    SEARCH_KEYWORDS = [
        "search", "find", "cases about", "show me", "lookup",
        "similar to", "like", "related", "involving"
    ]
    
    SIMILARITY_KEYWORDS = [
        "similar", "like this", "comparable", "related cases",
        "matching", "analogous"
    ]
    
    PREDICTION_KEYWORDS = [
        "predict", "outcome", "what will happen", "chances",
        "likely", "probability", "forecast", "estimate"
    ]
    
    # Role-specific keywords for RAG
    ROLE_KEYWORDS = {
        "facts": RhetoricalRole.FACTS,
        "fact": RhetoricalRole.FACTS,
        "background": RhetoricalRole.FACTS,
        "what happened": RhetoricalRole.FACTS,
        
        "issue": RhetoricalRole.ISSUE,
        "issues": RhetoricalRole.ISSUE,
        "question": RhetoricalRole.ISSUE,
        "legal question": RhetoricalRole.ISSUE,
        
        "reasoning": RhetoricalRole.REASONING,
        "rationale": RhetoricalRole.REASONING,
        "analysis": RhetoricalRole.REASONING,
        "court's reasoning": RhetoricalRole.REASONING,
        "why": RhetoricalRole.REASONING,
        
        "decision": RhetoricalRole.DECISION,
        "ruling": RhetoricalRole.DECISION,
        "judgment": RhetoricalRole.DECISION,
        "verdict": RhetoricalRole.DECISION,
        "conclusion": RhetoricalRole.DECISION,
        
        "petitioner": RhetoricalRole.ARGUMENTS_PETITIONER,
        "petitioner's argument": RhetoricalRole.ARGUMENTS_PETITIONER,
        "petitioner argues": RhetoricalRole.ARGUMENTS_PETITIONER,
        "aop": RhetoricalRole.ARGUMENTS_PETITIONER,
        
        "respondent": RhetoricalRole.ARGUMENTS_RESPONDENT,
        "respondent's argument": RhetoricalRole.ARGUMENTS_RESPONDENT,
        "respondent argues": RhetoricalRole.ARGUMENTS_RESPONDENT,
        "aor": RhetoricalRole.ARGUMENTS_RESPONDENT,
    }
    
    def detect_intent(
        self,
        query: str,
        has_file: bool = False,
        session_context: Optional[dict] = None
    ) -> Tuple[Intent, Optional[List[RhetoricalRole]]]:
        """
        Detect user intent from query and context.
        
        Args:
            query: User's query string
            has_file: Whether user uploaded a file
            session_context: Optional session context (active case, history, etc.)
        
        Returns:
            Tuple of (Intent, Optional[List[RhetoricalRole]])
            - Intent: Detected user intent
            - Role filter: List of roles if role-specific query, else None
        """
        query_lower = query.lower()
        
        # Priority 1: File Upload
        if has_file:
            return Intent.UPLOAD_AND_CLASSIFY, None
        
        # Priority 2: Prediction Intent
        if self._contains_keywords(query_lower, self.PREDICTION_KEYWORDS):
            return Intent.PREDICT_OUTCOME, None
        
        # Priority 3: Similarity Search
        # Check if user is asking for similar cases
        if self._contains_keywords(query_lower, self.SIMILARITY_KEYWORDS):
            # If there's an active case in session, it's similarity to that case
            if session_context and session_context.get("active_case_id"):
                return Intent.SIMILARITY_SEARCH, None
            # Otherwise, it's a general search
            return Intent.SEARCH_EXISTING_CASES, None
        
        # Priority 4: Role-Specific QA
        # Check if query mentions specific roles
        detected_roles = self._detect_roles(query_lower)
        if detected_roles:
            return Intent.ROLE_SPECIFIC_QA, detected_roles
        
        # Priority 5: Search for Existing Cases
        # Keywords like "find cases about", "search for"
        if self._contains_keywords(query_lower, self.SEARCH_KEYWORDS):
            return Intent.SEARCH_EXISTING_CASES, None
        
        # Priority 6: General QA
        # Default for questions about uploaded cases or general legal queries
        if session_context and session_context.get("active_case_id"):
            # User has an active case, likely asking about it
            return Intent.GENERAL_QA, None
        
        # No active case, no specific keywords -> general search
        return Intent.SEARCH_EXISTING_CASES, None
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords."""
        return any(keyword in text for keyword in keywords)
    
    def _detect_roles(self, query: str) -> Optional[List[RhetoricalRole]]:
        """Detect mentioned rhetorical roles in the query."""
        detected_roles = set()
        
        for keyword, role in self.ROLE_KEYWORDS.items():
            if keyword in query:
                detected_roles.add(role)
        
        return list(detected_roles) if detected_roles else None
    
    def extract_case_reference(self, query: str) -> Optional[str]:
        """
        Extract case ID or case name from query.
        
        Patterns:
        - "case_12345"
        - "Case #12345"
        - "Navtej Singh Johar v. Union"
        """
        # Pattern 1: case_XXXXX format
        case_id_pattern = r"case[_\s]?(\d+)"
        match = re.search(case_id_pattern, query.lower())
        if match:
            return f"case_{match.group(1)}"
        
        # Pattern 2: Case name (Title Case v. Title Case)
        case_name_pattern = r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)"
        match = re.search(case_name_pattern, query)
        if match:
            # Return normalized case name
            return f"{match.group(1)}_v_{match.group(2)}".replace(" ", "_").lower()
        
        return None
    
    def is_follow_up_question(self, query: str, session_context: Optional[dict] = None) -> bool:
        """
        Determine if the query is a follow-up question.
        
        Indicators:
        - Pronouns: "it", "that", "this", "the case"
        - Referential: "and", "also", "what about"
        - Short question with context available
        """
        if not session_context or not session_context.get("active_case_id"):
            return False
        
        query_lower = query.lower()
        
        # Pronouns and references
        referential_words = [
            "it", "that", "this", "the case", "that case",
            "and", "also", "what about", "how about"
        ]
        
        has_reference = any(word in query_lower for word in referential_words)
        is_short = len(query.split()) < 10
        
        return has_reference or (is_short and len(session_context.get("conversation_history", [])) > 0)


# Singleton instance
intent_detector = IntentDetector()
