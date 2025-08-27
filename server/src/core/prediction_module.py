"""
Judgment Prediction Module for Legal Cases

This module implements probable judgment prediction for pending cases 
by analyzing similar precedents and extracting patterns from historical decisions.
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

from langchain_community.chat_models import ChatVertexAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from .legal_rag import LegalRAGSystem, RhetoricalRole

logger = logging.getLogger(__name__)

class JudgmentOutcome(Enum):
    """Possible judgment outcomes"""
    ALLOWED = "allowed"
    DISMISSED = "dismissed"
    PARTLY_ALLOWED = "partly_allowed"
    REMANDED = "remanded"
    QUASHED = "quashed"
    STAYED = "stayed"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"

class CaseType(Enum):
    """Types of legal cases"""
    CIVIL = "civil"
    CRIMINAL = "criminal"
    CONSTITUTIONAL = "constitutional"
    COMMERCIAL = "commercial"
    FAMILY = "family"
    TAX = "tax"
    LABOR = "labor"
    PROPERTY = "property"

@dataclass
class PrecedentCase:
    """Precedent case information"""
    case_id: str
    case_name: str
    facts: str
    issues: str
    reasoning: str
    decision: str
    outcome: JudgmentOutcome
    case_type: CaseType
    court: str
    year: int
    citation: str
    similarity_score: float = 0.0

@dataclass
class PredictionResult:
    """Judgment prediction result"""
    predicted_outcome: JudgmentOutcome
    confidence: float
    probability_distribution: Dict[str, float]
    similar_cases: List[PrecedentCase]
    key_factors: List[str]
    reasoning: str
    disclaimer: str

class CaseSimilarityAnalyzer:
    """
    Analyzes similarity between cases based on facts, issues, and legal principles
    """
    
    def __init__(self):
        """Initialize similarity analyzer"""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.legal_keywords = self._load_legal_keywords()
        
    def _load_legal_keywords(self) -> List[str]:
        """Load legal domain-specific keywords for enhanced similarity"""
        return [
            # Constitutional terms
            "fundamental rights", "article", "constitution", "violation", "due process",
            "equal protection", "liberty", "privacy", "speech", "expression",
            
            # Civil law terms
            "contract", "breach", "damages", "negligence", "tort", "liability",
            "compensation", "injunction", "specific performance",
            
            # Criminal law terms
            "arrest", "bail", "conviction", "evidence", "procedure", "investigation",
            "charge", "prosecution", "defense", "sentencing",
            
            # Procedural terms
            "jurisdiction", "appeal", "revision", "review", "stay", "interim",
            "preliminary", "final", "ex-parte", "notice", "hearing",
            
            # Family law terms
            "marriage", "divorce", "custody", "maintenance", "dowry", "domestic",
            
            # Commercial terms
            "business", "corporate", "shareholder", "director", "merger", "acquisition",
            "insolvency", "bankruptcy", "debt", "credit"
        ]
    
    def extract_case_features(self, case_text: str, case_type: CaseType = None) -> Dict[str, Any]:
        """
        Extract relevant features from case text for similarity analysis
        
        Args:
            case_text: Full case text
            case_type: Type of case
            
        Returns:
            Dictionary of extracted features
        """
        features = {
            "legal_keywords": [],
            "entities": [],
            "case_citations": [],
            "statutory_references": [],
            "procedural_terms": [],
            "factual_complexity": 0,
            "case_length": len(case_text.split())
        }
        
        text_lower = case_text.lower()
        
        # Extract legal keywords
        for keyword in self.legal_keywords:
            if keyword in text_lower:
                features["legal_keywords"].append(keyword)
        
        # Extract case citations
        citation_patterns = [
            r'\(\d{4}\)\s+\d+\s+SCC\s+\d+',
            r'AIR\s+\d{4}\s+SC\s+\d+',
            r'\d{4}\s+\(\d+\)\s+SCC\s+\d+',
            r'[A-Z][a-zA-Z\s&\.]+\s+v\.?\s+[A-Z][a-zA-Z\s&\.]+'
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, case_text)
            features["case_citations"].extend(matches)
        
        # Extract statutory references
        statutory_patterns = [
            r'Section\s+\d+[A-Za-z]*',
            r'Article\s+\d+[A-Za-z]*',
            r'Rule\s+\d+[A-Za-z]*',
            r'Order\s+\d+[A-Za-z]*'
        ]
        
        for pattern in statutory_patterns:
            matches = re.findall(pattern, case_text, re.IGNORECASE)
            features["statutory_references"].extend(matches)
        
        # Calculate factual complexity (number of dates, names, amounts)
        date_patterns = r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b|\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        amount_patterns = r'Rs\.?\s*\d+[,\d]*|\$\s*\d+[,\d]*'
        
        dates = len(re.findall(date_patterns, case_text, re.IGNORECASE))
        amounts = len(re.findall(amount_patterns, case_text))
        
        features["factual_complexity"] = dates + amounts
        
        return features
    
    def calculate_similarity(self, case1_text: str, case2_text: str, 
                           case1_features: Dict[str, Any] = None,
                           case2_features: Dict[str, Any] = None) -> float:
        """
        Calculate similarity between two cases
        
        Args:
            case1_text: Text of first case
            case2_text: Text of second case
            case1_features: Pre-extracted features for case 1
            case2_features: Pre-extracted features for case 2
            
        Returns:
            Similarity score between 0 and 1
        """
        # Text similarity using TF-IDF
        try:
            tfidf_matrix = self.vectorizer.fit_transform([case1_text, case2_text])
            text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            text_similarity = 0.0
        
        # Feature-based similarity
        if case1_features is None:
            case1_features = self.extract_case_features(case1_text)
        if case2_features is None:
            case2_features = self.extract_case_features(case2_text)
        
        feature_similarity = self._calculate_feature_similarity(case1_features, case2_features)
        
        # Weighted combination
        final_similarity = 0.7 * text_similarity + 0.3 * feature_similarity
        
        return min(final_similarity, 1.0)
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], 
                                    features2: Dict[str, Any]) -> float:
        """Calculate similarity based on extracted features"""
        similarity_scores = []
        
        # Legal keywords similarity
        keywords1 = set(features1.get("legal_keywords", []))
        keywords2 = set(features2.get("legal_keywords", []))
        if keywords1 or keywords2:
            keyword_similarity = len(keywords1.intersection(keywords2)) / len(keywords1.union(keywords2))
            similarity_scores.append(keyword_similarity)
        
        # Statutory references similarity
        statutes1 = set(features1.get("statutory_references", []))
        statutes2 = set(features2.get("statutory_references", []))
        if statutes1 or statutes2:
            statute_similarity = len(statutes1.intersection(statutes2)) / len(statutes1.union(statutes2))
            similarity_scores.append(statute_similarity)
        
        # Factual complexity similarity
        complexity1 = features1.get("factual_complexity", 0)
        complexity2 = features2.get("factual_complexity", 0)
        if complexity1 + complexity2 > 0:
            complexity_similarity = 1 - abs(complexity1 - complexity2) / (complexity1 + complexity2)
            similarity_scores.append(complexity_similarity)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0

class JudgmentPredictor:
    """
    Main judgment prediction engine using precedent analysis and machine learning
    """
    
    def __init__(self, rag_system: LegalRAGSystem):
        """
        Initialize judgment predictor
        
        Args:
            rag_system: Legal RAG system for retrieving precedents
        """
        self.rag_system = rag_system
        self.similarity_analyzer = CaseSimilarityAnalyzer()
        self.llm = ChatVertexAI(temperature=0.1, model_name="gemini-2.5-flash")
        
        # Outcome patterns for extraction
        self.outcome_patterns = {
            JudgmentOutcome.ALLOWED: [
                "petition allowed", "appeal allowed", "application allowed",
                "writ allowed", "granted", "in favor of petitioner",
                "petition is allowed", "appeal is allowed"
            ],
            JudgmentOutcome.DISMISSED: [
                "petition dismissed", "appeal dismissed", "application dismissed",
                "writ dismissed", "dismissed", "rejected", "petition is dismissed"
            ],
            JudgmentOutcome.PARTLY_ALLOWED: [
                "partly allowed", "partially allowed", "allowed in part",
                "partly granted", "partially granted"
            ],
            JudgmentOutcome.REMANDED: [
                "remanded", "remand", "sent back", "returned to",
                "remanded for", "case remanded"
            ],
            JudgmentOutcome.QUASHED: [
                "quashed", "set aside", "quash", "orders quashed",
                "judgment quashed", "order set aside"
            ]
        }
        
        logger.info("Judgment Predictor initialized")
    
    def predict_judgment(self, case_facts: str, case_issues: str = None, 
                        case_type: CaseType = None, k_similar: int = 10) -> PredictionResult:
        """
        Predict judgment outcome for a pending case
        
        Args:
            case_facts: Facts of the pending case
            case_issues: Legal issues (optional)
            case_type: Type of case (optional)
            k_similar: Number of similar cases to consider
            
        Returns:
            Prediction result with outcome probabilities and reasoning
        """
        try:
            # Prepare case text for analysis
            case_text = case_facts
            if case_issues:
                case_text += f" Issues: {case_issues}"
            
            # Find similar precedent cases
            similar_cases = self._find_similar_cases(case_text, case_type, k_similar)
            
            if not similar_cases:
                return self._create_no_precedent_result(case_text)
            
            # Analyze outcomes of similar cases
            outcome_analysis = self._analyze_precedent_outcomes(similar_cases)
            
            # Generate prediction reasoning
            reasoning = self._generate_prediction_reasoning(
                case_text, similar_cases, outcome_analysis
            )
            
            # Determine key factors
            key_factors = self._extract_key_factors(case_text, similar_cases)
            
            # Create prediction result
            predicted_outcome = max(outcome_analysis["probabilities"].items(), 
                                  key=lambda x: x[1])[0]
            
            return PredictionResult(
                predicted_outcome=JudgmentOutcome(predicted_outcome),
                confidence=outcome_analysis["confidence"],
                probability_distribution=outcome_analysis["probabilities"],
                similar_cases=similar_cases[:5],  # Top 5 most similar
                key_factors=key_factors,
                reasoning=reasoning,
                disclaimer=self._get_legal_disclaimer()
            )
            
        except Exception as e:
            logger.error(f"Error in judgment prediction: {e}")
            return self._create_error_result(str(e))
    
    def _find_similar_cases(self, case_text: str, case_type: CaseType = None, 
                           k: int = 10) -> List[PrecedentCase]:
        """Find similar precedent cases using RAG system"""
        # Query RAG system for similar cases
        search_query = f"Cases with similar facts and issues: {case_text[:500]}"
        
        # Focus on Facts, Issues, and Decisions for similarity
        rag_response = self.rag_system.query_legal_rag(
            search_query,
            specific_roles=[
                RhetoricalRole.FACTS.value,
                RhetoricalRole.ISSUE.value,
                RhetoricalRole.REASONING.value,
                RhetoricalRole.DECISION.value
            ],
            k=k * 2  # Get more initially for filtering
        )
        
        # Extract case information from retrieved documents
        similar_cases = []
        retrieved_docs = rag_response.get("retrieved_docs", [])
        
        for doc in retrieved_docs:
            case_info = self._extract_case_info_from_doc(doc)
            if case_info:
                # Calculate similarity
                doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                similarity = self.similarity_analyzer.calculate_similarity(case_text, doc_text)
                case_info.similarity_score = similarity
                similar_cases.append(case_info)
        
        # Sort by similarity and return top k
        similar_cases.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_cases[:k]
    
    def _extract_case_info_from_doc(self, doc) -> Optional[PrecedentCase]:
        """Extract case information from retrieved document"""
        try:
            # Extract text content
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                metadata = getattr(doc, 'metadata', {})
            else:
                content = str(doc)
                metadata = {}
            
            # Extract outcome from decision text
            outcome = self._extract_outcome_from_text(content)
            if not outcome:
                return None
            
            # Extract basic case info
            case_name = metadata.get('case_name', 'Unknown Case')
            court = metadata.get('court', 'Unknown Court')
            citation = metadata.get('citation', 'No Citation')
            
            # Extract year from citation or content
            year_match = re.search(r'\b(19|20)\d{2}\b', citation + content)
            year = int(year_match.group()) if year_match else 2020
            
            # Determine case type (simplified)
            case_type = self._determine_case_type(content)
            
            return PrecedentCase(
                case_id=metadata.get('doc_id', f'case_{hash(content)}'),
                case_name=case_name,
                facts=content[:500],  # First 500 chars as facts
                issues="",  # Would need more sophisticated extraction
                reasoning="",  # Would need more sophisticated extraction
                decision=content[-200:],  # Last 200 chars as decision
                outcome=outcome,
                case_type=case_type,
                court=court,
                year=year,
                citation=citation
            )
            
        except Exception as e:
            logger.warning(f"Error extracting case info: {e}")
            return None
    
    def _extract_outcome_from_text(self, text: str) -> Optional[JudgmentOutcome]:
        """Extract judgment outcome from decision text"""
        text_lower = text.lower()
        
        for outcome, patterns in self.outcome_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return outcome
        
        return None
    
    def _determine_case_type(self, content: str) -> CaseType:
        """Determine case type from content (simplified classification)"""
        content_lower = content.lower()
        
        # Simple keyword-based classification
        if any(word in content_lower for word in ["criminal", "arrest", "bail", "conviction"]):
            return CaseType.CRIMINAL
        elif any(word in content_lower for word in ["constitutional", "fundamental rights", "article"]):
            return CaseType.CONSTITUTIONAL
        elif any(word in content_lower for word in ["contract", "commercial", "business"]):
            return CaseType.COMMERCIAL
        elif any(word in content_lower for word in ["marriage", "divorce", "family"]):
            return CaseType.FAMILY
        else:
            return CaseType.CIVIL
    
    def _analyze_precedent_outcomes(self, similar_cases: List[PrecedentCase]) -> Dict[str, Any]:
        """Analyze outcomes of similar precedent cases"""
        if not similar_cases:
            return {"probabilities": {}, "confidence": 0.0}
        
        # Count outcomes weighted by similarity
        outcome_weights = defaultdict(float)
        total_weight = 0.0
        
        for case in similar_cases:
            weight = case.similarity_score
            outcome_weights[case.outcome.value] += weight
            total_weight += weight
        
        # Calculate probabilities
        probabilities = {}
        for outcome, weight in outcome_weights.items():
            probabilities[outcome] = weight / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on agreement and similarity scores
        max_prob = max(probabilities.values()) if probabilities else 0.0
        avg_similarity = np.mean([case.similarity_score for case in similar_cases])
        confidence = min(max_prob * avg_similarity, 1.0)
        
        return {
            "probabilities": probabilities,
            "confidence": confidence,
            "total_cases": len(similar_cases),
            "avg_similarity": avg_similarity
        }
    
    def _generate_prediction_reasoning(self, case_text: str, 
                                     similar_cases: List[PrecedentCase],
                                     outcome_analysis: Dict[str, Any]) -> str:
        """Generate detailed reasoning for the prediction"""
        top_cases = similar_cases[:3]
        probabilities = outcome_analysis["probabilities"]
        
        reasoning_prompt = f"""
        As a legal AI analyst, provide a detailed reasoning for judgment prediction based on the following:
        
        PENDING CASE:
        {case_text[:1000]}
        
        SIMILAR PRECEDENT CASES:
        {self._format_precedent_cases_for_prompt(top_cases)}
        
        OUTCOME PROBABILITIES:
        {self._format_probabilities_for_prompt(probabilities)}
        
        Provide a structured analysis including:
        1. Key factual similarities with precedents
        2. Relevant legal principles from similar cases
        3. Factors favoring different outcomes
        4. Most likely outcome with reasoning
        5. Potential distinguishing factors
        
        Keep the analysis legal, precise, and based on precedent analysis.
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=reasoning_prompt)])
            return response.content
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return "Unable to generate detailed reasoning due to technical limitations."
    
    def _extract_key_factors(self, case_text: str, 
                           similar_cases: List[PrecedentCase]) -> List[str]:
        """Extract key factors that influence the prediction"""
        factors = []
        
        # Extract from case text
        case_lower = case_text.lower()
        
        # Legal concept factors
        legal_factors = [
            "fundamental rights violation", "due process", "natural justice",
            "constitutional validity", "statutory interpretation", "precedent binding",
            "factual dispute", "procedural irregularity", "jurisdiction issue"
        ]
        
        for factor in legal_factors:
            if factor in case_lower:
                factors.append(factor.title())
        
        # Outcome-influencing factors from similar cases
        outcome_factors = Counter()
        for case in similar_cases:
            case_content = (case.facts + case.reasoning + case.decision).lower()
            for factor in legal_factors:
                if factor in case_content:
                    outcome_factors[factor] += case.similarity_score
        
        # Add most relevant factors from precedents
        for factor, weight in outcome_factors.most_common(3):
            if factor.title() not in factors:
                factors.append(f"Precedent factor: {factor.title()}")
        
        return factors[:10]  # Limit to top 10 factors
    
    def _format_precedent_cases_for_prompt(self, cases: List[PrecedentCase]) -> str:
        """Format precedent cases for LLM prompt"""
        formatted = []
        for i, case in enumerate(cases, 1):
            formatted.append(f"""
            Case {i}: {case.case_name}
            Court: {case.court}
            Year: {case.year}
            Outcome: {case.outcome.value.title()}
            Similarity: {case.similarity_score:.2f}
            Facts: {case.facts[:300]}...
            """)
        return "\n".join(formatted)
    
    def _format_probabilities_for_prompt(self, probabilities: Dict[str, float]) -> str:
        """Format probabilities for LLM prompt"""
        formatted = []
        for outcome, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            formatted.append(f"- {outcome.title()}: {prob:.1%}")
        return "\n".join(formatted)
    
    def _get_legal_disclaimer(self) -> str:
        """Get legal disclaimer for predictions"""
        return """
        **IMPORTANT LEGAL DISCLAIMER**: This prediction is generated for informational 
        purposes only and should not be considered as legal advice. Actual court decisions 
        depend on numerous factors including specific facts, applicable law, judicial 
        interpretation, and procedural considerations that may not be fully captured in 
        this analysis. Always consult with qualified legal counsel for matters requiring 
        legal advice.
        """
    
    def _create_no_precedent_result(self, case_text: str) -> PredictionResult:
        """Create result when no similar precedents are found"""
        return PredictionResult(
            predicted_outcome=JudgmentOutcome.DISMISSED,  # Conservative default
            confidence=0.1,
            probability_distribution={"unknown": 1.0},
            similar_cases=[],
            key_factors=["No similar precedents found"],
            reasoning="Unable to make a reliable prediction due to lack of similar precedent cases in the database.",
            disclaimer=self._get_legal_disclaimer()
        )
    
    def _create_error_result(self, error_msg: str) -> PredictionResult:
        """Create result for error cases"""
        return PredictionResult(
            predicted_outcome=JudgmentOutcome.DISMISSED,
            confidence=0.0,
            probability_distribution={"error": 1.0},
            similar_cases=[],
            key_factors=["Analysis error"],
            reasoning=f"Error in prediction analysis: {error_msg}",
            disclaimer=self._get_legal_disclaimer()
        )

# Example usage
if __name__ == "__main__":
    # This would require initialized RAG system
    from .legal_rag import LegalRAGSystem
    
    # Initialize (in practice, would use existing RAG system)
    rag_system = LegalRAGSystem()
    predictor = JudgmentPredictor(rag_system)
    
    # Example case for prediction
    sample_facts = """
    The petitioner was arrested without warrant for alleged theft. 
    The arrest was made at night without following proper procedures. 
    The petitioner claims violation of fundamental rights under Article 21.
    No FIR was registered at the time of arrest.
    """
    
    sample_issues = """
    Whether arrest without warrant and proper procedure violates Article 21?
    Whether the arrest was justified under the circumstances?
    """
    
    # Make prediction
    result = predictor.predict_judgment(
        case_facts=sample_facts,
        case_issues=sample_issues,
        case_type=CaseType.CRIMINAL
    )
    
    print("Prediction Result:")
    print(f"Predicted Outcome: {result.predicted_outcome.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Probabilities: {result.probability_distribution}")
    print(f"Key Factors: {result.key_factors}")
    print(f"Reasoning: {result.reasoning[:500]}...")
    print(f"Similar Cases Found: {len(result.similar_cases)}")