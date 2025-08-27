"""
Prediction endpoints for judgment outcomes
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..core.prediction_module import JudgmentPredictor, CaseType
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/predict", tags=["prediction"])

class PredictionRequest(BaseModel):
    """Request model for judgment prediction"""
    case_facts: str = Field(..., description="Facts of the case")
    case_issues: Optional[str] = Field(None, description="Legal issues")
    case_type: Optional[str] = Field(None, description="Type of case")
    session_id: Optional[str] = Field(None, description="Conversation session ID")

class PredictionResponse(BaseModel):
    """Response model for judgment prediction"""
    predicted_outcome: str = Field(..., description="Predicted judgment outcome")
    confidence: float = Field(..., description="Prediction confidence")
    probability_distribution: Dict[str, float] = Field(..., description="Outcome probabilities")
    similar_cases: List[Dict[str, Any]] = Field(..., description="Similar precedent cases")
    key_factors: List[str] = Field(..., description="Key influencing factors")
    reasoning: str = Field(..., description="Prediction reasoning")
    disclaimer: str = Field(..., description="Legal disclaimer")

# Global prediction module (will be set by main.py)
prediction_module: Optional[JudgmentPredictor] = None

def set_prediction_module(predictor: JudgmentPredictor):
    """Set the global prediction module instance"""
    global prediction_module
    prediction_module = predictor

@router.post("/judgment", response_model=PredictionResponse)
async def predict_judgment(
    request: PredictionRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Predict judgment outcome for a pending case
    """
    try:
        if not prediction_module:
            raise HTTPException(status_code=503, detail="Prediction module not initialized")
        
        # Parse case type if provided
        case_type = None
        if request.case_type:
            try:
                case_type = CaseType(request.case_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid case type: {request.case_type}")
        
        # Make prediction
        prediction = prediction_module.predict_judgment(
            case_facts=request.case_facts,
            case_issues=request.case_issues,
            case_type=case_type
        )
        
        # Format similar cases for response
        similar_cases_data = []
        for case in prediction.similar_cases:
            similar_cases_data.append({
                "case_name": case.case_name,
                "court": case.court,
                "year": case.year,
                "outcome": case.outcome.value,
                "similarity_score": case.similarity_score,
                "citation": case.citation
            })
        
        return PredictionResponse(
            predicted_outcome=prediction.predicted_outcome.value,
            confidence=prediction.confidence,
            probability_distribution=prediction.probability_distribution,
            similar_cases=similar_cases_data,
            key_factors=prediction.key_factors,
            reasoning=prediction.reasoning,
            disclaimer=prediction.disclaimer
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in judgment prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))