"""
Role classification endpoints
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from ..models.role_classifier import RoleClassifier
from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/classify", tags=["classification"])

class RoleClassificationRequest(BaseModel):
    """Request for rhetorical role classification"""
    text: str = Field(..., description="Legal document text")
    context_mode: Optional[str] = Field("prev", description="Context mode for classification")

class RoleClassificationResponse(BaseModel):
    """Response for rhetorical role classification"""
    sentences: List[Dict[str, Any]] = Field(..., description="Classified sentences with roles")
    statistics: Dict[str, int] = Field(..., description="Role distribution statistics")

# Global role classifier (will be set by main.py)
role_classifier: Optional[RoleClassifier] = None

def set_role_classifier(classifier: RoleClassifier):
    """Set the global role classifier instance"""
    global role_classifier
    role_classifier = classifier

@router.post("/roles", response_model=RoleClassificationResponse)
async def classify_rhetorical_roles(
    request: RoleClassificationRequest,
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Classify rhetorical roles in legal document text
    """
    try:
        if not role_classifier:
            raise HTTPException(status_code=503, detail="Role classifier not initialized")
        
        # Classify rhetorical roles
        results = role_classifier.classify_document(
            document_text=request.text,
            context_mode=request.context_mode
        )
        
        # Calculate statistics
        role_stats = {}
        for result in results:
            role = result["role"]
            role_stats[role] = role_stats.get(role, 0) + 1
        
        return RoleClassificationResponse(
            sentences=results,
            statistics=role_stats
        )
        
    except Exception as e:
        logger.error(f"Error in role classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))