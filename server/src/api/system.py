"""
System information and health check endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends

from .auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/system", tags=["system"])

@router.get("/info")
async def get_system_info(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get system information and statistics
    """
    return {
        "system_name": "Nyaya - Legal Document Analysis System",
        "version": "1.0.0",
        "features": [
            "Rhetorical Role Classification",
            "Role-Aware RAG System",
            "Multi-turn Conversations",
            "Document Processing",
            "Judgment Prediction",
            "Precedent Analysis"
        ],
        "supported_formats": ["PDF", "TXT"],
        "rhetorical_roles": [
            "Facts",
            "Issue", 
            "Arguments of Petitioner",
            "Arguments of Respondent",
            "Reasoning",
            "Decision",
            "None"
        ],
        "case_types": [
            "civil",
            "criminal", 
            "constitutional",
            "commercial",
            "family",
            "tax",
            "labor",
            "property"
        ]
    }

@router.get("/stats")
async def get_system_stats(user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get system usage statistics
    """
    # This would be implemented with proper tracking
    return {
        "documents_processed": 0,
        "queries_handled": 0,
        "active_conversations": 0,
        "predictions_made": 0,
        "uptime": "System just started"
    }