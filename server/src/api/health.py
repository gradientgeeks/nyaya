"""
Health check endpoints
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": "Legal Document Analysis API",
        "version": "1.0.0"
    }