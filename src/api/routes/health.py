from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import torch
from typing import Dict

from src.database.database import get_db

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict:
    return {
        "status": "healthy",
        "service": "demographic-analysis-system",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)) -> Dict:
    cuda_available = torch.cuda.is_available()
    
    db_healthy = True
    try:
        db.execute("SELECT 1")
    except:
        db_healthy = False
    
    return {
        "status": "healthy" if db_healthy else "degraded",
        "service": "demographic-analysis-system",
        "version": "1.0.0",
        "components": {
            "database": "healthy" if db_healthy else "unhealthy",
            "cuda": "available" if cuda_available else "unavailable",
            "gpu_count": torch.cuda.device_count() if cuda_available else 0
        }
    }
