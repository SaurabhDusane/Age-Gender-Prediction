from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import uuid

from src.database.database import get_db
from src.database.repository import FaceRepository
from src.api.schemas import FaceResponse, FaceSearchQuery, FaceStatistics

router = APIRouter()


@router.get("/faces/{face_id}", response_model=FaceResponse)
async def get_face(face_id: str, db: Session = Depends(get_db)):
    try:
        face_uuid = uuid.UUID(face_id)
        face_repo = FaceRepository(db)
        face = face_repo.get_face_by_id(face_uuid)
        
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")
        
        return {
            "id": str(face.id),
            "session_id": str(face.session_id),
            "timestamp": face.timestamp,
            "bbox": [face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2],
            "age": face.age,
            "gender": face.gender,
            "emotion": face.emotion,
            "ethnicity": face.ethnicity,
            "quality_score": face.quality_score
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/faces/search", response_model=List[FaceResponse])
async def search_faces(
    age_min: Optional[float] = Query(None),
    age_max: Optional[float] = Query(None),
    gender: Optional[str] = Query(None),
    emotion: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    try:
        face_repo = FaceRepository(db)
        
        session_uuid = uuid.UUID(session_id) if session_id else None
        
        faces = face_repo.search_faces(
            age_min=age_min,
            age_max=age_max,
            gender=gender,
            emotion=emotion,
            session_id=session_uuid,
            limit=limit
        )
        
        return [
            {
                "id": str(face.id),
                "session_id": str(face.session_id),
                "timestamp": face.timestamp,
                "bbox": [face.bbox_x1, face.bbox_y1, face.bbox_x2, face.bbox_y2],
                "age": face.age,
                "gender": face.gender,
                "emotion": face.emotion,
                "ethnicity": face.ethnicity,
                "quality_score": face.quality_score
            }
            for face in faces
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/faces/statistics", response_model=FaceStatistics)
async def get_face_statistics(
    session_id: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    try:
        face_repo = FaceRepository(db)
        
        session_uuid = uuid.UUID(session_id) if session_id else None
        
        stats = face_repo.get_face_statistics(session_uuid)
        
        return stats
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/faces/{face_id}")
async def delete_face(face_id: str, db: Session = Depends(get_db)):
    try:
        face_uuid = uuid.UUID(face_id)
        face_repo = FaceRepository(db)
        
        face = face_repo.get_face_by_id(face_uuid)
        if not face:
            raise HTTPException(status_code=404, detail="Face not found")
        
        db.delete(face)
        db.commit()
        
        return {"success": True, "message": "Face deleted successfully"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
