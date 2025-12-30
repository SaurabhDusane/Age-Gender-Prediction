from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime

from src.database.database import get_db
from src.database.repository import SessionRepository, ConsentRepository
from src.api.schemas import SessionCreate, SessionResponse

router = APIRouter()


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db)
):
    try:
        session_repo = SessionRepository(db)
        
        session_dict = {
            "name": session_data.name,
            "description": session_data.description,
            "source_type": session_data.source_type,
            "source_identifier": session_data.source_identifier,
            "metadata": session_data.metadata
        }
        
        session = session_repo.create_session(session_dict)
        db.commit()
        
        if session_data.consent_given is not None:
            consent_repo = ConsentRepository(db)
            consent_repo.record_consent(
                session_id=session.id,
                consent_given=session_data.consent_given,
                consent_text="User provided consent for demographic analysis"
            )
            db.commit()
        
        return {
            "id": str(session.id),
            "name": session.name,
            "description": session.description,
            "start_time": session.start_time,
            "source_type": session.source_type,
            "source_identifier": session.source_identifier,
            "total_frames": session.total_frames,
            "total_faces": session.total_faces
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, db: Session = Depends(get_db)):
    try:
        session_uuid = uuid.UUID(session_id)
        session_repo = SessionRepository(db)
        session = session_repo.get_session_by_id(session_uuid)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "id": str(session.id),
            "name": session.name,
            "description": session.description,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "source_type": session.source_type,
            "source_identifier": session.source_identifier,
            "total_frames": session.total_frames,
            "total_faces": session.total_faces,
            "avg_fps": session.avg_fps
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    try:
        session_repo = SessionRepository(db)
        sessions = session_repo.list_sessions(limit=limit)
        
        return [
            {
                "id": str(s.id),
                "name": s.name,
                "description": s.description,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "source_type": s.source_type,
                "total_frames": s.total_frames,
                "total_faces": s.total_faces,
                "avg_fps": s.avg_fps
            }
            for s in sessions
        ]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sessions/{session_id}/end")
async def end_session(session_id: str, db: Session = Depends(get_db)):
    try:
        session_uuid = uuid.UUID(session_id)
        session_repo = SessionRepository(db)
        
        session_repo.update_session(
            session_uuid,
            {"end_time": datetime.utcnow()}
        )
        db.commit()
        
        return {"success": True, "message": "Session ended successfully"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    try:
        session_uuid = uuid.UUID(session_id)
        session_repo = SessionRepository(db)
        
        session_repo.delete_session(session_uuid)
        db.commit()
        
        return {"success": True, "message": "Session deleted successfully"}
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
