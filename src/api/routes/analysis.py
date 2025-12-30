from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import cv2
import numpy as np
from io import BytesIO
import uuid

from src.core.pipeline import DemographicPipeline
from src.database.database import get_db
from src.database.repository import FaceRepository, SessionRepository, AuditRepository
from src.api.schemas import AnalysisResponse, FaceData

router = APIRouter()

pipeline = DemographicPipeline(
    device="cuda",
    enable_tracking=False,
    enable_temporal_smoothing=False
)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    save_to_db: bool = Form(False),
    session_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        result = pipeline.process(frame)
        
        faces_data = []
        for face in result.faces:
            face_dict = {
                "bbox": face.bbox,
                "confidence": face.confidence,
                "age": face.age,
                "age_std": face.age_std,
                "age_confidence": face.age_confidence,
                "gender": face.gender,
                "gender_confidence": face.gender_confidence,
                "emotion": face.emotion,
                "emotion_confidence": face.emotion_confidence,
                "ethnicity": face.ethnicity,
                "ethnicity_confidence": face.ethnicity_confidence,
                "quality_score": face.quality_score,
                "attributes": face.attributes
            }
            faces_data.append(face_dict)
            
            if save_to_db:
                face_repo = FaceRepository(db)
                
                db_session_id = uuid.UUID(session_id) if session_id else uuid.uuid4()
                
                face_record_data = {
                    "session_id": db_session_id,
                    "bbox_x1": face.bbox[0],
                    "bbox_y1": face.bbox[1],
                    "bbox_x2": face.bbox[2],
                    "bbox_y2": face.bbox[3],
                    "age": face.age,
                    "age_std": face.age_std,
                    "age_confidence": face.age_confidence,
                    "gender": face.gender,
                    "gender_confidence": face.gender_confidence,
                    "emotion": face.emotion,
                    "emotion_confidence": face.emotion_confidence,
                    "ethnicity": face.ethnicity,
                    "ethnicity_confidence": face.ethnicity_confidence,
                    "quality_score": face.quality_score,
                    "blur_score": face.blur_score,
                }
                
                if face.pose_angles:
                    face_record_data.update({
                        "pose_yaw": face.pose_angles.get("yaw"),
                        "pose_pitch": face.pose_angles.get("pitch"),
                        "pose_roll": face.pose_angles.get("roll"),
                    })
                
                face_record = face_repo.create_face_record(face_record_data)
                
                if face.embedding is not None:
                    face_repo.create_embedding(face_record.id, face.embedding)
                
                db.commit()
                
                audit_repo = AuditRepository(db)
                audit_repo.log_action(
                    action="face_analysis",
                    entity_type="face_record",
                    entity_id=face_record.id
                )
                db.commit()
        
        return {
            "success": True,
            "num_faces": len(faces_data),
            "faces": faces_data,
            "processing_time": result.processing_time,
            "fps": result.fps
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/batch")
async def analyze_batch(
    files: List[UploadFile] = File(...),
    save_to_db: bool = Form(False),
    db: Session = Depends(get_db)
):
    results = []
    
    for file in files:
        try:
            result = await analyze_image(file, save_to_db, None, db)
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }
