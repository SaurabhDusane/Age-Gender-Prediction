from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class FaceData(BaseModel):
    bbox: List[float]
    confidence: float
    age: float
    age_std: float
    age_confidence: float
    gender: str
    gender_confidence: float
    emotion: str
    emotion_confidence: float
    ethnicity: Optional[str] = None
    ethnicity_confidence: Optional[float] = None
    quality_score: float
    attributes: Dict[str, Any]


class AnalysisResponse(BaseModel):
    success: bool
    num_faces: int
    faces: List[Dict]
    processing_time: float
    fps: float


class FaceResponse(BaseModel):
    id: str
    session_id: str
    timestamp: datetime
    bbox: List[float]
    age: float
    gender: str
    emotion: str
    ethnicity: Optional[str]
    quality_score: float


class FaceSearchQuery(BaseModel):
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    gender: Optional[str] = None
    emotion: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 100


class FaceStatistics(BaseModel):
    total_faces: int
    avg_age: float
    gender_distribution: Dict[str, int]
    emotion_distribution: Dict[str, int]


class SessionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    source_type: str
    source_identifier: Optional[str] = None
    metadata: Optional[Dict] = None
    consent_given: Optional[bool] = None


class SessionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    source_type: str
    source_identifier: Optional[str] = None
    total_frames: int
    total_faces: int
    avg_fps: Optional[float] = None
