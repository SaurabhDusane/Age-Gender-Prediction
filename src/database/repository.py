from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import numpy as np

from src.database.models import (
    FaceRecord, FaceEmbedding, Session as SessionModel,
    AuditLog, ConsentRecord, BiasMetric
)


class FaceRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def create_face_record(self, face_data: Dict[str, Any]) -> FaceRecord:
        face_record = FaceRecord(**face_data)
        self.session.add(face_record)
        self.session.flush()
        return face_record
    
    def create_embedding(self, face_record_id: uuid.UUID, embedding: np.ndarray) -> FaceEmbedding:
        embedding_record = FaceEmbedding(
            face_record_id=face_record_id,
            embedding=embedding.tolist()
        )
        self.session.add(embedding_record)
        self.session.flush()
        return embedding_record
    
    def get_face_by_id(self, face_id: uuid.UUID) -> Optional[FaceRecord]:
        return self.session.query(FaceRecord).filter(FaceRecord.id == face_id).first()
    
    def search_faces(
        self,
        age_min: Optional[float] = None,
        age_max: Optional[float] = None,
        gender: Optional[str] = None,
        emotion: Optional[str] = None,
        session_id: Optional[uuid.UUID] = None,
        limit: int = 100
    ) -> List[FaceRecord]:
        query = self.session.query(FaceRecord)
        
        if age_min is not None:
            query = query.filter(FaceRecord.age >= age_min)
        
        if age_max is not None:
            query = query.filter(FaceRecord.age <= age_max)
        
        if gender is not None:
            query = query.filter(FaceRecord.gender == gender)
        
        if emotion is not None:
            query = query.filter(FaceRecord.emotion == emotion)
        
        if session_id is not None:
            query = query.filter(FaceRecord.session_id == session_id)
        
        return query.order_by(FaceRecord.timestamp.desc()).limit(limit).all()
    
    def get_face_statistics(self, session_id: Optional[uuid.UUID] = None) -> Dict:
        query = self.session.query(FaceRecord)
        
        if session_id is not None:
            query = query.filter(FaceRecord.session_id == session_id)
        
        total_faces = query.count()
        
        avg_age = query.with_entities(func.avg(FaceRecord.age)).scalar() or 0
        
        gender_dist = dict(
            query.with_entities(
                FaceRecord.gender,
                func.count(FaceRecord.id)
            ).group_by(FaceRecord.gender).all()
        )
        
        emotion_dist = dict(
            query.with_entities(
                FaceRecord.emotion,
                func.count(FaceRecord.id)
            ).group_by(FaceRecord.emotion).all()
        )
        
        return {
            'total_faces': total_faces,
            'avg_age': float(avg_age),
            'gender_distribution': gender_dist,
            'emotion_distribution': emotion_dist
        }
    
    def delete_face_records(self, session_id: uuid.UUID):
        self.session.query(FaceRecord).filter(
            FaceRecord.session_id == session_id
        ).delete()


class SessionRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def create_session(self, session_data: Dict[str, Any]) -> SessionModel:
        session_model = SessionModel(**session_data)
        self.session.add(session_model)
        self.session.flush()
        return session_model
    
    def get_session_by_id(self, session_id: uuid.UUID) -> Optional[SessionModel]:
        return self.session.query(SessionModel).filter(
            SessionModel.id == session_id
        ).first()
    
    def update_session(self, session_id: uuid.UUID, updates: Dict[str, Any]):
        self.session.query(SessionModel).filter(
            SessionModel.id == session_id
        ).update(updates)
    
    def list_sessions(self, limit: int = 50) -> List[SessionModel]:
        return self.session.query(SessionModel).order_by(
            SessionModel.created_at.desc()
        ).limit(limit).all()
    
    def delete_session(self, session_id: uuid.UUID):
        self.session.query(SessionModel).filter(
            SessionModel.id == session_id
        ).delete()


class AuditRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def log_action(
        self,
        action: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[uuid.UUID] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict] = None
    ) -> AuditLog:
        audit_log = AuditLog(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            user_id=user_id,
            ip_address=ip_address,
            details=details
        )
        self.session.add(audit_log)
        self.session.flush()
        return audit_log
    
    def get_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditLog]:
        query = self.session.query(AuditLog)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        if action:
            query = query.filter(AuditLog.action == action)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()


class ConsentRepository:
    def __init__(self, session: Session):
        self.session = session
    
    def record_consent(
        self,
        session_id: uuid.UUID,
        consent_given: bool,
        user_identifier: Optional[str] = None,
        ip_address: Optional[str] = None,
        consent_text: Optional[str] = None
    ) -> ConsentRecord:
        consent = ConsentRecord(
            session_id=session_id,
            consent_given=consent_given,
            user_identifier=user_identifier,
            ip_address=ip_address,
            consent_text=consent_text
        )
        self.session.add(consent)
        self.session.flush()
        return consent
    
    def check_consent(self, session_id: uuid.UUID) -> bool:
        consent = self.session.query(ConsentRecord).filter(
            ConsentRecord.session_id == session_id
        ).first()
        
        return consent.consent_given if consent else False
