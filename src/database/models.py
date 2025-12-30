from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from datetime import datetime
import uuid

Base = declarative_base()


class FaceRecord(Base):
    __tablename__ = 'face_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    bbox_x1 = Column(Float, nullable=False)
    bbox_y1 = Column(Float, nullable=False)
    bbox_x2 = Column(Float, nullable=False)
    bbox_y2 = Column(Float, nullable=False)
    
    age = Column(Float, nullable=False)
    age_std = Column(Float, default=0.0)
    age_confidence = Column(Float, default=0.0)
    
    gender = Column(String(20), nullable=False)
    gender_confidence = Column(Float, default=0.0)
    
    emotion = Column(String(20))
    emotion_confidence = Column(Float, default=0.0)
    
    ethnicity = Column(String(50))
    ethnicity_confidence = Column(Float)
    
    track_id = Column(Integer, index=True)
    
    quality_score = Column(Float, default=0.0)
    blur_score = Column(Float, default=0.0)
    
    pose_yaw = Column(Float)
    pose_pitch = Column(Float)
    pose_roll = Column(Float)
    
    has_glasses = Column(Boolean, default=False)
    has_facial_hair = Column(Boolean, default=False)
    has_mask = Column(Boolean, default=False)
    
    skin_tone = Column(String(20))
    hair_color = Column(String(20))
    face_shape = Column(String(20))
    
    metadata = Column(JSON)
    
    frame_idx = Column(Integer)
    source_identifier = Column(String(255))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FaceEmbedding(Base):
    __tablename__ = 'face_embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    face_record_id = Column(UUID(as_uuid=True), ForeignKey('face_records.id'), index=True)
    
    embedding = Column(ARRAY(Float, dimensions=1))
    
    created_at = Column(DateTime, default=datetime.utcnow)


class Session(Base):
    __tablename__ = 'sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255))
    description = Column(Text)
    
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    
    source_type = Column(String(50))
    source_identifier = Column(String(255))
    
    total_frames = Column(Integer, default=0)
    total_faces = Column(Integer, default=0)
    avg_fps = Column(Float)
    
    metadata = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(UUID(as_uuid=True))
    
    user_id = Column(String(255))
    ip_address = Column(String(50))
    
    details = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class ConsentRecord(Base):
    __tablename__ = 'consent_records'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.id'))
    
    consent_given = Column(Boolean, nullable=False)
    consent_timestamp = Column(DateTime, default=datetime.utcnow)
    
    user_identifier = Column(String(255))
    ip_address = Column(String(50))
    
    consent_text = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class BiasMetric(Base):
    __tablename__ = 'bias_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    metric_name = Column(String(100), nullable=False)
    demographic_group = Column(String(100))
    
    value = Column(Float, nullable=False)
    threshold = Column(Float)
    
    passed = Column(Boolean)
    
    details = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
