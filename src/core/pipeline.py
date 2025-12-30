import torch
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import time
from collections import deque

from src.models.detector import FaceDetector
from src.models.age_estimator import AgeEstimator
from src.models.gender_classifier import GenderClassifier
from src.models.emotion_recognizer import EmotionRecognizer
from src.models.face_embedder import FaceEmbedder
from src.models.landmark_detector import LandmarkDetector
from src.models.attribute_detector import AttributeDetector
from src.core.quality_assessor import QualityAssessor
from src.core.tracker import FaceTracker
from src.utils.temporal_smoothing import TemporalSmoother


@dataclass
class FaceResult:
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
    embedding: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    quality_score: float = 0.0
    blur_score: float = 0.0
    pose_angles: Dict[str, float] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class FrameResult:
    faces: List[FaceResult]
    frame_idx: int
    timestamp: float
    fps: float
    processing_time: float
    num_faces: int


class DemographicPipeline:
    def __init__(
        self,
        device: str = "cuda",
        enable_tracking: bool = True,
        enable_temporal_smoothing: bool = True,
        enable_quality_check: bool = True,
        enable_landmarks: bool = True,
        enable_attributes: bool = True,
        config: Optional[Dict] = None
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.enable_tracking = enable_tracking
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.enable_quality_check = enable_quality_check
        self.enable_landmarks = enable_landmarks
        self.enable_attributes = enable_attributes
        
        self.config = config or {}
        
        self.detector = FaceDetector(device=self.device)
        self.age_estimator = AgeEstimator(device=self.device, ensemble=True)
        self.gender_classifier = GenderClassifier(device=self.device)
        self.emotion_recognizer = EmotionRecognizer(device=self.device)
        self.face_embedder = FaceEmbedder(device=self.device)
        
        if self.enable_landmarks:
            self.landmark_detector = LandmarkDetector()
        
        if self.enable_attributes:
            self.attribute_detector = AttributeDetector(device=self.device)
        
        if self.enable_quality_check:
            self.quality_assessor = QualityAssessor()
        
        if self.enable_tracking:
            self.tracker = FaceTracker()
        
        if self.enable_temporal_smoothing:
            self.temporal_smoother = TemporalSmoother(window_size=5)
        
        self.frame_count = 0
        self.fps_deque = deque(maxlen=30)
        self.last_time = time.time()
    
    def process(self, frame: np.ndarray, **kwargs) -> FrameResult:
        start_time = time.time()
        
        detections = self.detector.detect(frame)
        
        faces = []
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            
            face_img = self._crop_face(frame, bbox)
            
            if self.enable_quality_check:
                quality_result = self.quality_assessor.assess(face_img, bbox)
                if quality_result['quality_score'] < 0.3:
                    continue
            else:
                quality_result = {}
            
            age_result = self.age_estimator.predict(face_img)
            gender_result = self.gender_classifier.predict(face_img)
            emotion_result = self.emotion_recognizer.predict(face_img)
            embedding = self.face_embedder.extract(face_img)
            
            landmarks = None
            if self.enable_landmarks:
                landmarks = self.landmark_detector.detect(face_img)
            
            attributes = {}
            if self.enable_attributes:
                attributes = self.attribute_detector.detect(face_img)
            
            face_result = FaceResult(
                bbox=bbox,
                confidence=conf,
                age=age_result['age'],
                age_std=age_result.get('std', 0.0),
                age_confidence=age_result.get('confidence', 0.0),
                gender=gender_result['gender'],
                gender_confidence=gender_result['confidence'],
                emotion=emotion_result['emotion'],
                emotion_confidence=emotion_result['confidence'],
                ethnicity=attributes.get('ethnicity'),
                ethnicity_confidence=attributes.get('ethnicity_confidence'),
                embedding=embedding,
                landmarks=landmarks,
                quality_score=quality_result.get('quality_score', 0.0),
                blur_score=quality_result.get('blur_score', 0.0),
                pose_angles=quality_result.get('pose_angles', {}),
                attributes=attributes
            )
            
            faces.append(face_result)
        
        if self.enable_tracking and faces:
            faces = self.tracker.update(faces, frame)
        
        if self.enable_temporal_smoothing and faces:
            faces = self.temporal_smoother.smooth(faces)
        
        processing_time = time.time() - start_time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_deque.append(current_fps)
        avg_fps = np.mean(self.fps_deque)
        
        self.frame_count += 1
        
        return FrameResult(
            faces=faces,
            frame_idx=self.frame_count,
            timestamp=time.time(),
            fps=avg_fps,
            processing_time=processing_time,
            num_faces=len(faces)
        )
    
    def _crop_face(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]
    
    def reset(self):
        if self.enable_tracking:
            self.tracker.reset()
        if self.enable_temporal_smoothing:
            self.temporal_smoother.reset()
        self.frame_count = 0
        self.fps_deque.clear()
