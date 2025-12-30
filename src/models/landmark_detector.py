import cv2
import numpy as np
from typing import Optional, Dict
import mediapipe as mp


class LandmarkDetector:
    def __init__(
        self,
        num_landmarks: int = 468,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.num_landmarks = num_landmarks
        
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            self.use_mediapipe = True
        except:
            self.use_mediapipe = False
            self._init_dlib_detector()
    
    def _init_dlib_detector(self):
        try:
            import dlib
            predictor_path = "models/shape_predictor_68_face_landmarks.dat"
            self.predictor = dlib.shape_predictor(predictor_path)
            self.use_dlib = True
        except:
            self.use_dlib = False
    
    def detect(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        if self.use_mediapipe:
            return self._detect_mediapipe(face_img)
        elif hasattr(self, 'use_dlib') and self.use_dlib:
            return self._detect_dlib(face_img)
        else:
            return self._detect_simple(face_img)
    
    def _detect_mediapipe(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = face_img.shape[:2]
            
            points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])
            
            return np.array(points, dtype=np.float32)
        
        return None
    
    def _detect_dlib(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        import dlib
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        
        shape = self.predictor(gray, rect)
        
        points = []
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            points.append([x, y])
        
        return np.array(points, dtype=np.float32)
    
    def _detect_simple(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        h, w = face_img.shape[:2]
        
        landmarks = np.array([
            [w * 0.3, h * 0.4],
            [w * 0.7, h * 0.4],
            [w * 0.5, h * 0.6],
            [w * 0.35, h * 0.75],
            [w * 0.65, h * 0.75],
        ], dtype=np.float32)
        
        return landmarks
    
    def get_face_shape(self, landmarks: np.ndarray) -> str:
        if landmarks is None or len(landmarks) < 10:
            return "unknown"
        
        width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        ratio = height / width if width > 0 else 1.0
        
        if ratio > 1.3:
            return "oval"
        elif ratio < 0.9:
            return "round"
        elif 1.0 <= ratio <= 1.2:
            return "square"
        else:
            return "heart"
