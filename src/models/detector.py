import torch
import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path


class FaceDetector:
    def __init__(
        self,
        device: str = "cuda",
        model_type: str = "yolov8",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_face_size: int = 30
    ):
        self.device = device
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_face_size = min_face_size
        
        self._load_model()
    
    def _load_model(self):
        if self.model_type == "yolov8":
            try:
                from ultralytics import YOLO
                model_path = "models/yolov8n-face.pt"
                if not Path(model_path).exists():
                    model_path = "yolov8n.pt"
                self.model = YOLO(model_path)
                if torch.cuda.is_available() and self.device == "cuda":
                    self.model.to(self.device)
            except ImportError:
                self._load_opencv_detector()
        
        elif self.model_type == "retinaface":
            try:
                from retinaface import RetinaFace
                self.model = RetinaFace
            except ImportError:
                self._load_opencv_detector()
        
        else:
            self._load_opencv_detector()
    
    def _load_opencv_detector(self):
        prototxt = "models/deploy.prototxt"
        weights = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            self.model = cv2.dnn.readNetFromCaffe(prototxt, weights)
            self.model_type = "opencv"
        except:
            self.model = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.model_type = "haar"
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        if self.model_type == "yolov8":
            return self._detect_yolo(frame)
        elif self.model_type == "retinaface":
            return self._detect_retinaface(frame)
        elif self.model_type == "opencv":
            return self._detect_opencv(frame)
        else:
            return self._detect_haar(frame)
    
    def _detect_yolo(self, frame: np.ndarray) -> List[Dict]:
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': conf
                    })
        
        return detections
    
    def _detect_retinaface(self, frame: np.ndarray) -> List[Dict]:
        faces = self.model.detect_faces(frame)
        
        detections = []
        for key, face_data in faces.items():
            x1, y1, x2, y2 = face_data['facial_area']
            conf = face_data['score']
            
            if conf >= self.confidence_threshold:
                if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'landmarks': face_data.get('landmarks', {})
                    })
        
        return detections
    
    def _detect_opencv(self, frame: np.ndarray) -> List[Dict]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self.model.setInput(blob)
        detections_raw = self.model.forward()
        
        detections = []
        for i in range(detections_raw.shape[2]):
            confidence = detections_raw[0, 0, i, 2]
            
            if confidence >= self.confidence_threshold:
                box = detections_raw[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence)
                    })
        
        return detections
    
    def _detect_haar(self, frame: np.ndarray) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [float(x), float(y), float(x + w), float(y + h)],
                'confidence': 1.0
            })
        
        return detections
