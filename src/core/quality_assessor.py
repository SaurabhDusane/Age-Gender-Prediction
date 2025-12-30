import cv2
import numpy as np
from typing import Dict, List


class QualityAssessor:
    def __init__(
        self,
        blur_threshold: float = 100.0,
        min_quality_score: float = 0.3,
        max_pose_angle: float = 45.0
    ):
        self.blur_threshold = blur_threshold
        self.min_quality_score = min_quality_score
        self.max_pose_angle = max_pose_angle
    
    def assess(self, face_img: np.ndarray, bbox: List[float]) -> Dict:
        blur_score = self._assess_blur(face_img)
        brightness_score = self._assess_brightness(face_img)
        pose_angles = self._estimate_pose(face_img)
        occlusion_score = self._assess_occlusion(face_img)
        
        quality_score = self._compute_quality_score(
            blur_score, brightness_score, pose_angles, occlusion_score
        )
        
        return {
            'quality_score': quality_score,
            'blur_score': blur_score,
            'brightness_score': brightness_score,
            'pose_angles': pose_angles,
            'occlusion_score': occlusion_score,
            'is_acceptable': quality_score >= self.min_quality_score
        }
    
    def _assess_blur(self, face_img: np.ndarray) -> float:
        if face_img.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        blur_score = min(laplacian_var / self.blur_threshold, 1.0)
        return float(blur_score)
    
    def _assess_brightness(self, face_img: np.ndarray) -> float:
        if face_img.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 50:
            return mean_brightness / 50.0
        elif mean_brightness > 200:
            return (255 - mean_brightness) / 55.0
        else:
            return 1.0
    
    def _estimate_pose(self, face_img: np.ndarray) -> Dict[str, float]:
        h, w = face_img.shape[:2]
        
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, fw, fh = faces[0]
                center_x = x + fw // 2
                center_y = y + fh // 2
                
                yaw = ((center_x - w // 2) / (w // 2)) * 45
                pitch = ((center_y - h // 2) / (h // 2)) * 30
                roll = 0.0
            else:
                yaw, pitch, roll = 0.0, 0.0, 0.0
        
        except:
            yaw, pitch, roll = 0.0, 0.0, 0.0
        
        return {
            'yaw': float(yaw),
            'pitch': float(pitch),
            'roll': float(roll)
        }
    
    def _assess_occlusion(self, face_img: np.ndarray) -> float:
        if face_img.size == 0:
            return 1.0
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        white_pixels = np.sum(binary == 255)
        total_pixels = binary.size
        visible_ratio = white_pixels / total_pixels
        
        occlusion_score = 1.0 - abs(visible_ratio - 0.5) * 2
        return float(max(0.0, occlusion_score))
    
    def _compute_quality_score(
        self,
        blur_score: float,
        brightness_score: float,
        pose_angles: Dict[str, float],
        occlusion_score: float
    ) -> float:
        pose_score = 1.0
        for angle_name, angle_value in pose_angles.items():
            if angle_name in ['yaw', 'pitch']:
                pose_score *= max(0.0, 1.0 - abs(angle_value) / self.max_pose_angle)
        
        quality_score = (
            0.3 * blur_score +
            0.2 * brightness_score +
            0.3 * pose_score +
            0.2 * occlusion_score
        )
        
        return float(max(0.0, min(1.0, quality_score)))
