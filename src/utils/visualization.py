import cv2
import numpy as np
from typing import List, Optional


def draw_face_box(
    frame: np.ndarray,
    bbox: List[float],
    color: tuple = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_face_info(
    frame: np.ndarray,
    bbox: List[float],
    age: float,
    gender: str,
    emotion: str,
    confidence: Optional[float] = None,
    track_id: Optional[int] = None,
    font_scale: float = 0.6,
    thickness: int = 2
) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    
    info_lines = [
        f"Age: {age:.1f}",
        f"Gender: {gender}",
        f"Emotion: {emotion}"
    ]
    
    if track_id is not None:
        info_lines.insert(0, f"ID: {track_id}")
    
    if confidence is not None:
        info_lines.append(f"Conf: {confidence:.2f}")
    
    y_offset = y1 - 10
    for line in reversed(info_lines):
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        cv2.rectangle(
            frame,
            (x1, y_offset - text_size[1] - 5),
            (x1 + text_size[0] + 5, y_offset),
            (0, 0, 0),
            -1
        )
        
        cv2.putText(
            frame,
            line,
            (x1 + 2, y_offset - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        y_offset -= text_size[1] + 10
    
    return frame


def draw_landmarks(
    frame: np.ndarray,
    landmarks: np.ndarray,
    color: tuple = (0, 255, 255),
    radius: int = 1
) -> np.ndarray:
    for point in landmarks:
        x, y = map(int, point)
        cv2.circle(frame, (x, y), radius, color, -1)
    
    return frame


def create_dashboard(
    frame: np.ndarray,
    fps: float,
    num_faces: int,
    processing_time: float
) -> np.ndarray:
    h, w = frame.shape[:2]
    
    dashboard_height = 60
    dashboard = np.zeros((dashboard_height, w, 3), dtype=np.uint8)
    
    info_text = [
        f"FPS: {fps:.1f}",
        f"Faces: {num_faces}",
        f"Time: {processing_time*1000:.1f}ms"
    ]
    
    x_offset = 10
    for text in info_text:
        cv2.putText(
            dashboard,
            text,
            (x_offset, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        x_offset += 200
    
    result = np.vstack([frame, dashboard])
    return result
