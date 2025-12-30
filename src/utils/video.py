import cv2
import numpy as np
from typing import Optional, Generator
import time


class VideoStream:
    def __init__(
        self,
        source: int = 0,
        fps: Optional[int] = None,
        resolution: Optional[tuple] = None
    ):
        self.source = source
        self.fps = fps
        self.resolution = resolution
        
        self.cap = None
        self.is_running = False
        self._init_capture()
    
    def _init_capture(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source)
        elif isinstance(self.source, str):
            self.cap = cv2.VideoCapture(self.source)
        else:
            raise ValueError(f"Invalid source: {self.source}")
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        
        if self.resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.is_running = True
    
    def __iter__(self) -> Generator[np.ndarray, None, None]:
        while self.is_running:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            yield frame
    
    def read(self) -> tuple:
        return self.cap.read()
    
    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.is_running = False
    
    def __del__(self):
        self.release()


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        resolution: tuple = (640, 480),
        codec: str = 'mp4v'
    ):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, resolution
        )
    
    def write(self, frame: np.ndarray):
        self.writer.write(frame)
    
    def release(self):
        if self.writer is not None:
            self.writer.release()
    
    def __del__(self):
        self.release()
