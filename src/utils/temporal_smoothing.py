import numpy as np
from typing import List, Dict
from collections import defaultdict, deque


class TemporalSmoother:
    def __init__(
        self,
        window_size: int = 5,
        method: str = "exponential",
        alpha: float = 0.3
    ):
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        
        self.age_history = defaultdict(lambda: deque(maxlen=window_size))
        self.gender_history = defaultdict(lambda: deque(maxlen=window_size))
        self.emotion_history = defaultdict(lambda: deque(maxlen=window_size))
    
    def smooth(self, faces: List) -> List:
        for face in faces:
            if face.track_id is None:
                continue
            
            track_id = face.track_id
            
            self.age_history[track_id].append(face.age)
            self.gender_history[track_id].append(face.gender)
            self.emotion_history[track_id].append(face.emotion)
            
            if self.method == "exponential":
                face.age = self._exponential_smoothing(
                    self.age_history[track_id], face.age
                )
            elif self.method == "moving_average":
                face.age = np.mean(list(self.age_history[track_id]))
            elif self.method == "median":
                face.age = np.median(list(self.age_history[track_id]))
            
            face.gender = self._mode(self.gender_history[track_id])
            face.emotion = self._mode(self.emotion_history[track_id])
        
        return faces
    
    def _exponential_smoothing(self, history: deque, current_value: float) -> float:
        if len(history) == 1:
            return current_value
        
        prev_values = list(history)[:-1]
        if len(prev_values) == 0:
            return current_value
        
        prev_smoothed = prev_values[-1]
        smoothed = self.alpha * current_value + (1 - self.alpha) * prev_smoothed
        
        return smoothed
    
    def _mode(self, history: deque) -> str:
        if len(history) == 0:
            return "unknown"
        
        values = list(history)
        return max(set(values), key=values.count)
    
    def reset(self):
        self.age_history.clear()
        self.gender_history.clear()
        self.emotion_history.clear()
