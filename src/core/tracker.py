import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass
import time


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    embedding: np.ndarray
    age: float
    gender: str
    emotion: str
    age_history: deque
    gender_history: deque
    emotion_history: deque
    hits: int = 0
    age_since_update: int = 0
    last_seen: float = 0.0


class FaceTracker:
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 0.6,
        nn_budget: int = 100
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self.nn_budget = nn_budget
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, faces: List, frame: np.ndarray) -> List:
        self.frame_count += 1
        
        if len(self.tracks) == 0:
            for face in faces:
                self._initiate_track(face)
        else:
            matched_indices = self._match_faces_to_tracks(faces)
            
            for face_idx, track_idx in matched_indices:
                if track_idx is not None:
                    self._update_track(self.tracks[track_idx], faces[face_idx])
                else:
                    self._initiate_track(faces[face_idx])
            
            self._remove_dead_tracks()
        
        for i, face in enumerate(faces):
            for track in self.tracks:
                if self._compute_iou(face.bbox, track.bbox) > self.iou_threshold:
                    if track.hits >= self.min_hits:
                        face.track_id = track.track_id
                    break
        
        return faces
    
    def _match_faces_to_tracks(self, faces: List) -> List[tuple]:
        if len(self.tracks) == 0 or len(faces) == 0:
            return [(i, None) for i in range(len(faces))]
        
        cost_matrix = np.zeros((len(faces), len(self.tracks)))
        
        for i, face in enumerate(faces):
            for j, track in enumerate(self.tracks):
                iou = self._compute_iou(face.bbox, track.bbox)
                
                if face.embedding is not None and track.embedding is not None:
                    embedding_sim = np.dot(face.embedding, track.embedding)
                    cost = 1.0 - (0.5 * iou + 0.5 * embedding_sim)
                else:
                    cost = 1.0 - iou
                
                cost_matrix[i, j] = cost
        
        matches = []
        assigned_tracks = set()
        assigned_faces = set()
        
        for face_idx in range(len(faces)):
            if face_idx in assigned_faces:
                continue
            
            min_cost = float('inf')
            best_track = None
            
            for track_idx in range(len(self.tracks)):
                if track_idx in assigned_tracks:
                    continue
                
                cost = cost_matrix[face_idx, track_idx]
                if cost < min_cost and cost < self.max_distance:
                    min_cost = cost
                    best_track = track_idx
            
            if best_track is not None:
                matches.append((face_idx, best_track))
                assigned_tracks.add(best_track)
                assigned_faces.add(face_idx)
            else:
                matches.append((face_idx, None))
        
        return matches
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _initiate_track(self, face):
        track = Track(
            track_id=self.next_id,
            bbox=face.bbox,
            embedding=face.embedding,
            age=face.age,
            gender=face.gender,
            emotion=face.emotion,
            age_history=deque(maxlen=self.nn_budget),
            gender_history=deque(maxlen=self.nn_budget),
            emotion_history=deque(maxlen=self.nn_budget),
            hits=1,
            age_since_update=0,
            last_seen=time.time()
        )
        
        track.age_history.append(face.age)
        track.gender_history.append(face.gender)
        track.emotion_history.append(face.emotion)
        
        self.tracks.append(track)
        self.next_id += 1
    
    def _update_track(self, track: Track, face):
        track.bbox = face.bbox
        track.embedding = face.embedding
        track.age = face.age
        track.gender = face.gender
        track.emotion = face.emotion
        
        track.age_history.append(face.age)
        track.gender_history.append(face.gender)
        track.emotion_history.append(face.emotion)
        
        track.hits += 1
        track.age_since_update = 0
        track.last_seen = time.time()
    
    def _remove_dead_tracks(self):
        self.tracks = [
            track for track in self.tracks
            if track.age_since_update < self.max_age
        ]
        
        for track in self.tracks:
            track.age_since_update += 1
    
    def reset(self):
        self.tracks.clear()
        self.next_id = 1
        self.frame_count = 0
