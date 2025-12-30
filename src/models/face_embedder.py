import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Optional
from pathlib import Path


class ArcFaceModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        from torchvision.models import resnet100
        try:
            self.backbone = resnet100(pretrained=False)
        except:
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=True)
        
        self.backbone.fc = nn.Linear(2048, embedding_dim)
    
    def forward(self, x):
        return self.backbone(x)


class FaceEmbedder:
    def __init__(
        self,
        device: str = "cuda",
        embedding_dim: int = 512,
        normalize: bool = True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        self.model = self._load_model()
    
    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0 if self.device == "cuda" else -1)
            return app
        except:
            model = ArcFaceModel(embedding_dim=self.embedding_dim)
            model_path = Path("models/arcface_resnet100.pth")
            
            if model_path.exists():
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(state_dict)
                except:
                    pass
            
            model.to(self.device)
            model.eval()
            return model
    
    def preprocess(self, face_img: np.ndarray, target_size=(112, 112)) -> torch.Tensor:
        if face_img.size == 0:
            return torch.zeros(1, 3, *target_size).to(self.device)
        
        img = cv2.resize(face_img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img.unsqueeze(0).to(self.device)
        return img
    
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'get'):
            faces = self.model.get(face_img)
            if len(faces) > 0:
                embedding = faces[0].embedding
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                return embedding
        
        img_tensor = self.preprocess(face_img)
        
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = embedding.cpu().numpy().flatten()
            
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
        
        return embedding
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        return float(np.dot(emb1, emb2))
