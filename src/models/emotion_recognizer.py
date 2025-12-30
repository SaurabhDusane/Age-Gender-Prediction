import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path


class EmotionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        from torchvision.models import resnet34
        self.backbone = resnet34(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class EmotionRecognizer:
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.6):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.classes = [
            "neutral", "happy", "sad", "surprise",
            "fear", "disgust", "anger", "contempt"
        ]
        
        self.model = self._load_model()
    
    def _load_model(self):
        model = EmotionModel(num_classes=len(self.classes))
        model_path = Path("models/affectnet_8.pth")
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except:
                pass
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess(self, face_img: np.ndarray, target_size=(224, 224)) -> torch.Tensor:
        if face_img.size == 0:
            return torch.zeros(1, 3, *target_size).to(self.device)
        
        img = cv2.resize(face_img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img = img.unsqueeze(0).to(self.device)
        return img
    
    def predict(self, face_img: np.ndarray) -> Dict:
        img_tensor = self.preprocess(face_img)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            emotion = self.classes[pred_idx.item()]
            conf = confidence.item()
        
        probabilities = {
            cls: float(probs[0][i]) 
            for i, cls in enumerate(self.classes)
        }
        
        return {
            'emotion': emotion,
            'confidence': float(conf),
            'probabilities': probabilities
        }
