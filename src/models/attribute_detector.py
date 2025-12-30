import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path


class AttributeModel(nn.Module):
    def __init__(self, num_attributes=10):
        super().__init__()
        from torchvision.models import resnet18
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_attributes)
    
    def forward(self, x):
        return self.backbone(x)


class AttributeDetector:
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.ethnicity_classes = [
            "White", "Black", "Latino_Hispanic", "East Asian",
            "Southeast Asian", "Indian", "Middle Eastern"
        ]
        
        self.attribute_classes = [
            "glasses", "sunglasses", "hat", "earrings",
            "mask", "facial_hair", "makeup", "smile"
        ]
        
        self.models = self._load_models()
    
    def _load_models(self):
        models = {}
        
        ethnicity_model = AttributeModel(num_attributes=len(self.ethnicity_classes))
        ethnicity_path = Path("models/fairface_ethnicity.pth")
        if ethnicity_path.exists():
            try:
                ethnicity_model.load_state_dict(
                    torch.load(ethnicity_path, map_location=self.device)
                )
            except:
                pass
        ethnicity_model.to(self.device)
        ethnicity_model.eval()
        models['ethnicity'] = ethnicity_model
        
        attribute_model = AttributeModel(num_attributes=len(self.attribute_classes))
        attribute_path = Path("models/attributes.pth")
        if attribute_path.exists():
            try:
                attribute_model.load_state_dict(
                    torch.load(attribute_path, map_location=self.device)
                )
            except:
                pass
        attribute_model.to(self.device)
        attribute_model.eval()
        models['attributes'] = attribute_model
        
        return models
    
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
    
    def detect(self, face_img: np.ndarray) -> Dict:
        results = {}
        
        results.update(self._detect_ethnicity(face_img))
        results.update(self._detect_attributes(face_img))
        results.update(self._detect_simple_attributes(face_img))
        
        return results
    
    def _detect_ethnicity(self, face_img: np.ndarray) -> Dict:
        img_tensor = self.preprocess(face_img)
        
        with torch.no_grad():
            output = self.models['ethnicity'](img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            ethnicity = self.ethnicity_classes[pred_idx.item()]
            conf = confidence.item()
        
        return {
            'ethnicity': ethnicity,
            'ethnicity_confidence': float(conf),
            'ethnicity_probabilities': {
                cls: float(probs[0][i]) 
                for i, cls in enumerate(self.ethnicity_classes)
            }
        }
    
    def _detect_attributes(self, face_img: np.ndarray) -> Dict:
        img_tensor = self.preprocess(face_img)
        
        with torch.no_grad():
            output = self.models['attributes'](img_tensor)
            probs = torch.sigmoid(output)
        
        attributes = {}
        for i, attr in enumerate(self.attribute_classes):
            prob = float(probs[0][i])
            attributes[attr] = prob > 0.5
            attributes[f"{attr}_confidence"] = prob
        
        return attributes
    
    def _detect_simple_attributes(self, face_img: np.ndarray) -> Dict:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        skin_tone = self._estimate_skin_tone(face_img)
        
        hair_region = face_img[:int(face_img.shape[0] * 0.3), :]
        hair_color = self._estimate_hair_color(hair_region)
        
        return {
            'skin_tone': skin_tone,
            'hair_color': hair_color,
            'face_shape': 'unknown'
        }
    
    def _estimate_skin_tone(self, face_img: np.ndarray) -> str:
        center_region = face_img[
            int(face_img.shape[0] * 0.3):int(face_img.shape[0] * 0.7),
            int(face_img.shape[1] * 0.3):int(face_img.shape[1] * 0.7)
        ]
        
        if center_region.size == 0:
            return "unknown"
        
        avg_color = np.mean(center_region, axis=(0, 1))
        brightness = np.mean(avg_color)
        
        if brightness < 80:
            return "dark"
        elif brightness < 140:
            return "medium"
        else:
            return "light"
    
    def _estimate_hair_color(self, hair_region: np.ndarray) -> str:
        if hair_region.size == 0:
            return "unknown"
        
        avg_color = np.mean(hair_region, axis=(0, 1))
        
        if np.mean(avg_color) < 50:
            return "black"
        elif np.mean(avg_color) < 100:
            return "brown"
        elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
            return "blonde"
        else:
            return "other"
