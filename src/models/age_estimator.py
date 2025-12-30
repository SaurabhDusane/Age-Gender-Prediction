import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Dict, List
from pathlib import Path


class DEXModel(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        from torchvision.models import vgg16
        vgg = vgg16(pretrained=False)
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class SSRNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.age_pred = nn.Linear(64 * 16 * 16, 1)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.age_pred(x)


class AgeEstimator:
    def __init__(
        self,
        device: str = "cuda",
        ensemble: bool = True,
        model_weights: Dict[str, float] = None
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.ensemble = ensemble
        
        if model_weights is None:
            self.model_weights = {
                'dex': 0.4,
                'ssrnet': 0.3,
                'vit': 0.3
            }
        else:
            self.model_weights = model_weights
        
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        if self.ensemble:
            self.models['dex'] = self._load_dex()
            self.models['ssrnet'] = self._load_ssrnet()
        else:
            self.models['dex'] = self._load_dex()
    
    def _load_dex(self):
        model = DEXModel()
        model_path = Path("models/dex_imdb_wiki.pth")
        
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
            except:
                pass
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_ssrnet(self):
        model = SSRNetModel()
        model_path = Path("models/ssrnet_3_3_3_64_1.0_1.0.pth")
        
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
        if not self.ensemble:
            return self._predict_single(face_img, 'dex')
        
        predictions = []
        weights = []
        
        for model_name, model in self.models.items():
            if model_name in self.model_weights:
                pred = self._predict_single(face_img, model_name)
                predictions.append(pred['age'])
                weights.append(self.model_weights[model_name])
        
        if predictions:
            weights_sum = sum(weights)
            weighted_age = sum(p * w for p, w in zip(predictions, weights)) / weights_sum
            std = np.std(predictions)
            
            return {
                'age': float(weighted_age),
                'std': float(std),
                'confidence': float(1.0 - min(std / 10.0, 1.0)),
                'individual_predictions': {
                    name: predictions[i] 
                    for i, name in enumerate(self.models.keys())
                }
            }
        
        return {'age': 30.0, 'std': 10.0, 'confidence': 0.0}
    
    def _predict_single(self, face_img: np.ndarray, model_name: str) -> Dict:
        model = self.models.get(model_name)
        if model is None:
            return {'age': 30.0, 'confidence': 0.0}
        
        img_tensor = self.preprocess(face_img)
        
        with torch.no_grad():
            output = model(img_tensor)
            
            if model_name == 'dex':
                probs = torch.softmax(output, dim=1)
                ages = torch.arange(0, 101).float().to(self.device)
                age = torch.sum(probs * ages, dim=1).item()
            else:
                age = output.item()
            
            age = max(0, min(100, age))
        
        return {'age': float(age), 'confidence': 0.8}
