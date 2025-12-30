"""
Advanced Demographic Analysis System
Production-grade real-time facial attribute detection and tracking
"""

__version__ = "1.0.0"
__author__ = "Demographic Analysis Team"
__license__ = "MIT"

from src.core.pipeline import DemographicPipeline
from src.models.detector import FaceDetector
from src.models.age_estimator import AgeEstimator
from src.models.gender_classifier import GenderClassifier
from src.models.emotion_recognizer import EmotionRecognizer

__all__ = [
    "DemographicPipeline",
    "FaceDetector",
    "AgeEstimator",
    "GenderClassifier",
    "EmotionRecognizer",
]
