import pytest
import numpy as np
import torch
from src.models.detector import FaceDetector
from src.models.age_estimator import AgeEstimator
from src.models.gender_classifier import GenderClassifier
from src.models.emotion_recognizer import EmotionRecognizer


@pytest.fixture
def sample_frame():
    """Generate a sample frame for testing"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_face():
    """Generate a sample face crop"""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


def test_face_detector_initialization():
    """Test face detector initialization"""
    detector = FaceDetector(device="cpu")
    assert detector is not None
    assert detector.device == "cpu"


def test_face_detector_detection(sample_frame):
    """Test face detection"""
    detector = FaceDetector(device="cpu")
    detections = detector.detect(sample_frame)
    assert isinstance(detections, list)


def test_age_estimator_initialization():
    """Test age estimator initialization"""
    estimator = AgeEstimator(device="cpu", ensemble=False)
    assert estimator is not None
    assert estimator.device == "cpu"


def test_age_estimator_prediction(sample_face):
    """Test age prediction"""
    estimator = AgeEstimator(device="cpu", ensemble=False)
    result = estimator.predict(sample_face)
    
    assert 'age' in result
    assert 0 <= result['age'] <= 100
    assert 'confidence' in result


def test_gender_classifier_initialization():
    """Test gender classifier initialization"""
    classifier = GenderClassifier(device="cpu")
    assert classifier is not None
    assert len(classifier.classes) == 2


def test_gender_classifier_prediction(sample_face):
    """Test gender prediction"""
    classifier = GenderClassifier(device="cpu")
    result = classifier.predict(sample_face)
    
    assert 'gender' in result
    assert result['gender'] in ['male', 'female']
    assert 'confidence' in result
    assert 0 <= result['confidence'] <= 1


def test_emotion_recognizer_initialization():
    """Test emotion recognizer initialization"""
    recognizer = EmotionRecognizer(device="cpu")
    assert recognizer is not None
    assert len(recognizer.classes) > 0


def test_emotion_recognizer_prediction(sample_face):
    """Test emotion prediction"""
    recognizer = EmotionRecognizer(device="cpu")
    result = recognizer.predict(sample_face)
    
    assert 'emotion' in result
    assert result['emotion'] in recognizer.classes
    assert 'confidence' in result
    assert 'probabilities' in result
