import pytest
import numpy as np
from src.core.pipeline import DemographicPipeline, FaceResult, FrameResult


@pytest.fixture
def sample_frame():
    """Generate a sample frame"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def pipeline():
    """Initialize pipeline"""
    return DemographicPipeline(
        device="cpu",
        enable_tracking=False,
        enable_temporal_smoothing=False
    )


def test_pipeline_initialization(pipeline):
    """Test pipeline initialization"""
    assert pipeline is not None
    assert pipeline.device == "cpu"
    assert pipeline.detector is not None
    assert pipeline.age_estimator is not None
    assert pipeline.gender_classifier is not None


def test_pipeline_process(pipeline, sample_frame):
    """Test pipeline processing"""
    result = pipeline.process(sample_frame)
    
    assert isinstance(result, FrameResult)
    assert isinstance(result.faces, list)
    assert result.frame_idx > 0
    assert result.processing_time > 0
    assert result.fps >= 0


def test_pipeline_multiple_frames(pipeline, sample_frame):
    """Test processing multiple frames"""
    results = []
    for _ in range(5):
        result = pipeline.process(sample_frame)
        results.append(result)
    
    assert len(results) == 5
    assert all(isinstance(r, FrameResult) for r in results)


def test_pipeline_reset(pipeline):
    """Test pipeline reset"""
    pipeline.frame_count = 10
    pipeline.reset()
    assert pipeline.frame_count == 0


def test_face_result_creation():
    """Test FaceResult dataclass"""
    face_result = FaceResult(
        bbox=[100, 100, 200, 200],
        confidence=0.95,
        age=30.0,
        age_std=3.0,
        age_confidence=0.85,
        gender="male",
        gender_confidence=0.92,
        emotion="happy",
        emotion_confidence=0.88
    )
    
    assert face_result.bbox == [100, 100, 200, 200]
    assert face_result.age == 30.0
    assert face_result.gender == "male"
    assert face_result.emotion == "happy"
