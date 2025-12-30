#!/usr/bin/env python3
"""
Test the demographic analysis pipeline with sample images
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.pipeline import DemographicPipeline
from src.utils.visualization import draw_face_box, draw_face_info


def test_with_webcam():
    """Test pipeline with webcam"""
    print("Testing with webcam...")
    pipeline = DemographicPipeline(device="cpu")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Could not open webcam")
        return
    
    print("✓ Webcam opened. Press 'q' to quit.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = pipeline.process(frame)
        
        for face in result.faces:
            draw_face_box(frame, face.bbox)
            draw_face_info(frame, face.bbox, face.age, face.gender, face.emotion)
        
        cv2.putText(
            frame, f"FPS: {result.fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        cv2.imshow('Test Pipeline', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, FPS: {result.fps:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ Test complete. Processed {frame_count} frames")


def test_with_synthetic_image():
    """Test pipeline with synthetic image"""
    print("Testing with synthetic image...")
    
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    pipeline = DemographicPipeline(device="cpu")
    
    try:
        result = pipeline.process(img)
        print(f"✓ Pipeline processed image successfully")
        print(f"  Detected {result.num_faces} faces")
        print(f"  Processing time: {result.processing_time*1000:.2f}ms")
        print(f"  FPS: {result.fps:.2f}")
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")


def main():
    print("=== Testing Demographic Analysis Pipeline ===\n")
    
    print("1. Testing with synthetic image...")
    test_with_synthetic_image()
    
    print("\n2. Would you like to test with webcam? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        test_with_webcam()
    else:
        print("Skipping webcam test")
    
    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
