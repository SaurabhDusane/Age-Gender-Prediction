#!/usr/bin/env python3
"""
Main entry point for the Demographic Analysis System
"""
import argparse
import cv2
import sys
from pathlib import Path

from src.core.pipeline import DemographicPipeline
from src.utils.video import VideoStream, VideoWriter
from src.utils.visualization import draw_face_box, draw_face_info, create_dashboard


def run_webcam(args):
    print("Starting webcam analysis...")
    
    pipeline = DemographicPipeline(
        device=args.device,
        enable_tracking=True,
        enable_temporal_smoothing=True
    )
    
    stream = VideoStream(source=args.source)
    
    if args.output:
        writer = VideoWriter(
            args.output,
            fps=30,
            resolution=(640, 480)
        )
    
    try:
        for frame in stream:
            result = pipeline.process(frame)
            
            for face in result.faces:
                draw_face_box(frame, face.bbox)
                draw_face_info(
                    frame,
                    face.bbox,
                    face.age,
                    face.gender,
                    face.emotion,
                    face.confidence,
                    face.track_id
                )
            
            frame = create_dashboard(
                frame,
                result.fps,
                result.num_faces,
                result.processing_time
            )
            
            cv2.imshow('Demographic Analysis', frame)
            
            if args.output:
                writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        stream.release()
        if args.output:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete. Average FPS: {result.fps:.2f}")


def run_video(args):
    print(f"Processing video: {args.input}")
    
    pipeline = DemographicPipeline(
        device=args.device,
        enable_tracking=True,
        enable_temporal_smoothing=True
    )
    
    stream = VideoStream(source=args.input)
    
    if args.output:
        writer = VideoWriter(args.output, fps=30, resolution=(640, 480))
    
    try:
        frame_count = 0
        for frame in stream:
            result = pipeline.process(frame)
            
            for face in result.faces:
                draw_face_box(frame, face.bbox)
                draw_face_info(
                    frame,
                    face.bbox,
                    face.age,
                    face.gender,
                    face.emotion,
                    face.confidence,
                    face.track_id
                )
            
            frame = create_dashboard(
                frame,
                result.fps,
                result.num_faces,
                result.processing_time
            )
            
            if not args.headless:
                cv2.imshow('Demographic Analysis', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if args.output:
                writer.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, FPS: {result.fps:.2f}")
    
    finally:
        stream.release()
        if args.output:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"\nProcessing complete. Total frames: {frame_count}")


def run_image(args):
    print(f"Processing image: {args.input}")
    
    pipeline = DemographicPipeline(device=args.device)
    
    frame = cv2.imread(args.input)
    if frame is None:
        print(f"Error: Could not load image {args.input}")
        return
    
    result = pipeline.process(frame)
    
    print(f"\nFound {result.num_faces} face(s):")
    for i, face in enumerate(result.faces, 1):
        print(f"\nFace {i}:")
        print(f"  Age: {face.age:.1f} ± {face.age_std:.1f} years")
        print(f"  Gender: {face.gender} ({face.gender_confidence:.2%})")
        print(f"  Emotion: {face.emotion} ({face.emotion_confidence:.2%})")
        if face.ethnicity:
            print(f"  Ethnicity: {face.ethnicity} ({face.ethnicity_confidence:.2%})")
        print(f"  Quality Score: {face.quality_score:.2f}")
        
        draw_face_box(frame, face.bbox)
        draw_face_info(
            frame,
            face.bbox,
            face.age,
            face.gender,
            face.emotion,
            face.confidence
        )
    
    if args.output:
        cv2.imwrite(args.output, frame)
        print(f"\nSaved result to: {args.output}")
    
    if not args.headless:
        cv2.imshow('Demographic Analysis', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_api(args):
    print("Starting API server...")
    import uvicorn
    from src.api.app import app
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Demographic Analysis System"
    )
    
    parser.add_argument(
        "mode",
        choices=["webcam", "video", "image", "api"],
        help="Operation mode"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file path (for video/image mode)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=int,
        default=0,
        help="Camera source (default: 0)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="API port (default: 8000)"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of API workers (default: 4)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for API"
    )
    
    args = parser.parse_args()
    
    if args.mode == "webcam":
        run_webcam(args)
    
    elif args.mode == "video":
        if not args.input:
            print("Error: --input is required for video mode")
            sys.exit(1)
        run_video(args)
    
    elif args.mode == "image":
        if not args.input:
            print("Error: --input is required for image mode")
            sys.exit(1)
        run_image(args)
    
    elif args.mode == "api":
        run_api(args)


if __name__ == "__main__":
    main()
