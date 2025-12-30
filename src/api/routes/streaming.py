from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import cv2
import numpy as np
import json
import base64
from typing import Dict, List
import asyncio

from src.core.pipeline import DemographicPipeline
from src.database.database import get_db
from src.utils.video import VideoStream

router = APIRouter()


class StreamManager:
    def __init__(self):
        self.active_streams: Dict[str, Dict] = {}
        self.pipelines: Dict[str, DemographicPipeline] = {}
    
    def create_stream(self, stream_id: str, source: int = 0):
        pipeline = DemographicPipeline(
            device="cuda",
            enable_tracking=True,
            enable_temporal_smoothing=True
        )
        
        self.pipelines[stream_id] = pipeline
        self.active_streams[stream_id] = {
            "source": source,
            "active": True
        }
    
    def stop_stream(self, stream_id: str):
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["active"] = False
            del self.active_streams[stream_id]
        
        if stream_id in self.pipelines:
            del self.pipelines[stream_id]
    
    def get_pipeline(self, stream_id: str) -> DemographicPipeline:
        return self.pipelines.get(stream_id)


stream_manager = StreamManager()


@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    stream_id = str(id(websocket))
    
    try:
        init_data = await websocket.receive_json()
        source = init_data.get("source", 0)
        
        stream_manager.create_stream(stream_id, source)
        pipeline = stream_manager.get_pipeline(stream_id)
        
        video_stream = VideoStream(source=source)
        
        for frame in video_stream:
            if not stream_manager.active_streams.get(stream_id, {}).get("active", False):
                break
            
            result = pipeline.process(frame)
            
            faces_data = []
            for face in result.faces:
                faces_data.append({
                    "bbox": face.bbox,
                    "age": face.age,
                    "gender": face.gender,
                    "emotion": face.emotion,
                    "track_id": face.track_id,
                    "confidence": face.confidence
                })
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            response = {
                "frame": frame_base64,
                "faces": faces_data,
                "num_faces": len(faces_data),
                "fps": result.fps,
                "processing_time": result.processing_time
            }
            
            await websocket.send_json(response)
            await asyncio.sleep(0.01)
        
        video_stream.release()
    
    except WebSocketDisconnect:
        stream_manager.stop_stream(stream_id)
    
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        stream_manager.stop_stream(stream_id)


@router.post("/stream/start")
async def start_stream(source: int = 0):
    stream_id = f"stream_{source}"
    stream_manager.create_stream(stream_id, source)
    
    return {
        "success": True,
        "stream_id": stream_id,
        "message": "Stream started successfully"
    }


@router.post("/stream/stop/{stream_id}")
async def stop_stream(stream_id: str):
    stream_manager.stop_stream(stream_id)
    
    return {
        "success": True,
        "message": "Stream stopped successfully"
    }
