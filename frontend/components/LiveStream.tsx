'use client';

import { useState, useEffect, useRef } from 'react';
import { Play, Square, Camera } from 'lucide-react';

export default function LiveStream({ apiUrl }: { apiUrl: string }) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentFrame, setCurrentFrame] = useState<string | null>(null);
  const [faces, setFaces] = useState<any[]>([]);
  const [fps, setFps] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const startStream = () => {
    const ws = new WebSocket(`ws://localhost:8000/api/v1/stream`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      ws.send(JSON.stringify({ source: 0 }));
      setIsStreaming(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.frame) {
        setCurrentFrame(`data:image/jpeg;base64,${data.frame}`);
        setFaces(data.faces || []);
        setFps(data.fps || 0);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsStreaming(false);
    };

    wsRef.current = ws;
  };

  const stopStream = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreaming(false);
    setCurrentFrame(null);
    setFaces([]);
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Live Stream</h2>
        <div className="flex space-x-2">
          {!isStreaming ? (
            <button
              onClick={startStream}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              <Play className="h-5 w-5 mr-2" />
              Start Stream
            </button>
          ) : (
            <button
              onClick={stopStream}
              className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
            >
              <Square className="h-5 w-5 mr-2" />
              Stop Stream
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <div className="relative bg-gray-900 rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
            {currentFrame ? (
              <img
                src={currentFrame}
                alt="Live stream"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="flex items-center justify-center h-full">
                <div className="text-center text-gray-400">
                  <Camera className="h-16 w-16 mx-auto mb-4" />
                  <p>No stream active</p>
                </div>
              </div>
            )}
            {isStreaming && (
              <div className="absolute top-4 right-4 bg-black bg-opacity-50 text-white px-3 py-1 rounded-full text-sm">
                {fps.toFixed(1)} FPS
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-3">Detected Faces</h3>
            <div className="text-3xl font-bold text-blue-600">{faces.length}</div>
          </div>

          <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
            <h3 className="font-semibold text-gray-900 mb-3">Face Details</h3>
            {faces.length === 0 ? (
              <p className="text-gray-500 text-sm">No faces detected</p>
            ) : (
              <div className="space-y-3">
                {faces.map((face, idx) => (
                  <div key={idx} className="bg-white rounded p-3 text-sm border border-gray-200">
                    <div className="font-medium text-gray-900 mb-2">
                      {face.track_id ? `Track #${face.track_id}` : `Face ${idx + 1}`}
                    </div>
                    <div className="space-y-1 text-gray-600">
                      <div>Age: {face.age?.toFixed(1)} years</div>
                      <div>Gender: {face.gender}</div>
                      <div>Emotion: {face.emotion}</div>
                      <div className="text-xs text-gray-400">
                        Confidence: {(face.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
