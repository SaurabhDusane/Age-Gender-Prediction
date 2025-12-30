from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    APP_NAME: str = "Demographic Analysis System"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "demographics"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    DEVICE: str = "cuda"
    BATCH_SIZE: int = 8
    
    FACE_DETECTOR: str = "yolov8"
    FACE_CONFIDENCE_THRESHOLD: float = 0.5
    
    MODEL_DIR: str = "./models"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
