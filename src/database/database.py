from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import redis
from typing import Generator
import os

from src.database.models import Base
from src.config.settings import settings


class Database:
    def __init__(self, database_url: str = None):
        self.database_url = database_url or self._get_database_url()
        
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def _get_database_url(self) -> str:
        return f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    
    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class RedisCache:
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or self._get_redis_url()
        self.client = redis.from_url(
            self.redis_url,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5
        )
    
    def _get_redis_url(self) -> str:
        return f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
    
    def set(self, key: str, value: bytes, expire: int = None):
        self.client.set(key, value, ex=expire)
    
    def get(self, key: str) -> bytes:
        return self.client.get(key)
    
    def delete(self, key: str):
        self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        return self.client.exists(key)
    
    def clear(self):
        self.client.flushdb()


db = Database()
cache = RedisCache()


def get_db() -> Generator[Session, None, None]:
    with db.get_session() as session:
        yield session
