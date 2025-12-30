#!/usr/bin/env python3
"""
Initialize the database with required tables and extensions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.database import db
from src.database.models import Base


def init_database():
    """Create all database tables"""
    print("Initializing database...")
    
    try:
        print("Creating tables...")
        db.create_tables()
        print("✓ Database tables created successfully")
        
        print("\nNote: For pgvector support, run in PostgreSQL:")
        print("CREATE EXTENSION IF NOT EXISTS vector;")
        print("CREATE INDEX ON face_embeddings USING ivfflat (embedding vector_cosine_ops);")
        
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_database()
