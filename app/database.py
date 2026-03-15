"""
database.py - SQLite chat history persistence via SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base   # fixed: no longer ext.declarative
from datetime import datetime

DATABASE_URL = "sqlite:///./chat_history.db"
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer,  primary_key=True, index=True)
    session_id = Column(String,   index=True)
    role       = Column(String)                  # "user" or "assistant"
    message    = Column(Text)
    sources    = Column(Text, nullable=True)     # JSON-encoded citations
    timestamp  = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create tables if they don't exist. Call once at startup."""
    Base.metadata.create_all(bind=engine)


def save_message(session_id: str, role: str, message: str, sources: str = None):
    db = SessionLocal()
    try:
        db.add(ChatHistory(
            session_id=session_id, role=role,
            message=message, sources=sources,
        ))
        db.commit()
    finally:
        db.close()


def get_chat_history(session_id: str) -> list[ChatHistory]:
    db = SessionLocal()
    try:
        return (
            db.query(ChatHistory)
            .filter(ChatHistory.session_id == session_id)
            .order_by(ChatHistory.timestamp)
            .all()
        )
    finally:
        db.close()


def get_all_sessions() -> list[str]:
    db = SessionLocal()
    try:
        return [s[0] for s in db.query(ChatHistory.session_id).distinct().all()]
    finally:
        db.close()