# src/models.py
"""
SQLAlchemy ORM models and DB initialization helpers.

This uses SQLAlchemy 2.x style ORM and is intentionally minimal for MVP.
Switch DB_URL in env/config to use Postgres in production.
"""
from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Text,
    JSON,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, UTC
from src.config import cfg

Base = declarative_base()

# Page class defines the database pages with each column definition and their data types so we can sqlite into it and see all the URLs we fetched and extracted

class Page(Base):
    __tablename__ = "pages"
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, index=True, nullable=False)
    fetched_at = Column(DateTime, default=lambda:datetime.now(UTC))
    status_code = Column(Integer, nullable=True)
    title = Column(String, nullable=True)
    lang = Column(String(16), nullable=True)
    text_excerpt = Column(Text, nullable=True)
    html_path = Column(String, nullable=True)
    raw_meta = Column(JSON, nullable=True)

def get_engine():
    """
    Create and return an SQLAlchemy engine.
    For sqlite, we pass check_same_thread=False to avoid threading issues.
    """
    connect_args = {"check_same_thread": False} if cfg.DB_URL.startswith("sqlite") else {}
    return create_engine(cfg.DB_URL, connect_args=connect_args)

def init_db():
    """
    Create tables (if missing) and return a sessionmaker factory.
    Call this at app startup.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
