"""
Database setup.

Defaults to a local SQLite file so the project runs with zero setup. Swap
to MySQL/Postgres by setting DATABASE_URL in .env, e.g.:
    DATABASE_URL=mysql+pymysql://user:pass@localhost/duplicate_questions
    DATABASE_URL=postgresql+psycopg2://user:pass@localhost/duplicate_questions
(install the matching driver: PyMySQL / psycopg2-binary — see requirements.txt)
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./duplicate_questions.db")

# check_same_thread is only needed for SQLite (FastAPI uses one connection
# across threads); other databases ignore this argument.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create tables that don't exist yet. Call once at app startup."""
    from src.db import models  # noqa: F401  (import so the model is registered on Base)

    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """FastAPI dependency: yields a session, always closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()