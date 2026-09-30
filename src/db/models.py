"""
ORM models.

Embeddings are stored as raw float32 bytes (via numpy .tobytes()/.frombuffer())
rather than a JSON array — much smaller, and trivial to load straight back
into a numpy array of the index's dimension.
"""

import datetime

from sqlalchemy import DateTime, Integer, LargeBinary, String, func
from sqlalchemy.orm import Mapped, mapped_column

from src.db.database import Base


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(String(1000), nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # The root id of this question's Union-Find group at the time it was last
    # updated. NULL means "not yet clustered with anything" (a singleton).
    # Indexed because GET /clusters/{id} and duplicate lookups filter on this.
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"Question(id={self.id}, cluster_id={self.cluster_id}, text={self.text[:40]!r})"