from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Index, Integer, JSON, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Author(Base):
    """OpenAlex-aligned researcher row — id matches FAISS row labels (1..N)."""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    openalex_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    display_name: Mapped[str] = mapped_column(String(500), nullable=False)
    orcid: Mapped[Optional[str]] = mapped_column(String(500))
    profile_text: Mapped[Optional[str]] = mapped_column(Text)
    institution_name: Mapped[Optional[str]] = mapped_column(String(500))
    institution_country: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    h_index: Mapped[Optional[int]] = mapped_column(Integer)
    works_count: Mapped[Optional[int]] = mapped_column(Integer)
    cited_by_count: Mapped[Optional[int]] = mapped_column(Integer)
    i10_index: Mapped[Optional[int]] = mapped_column(Integer)
    two_year_mean_citedness: Mapped[Optional[float]] = mapped_column(Float)
    top_concepts: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    recent_works: Mapped[Optional[list[Any]]] = mapped_column(JSON)
    last_known_activity_year: Mapped[Optional[int]] = mapped_column(Integer)

    embedding_vector: Mapped[Optional[bytes]] = mapped_column(LargeBinary)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_authors_openalex_id", "openalex_id"),
        Index("ix_authors_h_index", "h_index"),
    )

    def __repr__(self) -> str:
        return f"<Author id={self.id} openalex_id={self.openalex_id!r}>"
