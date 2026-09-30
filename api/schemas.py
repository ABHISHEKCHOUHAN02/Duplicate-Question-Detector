from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class QuestionCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, examples=["How do I learn Python?"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


class DuplicateMatch(BaseModel):
    id: int
    text: str
    similarity: float


class QuestionCreateResponse(BaseModel):
    id: int
    text: str
    cluster_id: int | None
    is_new_cluster: bool
    duplicates: list[DuplicateMatch]


class QuestionRead(BaseModel):
    id: int
    text: str
    cluster_id: int | None
    created_at: datetime

    model_config = {"from_attributes": True}   # lets this build straight from the ORM object


class ClusterSummary(BaseModel):
    cluster_id: int
    size: int


class ClusterDetail(BaseModel):
    cluster_id: int
    size: int
    questions: list[QuestionRead]