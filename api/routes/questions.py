import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.schemas import DuplicateMatch, QuestionCreate, QuestionCreateResponse, QuestionRead
from api.state import AppState, add_question
from src.db.database import get_db
from src.db.models import Question

router = APIRouter(prefix="/questions", tags=["questions"])


def get_state(request: Request) -> AppState:
    """Retrieves the shared AppState set on the app in main.py's lifespan."""
    return request.app.state.app_state


@router.post("", response_model=QuestionCreateResponse, status_code=201)
def create_question(payload: QuestionCreate, request: Request, db: Session = Depends(get_db)):
    state = get_state(request)
    try:
        row, candidates, is_new_cluster = add_question(state, db, payload.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    id_to_text = {}
    if candidates:
        rows = db.query(Question).filter(Question.id.in_([c for c, _ in candidates])).all()
        id_to_text = {r.id: r.text for r in rows}

    return QuestionCreateResponse(
        id=row.id,
        text=row.text,
        cluster_id=row.cluster_id,
        is_new_cluster=is_new_cluster,
        duplicates=[
            DuplicateMatch(id=cid, text=id_to_text.get(cid, ""), similarity=round(sim, 4))
            for cid, sim in candidates
        ],
    )


@router.get("", response_model=list[QuestionRead])
def list_questions(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """All questions, most recently added first. Use skip/limit to page through them."""
    return (
        db.query(Question)
        .order_by(Question.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get("/{question_id}", response_model=QuestionRead)
def get_question(question_id: int, db: Session = Depends(get_db)):
    row = db.get(Question, question_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Question not found")
    return row


@router.get("/{question_id}/duplicates", response_model=list[DuplicateMatch])
def get_duplicates(question_id: int, db: Session = Depends(get_db)):
    """
    All OTHER questions in the same cluster, ranked by true cosine similarity
    to this question (recomputed from stored embeddings — this also covers
    questions linked only transitively, not just direct LSH neighbours).
    """
    row = db.get(Question, question_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Question not found")
    if row.cluster_id is None:
        return []

    others = (
        db.query(Question)
        .filter(Question.cluster_id == row.cluster_id, Question.id != question_id)
        .all()
    )
    if not others:
        return []

    query_vec = np.frombuffer(row.embedding, dtype=np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    results = []
    for other in others:
        other_vec = np.frombuffer(other.embedding, dtype=np.float32)
        sim = float(np.dot(query_vec, other_vec / np.linalg.norm(other_vec)))
        results.append(DuplicateMatch(id=other.id, text=other.text, similarity=round(sim, 4)))

    results.sort(key=lambda d: d.similarity, reverse=True)
    return results