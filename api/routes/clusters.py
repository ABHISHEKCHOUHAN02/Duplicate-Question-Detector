from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.schemas import ClusterDetail, ClusterSummary
from src.db.database import get_db
from src.db.models import Question

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("", response_model=list[ClusterSummary])
def list_clusters(min_size: int = 2, limit: int = 50, db: Session = Depends(get_db)):
    """
    All clusters with at least `min_size` members, largest first.
    min_size=2 (the default) hides singletons — questions with no known
    duplicate yet — since they aren't really "clusters".
    """
    rows = (
        db.query(Question.cluster_id, func.count(Question.id).label("size"))
        .filter(Question.cluster_id.isnot(None))
        .group_by(Question.cluster_id)
        .having(func.count(Question.id) >= min_size)
        .order_by(func.count(Question.id).desc())
        .limit(limit)
        .all()
    )
    return [ClusterSummary(cluster_id=cid, size=size) for cid, size in rows]


@router.get("/{cluster_id}", response_model=ClusterDetail)
def get_cluster(cluster_id: int, db: Session = Depends(get_db)):
    questions = db.query(Question).filter(Question.cluster_id == cluster_id).all()
    if not questions:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return ClusterDetail(cluster_id=cluster_id, size=len(questions), questions=questions)