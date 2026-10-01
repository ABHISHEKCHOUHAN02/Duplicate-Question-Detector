from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import clusters, questions
from api.state import AppState
from src.db.database import SessionLocal, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    init_db()
    app.state.app_state = AppState()

    db = SessionLocal()
    try:
        app.state.app_state.rebuild_from_db(db)
    finally:
        db.close()

    print(
        f"Loaded {len(app.state.app_state.lsh)} questions, "
        f"{app.state.app_state.uf.num_clusters} clusters"
    )
    yield
    # --- shutdown --- (nothing to clean up: SQLAlchemy's engine manages its own pool)


app = FastAPI(
    title="Duplicate Question Detector",
    description="Detects and clusters duplicate questions using SBERT embeddings, "
                "an LSH index, and Union-Find.",
    version="0.1.0",
    lifespan=lifespan,
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(questions.router)
app.include_router(clusters.router)


@app.get("/health")
def health():
    return {"status": "ok"}