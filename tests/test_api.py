"""
Tests for the FastAPI app (Phase 6).

Run from the project root:
    pytest tests/test_api.py -v

Uses a real HTTP-style TestClient against a real (temporary, file-based)
SQLite database — this exercises the full stack: request -> route -> DB ->
LSH index -> Union-Find -> DB update -> response.

The real SBERT model is NOT used here (no network access needed to run
these tests). A small deterministic fake embedder stands in for it: any
text containing a topic keyword ("python", "java", "pasta") maps to a
vector near that topic's fixed random center, plus a little noise so two
different phrasings of the same topic are close but not identical - exactly
how real SBERT embeddings behave for paraphrases.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_api_pytest.db")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state import AppState
from src.db.database import Base, SessionLocal, engine

DIM = 384
TOPIC_CENTERS = {
    "python": np.random.default_rng(1).normal(size=DIM),
    "java": np.random.default_rng(2).normal(size=DIM),
    "pasta": np.random.default_rng(3).normal(size=DIM),
}


def fake_embed(text: str) -> np.ndarray:
    """Deterministic stand-in for SBERT: same topic keyword -> nearby vectors."""
    low = text.lower()
    seed = abs(hash(text)) % (2**31)
    for key, base in TOPIC_CENTERS.items():
        if key in low:
            noise = np.random.default_rng(seed).normal(scale=0.05, size=DIM)
            return (base + noise).astype(np.float32)
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


@pytest.fixture()
def client():
    """Fresh, empty database and a fresh AppState for every test."""
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as c:                          # runs the lifespan (init_db + rebuild)
        c.app.state.app_state.embed_fn = fake_embed      # swap in the fake embedder
        yield c
    Base.metadata.drop_all(bind=engine)


def create(client, text):
    r = client.post("/questions", json={"text": text})
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Basic create behaviour
# ---------------------------------------------------------------------------

def test_first_question_has_no_cluster_or_duplicates(client):
    r = create(client, "How do I learn Python?")
    assert r["cluster_id"] is None
    assert r["duplicates"] == []
    assert r["is_new_cluster"] is False


def test_near_duplicate_forms_a_new_cluster(client):
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")

    assert r2["cluster_id"] is not None
    assert r2["is_new_cluster"] is True
    assert [d["id"] for d in r2["duplicates"]] == [r1["id"]]
    assert r2["duplicates"][0]["similarity"] > 0.8


def test_original_question_cluster_id_is_retroactively_updated(client):
    """r1 starts with cluster_id=None; once r2 matches it, r1 must be updated too."""
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")

    r1_after = client.get(f"/questions/{r1['id']}").json()
    assert r1_after["cluster_id"] == r2["cluster_id"]


def test_third_duplicate_joins_existing_cluster_instead_of_making_a_new_one(client):
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")
    r3 = create(client, "Best resources to learn Python programming?")

    assert r3["cluster_id"] == r2["cluster_id"]
    assert r3["is_new_cluster"] is False


def test_unrelated_question_gets_no_cluster(client):
    create(client, "How do I learn Python?")
    r = create(client, "How do I cook pasta?")
    assert r["cluster_id"] is None
    assert r["duplicates"] == []


def test_two_topics_form_two_separate_clusters(client):
    create(client, "How do I learn Python?")
    create(client, "What is the best way to learn Python?")
    create(client, "How do I learn Java?")
    r = create(client, "What is the best way to learn Java?")

    python_cluster = client.get("/clusters").json()
    ids = {c["cluster_id"] for c in python_cluster}
    assert r["cluster_id"] in ids
    assert len(ids) == 2


# ---------------------------------------------------------------------------
# Reading back: /questions/{id}/duplicates, /clusters, /clusters/{id}
# ---------------------------------------------------------------------------

def test_duplicates_endpoint_returns_other_members_sorted_by_similarity(client):
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")
    r3 = create(client, "Best resources to learn Python programming?")

    dups = client.get(f"/questions/{r2['id']}/duplicates").json()
    assert {d["id"] for d in dups} == {r1["id"], r3["id"]}
    sims = [d["similarity"] for d in dups]
    assert sims == sorted(sims, reverse=True)


def test_duplicates_endpoint_empty_for_unclustered_question(client):
    r = create(client, "How do I cook pasta?")
    assert client.get(f"/questions/{r['id']}/duplicates").json() == []


def test_list_clusters_orders_largest_first_and_hides_singletons(client):
    create(client, "How do I learn Python?")
    create(client, "What is the best way to learn Python?")
    create(client, "Best resources to learn Python programming?")   # cluster of 3
    create(client, "How do I learn Java?")
    create(client, "What is the best way to learn Java?")           # cluster of 2
    create(client, "How do I cook pasta?")                          # singleton, hidden

    clusters = client.get("/clusters").json()
    assert [c["size"] for c in clusters] == sorted((c["size"] for c in clusters), reverse=True)
    assert clusters[0]["size"] == 3
    assert all(c["size"] >= 2 for c in clusters)   # no singletons leaked in


def test_cluster_detail_returns_full_question_objects(client):
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")

    detail = client.get(f"/clusters/{r2['cluster_id']}").json()
    assert detail["size"] == 2
    assert {q["id"] for q in detail["questions"]} == {r1["id"], r2["id"]}


# ---------------------------------------------------------------------------
# Errors and validation
# ---------------------------------------------------------------------------

def test_get_missing_question_is_404(client):
    assert client.get("/questions/99999").status_code == 404


def test_get_duplicates_for_missing_question_is_404(client):
    assert client.get("/questions/99999/duplicates").status_code == 404


def test_get_missing_cluster_is_404(client):
    assert client.get("/clusters/99999").status_code == 404


def test_blank_text_is_rejected(client):
    r = client.post("/questions", json={"text": "   "})
    assert r.status_code == 422


def test_missing_text_field_is_rejected(client):
    r = client.post("/questions", json={})
    assert r.status_code == 422


def test_health_check(client):
    assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Persistence across a simulated restart
# ---------------------------------------------------------------------------

def test_clustering_survives_a_rebuild_from_the_database(client):
    """Simulates a server restart: a brand-new AppState rebuilt only from the
    DB should reproduce the same clusters, without touching the LSH index or
    Union-Find that were live during the original requests."""
    r1 = create(client, "How do I learn Python?")
    r2 = create(client, "What is the best way to learn Python?")
    r3 = create(client, "Best resources to learn Python programming?")
    r4 = create(client, "How do I cook pasta?")

    fresh_state = AppState()
    db = SessionLocal()
    try:
        fresh_state.rebuild_from_db(db)
    finally:
        db.close()

    assert fresh_state.uf.connected(r1["id"], r3["id"])       # linked only transitively via r2
    assert not fresh_state.uf.connected(r1["id"], r4["id"])
    assert len(fresh_state.lsh) == 4
    assert fresh_state.uf.num_clusters == 2                   # {r1,r2,r3} + {r4}