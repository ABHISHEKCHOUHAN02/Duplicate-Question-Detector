"""
Embedding a single new question at request time.

This is CPU inference on ONE short sentence, which is fast (well under
100ms typically) — no GPU needed here, unlike the bulk Colab embedding step
in Phase 2. The model is loaded once (singleton) and reused across requests.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None
finetuned_model = r"D:\vscode\ML\Duplicate Question Detector\data\finetuned_sbert"
EMBEDDING_DIM = 384
import os
assert os.path.isdir(finetuned_model), f"Not found: {finetuned_model}"

def _get_model():
    global _MODEL
    if _MODEL is None:
         # imported lazily so
        _MODEL = SentenceTransformer(finetuned_model)                # tests can skip this
    return _MODEL


def embed_text(text: str) -> np.ndarray:
    """Encode one question into a (EMBEDDING_DIM,) float32 numpy vector."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    vector = _get_model().encode(text, convert_to_numpy=True)
    return vector.astype(np.float32)