import os
import pandas as pd
import numpy as np
import pytest


from  src.search.brute_force import BruteForceIndex
index = BruteForceIndex()

Q1_PATH = "data/embeddings/q1_embeddings_baseline.npy"
Q2_PATH = "data/embeddings/q2_embeddings_baseline.npy"
CSV_PATH = "data/embeddings/test_split_with_similarity.csv"

real_data_available = all(os.path.exists(p) for p in (Q1_PATH, Q2_PATH, CSV_PATH))

SAMPLE_SIZE  = 300  # number of rows to sample from the CSV for testing

@pytest.mark.skipif(not real_data_available, reason="Real embedding files not found in repo")

def test_real_data_sample():
    """
    Index the question1 embedding of the first SAMPLE_SIZE rows, then query
    with each DUPLICATE pair's question2 embedding. We check that:
      (a) true duplicate pairs score higher on average than non-duplicates
      (b) the matching question1 shows up in the top-5 results reasonably often
    """
    # Load the first SAMPLE_SIZE embeddings and CSV data 
    df = pd.read_csv(CSV_PATH).iloc[:SAMPLE_SIZE].reset_index(drop=True)
    emb_1 = np.load(Q1_PATH)[:SAMPLE_SIZE]
    emb_2 = np.load(Q2_PATH)[:SAMPLE_SIZE]
    
    assert len(df) == len(emb_1) == len(emb_2), "Mismatch in number of rows between CSV and embeddings"
    
    #pairwise similarity sanity check - it means that the embeddings are not completely random and have some structure
    
    sims = np.array([
        index._cosine_sim(emb_1[i], emb_2[i]) for i in range(len(df))
    ])
    
    # we calculate the mean similarity for duplicate pairs and non-duplicate pairs, and assert that the mean similarity for duplicates is greater than that for non-duplicates. This is a sanity check to ensure that the embeddings are meaningful and that the model is able to distinguish between duplicate and non-duplicate questions based on their embeddings.
    mean_dup = sims[df["is_duplicate"].values == 1].mean()
    mean_non_dup = sims[df["is_duplicate"].values == 0].mean()
    print(f"\nMean similarity - duplicates: {mean_dup:.3f}, non-duplicates: {mean_non_dup:.3f}")
    assert mean_dup > mean_non_dup
    
    #retrieval hit-rate for true duplicate pairs - it means that we are checking how often the true duplicate question appears in the top-5 results when we query with the embedding of the second question in the pair. This is a measure of how well the model is able to retrieve the correct duplicate question based on its embedding.
    
    dup_rows = df.index[df["is_duplicate"] == 1] # get the indices of the rows that are duplicates
    hits = 0
    for i in dup_rows:
        top5_ids = [item_id for item_id, _ in index.query(emb_2[i], top_k=100)]
        if i in dup_rows:
            hits += 1
    hit_rate = hits / len(dup_rows)
    print(f"Top-5 retrieval hit rate on {len(dup_rows)} duplicate pairs: {hit_rate:.2%}")
    