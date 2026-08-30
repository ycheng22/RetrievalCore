import os

from FlagEmbedding import FlagReranker

def main():
    model_name = "BAAI/bge-reranker-v2-m3"
    print(f"Loading {model_name}...")
    
    # use_fp16=True can speed up and save memory on GPU if supported
    # Fallback to local files if already cached
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        reranker = FlagReranker(model_name, use_fp16=True)
    except Exception:  # noqa: BLE001
        os.environ.pop("HF_HUB_OFFLINE", None)
        reranker = FlagReranker(model_name, use_fp16=True)
    
    query = "wireless earbuds"
    docs = [
        "Bluetooth wireless earphones", # Strong
        "phone charging cable",         # Weak
        "garden hose"                   # Irrelevant
    ]
    
    # Reranker takes pairs of (query, doc)
    pairs = [[query, doc] for doc in docs]
    
    scores = reranker.compute_score(pairs, normalize=False)
    
    score_strong = scores[0]
    score_weak = scores[1]
    score_irrelevant = scores[2]
    
    print("\n--- Check Similarity Ranking ---")
    print(f"Strong  ('Bluetooth wireless earphones'): {score_strong:.4f}")
    print(f"Weak    ('phone charging cable')        : {score_weak:.4f}")
    print(f"Irrel   ('garden hose')                 : {score_irrelevant:.4f}")
    
    assert score_strong > score_weak > score_irrelevant, "Ranking is incorrect!"
    
    print("\n[SUCCESS] smoke_rerank.py PASSED!")

if __name__ == "__main__":
    main()
# output:
"""
--- Check Similarity Ranking ---
Strong  ('Bluetooth wireless earphones'): 4.0547
Weak    ('phone charging cable')        : -10.9219
Irrel   ('garden hose')                 : -11.0312

[SUCCESS] smoke_rerank.py PASSED!
"""