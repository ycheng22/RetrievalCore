import os
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()

print(f"HF_HOME: {os.getenv('HF_HOME')}")
print(f"HF_ENDPOINT: {os.getenv('HF_ENDPOINT')}")

def main():
    model_name = "BAAI/bge-reranker-v2-m3"
    print(f"Loading {model_name}...")
    
    # use_fp16=True can speed up and save memory on GPU if supported
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
