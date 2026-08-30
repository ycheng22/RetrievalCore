
import torch
from sentence_transformers import SentenceTransformer


def main():
    model_name = "BAAI/bge-m3"
    print(f"Loading {model_name}...")
    
    # We use SentenceTransformer for the dense embedding part of BGE-M3
    # local_files_only=True uses the local cache without pinging HuggingFace Hub
    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except Exception:  # noqa: BLE001
        model = SentenceTransformer(model_name)
    
    query = "wireless earbuds"
    docs = [
        "Bluetooth wireless earphones", # Strong
        "phone charging cable",         # Weak
        "garden hose"                   # Irrelevant
    ]
    
    # Encode query and docs
    # Note: BGE models typically don't require instructions for retrieval unless specifically trained for it,
    # but BGE-M3 just uses standard encoding.
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(docs, convert_to_tensor=True)
    
    print("\n--- (a) Check Shape ---")
    print(f"Query shape: {query_emb.shape} (Expected: 1024)")
    print(f"Docs shape: {doc_embs.shape} (Expected: 3, 1024)")
    assert query_emb.shape[-1] == 1024, f"Expected dim 1024, got {query_emb.shape[-1]}"
    
    print("\n--- (b) Check L2 Normalization ---")
    query_norm = torch.norm(query_emb, p=2)
    doc_norms = torch.norm(doc_embs, p=2, dim=-1)
    print(f"Query norm: {query_norm.item():.4f}")
    for i, norm in enumerate(doc_norms):
        print(f"Doc {i} norm: {norm.item():.4f}")
    assert torch.isclose(query_norm, torch.tensor(1.0), atol=1e-3), "Query is not L2 normalized"
    
    print("\n--- (c) Check Similarity Ranking ---")
    # Cosine similarity is just dot product since vectors are L2 normalized
    similarities = torch.matmul(doc_embs, query_emb.T).squeeze()
    
    score_strong = similarities[0].item()
    score_weak = similarities[1].item()
    score_irrelevant = similarities[2].item()
    
    print(f"Strong  ('Bluetooth wireless earphones'): {score_strong:.4f}")
    print(f"Weak    ('phone charging cable')        : {score_weak:.4f}")
    print(f"Irrel   ('garden hose')                 : {score_irrelevant:.4f}")
    
    assert score_strong > score_weak > score_irrelevant, "Ranking is incorrect!"
    
    print("\n[SUCCESS] smoke_embed.py PASSED!")

if __name__ == "__main__":
    main()

# output
"""
--- (a) Check Shape ---
Query shape: torch.Size([1024]) (Expected: 1024)
Docs shape: torch.Size([3, 1024]) (Expected: 3, 1024)

--- (b) Check L2 Normalization ---
Query norm: 1.0000
Doc 0 norm: 1.0000
Doc 1 norm: 1.0000
Doc 2 norm: 1.0000

--- (c) Check Similarity Ranking ---
d:\\Github_Clones\\RetrievalCore\\scripts\\smoke_embed.py:47: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at C:\actions-runner\\_work\\pytorch\\pytorch\\pytorch\aten\\src\\ATen\native\\TensorShape.cpp:3729.)
shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at C:\actions-runner\\_work\\pytorch\\pytorch\\pytorch\aten\\src\\ATen\native\\TensorShape.cpp:3729.)
sider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at C:\actions-runner\\_work\\pytorch\\pytorch\\pytorch\aten\\src\\ATen\native\\TensorShape.cpp:3729.)
red internally at C:\actions-runner\\_work\\pytorch\\pytorch\\pytorch\aten\\src\\ATen\native\\TensorShape.cpp:3729.)
src\\ATen\native\\TensorShape.cpp:3729.)
  similarities = torch.matmul(doc_embs, query_emb.T).squeeze()
Strong  ('Bluetooth wireless earphones'): 0.8676
Weak    ('phone charging cable')        : 0.5798
Irrel   ('garden hose')                 : 0.4929

[SUCCESS] smoke_embed.py PASSED!
"""