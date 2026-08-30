#!/usr/bin/env python
"""Measure the safe BGE-M3 embedding batch size on this machine.

Stop Ollama before running. Numbers measured with another process resident
on the GPU are void. Config comes from CLI args only (AGENTS.md 4).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import pandas as pd

MODEL_NAME = "BAAI/bge-m3"


def assert_gpu_is_free(limit_mib: int = 500) -> None:
    """Ensure the GPU is not actively used by another process like Ollama."""
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible")
    
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 1024**2
    
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory Status: Total {total / 1024**3:.1f} GiB, Used {used:.0f} MiB\n")
    
    if used > limit_mib:
        raise SystemExit(
            "⚠️ GPU is not idle. Run `ollama stop <model>` (or stop the service) and retry.\n"
            "If Ollama or other processes are running in the background, the measured memory ceiling will be inaccurate."
        )


def load_and_concat_texts(path: Path | str, sample_size: int = 512) -> list[str]:
    """Load parquet data, concatenate title and description, and sample to retain real length distribution."""
    print(f"Loading data from {path}...")
    df = pd.read_parquet(path)
    
    # Concatenate product title and description
    df['product_title'] = df['product_title'].fillna('').astype(str)
    df['product_description'] = df['product_description'].fillna('').astype(str)
    combined = (df['product_title'] + " " + df['product_description']).str.strip()
    
    # Filter out empty rows
    texts = combined[combined != ""].tolist()
    if not texts:
        raise ValueError("No valid texts found after cleaning.")
        
    # Take a random/sequential sample to keep the real-world text length distribution.
    # We do not use fake dummy strings because real memory usage depends on the maximum sequence length in a batch.
    sampled = texts[:sample_size]
    
    lens = sorted(len(t) for t in sampled)
    print(f"Sampled {len(sampled)} representative texts.")
    print(f"Character length distribution: P50={lens[len(lens)//2]}, P95={lens[int(len(lens)*0.95)]}, Max={lens[-1]}")
    
    return sampled


@torch.inference_mode()
def encode(model, tok, batch: list[str], max_len: int) -> torch.Tensor:
    """Tokenize and encode a batch of strings, returning normalized embeddings."""
    enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    cls = model(**enc).last_hidden_state[:, 0]
    return F.normalize(cls, dim=-1)


def bench(model, tok, pool: list[str], bs: int, max_len: int, reps: int) -> tuple[float, float]:
    """
    Step 4: Benchmarking Core Logic
    
    This function measures the maximum throughput (items per second) and peak memory (GiB)
    for a specific batch size (`bs`).
    
    How it works:
    1. We aggressively clear the PyTorch CUDA cache and reset peak memory statistics so 
       we can accurately measure the memory footprint of *just this batch size*.
    2. We build `reps` number of batches by cycling through the `pool` of real texts.
       Cycling ensures that the batches have consistent real-world length distributions.
    3. We run a 'warm-up' iteration that is NOT timed, ensuring CUDA kernels are initialized.
    4. We run the actual timed loop over all batches, using `torch.cuda.synchronize()` 
       to block the CPU until the GPU finishes computing, ensuring our timer is accurate.
    5. Finally, we read the peak memory reserved by PyTorch during the execution.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create batches by cycling through the text pool
    batches = []
    for i in range(reps):
        batch = [pool[(i * bs + j) % len(pool)] for j in range(bs)]
        batches.append(batch)

    # Warm-up (not timed) to avoid initialization overhead
    encode(model, tok, batches[0], max_len)  
    torch.cuda.synchronize()

    # Timed benchmarking
    t0 = time.perf_counter()
    for b in batches:
        encode(model, tok, b, max_len)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak_gib = torch.cuda.max_memory_reserved() / 1024**3
    return (bs * reps) / dt, peak_gib


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--texts", type=Path, default=Path("data/raw/shopping_queries_dataset_products.parquet"),
                   help="Path to the parquet dataset")
    p.add_argument("--sizes", default="32,64,128,256,512",
                   help="Comma-separated batch sizes to test")
    p.add_argument("--max-len", type=int, default=512,
                   help="Maximum token length for the model")
    p.add_argument("--reps", type=int, default=3,
                   help="Number of batches to run per size")
    p.add_argument("--n-products", type=int, default=50_000,
                   help="Corpus size, used to estimate the Sunday ETA")
    p.add_argument("--out", type=Path, default=Path("docs/bench_batch_size.json"))
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    assert_gpu_is_free()

    base_texts = load_and_concat_texts(args.texts, sample_size=512)

    print(f"\nLoading model {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda").eval()

    results = []
    print(f"\n{'Batch':>6} {'Items/s':>9} {'Peak(GiB)':>9}")
    print("-" * 27)

    for bs in sizes:
        try:
            tp, peak = bench(model, tok, base_texts, bs, args.max_len, args.reps)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"{bs:>6} {'OOM':>9}")
            torch.cuda.empty_cache()
            break
            
        results.append({"batch_size": bs, "items_per_s": round(tp, 1), "peak_gib": round(peak, 2)})
        print(f"{bs:>6} {tp:>9.1f} {peak:>9.2f}")

    if not results:
        raise SystemExit("\nEven the smallest batch (32) triggered an OOM. Reduce max_len or use a smaller model.")

    # --- Step 5: Decision Logic (Knee of the curve & Fallback) ---
    # 
    # The absolute maximum batch size is determined by hardware memory limits (OOM).
    # However, the optimal batch size is the "knee" of the throughput curve.
    # As batch size increases, throughput (items/s) usually plateaus because the GPU 
    # compute units are fully saturated. Increasing batch size further only wastes memory.
    
    best_tp = max(r["items_per_s"] for r in results)

    # 1. Find the "knee": we define this as the smallest batch size that achieves at least 95% of peak throughput.
    knee_result = next(r for r in results if r["items_per_s"] >= 0.95 * best_tp)
    largest_result = results[-1]

    # 2. Check if we hit the memory wall BEFORE reaching the throughput plateau.
    # If the knee is equal to the largest batch we tested without OOM, AND we didn't finish the test array (meaning we OOMed on the next size),
    # it means throughput was still climbing, but we crashed into the memory ceiling.
    hit_memory_wall_before_plateau = (knee_result["batch_size"] == largest_result["batch_size"]) and (len(results) < len(sizes))

    chosen = knee_result["batch_size"]
    if hit_memory_wall_before_plateau:
        # Fallback: We hit the memory limit, but real-world data at scale has unpredictable lengths.
        # Long runs cause memory fragmentation. To be safe for an overnight job, we step down one tier (or take 80%).
        chosen = max(8, int(chosen * 0.8) // 8 * 8)
        note = "Throughput hadn't plateaued before OOM. Applied a 20% safety margin to prevent fragmentation crashes overnight."
    else:
        # We hit the throughput knee BEFORE running out of memory. 
        # By picking the knee, we naturally have plenty of unused memory left over as a safety buffer.
        note = "Throughput reached its plateau (knee of the curve). Chosen value naturally leaves a memory safety buffer."

    tp = next((r["items_per_s"] for r in results if r["batch_size"] == chosen), knee_result["items_per_s"])

    # Calculate ETAs
    pure_min = args.n_products / tp / 60
    total_min_est_lower = pure_min * 1.5
    total_min_est_upper = pure_min * 2.0

    print(f"\n=== Final Recommendation ===")
    print(f"Recommended Batch Size : {chosen}  ({note})")
    print(f"Expected Throughput    : {tp:.1f} items/s")
    print(f"Peak Memory            : {knee_result['peak_gib']:.2f} GiB (Stress peak)")

    print(f"\n=== Sunday ETA (based on {args.n_products:,} products) ===")
    print(f"Pure Inference Time    : {pure_min:.0f} mins")
    print(f"Total Time (w/ I/O)    : ~{total_min_est_lower:.0f} - {total_min_est_upper:.0f} mins")

    if total_min_est_upper > 90:
        print("\n⚠️ WARNING: Estimated total time exceeds 90 minutes!")
        print("Suggested Action: Consider reducing the sampling scale or start the embedding task earlier on Saturday night.")

    # Generate DECISIONS.md output
    date_str = time.strftime('%Y-%m-%d')
    device_name = torch.cuda.get_device_name(0)

    markdown_entry = f"""### {date_str} - BGE-M3 Embedding Batch Size Configuration

* **Batch Size**: `{chosen}`
* **Peak Memory**: `{knee_result['peak_gib']:.2f} GiB`
* **Throughput**: `{tp:.0f} items/s`
* **ETA**: `~{total_min_est_upper:.0f} min` for {args.n_products:,} products (including I/O & Upsert overhead)

**Notes**: Measured on {device_name} with `max_len={args.max_len}` using fp16. {note}
"""

    print("\n--- Copy and paste the following into docs/DECISIONS.md ---\n")
    print(markdown_entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "date": date_str,
        "device": device_name,
        "max_len": args.max_len,
        "total_products": args.n_products,
        "results": results,
        "chosen": chosen,
        "note": note
    }, indent=2), encoding="utf-8")
    print(f"\nJSON benchmark results saved to: {args.out}")


if __name__ == "__main__":
    main()