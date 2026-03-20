# Results — Chess Transformer Ablation Study

## Benchmark Table

| Variant | Params | Val Loss | Perplexity | Move Legality | Tok/s |
|---|---|---|---|---|---|
| Vanilla MHA | 1,400,832 | 5.3204 | 204.47 | 81.5% | 184,629 |
| **RoPE MHA** | **1,400,832** | **4.3611** | **78.34** | **88.3%** | 132,760 |
| GQA (kv\_heads=2) | 1,335,296 | 5.3075 | 201.85 | 83.0% | 164,689 |
| Sliding Window | 1,400,832 | 5.3596 | 212.65 | 81.5% | 184,579 |

**Training setup:** 120,266 Lichess games (2013-01), 4,782-token vocabulary, 3,000 steps, cosine LR decay (3e-4 → 3e-5) with 300-step warmup. `d_model=128`, `n_heads=4`, `n_layers=4`, `seq_len=128`, `batch_size=64`. All variants trained on identical data splits with the same random seed.

---

## Key Takeaways

### 1. RoPE dramatically outperforms absolute positional encoding on sequential data

RoPE achieves 2.6× lower perplexity (78.34 vs 204.47) and +6.8 percentage points in move legality (88.3% vs 81.5%) over vanilla MHA with an identical parameter count. The mechanism matters: chess moves are inherently relational — whether `Nf3` is a good move depends on what happened 2–4 moves earlier, not on its absolute position in the game. RoPE encodes this through rotation of query/key vectors, which means the attention score between positions i and j depends only on their *relative* offset (i−j). Sinusoidal PE added to the embedding does not have this property; the model must learn the relative structure implicitly, which is far harder with a small model and limited data.

This result mirrors what the broader literature found when RoPE was introduced in RoFormer and later adopted by LLaMA, Gemma, and Mistral: on tasks with strong local dependency structure, relative position encoding consistently outperforms absolute encoding, especially at smaller model sizes where the model cannot afford to learn the equivalence from scratch.

### 2. GQA achieves the same quality for less memory — the right inference tradeoff

GQA (kv_heads=2, n_heads=4) uses 4.7% fewer parameters than vanilla MHA and produces essentially identical results: val loss 5.3075 vs 5.3204, move legality 83.0% vs 81.5% — differences well within noise given the stochastic training. The key benefit does not show up in this table: at inference, the KV cache is 2× smaller (one K/V head stored per 2 query heads). This is precisely why LLaMA-2 70B and Mistral 7B adopted GQA — not for quality, but because KV cache is the dominant memory bottleneck when serving large models at scale.

In this project, GQA's cache is stored at the `kv_heads` dimension in the attention module, so the memory saving is real and measurable: with `kv_heads=2` and `n_heads=4`, the cache tensor is half the size of vanilla MHA's. The tradeoff is clear: if deployment memory matters, GQA is a free lunch.

### 3. Sliding window attention needs custom kernels to deliver its theoretical gains

Sliding window attention matches vanilla MHA in both quality (5.3596 vs 5.3204 val loss) and throughput (184K vs 184K tok/s). The result is expected: PyTorch's dense matrix multiply with a `-inf` mask does not skip any FLOPs — it computes the full O(T²) attention matrix and then zeros out the masked positions. The theoretical O(T·W) complexity only materializes with custom CUDA kernels (as used in Longformer and BigBird) or with Flash Attention's tiling approach. This implementation is pedagogically valuable for understanding the mask structure, but production deployments that need long-context efficiency require a lower-level implementation.

### 4. Throughput vs quality is a genuine tradeoff

RoPE is the best model but the slowest at inference (132,760 tok/s vs 184,629 for vanilla). The overhead comes from precomputing and applying the cosine/sine rotation to every Q/K tensor. In practice this is handled by fusing the RoPE application into the attention kernel (which Flash Attention 2 does), but with PyTorch eager mode it adds a measurable overhead. For latency-sensitive deployments, the GQA model (164,689 tok/s, essentially identical quality to vanilla) is a better baseline than vanilla MHA because it delivers the same performance with half the KV cache footprint.

### 5. torch.compile on T4: no speedup

`benchmark.py` measured `torch.compile()` speedup on the vanilla variant: **0.90×** (eager: 187,777 tok/s, compiled: 169,637 tok/s). The T4 does not have enough streaming multiprocessors to use `max_autotune_gemm` mode, so the compiler falls back to a less aggressive kernel that adds overhead without recovering it. This is consistent with PyTorch's documented guidance that `torch.compile` gains are most significant on A100/H100. The flag remains useful for larger GPUs and does not hurt correctness on any hardware.

---

## Scaling Law Experiment

Three model sizes were trained on the same data to check whether validation loss follows a power law in compute (Chinchilla / Hoffmann et al. 2022 style):

| Size | d\_model | Layers | Params | Best Val Loss | Best PPL |
|---|---|---|---|---|---|
| Small | 128 | 4 | 1,400,832 | 5.0671 | 158.72 |
| Medium | 256 | 6 | 5,949,440 | 4.7494 | 115.51 |
| Large | 512 | 6 | 21,336,064 | 4.7844 | 119.63 |

All three sizes trained for 3,000 steps on the same 50K games. Loss decreases from small → medium as expected. The large model (21M params) converges slightly worse than medium (5M) at this compute budget — a known effect where larger models need more data and steps to reach their optimal loss. With 3,000 steps the large model is compute-limited, not data-limited. The script fits a power law L(C) = a · C^b in log space and overlays the fit on a log-log plot (`plots/scaling_laws.png`).

---

## Reproducing

```bash
# Train all 4 variants
for variant in vanilla rope gqa sparse; do
    python train.py --variant $variant --max_steps 3000
done

# Generate benchmark table + plots
python benchmark.py --checkpoint_dir checkpoints

# Scaling experiment
python scale.py --pgn_path data/games.pgn --max_steps 3000
```

Plots are saved to `plots/loss_curves.png`, `plots/benchmark.png`, and `plots/scaling_laws.png`.
