# stumpy-jax

A JAX implementation of [STUMPY](https://github.com/stumpy-dev/stumpy)'s
z-normalized matrix profile (`stumpy.stump`): self-joins, AB-joins, top-k,
left/right profiles, NaN/inf and constant-subsequence handling.

```python
import jax
jax.config.update("jax_enable_x64", True)   # optional: float64 for exact stumpy parity
import stumpy_jax

mp = stumpy_jax.stump(T, m=100)              # MatrixProfile(P, I, left_P, left_I, right_P, right_I)
mp.P[:, 0], mp.I[:, 0]
mp.to_stumpy_array()                         # stumpy's (l, 2k+2) object layout
```

`dtype` defaults to float64 when x64 is enabled, else float32 (pass `dtype=` to override).

## How it works
The distance matrix is computed in tiles of Pearson correlations, each a single
matmul of on-the-fly z-normalized windows (`precision=HIGHEST`, no TF32). Self-joins
visit only the upper triangle, updating rows (right profile) and columns (left
profile). See `plans/001-stump-jax.md`.

## Benchmarks (RTX PRO 6000 Blackwell; stumpy on 24 CPU threads; random walk)

| n | m | stumpy (numba cpu) | stumpy.gpu_stump | jax float32 | jax float64 |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 100 | 0.12 s | 0.70 s | 0.004 s | 0.025 s |
| 65,536 | 100 | 0.86 s | 2.79 s | 0.039 s | 0.29 s |
| 131,072 | 100 | 3.31 s | 5.60 s | 0.13 s | 1.12 s |
| 262,144 | 100 | — | 11.2 s | 0.50 s | 4.4 s |
| 65,536 | 1000 | 1.37 s | 2.83 s | 0.10 s | 2.2 s |

Compile/JIT time is excluded for every method. float64 matches stumpy to ~1e-11
(`gpu_stump` to ~1e-9); float32 to ~1e-4 in distance. `gpu_stump` (float64) launches
one kernel per row, so at these sizes it is launch-bound: linear in n and
independent of m. The JAX cost scales with `m` (matmul), so on GPUs with weak
FP64 prefer float32 for large windows.

The `gpu_stump` comparison needs the dev dependency `numba-cuda[cu13]`; the
benchmark shims `np.row_stack` (removed in numpy 2.5, still referenced by
numba-cuda 0.30).

```
uv run pytest tests            # correctness vs stumpy
uv run python benchmarks/bench_stump.py --n 65536 --m 100
```
