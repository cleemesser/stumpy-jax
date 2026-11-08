# Add a JAX implementation of `stump` (z-normalized matrix profile)

## Summary

This adds `stumpy_jax.stump`, a JAX implementation of `stumpy.stump` that runs
on GPU and CPU. It computes the same z-normalized (top-k) matrix profile: P, I,
and the left/right P and I, with stumpy's NaN/inf, constant-subsequence and
exclusion-zone semantics.

On an RTX PRO 6000 Blackwell it is **14–30× faster than `stumpy.stump` on 24
CPU threads in float32**, and **3–5× faster in float64** at m = 100. It is also
faster than `stumpy.gpu_stump` in every configuration measured.

## Approach

`stumpy.stump` uses an O(1) per-step covariance update along each diagonal.
That is sequential and maps poorly to XLA. Instead, the distance matrix is
built from tiles of Pearson correlations. Each tile is one matmul of windows
that are z-normalized as the tile is computed:

    rho[i, j] = Z_A[i] . Z_B[j] / m

* Memory is `O(n + tile_rows * tile_cols)`; the `l x m` window matrix is never
  stored.
* Self-joins visit only the upper triangle. Row reductions give right
  neighbours and column reductions give left neighbours, so each pair is
  computed once and top-k has no duplicates.
* Correlations are computed directly, with no recurrence, so float32 is
  accurate. Matmuls use `precision=HIGHEST` (TF32 off).
* The whole computation is one `jit`, with nested `fori_loop`s over row and
  column tiles and in-place `dynamic_update_slice` for column results.

Full details are in `docs/implementation.md`.

## API

```python
mp = stumpy_jax.stump(T_A, m, T_B=None, ignore_trivial=True, k=1,
                      T_A_subseq_isconstant=None, T_B_subseq_isconstant=None,
                      dtype=None, tile_rows=None, tile_cols=None)
mp.P, mp.I, mp.left_P, mp.left_I, mp.right_P, mp.right_I
mp.to_stumpy_array()   # stumpy's (l, 2k + 2) object layout
```

`dtype` defaults to float64 when `jax_enable_x64` is on, and float32 otherwise.

## Benchmarks

The hardware was an RTX PRO 6000 Blackwell (FP32:FP64 = 64:1) and a 24-thread
CPU. Compile time is excluded and inputs are random walks.

| n | m | stumpy (numba cpu) | stumpy.gpu_stump | jax float32 | jax float64 |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 100 | 0.12 s | 0.70 s | 0.004 s | 0.025 s |
| 65,536 | 100 | 0.86 s | 2.79 s | 0.039 s | 0.29 s |
| 131,072 | 100 | 3.31 s | 5.60 s | 0.13 s | 1.12 s |
| 262,144 | 100 | n/a | 11.2 s | 0.50 s | 4.4 s |
| 65,536 | 1000 | 1.37 s | 2.83 s | 0.10 s | 2.2 s |

Accuracy against `stumpy.stump`: float64 distances match to about 1e-11 and
float32 to 4e-5 (m = 100). Where there are no ties, indices are identical.

## Testing

* `tests/test_stump.py` has 12 tests against `stumpy.stump`. They use small
  tiles so that padding, multiple tiles and partially masked diagonal tiles are
  exercised.
* Cases covered: self-join and AB-join, k = 1 and k = 3, `m = 3`, NaN, inf and
  constant regions, a custom isconstant mask, and float32 with a large offset.
* Every reported index is checked against a naive distance computation.
* `benchmarks/bench_stump.py` reproduces the table above.
* `uv run pytest tests` passes and `ruff check` is clean.

## Known differences and limitations

* **Ties.** The lowest index wins. stumpy's choice depends on how its threads
  split the diagonals, so indices can differ where several neighbours are
  equally close, such as around constant subsequences. Distances are unaffected.
* **Return type.** It returns a `MatrixProfile` NamedTuple, not stumpy's
  `mparray`; `.to_stumpy_array()` gives stumpy's layout. Pre-1.0, so this may
  change.
* **float64 at large m.** It is slower than numba at m = 1000 on GPUs with weak
  FP64, because the matmul does O(m) work per pair. Use float32 there.
* **Not implemented:** `normalize=False` (`aamp`), `mstump`, `stumpi`,
  `scrump`, multi-GPU.
* **Fused Pallas kernel.** A fused Pallas (Triton) float32 kernel is included in
  `src/stumpy_jax/_pallas.py`. Standalone it is 1.6× faster per tile using
  TF32 x3, with no loss of accuracy, but it is not yet used by `stump`. That is
  follow-up work.
* **Benchmark dependencies.** The `gpu_stump` comparison needs the dev
  dependency `numba-cuda[cu13]`. The benchmark adds an `np.row_stack` shim,
  because numba-cuda 0.30 still uses `np.row_stack`, which numpy 2.5 removed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
