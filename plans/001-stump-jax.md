# Plan: JAX implementation of STUMPY's `stump` (z-normalized matrix profile)

## Goal
A JAX (GPU-first, CPU-capable) drop-in for `stumpy.stump` that matches its
results (P, I, left/right P & I, top-k) to numerical precision, and is fast
on the local RTX PRO 6000 Blackwell.

## Algorithm choice
STUMPY's numba kernel walks diagonals with an O(1) covariance recurrence.
That is inherently sequential per diagonal and maps poorly to XLA. Instead:

* **Tiled correlation via matmul.** z-normalize sliding windows on the fly
  (`Z = (W - mu) * sigma_inv`, built by gather per tile), then
  `rho_tile = Z_A[rows] @ Z_B[cols].T / m`. Matmul is what XLA/GPUs do best,
  and there is no recurrence drift, so float32 is viable
  (with `precision=HIGHEST` to avoid TF32).
* **Self-join symmetry.** Only upper-triangle tile pairs (c >= r) are
  computed. Each tile updates its rows (right profile + top-k) and its columns
  (left profile + top-k) -> ~2x less work, and every pair (i, j) is visited
  exactly once (no duplicate neighbours in top-k).
* **AB-join.** All tile pairs, row updates only.
* Nested `fori_loop` (outer row tiles, inner column tiles) inside one `jit`;
  row accumulators live in the inner-loop carry, column results are
  `dynamic_update_slice`d into global buffers (in-place in XLA).
* k == 1 uses max/argmax; k > 1 uses `lax.top_k` on [running, candidates].
* Ties -> lowest index (earliest-visited) wins.

## Preprocessing (on device, two-pass so it is exact)
* rolling isfinite (cumsum of non-finite counts), non-finite -> 0
* mean / std per window via chunked gather (two-pass), isconstant = ptp == 0
  (or user-supplied bool array / callable, as stumpy)
* sigma of constant windows -> 1; constant semantics match stumpy:
  both constant -> rho = 1, one constant -> rho = 0.5, non-finite -> -inf.
* float32 mode: subtract global nanmean on host in float64 first.

## Precision / dtype
* float64 when `jax_enable_x64` is on (default for exact stumpy parity),
  float32 otherwise or on request. FP64 on this GPU is ~1/64 rate, so
  float32 is the fast path; report error vs stumpy in benchmarks.

## API (pre-1.0, not bound by stumpy's return type)
`stump(T_A, m, T_B=None, ignore_trivial=True, k=1, T_A_subseq_isconstant=None,
T_B_subseq_isconstant=None, tile_size=None, dtype=None) -> MatrixProfile`
NamedTuple (P, I, left_P, left_I, right_P, right_I) with
`.to_stumpy_array()` giving stumpy's (l, 2k+2) object layout.

## Testing / benchmarking
* pytest tests vs `stumpy.stump` (self-join, AB-join, top-k, NaN/inf,
  constant subsequences, odd lengths not a multiple of tile size).
* `benchmarks/bench_stump.py`: stumpy (numba CPU), stumpy.gpu_stump (numba-cuda), jax f32/f64.

## Later
non-normalized (`aamp`), multi-GPU sharding, `mstump`, diagonal-recurrence
kernel for very large m in fp64.

## Status (2026-11-07)
Implemented in `src/stumpy_jax/stump.py`; tests in `tests/test_stump.py` (12 pass);
benchmark table in README. Default tiles 2048 x 4096. Index differences vs stumpy
arise only from ties (constant subsequences), where we pick the lowest index.
float32 reaches ~9 TFLOPS effective (n=2**16, m=100), well below peak: the
elementwise epilogue / reductions dominate -> candidate for a Pallas kernel.
