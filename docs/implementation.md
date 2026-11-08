# stumpy-jax: implementation approach and benchmarks

`stumpy_jax.stump` computes the same z-normalized (top-k) matrix profile as
`stumpy.stump`: self-joins and AB-joins, top-k, left/right profiles, NaN/inf
handling, constant subsequences and a user-supplied `T_subseq_isconstant`.

## 1. Why not port the numba kernel directly

`stumpy.stump` walks each diagonal of the distance matrix with an O(1)
covariance update (Zhu et al. 2016 / Zimmerman et al. 2019). Each step depends
on the previous one, and parallelism comes from giving each CPU thread a
contiguous range of diagonals. Under XLA that becomes a `lax.scan` with tens of
thousands of short sequential steps. It is limited by memory bandwidth and
per-step overhead, and the recurrence drifts in float32.

## 2. Approach: tiled correlations as matmuls

For windows of length `m`, with `Z[i] = (T[i:i+m] - mu[i]) / sigma[i]`:

    rho[i, j] = Z_A[i] . Z_B[j] / m          D[i, j] = sqrt(2 m (1 - rho[i, j]))

* **Tiles.** The distance matrix is split into `tile_rows x tile_cols` tiles
  (default 2048 x 4096). Each tile is one matmul,
  `Z_A[rows] @ Z_B[cols].T`. The windows are gathered from `T` and
  z-normalized as each tile is computed, so memory is
  `O(n + tile_rows * tile_cols)`. The full `l x m` window matrix is never stored.
* **No drift.** Every correlation is computed directly rather than by
  recurrence, so float32 is usable. Matmuls use `precision=HIGHEST`, because
  XLA would otherwise use TF32 on Ampere and later GPUs, which keeps only a
  10-bit mantissa.
* **Self-join symmetry.** Only tiles on or above the diagonal are visited. A
  tile's row reductions give *right* neighbours, and its column reductions give
  *left* neighbours. Each pair (i, j) is therefore computed exactly once, which
  halves the work and means top-k never contains duplicates. Tiles on the
  diagonal mask `j - i <= excl_zone`.
* **AB-join.** All tiles are visited and only rows are updated. Left and right
  profiles are `inf` / `-1`, as in stumpy.
* **Control flow.** The whole computation is one `jit`. An outer `fori_loop`
  runs over row tiles and an inner loop over column tiles. Row accumulators
  live in the inner-loop carry. Column results are written to global buffers
  with `dynamic_update_slice`, which XLA does in place.
* **k = 1 fast path.** For a self-join only the left and right bests are
  tracked, and `P = max(left, right)`. On ties the left neighbour is preferred,
  because every left index is smaller than every right index. For `k > 1`,
  `lax.top_k` runs over `[running, candidates]`.
* **stumpy semantics.**
  * A window containing NaN or inf gets `rho = -inf`, so `P = inf` and `I = -1`.
  * Both windows constant gives `rho = 1`; one constant gives `rho = 0.5`.
  * Constant windows get `sigma := 1`.
  * `rho` is clipped to 1, and `p_norm < 1e-14` becomes 0.
  * The exclusion zone is `ceil(m / 4)`.
* **Preprocessing.** Rolling isfinite is computed on the host from cumulative
  sums. Per-window mean, standard deviation and the constant check (max == min)
  are computed on the device in two passes, so they are exact. In float32 mode
  the global mean is subtracted on the host in float64 first, to avoid
  cancellation for series with a large offset.
* **Ties.** The lowest index wins. stumpy's choice depends on how its threads
  split the diagonals, so the two can differ only where several neighbours are
  equally close (for example around constant subsequences).

## 3. Correctness

`tests/test_stump.py` (12 tests) compares against `stumpy.stump` 1.14.1. It
uses small tiles so that padding, multiple tiles and partially masked diagonal
tiles are all exercised. Cases covered:

* self-join and AB-join, k = 1 and k = 3, `m = 3`
* NaN, inf and constant regions
* a custom isconstant mask
* float32 with a 1e4 offset

Every reported index is re-checked with a naive distance computation, so
differences caused by ties cannot hide a real error. Where there are no ties,
P, I, left_I and right_I equal stumpy's exactly.

| mode | max \|P - P_stumpy\| | index agreement (no ties) |
|---|---|---|
| float64 | 1e-12 to 1e-11 | 100% |
| float32 | 3e-5 to 4e-5 (m = 100), 2e-4 (m = 1000) | 100% (m = 100), 99.8% (m = 1000) |

## 4. Benchmarks

The hardware was an NVIDIA RTX PRO 6000 Blackwell Workstation Edition
(sm_120, 96 GB, FP32:FP64 = 64:1) and a 24-thread CPU. The software was
JAX 0.12.0.1 with CUDA 13, stumpy 1.14.1, numba 0.68.1 and numba-cuda 0.30.4.
Inputs are random walks. Compile and JIT time is excluded: stumpy has one
warm-up call and one timed run; JAX has one warm-up call and the best of 3.

| n | m | stumpy (numba, 24 threads) | stumpy.gpu_stump | jax float32 | jax float64 |
|---:|---:|---:|---:|---:|---:|
| 16,384 | 100 | 0.12 s | 0.70 s | **0.004 s** | 0.025 s |
| 65,536 | 100 | 0.86 s | 2.79 s | **0.039 s** | 0.29 s |
| 131,072 | 100 | 3.31 s | 5.60 s | **0.13 s** | 1.12 s |
| 262,144 | 100 | n/a | 11.2 s | **0.50 s** | 4.4 s |
| 65,536 | 1000 | 1.37 s | 2.83 s | **0.10 s** | 2.2 s |

Observations:

* **float32 against stumpy.** float32 is 14–30× faster than numba on the CPU (22–30× at m = 100)
  and 22–175× faster than `gpu_stump`.
* **float64 against stumpy.** float64 is 3–5× faster than numba at m = 100. At
  m = 1000 it is slower than numba, because the matmul does O(m) work per pair
  while the diagonal recurrence does O(1). With the card's 1/64-rate FP64 the
  float64 path sustains about 1.4 TFLOPS, so it is limited by FP64
  throughput.
* **`gpu_stump` is launch-bound.** It launches one kernel per row, so at these
  sizes its time is linear in n and independent of m.
* **Tile size.** Tile size matters most for small tiles: at n = 65,536,
  512 x 512 tiles took 0.27 s against 0.047 s for 2048 x 4096. Tile sizes from
  1024 x 4096 to 4096 x 4096 are within 5% of each other at that size. At
  n = 262,144, 2048 x 4096 was the best of those tried.

## 5. In progress: fused Pallas kernel for float32

**Profile of the current XLA path.** One 2048 x 4096 tile (m = 100) takes
0.047 ms for the cuBLAS matmul and 0.071 ms with the mask and max/argmax work
added. Across the 272 tiles at n = 65,536 that comes to about 19 ms of the
39 ms total. The rest is per-tile overhead: gathering windows, merging results,
updating slices and loop control.

**Kernel.** `src/stumpy_jax/_pallas.py` is a Triton-backend Pallas kernel.
Mosaic GPU targets sm_90a/sm_100a, so it is not an option on sm_120. Each
program computes one `BM x BN` block of `rho` in registers, with a loop over K
in chunks of 32. It then applies the finite, constant and exclusion masks and
writes only the block's per-row and per-column max/argmax. The `R x C`
correlation matrix never reaches global memory.

**Standalone results** on the same 2048 x 4096 tile:

| dot precision | best config (BM, BN, BK, warps, stages) | time | max \|rho error\| vs float64 |
|---|---|---:|---:|
| XLA path (cuBLAS HIGHEST + epilogue) | n/a | 0.071 ms | n/a |
| IEEE float32 (`HIGHEST`) | 128, 64, 32, 4, 3 | 0.064 ms | 4.8e-7 |
| **TF32 x3** (tensor cores) | 64, 64, 32, 4, 3 | **0.045 ms** | 3.2e-7 |
| BF16 x6 | 64, 128, 32, 4, 3 | 0.049 ms | 2.2e-7 |
| BF16 x9 | 64, 128, 32, 4, 3 | 0.060 ms | 2.2e-7 |

TF32 x3 is 1.6× faster per tile than the XLA path with no loss of accuracy,
and the argmax indices agree 100%. The kernel is **not yet wired into
`stump`**. Remaining steps:

1. Use it for k = 1 on GPU.
2. Skip blocks that lie wholly below the diagonal.
3. Use larger super-tiles to amortize loop overhead.
4. Measure end to end.

Most of the expected gain is from removing per-tile overhead, so the end-to-end
speedup could be larger or smaller than 1.6×.

## 6. Not yet implemented

* `normalize=False` (`aamp`, non-normalized distances with p-norm)
* `mstump`, `stumpi`, `scrump`
* multi-GPU sharding
* a diagonal-recurrence kernel for float64 with large m
