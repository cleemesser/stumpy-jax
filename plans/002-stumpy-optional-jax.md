# Plan: JAX as an optional dependency of stumpy

Status: proposal for discussion. Nothing has been changed in the stumpy repo yet.

## How stumpy handles optional backends today

Observed in the `stumpy/` clone at commit 51475d6:

* **Hard dependencies** are only numpy, scipy and numba. The only extras group
  is `ci`.
* **GPU (numba-cuda).** `stumpy/__init__.py` calls
  `numba.cuda.is_available()`. If CUDA is present, the `gpu_*` functions are
  added to `_lazy_imports`, a PEP 562 module `__getattr__` mechanism.
  Otherwise each `gpu_*` name is bound to a dummy (`core._gpu_stump_dnf`) that
  raises a "driver not found" error, and the docstring is copied in from the
  source so the API docs still render.
* **dask / ray.** These are never imported. `stumped` and friends take a client
  object and detect its type at runtime, so neither appears in the package
  dependencies.
* **`normalize=False`.** The `@core.non_normalized(aamp)` decorator redirects
  each z-normalized function to its non-normalized counterpart.

## Proposal

### 1. Packaging
* Add extras to `pyproject.toml`:
  * `jax = ["jax >= X"]` (CPU). The minimum is not chosen yet; it must have
    the scoped `jax.enable_x64` context manager, which I have only verified on
    0.12.
  * `jax-cuda = ["jax[cuda12] >= X"]`

  GPU wheels depend on the platform. The docs should point users to JAX's
  install guide rather than stumpy trying to pick the right wheel (cuda12,
  cuda13, ROCm or TPU).
* Add jax to the conda/pixi environment for CI. conda-forge ships `jax` and
  `jaxlib`.
* jax must **never** be imported by `import stumpy`. JAX takes about 0.5–1 s to
  import, and it reserves GPU memory as soon as it initializes.

### 2. Public API
* Add `stumpy.jax_stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True,
  p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None,
  dtype=np.float64)`. It would return an `mparray`, exactly like `stump`.
  * The name follows the `gpu_stump` / `stumped` pattern, and it is
    backend-neutral because JAX also runs on CPU and TPU.
  * Alternative: a `backend="jax"` keyword on `stump`. That keeps the API
    smaller, but it mixes concerns and is unlike how stumpy exposes its other
    backends.
* Register it in `_lazy_imports`, gated on
  `importlib.util.find_spec("jax") is not None`. If JAX is missing, bind a
  dummy `_jax_stump_dnf` that raises
  `ImportError("jax_stump requires JAX: pip install stumpy[jax]")`. Copy the
  docstring as is already done for the GPU functions.
* `normalize=False`: add `jax_aamp` and use `@core.non_normalized(jax_aamp)`.
  The tiled kernel adapts directly: `-||a - b||^2 = 2 a.b - |a|^2 - |b|^2` for
  p = 2. Other p-norms could fall back to a slower elementwise tile.

### 3. Implementation inside stumpy
* **Reuse stumpy's preprocessing.** Call `core.preprocess_diagonal` on the
  host to get `T`, `mu`, `sigma_inv`, `isfinite` and `isconstant`, then copy
  them to the device. This drops our duplicated host and device preprocessing
  and guarantees identical isconstant semantics, including callables and
  `fix_isconstant_isfinite_conflicts`. The device then only runs the tiled
  kernel.
* **Respect stumpy's config.** Read `config.STUMPY_EXCL_ZONE_DENOM` and
  `STUMPY_P_NORM_THRESHOLD` at call time rather than hard-coding them.
* **Precision without global side effects.** Wrap the call in
  `with jax.enable_x64(True):`, which is scoped (verified with JAX 0.12), so
  the user's global JAX config is left alone. The default should be float64
  for parity with the rest of stumpy and its tests; `dtype=np.float32` is the
  documented fast path on GPUs.
* **Module layout.** `stumpy/jax_stump.py` would hold the public function.
  The XLA kernel and the Pallas kernel would go in a private `stumpy/_jax/`
  package so the Pallas code can be dropped without API churn.
* **Pallas.** Use the Triton Pallas path only when the device is a CUDA GPU
  and k = 1. Use the pure-XLA path everywhere else (CPU, TPU, ROCm, k > 1).
  Tests can run the Pallas kernel on CPU CI with `interpret=True`.

### 4. Tests
* Add `tests/test_jax_stump.py` with `pytest.importorskip("jax")`. Mirror
  `test_stump.py`, comparing against `naive.stump` and `stumpy.stump`, using
  small tiles to exercise the multi-tile paths.
* Index assertions in tests with ties (identical or constant subsequences)
  need care. stumpy's tie-breaking depends on thread layout, and ours always
  picks the lowest index. Either assert distances only there, or make
  `naive.stump`'s tie rule explicit.
* CI: one matrix entry with `stumpy[jax]` on CPU. GPU stays manual or
  self-hosted, as for numba-cuda today.

### 5. Documentation
* Add API reference entries, a section in the install guide, and a short
  "choosing a backend" note covering numba CPU, numba CUDA, dask/ray and JAX,
  with the float32/float64 trade-off and the benchmark table.
* JAX and numba-cuda in one process: JAX preallocates 75% of GPU memory by
  default, so mixing it with `gpu_stump` needs
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Risks and open questions
* **Maintainer appetite.** stumpy has kept its dependencies minimal and favours
  numba throughout. Open a GitHub discussion or issue with the benchmark table
  before writing the PR.
* **Version churn.** The JAX API moves quickly; `pl.dot` was removed in 0.12,
  for example. The Pallas path is most exposed. Pin a minimum version and keep
  the pure-XLA fallback working.
* **Scope.** Start with `jax_stump` / `jax_aamp` only. `mstump`, `stimp`,
  `ostinato` and the others can follow once the pattern is accepted.
* **Alternative: keep `stumpy-jax` as a separate package.** stumpy would only
  document it. Zero burden on stumpy, but discoverability is worse and
  preprocessing gets duplicated. This is a reasonable fallback if maintainers
  decline.

## Proposed steps
1. Open a discussion on stumpy with the benchmarks and this outline.
2. Refactor `stumpy_jax` to take stumpy-preprocessed inputs and return an
   `mparray`, so the code is drop-in ready.
3. Wire in the Pallas k = 1 path and measure end to end.
4. Open the PR: `stumpy/jax_stump.py`, `stumpy/_jax/`, extras, lazy import,
   tests and docs.
