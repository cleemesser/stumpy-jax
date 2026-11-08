"""JAX implementation of the z-normalized matrix profile (``stumpy.stump``).

Rather than STUMPY's per-diagonal covariance recurrence (sequential, a poor fit
for XLA), the distance matrix is computed in tiles of Pearson correlations::

    rho[i, j] = Z_A[i] . Z_B[j] / m,   Z[i] = (T[i:i+m] - mu[i]) / sigma[i]

Each tile is a single matmul of z-normalized windows that are gathered on the
fly, so memory stays O(n + tile_rows * tile_cols). For a self-join only tiles on
or above the diagonal are visited; each tile updates both its rows (right
neighbours) and its columns (left neighbours), so every pair (i, j) is computed
exactly once.
"""

import math
import warnings
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

EXCL_ZONE_DENOM = 4
P_NORM_THRESHOLD = 1e-14


class MatrixProfile(NamedTuple):
    """Matrix profile result. ``P``/``I`` have shape ``(l, k)``, the others ``(l,)``."""

    P: np.ndarray
    I: np.ndarray
    left_P: np.ndarray
    left_I: np.ndarray
    right_P: np.ndarray
    right_I: np.ndarray

    def to_stumpy_array(self):
        """Return STUMPY's ``(l, 2k + 2)`` object array layout: P, I, left_I, right_I."""
        l, k = self.P.shape
        out = np.empty((l, 2 * k + 2), dtype=object)
        out[:, :k] = self.P
        out[:, k:] = np.column_stack((self.I, self.left_I, self.right_I))
        return out


def _default_dtype():
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def _host_preprocess(T, m, T_subseq_isconstant, dtype):
    """Host-side cleanup: non-finite handling and user-supplied isconstant."""
    T = np.array(T, dtype=np.float64, copy=True)
    if T.ndim != 1:
        raise ValueError(f"T is {T.ndim}-dimensional and must be 1-dimensional")
    if m <= 2:
        raise ValueError("All window sizes must be greater than or equal to three")
    if m > T.shape[0]:
        raise ValueError(f"The window size must be less than or equal to {T.shape[0]}")
    l = T.shape[0] - m + 1

    finite = np.isfinite(T)
    n_bad = np.concatenate(([0], np.cumsum(~finite)))
    subseq_isfinite = (n_bad[m:] - n_bad[:-m]) == 0
    T[~finite] = np.nan

    if callable(T_subseq_isconstant):
        T_subseq_isconstant = np.asarray(T_subseq_isconstant(T, m))
    if T_subseq_isconstant is not None:
        T_subseq_isconstant = np.asarray(T_subseq_isconstant)
        if T_subseq_isconstant.dtype != np.bool_ or T_subseq_isconstant.shape != (l,):
            raise ValueError(
                f"`T_subseq_isconstant` must be a bool array of shape ({l},)"
            )
        T_subseq_isconstant = T_subseq_isconstant & subseq_isfinite

    T[~finite] = 0.0
    if np.dtype(dtype) != np.float64 and finite.any():
        # z-normalization is offset invariant; removing the global offset in
        # float64 avoids catastrophic cancellation when computing in float32
        T -= T[finite].mean()
    return T, subseq_isfinite, T_subseq_isconstant


def _pad(x, length, value):
    return np.concatenate((x, np.full(length - x.shape[0], value, dtype=x.dtype)))


@partial(jax.jit, static_argnames=("m", "chunk"))
def _window_stats(T, m, chunk):
    """Two-pass per-window mean, std and ptp == 0 for ``T.shape[0] - m + 1`` windows
    (must be a multiple of ``chunk``)."""
    n_chunks = (T.shape[0] - m + 1) // chunk
    win = jnp.arange(chunk)[:, None] + jnp.arange(m)[None, :]

    def stats(start):
        W = lax.dynamic_slice(T, (start,), (chunk + m - 1,))[win]
        mu = W.mean(axis=1)
        sigma = jnp.sqrt(jnp.square(W - mu[:, None]).mean(axis=1))
        return mu, sigma, W.max(axis=1) == W.min(axis=1)

    mu, sigma, isconstant = lax.map(stats, jnp.arange(n_chunks) * chunk)
    return mu.reshape(-1), sigma.reshape(-1), isconstant.reshape(-1)


def _device_preprocess(T, m, isfinite, isconstant, L_pad, dtype):
    """Pad to ``L_pad`` windows and compute (T, mu, sigma_inv, isfinite, isconstant)
    on device."""
    chunk = min(_round_up(max(1, 2**22 // m), 128), L_pad)
    L_stats = _round_up(L_pad, chunk)
    T_d = jnp.asarray(_pad(T, L_stats + m - 1, 0.0), dtype=dtype)
    mu, sigma, ptp_zero = (x[:L_pad] for x in _window_stats(T_d, m, chunk))
    T_d = T_d[: L_pad + m - 1]
    isfinite = jnp.asarray(_pad(isfinite, L_pad, False))
    if isconstant is None:
        isconstant = ptp_zero & isfinite
    else:
        isconstant = jnp.asarray(_pad(isconstant, L_pad, False))
    sigma_inv = 1.0 / jnp.where(isconstant | (sigma == 0), 1.0, sigma)
    return T_d, mu, sigma_inv, isfinite, isconstant


def _merge_topk(vals, idx, cand_vals, cand_idx, k):
    """Merge candidates (R, C) with ids (C,) into running top-k (R, k).

    Ties keep the running (earlier) entry, then the lowest candidate position."""
    if k == 1:
        pos = jnp.argmax(cand_vals, axis=1)
        best = jnp.take_along_axis(cand_vals, pos[:, None], axis=1)
        better = best > vals
        return jnp.where(better, best, vals), jnp.where(
            better, cand_idx[pos][:, None], idx
        )
    all_vals = jnp.concatenate((vals, cand_vals), axis=1)
    all_idx = jnp.concatenate(
        (idx, jnp.broadcast_to(cand_idx, cand_vals.shape)), axis=1
    )
    vals, pos = lax.top_k(all_vals, k)
    return vals, jnp.take_along_axis(all_idx, pos, axis=1)


@partial(
    jax.jit,
    static_argnames=("m", "k", "tile_rows", "tile_cols", "self_join", "excl_zone"),
)
def _stump_kernel(A, B, m, k, tile_rows, tile_cols, self_join, excl_zone):
    """Return the top-k / left / right Pearson correlations and indices.

    ``A`` and ``B`` are (T, mu, sigma_inv, isfinite, isconstant) tuples, padded so
    that the number of windows is a multiple of ``tile_rows`` (A) and ``tile_cols``
    (B). For a self-join ``A is B`` and ``tile_cols % tile_rows == 0``."""
    dtype = A[0].dtype
    L_A, L_B = A[1].shape[0], B[1].shape[0]
    n_row_tiles, n_col_tiles = L_A // tile_rows, L_B // tile_cols
    inv_m = jnp.asarray(1.0 / m, dtype)
    neg_inf = jnp.asarray(-jnp.inf, dtype)
    offs = jnp.arange(m)

    def load(X, start, size):
        T, mu, sigma_inv, isfinite, isconstant = X
        W = lax.dynamic_slice(T, (start,), (size + m - 1,))[
            jnp.arange(size)[:, None] + offs[None, :]
        ]
        mu, sigma_inv, isfinite, isconstant = (
            lax.dynamic_slice(x, (start,), (size,))
            for x in (mu, sigma_inv, isfinite, isconstant)
        )
        return (W - mu[:, None]) * sigma_inv[:, None], isfinite, isconstant

    def pearson(a, b):
        Z_a, fin_a, const_a = a
        Z_b, fin_b, const_b = b
        rho = jnp.minimum(
            jnp.matmul(Z_a, Z_b.T, precision=lax.Precision.HIGHEST) * inv_m, 1.0
        )
        both = const_a[:, None] & const_b[None, :]
        either = const_a[:, None] | const_b[None, :]
        rho = jnp.where(both, 1.0, jnp.where(either, 0.5, rho))
        return jnp.where(fin_a[:, None] & fin_b[None, :], rho, neg_inf)

    def init(n, width):
        return jnp.full((n, width), neg_inf), jnp.full((n, width), -1, dtype=jnp.int32)

    # global state: top-k (only needed separately when k > 1 or for AB-joins),
    # left (self-join columns) and right (self-join rows)
    track_topk = k > 1 or not self_join
    widths = {"topk": k, "left": 1, "right": 1}
    glob = {
        "topk": init(L_A, k) if track_topk else None,
        "left": init(L_A, 1) if self_join else None,
        "right": init(L_A, 1) if self_join else None,
    }

    def row_tile(r, glob):
        r0 = r * tile_rows
        a = load(A, r0, tile_rows)
        row_ids = (r0 + jnp.arange(tile_rows)).astype(jnp.int32)

        def col_tile(c, carry):
            row_state, glob = carry
            c0 = c * tile_cols
            col_ids = (c0 + jnp.arange(tile_cols)).astype(jnp.int32)
            rho = pearson(a, load(B, c0, tile_cols))
            if self_join:
                rho = jnp.where(
                    col_ids[None, :] - row_ids[:, None] > excl_zone, rho, neg_inf
                )
            row_state = {
                key: None if st is None else _merge_topk(*st, rho, col_ids, widths[key])
                for key, st in row_state.items()
            }
            if self_join:
                glob = dict(glob)
                rho_t = rho.T
                for key in ("topk", "left"):
                    if glob[key] is None:
                        continue
                    vals, idx = (
                        lax.dynamic_slice_in_dim(x, c0, tile_cols) for x in glob[key]
                    )
                    vals, idx = _merge_topk(vals, idx, rho_t, row_ids, widths[key])
                    glob[key] = (
                        lax.dynamic_update_slice_in_dim(glob[key][0], vals, c0, 0),
                        lax.dynamic_update_slice_in_dim(glob[key][1], idx, c0, 0),
                    )
            return row_state, glob

        row_state = {
            "topk": init(tile_rows, k) if track_topk else None,
            "right": init(tile_rows, 1) if self_join else None,
        }
        c_start = r0 // tile_cols if self_join else 0
        row_state, glob = lax.fori_loop(
            c_start, n_col_tiles, col_tile, (row_state, glob)
        )

        glob = dict(glob)
        for key in ("topk", "right"):
            if row_state[key] is None:
                continue
            vals, idx = row_state[key]
            if self_join and key == "topk":
                # merge with left contributions already written by column updates
                g_vals, g_idx = (
                    lax.dynamic_slice_in_dim(x, r0, tile_rows) for x in glob[key]
                )
                all_vals = jnp.concatenate((g_vals, vals), axis=1)
                all_idx = jnp.concatenate((g_idx, idx), axis=1)
                vals, pos = lax.top_k(all_vals, k)
                idx = jnp.take_along_axis(all_idx, pos, axis=1)
            glob[key] = (
                lax.dynamic_update_slice_in_dim(glob[key][0], vals, r0, 0),
                lax.dynamic_update_slice_in_dim(glob[key][1], idx, r0, 0),
            )
        return glob

    glob = lax.fori_loop(0, n_row_tiles, row_tile, glob)

    if self_join:
        left, right = glob["left"], glob["right"]
        if track_topk:
            topk = glob["topk"]
        else:
            # k == 1: every left index is smaller than every right index, so
            # preferring left on ties matches lowest-index tie breaking
            use_left = left[0] >= right[0]
            topk = (
                jnp.where(use_left, left[0], right[0]),
                jnp.where(use_left, left[1], right[1]),
            )
    else:
        topk = glob["topk"]
        left = right = init(L_A, 1)

    def to_dist(rho):
        p_norm = jnp.abs(2.0 * m * (1.0 - rho))
        return jnp.sqrt(jnp.where(p_norm < P_NORM_THRESHOLD, 0.0, p_norm))

    return (
        to_dist(topk[0]),
        topk[1],
        to_dist(left[0][:, 0]),
        left[1][:, 0],
        to_dist(right[0][:, 0]),
        right[1][:, 0],
    )


def _round_up(x, multiple):
    return -(-x // multiple) * multiple


def stump(
    T_A,
    m,
    T_B=None,
    ignore_trivial=True,
    k=1,
    T_A_subseq_isconstant=None,
    T_B_subseq_isconstant=None,
    dtype=None,
    tile_rows=None,
    tile_cols=None,
):
    """Compute the z-normalized (top-k) matrix profile with JAX.

    Parameters mirror :func:`stumpy.stump` (``normalize=False``/``p`` are not
    supported). ``dtype`` defaults to float64 when ``jax_enable_x64`` is enabled,
    otherwise float32. ``tile_rows``/``tile_cols`` set the size of each correlation
    tile; ``tile_cols`` is rounded up to a multiple of ``tile_rows``.

    Returns a :class:`MatrixProfile` of NumPy arrays. Distances of subsequences
    without a valid neighbour are ``inf`` with index ``-1``. Ties go to the lowest
    index.
    """
    dtype = jnp.dtype(dtype or _default_dtype())
    if dtype == jnp.float64 and not jax.config.jax_enable_x64:
        raise ValueError(
            "dtype=float64 requires jax.config.update('jax_enable_x64', True)"
        )
    if k < 1:
        raise ValueError("k must be a positive integer")
    m = int(m)

    self_join = T_B is None
    if not self_join and ignore_trivial:
        # stumpy: ignore_trivial with an explicit T_B only applies if T_B is T_A
        self_join = np.array_equal(np.asarray(T_A), np.asarray(T_B), equal_nan=True)
    if self_join:
        T_B_subseq_isconstant = T_A_subseq_isconstant

    T_A, fin_A, const_A = _host_preprocess(T_A, m, T_A_subseq_isconstant, dtype)
    l_A = T_A.shape[0] - m + 1
    excl_zone = math.ceil(m / EXCL_ZONE_DENOM)
    if self_join and l_A // 2 <= excl_zone:
        warnings.warn(
            f"The window size, 'm = {m}', may be too large and could lead to "
            "meaningless results. Consider reducing 'm' where necessary"
        )

    # defaults tuned on an RTX PRO 6000 Blackwell (n = 2**16 .. 2**18, m = 100)
    tile_rows = int(tile_rows or 2048)
    tile_cols = int(tile_cols or 4096)
    if self_join:
        tile_rows = min(tile_rows, _round_up(l_A, 128))
        tile_cols = min(_round_up(tile_cols, tile_rows), _round_up(l_A, tile_rows))
        A = B = _device_preprocess(
            T_A, m, fin_A, const_A, _round_up(l_A, tile_cols), dtype
        )
    else:
        T_B, fin_B, const_B = _host_preprocess(T_B, m, T_B_subseq_isconstant, dtype)
        l_B = T_B.shape[0] - m + 1
        tile_rows = min(tile_rows, _round_up(l_A, 128))
        tile_cols = min(tile_cols, _round_up(l_B, 128))
        A = _device_preprocess(T_A, m, fin_A, const_A, _round_up(l_A, tile_rows), dtype)
        B = _device_preprocess(T_B, m, fin_B, const_B, _round_up(l_B, tile_cols), dtype)

    out = _stump_kernel(
        A,
        B,
        m=m,
        k=int(k),
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        self_join=self_join,
        excl_zone=excl_zone,
    )
    P, I, PL, IL, PR, IR = (np.asarray(x)[:l_A] for x in out)
    return MatrixProfile(
        P.astype(np.float64),
        I.astype(np.int64),
        PL.astype(np.float64),
        IL.astype(np.int64),
        PR.astype(np.float64),
        IR.astype(np.int64),
    )
