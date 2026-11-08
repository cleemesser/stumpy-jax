"""Fused Pallas (Triton) kernel: Pearson tile -> per-row and per-column best (k = 1).

For a tile of z-normalized windows ``Z_A`` (R, K) and ``Z_B`` (C, K) the kernel
computes ``rho = Z_A @ Z_B.T / m`` block by block in registers, applies the
non-finite / constant / exclusion-zone rules and writes only the max and argmax
of every block row and block column. The (R, C) correlation matrix never touches
global memory.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltr

INT_MAX = 2**31 - 1
FINITE = 1
CONSTANT = 2


def _argbest(rho, best, ids, axis):
    """Lowest id achieving ``best`` along ``axis`` (ties -> lowest index)."""
    hit = rho == jnp.expand_dims(best, axis)
    return jnp.min(jnp.where(hit, jnp.expand_dims(ids, 1 - axis), INT_MAX), axis=axis)


def _kernel(
    za_ref,
    zb_ref,
    ida_ref,
    fa_ref,
    idb_ref,
    fb_ref,
    *out_refs,
    m,
    excl_zone,
    self_join,
    block_k,
    precision,
):
    block_m, block_n = za_ref.shape[0], zb_ref.shape[0]
    n_k = za_ref.shape[1] // block_k

    def body(kk, acc):
        a = za_ref[:, pl.ds(kk * block_k, block_k)]
        b = zb_ref[:, pl.ds(kk * block_k, block_k)]
        return acc + pltr.dot(a, b, trans_b=True, precision=precision)

    acc = lax.fori_loop(0, n_k, body, jnp.zeros((block_m, block_n), za_ref.dtype))
    rho = jnp.minimum(acc * (1.0 / m), 1.0)

    ida, idb = ida_ref[...], idb_ref[...]
    fa, fb = fa_ref[...], fb_ref[...]
    const_a = ((fa & CONSTANT) != 0)[:, None]
    const_b = ((fb & CONSTANT) != 0)[None, :]
    rho = jnp.where(const_a & const_b, 1.0, jnp.where(const_a | const_b, 0.5, rho))
    valid = ((fa & FINITE) != 0)[:, None] & ((fb & FINITE) != 0)[None, :]
    if self_join:
        valid = valid & (idb[None, :] - ida[:, None] > excl_zone)
    rho = jnp.where(valid, rho, -jnp.inf)

    row_v, row_i = out_refs[0], out_refs[1]
    best = jnp.max(rho, axis=1)
    row_v[...] = best[None, :]
    row_i[...] = _argbest(rho, best, idb, axis=1)[None, :]
    if self_join:
        col_v, col_i = out_refs[2], out_refs[3]
        best = jnp.max(rho, axis=0)
        col_v[...] = best[None, :]
        col_i[...] = _argbest(rho, best, ida, axis=0)[None, :]


@partial(
    jax.jit,
    static_argnames=(
        "m",
        "excl_zone",
        "self_join",
        "block_m",
        "block_n",
        "block_k",
        "num_warps",
        "num_stages",
        "precision",
    ),
)
def tile_best(
    Z_a,
    Z_b,
    ids_a,
    flags_a,
    ids_b,
    flags_b,
    *,
    m,
    excl_zone,
    self_join,
    block_m=64,
    block_n=64,
    block_k=32,
    num_warps=4,
    num_stages=2,
    precision=lax.Precision.HIGHEST,
):
    """Return per-block row bests ``(C // block_n, R)`` (values, ids) and, for a
    self-join, per-block column bests ``(R // block_m, C)``.

    ``Z_a``/``Z_b`` must have the same number of columns, a multiple of ``block_k``
    (zero padding is harmless); ``R``/``C`` must be multiples of ``block_m``/``block_n``.
    ``flags`` hold ``FINITE | CONSTANT`` bits per window."""
    R, K = Z_a.shape
    C = Z_b.shape[0]
    n_bm, n_bn = R // block_m, C // block_n
    dtype = Z_a.dtype

    out_shape = [
        jax.ShapeDtypeStruct((n_bn, R), dtype),
        jax.ShapeDtypeStruct((n_bn, R), jnp.int32),
    ]
    out_specs = [
        pl.BlockSpec((1, block_m), lambda ii, jj: (jj, ii)),
        pl.BlockSpec((1, block_m), lambda ii, jj: (jj, ii)),
    ]
    if self_join:
        out_shape += [
            jax.ShapeDtypeStruct((n_bm, C), dtype),
            jax.ShapeDtypeStruct((n_bm, C), jnp.int32),
        ]
        out_specs += [
            pl.BlockSpec((1, block_n), lambda ii, jj: (ii, jj)),
            pl.BlockSpec((1, block_n), lambda ii, jj: (ii, jj)),
        ]

    row_spec = pl.BlockSpec((block_m,), lambda ii, jj: (ii,))
    col_spec = pl.BlockSpec((block_n,), lambda ii, jj: (jj,))
    return pl.pallas_call(
        partial(
            _kernel,
            m=m,
            excl_zone=excl_zone,
            self_join=self_join,
            block_k=block_k,
            precision=precision,
        ),
        grid=(n_bm, n_bn),
        in_specs=[
            pl.BlockSpec((block_m, K), lambda ii, jj: (ii, 0)),
            pl.BlockSpec((block_n, K), lambda ii, jj: (jj, 0)),
            row_spec,
            row_spec,
            col_spec,
            col_spec,
        ],
        out_specs=out_specs,
        out_shape=out_shape,
        compiler_params=pltr.CompilerParams(num_warps=num_warps, num_stages=num_stages),
    )(Z_a, Z_b, ids_a, flags_a, ids_b, flags_b)


def reduce_blocks(vals, ids):
    """Collapse per-block bests ``(n_blocks, N)`` to ``(N,)``; ties -> first block."""
    pos = jnp.argmax(vals, axis=0)
    take = partial(jnp.take_along_axis, indices=pos[None, :], axis=0)
    return take(vals)[0], take(ids)[0]
