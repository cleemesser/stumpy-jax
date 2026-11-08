"""Compare stumpy_jax.stump against stumpy.stump.

Run with ``pytest tests`` or directly with ``python tests/test_stump.py``.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

import stumpy
import stumpy_jax

# small tiles so that multi-tile code paths, padding and partial diagonal tiles
# are exercised on short series
TILES = {"tile_rows": 128, "tile_cols": 256}


def naive_distance(T_A, T_B, m, ii, jj):
    """z-normalized distance between T_A[ii:ii+m] and T_B[jj:jj+m] (stumpy semantics)."""
    a, b = T_A[ii : ii + m], T_B[jj : jj + m]
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.inf
    a_const, b_const = np.ptp(a) == 0, np.ptp(b) == 0
    if a_const and b_const:
        return 0.0
    if a_const or b_const:
        return np.sqrt(m)
    rho = min(1.0, np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))
    return np.sqrt(abs(2 * m * (1 - rho)))


def check_indices(T_A, T_B, m, P, I):
    """Each reported index must actually achieve the reported distance (robust to ties)."""
    for ii in range(P.shape[0]):
        if I[ii] < 0:
            assert np.isinf(P[ii])
            continue
        np.testing.assert_allclose(
            naive_distance(T_A, T_B, m, ii, I[ii]), P[ii], atol=1e-7
        )


def make_series(n, seed=0, nan=True, constant=True):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal(n).cumsum()
    if constant:
        T[n // 5 : n // 5 + 60] = 3.0
    if nan:
        T[n // 2] = np.nan
        T[2 * n // 3] = np.inf
    return T


@pytest.mark.parametrize("n, m", [(700, 20), (1000, 50), (513, 3)])
@pytest.mark.parametrize("k", [1, 3])
def test_self_join(n, m, k):
    T = make_series(n)
    ref = stumpy.stump(T, m, k=k)
    out = stumpy_jax.stump(T, m, k=k, **TILES)
    ref_P = ref[:, :k].astype(np.float64).reshape(-1, k)
    np.testing.assert_allclose(out.P, ref_P, atol=1e-8)
    for kk in range(k):
        check_indices(T, T, m, out.P[:, kk], out.I[:, kk])
    check_indices(T, T, m, out.left_P, out.left_I)
    check_indices(T, T, m, out.right_P, out.right_I)
    # left neighbours are to the left, right neighbours to the right
    valid = out.left_I >= 0
    assert (out.left_I[valid] < np.arange(len(out.left_I))[valid]).all()
    valid = out.right_I >= 0
    assert (out.right_I[valid] > np.arange(len(out.right_I))[valid]).all()
    # indices can differ from stumpy only through ties (e.g. the constant region);
    # exact agreement without ties is checked in test_left_right_profiles_match_stumpy
    ref_I = ref[:, k : 2 * k].astype(np.int64)
    print(
        f"n={n} m={m} k={k}: index agreement with stumpy {(ref_I == out.I).mean():.4f}"
    )


@pytest.mark.parametrize("k", [1, 2])
def test_ab_join(k):
    T_A = make_series(900, seed=1)
    T_B = make_series(650, seed=2, constant=False)
    m = 30
    ref = stumpy.stump(T_A, m, T_B, ignore_trivial=False, k=k)
    out = stumpy_jax.stump(T_A, m, T_B, ignore_trivial=False, k=k, **TILES)
    np.testing.assert_allclose(
        out.P, ref[:, :k].astype(np.float64).reshape(-1, k), atol=1e-8
    )
    for kk in range(k):
        check_indices(T_A, T_B, m, out.P[:, kk], out.I[:, kk])
    assert (out.left_I == -1).all() and (out.right_I == -1).all()
    assert np.isinf(out.left_P).all() and np.isinf(out.right_P).all()


def test_left_right_profiles_match_stumpy():
    T = make_series(800, seed=3, constant=False)
    m = 25
    ref = stumpy.stump(T, m)
    out = stumpy_jax.stump(T, m, **TILES)
    np.testing.assert_array_equal(out.left_I, ref.left_I_)
    np.testing.assert_array_equal(out.right_I, ref.right_I_)
    np.testing.assert_array_equal(out.I[:, 0], ref.I_)


def test_default_tiles_and_stumpy_layout():
    T = make_series(5000, seed=4, nan=False, constant=False)
    m = 100
    ref = stumpy.stump(T, m)
    out = stumpy_jax.stump(T, m)
    arr = out.to_stumpy_array()
    assert arr.shape == ref.shape
    np.testing.assert_allclose(
        arr[:, 0].astype(float), ref[:, 0].astype(float), atol=1e-8
    )
    np.testing.assert_array_equal(
        arr[:, 1:].astype(np.int64), ref[:, 1:].astype(np.int64)
    )


def test_float32():
    T = make_series(3000, seed=5, nan=False, constant=False) + 1e4  # large offset
    m = 64
    ref = stumpy.stump(T, m)
    out = stumpy_jax.stump(T, m, dtype="float32")
    err = np.abs(out.P[:, 0] - ref.P_)
    print(
        f"float32: max |dP| = {err.max():.2e}, index agreement {(out.I[:, 0] == ref.I_).mean():.4f}"
    )
    assert err.max() < 1e-2


def test_isconstant_override():
    T = make_series(600, seed=6, nan=False, constant=False)
    m = 20
    isconstant = np.zeros(len(T) - m + 1, dtype=bool)
    isconstant[::50] = True
    ref = stumpy.stump(T, m, T_A_subseq_isconstant=isconstant)
    out = stumpy_jax.stump(T, m, T_A_subseq_isconstant=isconstant, **TILES)
    np.testing.assert_allclose(out.P[:, 0], ref.P_, atol=1e-8)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
