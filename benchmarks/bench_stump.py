"""Benchmark stumpy_jax.stump against stumpy.stump (numba, CPU) and stumpy.gpu_stump
(numba-cuda, float64).

usage: python benchmarks/bench_stump.py [--n 16384 65536] [--m 100] [--tiles 1024x2048 ...]
"""

import argparse
import time
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

import stumpy_jax

# numba-cuda 0.30 still registers an overload for np.row_stack, which numpy 2.5
# removed (it was an alias of vstack); needed before stumpy.gpu_stump is imported
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack


def timeit(fn, repeat=3):
    fn()  # compile / warm up
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[2**14, 2**16, 2**17])
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--tiles", nargs="+", default=[None], help="ROWSxCOLS")
    parser.add_argument("--no-stumpy", action="store_true")
    parser.add_argument("--stumpy-max-n", type=int, default=2**17)
    parser.add_argument("--no-gpu-stump", action="store_true")
    parser.add_argument("--gpu-stump-max-n", type=int, default=2**18)
    args = parser.parse_args()

    for n in args.n:
        T = np.random.default_rng(0).standard_normal(n).cumsum()
        ref = None
        if not args.no_stumpy and n <= args.stumpy_max_n:
            import stumpy

            t, ref = timeit(partial(stumpy.stump, T, args.m), repeat=1)
            print(f"n={n:>8} m={args.m} stumpy (numba cpu)   {t:8.3f} s", flush=True)
        if not args.no_gpu_stump and n <= args.gpu_stump_max_n:
            import stumpy

            t, out = timeit(partial(stumpy.gpu_stump, T, args.m), repeat=1)
            msg = f"n={n:>8} m={args.m} stumpy.gpu_stump      {t:8.3f} s"
            if ref is not None:
                err = np.abs(out.P_ - ref.P_).max()
                msg += f"  max|dP|={err:.1e}  I agree={(out.I_ == ref.I_).mean():.4f}"
            print(msg, flush=True)
        for tiles in args.tiles:
            kw = {}
            if tiles:
                rows, cols = (int(x) for x in tiles.split("x"))
                kw = {"tile_rows": rows, "tile_cols": cols}
            for dtype in ("float32", "float64"):
                t, out = timeit(partial(stumpy_jax.stump, T, args.m, dtype=dtype, **kw))
                msg = f"n={n:>8} m={args.m} jax {dtype} {tiles or 'default':>10}  {t:8.3f} s"
                if ref is not None:
                    err = np.abs(out.P[:, 0] - ref.P_).max()
                    msg += f"  max|dP|={err:.1e}  I agree={(out.I[:, 0] == ref.I_).mean():.4f}"
                print(msg, flush=True)


if __name__ == "__main__":
    main()
