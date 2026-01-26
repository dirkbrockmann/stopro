from __future__ import annotations

import argparse
import time
import numpy as np
from tqdm import trange

from benchmarks.common import add_common_args
from stopro import kimura_replicator


def make_cov(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    C = A @ A.T
    C /= np.max(np.abs(C))
    return C


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args(argv)

    cov = make_cov(args.N, args.seed)

    # warmup
    for _ in trange(args.warmup, desc="kimura_replicator warmup", unit="run", leave=False):
        kimura_replicator(args.T, N=args.N, steps=args.steps, gap=args.gap, samples=args.samples, covariance=cov)

    times: list[float] = []
    for _ in trange(args.repeats, desc="kimura_replicator", unit="run"):
        t0 = time.perf_counter()
        kimura_replicator(args.T, N=args.N, steps=args.steps, gap=args.gap, samples=args.samples, covariance=cov)
        times.append(time.perf_counter() - t0)

    t = np.array(times, dtype=float)
    print(f"N={args.N} samples={args.samples} steps={args.steps} T={args.T} gap={args.gap}")
    print(f"best   : {t.min():.6f} s")
    print(f"median : {np.median(t):.6f} s")
    print(f"mean   : {t.mean():.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())