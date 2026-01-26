from __future__ import annotations

import argparse
import time
import numpy as np

from stopro import kimura_replicator


def make_cov(N: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N))
    C = A @ A.T
    # scale so values are well-conditioned-ish (optional but helpful)
    C /= np.max(np.abs(C))
    return C


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--gap", type=int, default=1)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cov = make_cov(args.N, args.seed)

    # warmup
    for _ in range(args.warmup):
        kimura_replicator(args.T, N=args.N, steps=args.steps, gap=args.gap, samples=args.samples, covariance=cov)

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        kimura_replicator(args.T, N=args.N, steps=args.steps, gap=args.gap, samples=args.samples, covariance=cov)
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    print(f"N={args.N} samples={args.samples} steps={args.steps} T={args.T} gap={args.gap}")
    print(f"best   : {t.min():.6f} s")
    print(f"median : {np.median(t):.6f} s")
    print(f"mean   : {t.mean():.6f} s")
    return 0